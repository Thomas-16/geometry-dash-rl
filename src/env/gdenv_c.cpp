#include "env/gdenv_c.h"

#include "core/level.hpp"
#include "core/physics.hpp"
#include "env/contracts.hpp"
#include "env/episode.hpp"
#include "env/obs.hpp"
#include "env/solve.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string g_lastError;

// Persistent per-handle worker pool for the env loops. Envs are fully
// independent (each touches only its own EnvInstance and disjoint output
// rows; LevelGeometry/PhysicsConfig are read-only), and no float op crosses
// envs, so parallel stepping is bit-identical to serial. Work is handed out
// in small dynamic chunks because macro steps span 4-104 ticks — static
// partitioning imbalances badly.
class ThreadPool {
public:
    // nThreads counts the calling thread too: nThreads <= 1 spawns no
    // workers and run() executes inline (the exact serial fallback).
    explicit ThreadPool(int nThreads) {
        for (int w = 0; w < nThreads - 1; ++w)
            workers_.emplace_back([this] { workerLoop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cvStart_.notify_all();
        for (auto& t : workers_) t.join();
    }

    void run(int n, const std::function<void(int)>& body) {
        if (workers_.empty()) {
            for (int i = 0; i < n; ++i) body(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            job_ = &body;
            jobN_ = n;
            next_.store(0, std::memory_order_relaxed);
            pending_ = static_cast<int>(workers_.size());
            ++jobId_;
        }
        cvStart_.notify_all();
        work(n, body); // the calling thread participates
        std::unique_lock<std::mutex> lock(m_);
        cvDone_.wait(lock, [this] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    static constexpr int kChunk = 4;

    void work(int n, const std::function<void(int)>& body) {
        for (;;) {
            int i0 = next_.fetch_add(kChunk, std::memory_order_relaxed);
            if (i0 >= n) break;
            int i1 = std::min(n, i0 + kChunk);
            for (int i = i0; i < i1; ++i) body(i);
        }
    }

    void workerLoop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* job;
            int n;
            {
                std::unique_lock<std::mutex> lock(m_);
                cvStart_.wait(lock,
                              [&] { return stop_ || jobId_ != seen; });
                if (stop_) return;
                seen = jobId_;
                job = job_;
                n = jobN_;
            }
            work(n, *job);
            {
                std::lock_guard<std::mutex> lock(m_);
                if (--pending_ == 0) cvDone_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cvStart_, cvDone_;
    const std::function<void(int)>* job_ = nullptr;
    int jobN_ = 0;
    std::atomic<int> next_{0};
    int pending_ = 0;
    uint64_t jobId_ = 0;
    bool stop_ = false;
};

int resolveThreadCount() {
    if (const char* env = std::getenv("GDENV_THREADS")) {
        int v = std::atoi(env);
        if (v >= 1) return v;
    }
    unsigned hc = std::thread::hardware_concurrency();
    return hc > 1 ? static_cast<int>(hc) - 1 : 1;
}

const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

std::string resolveLevelPath(const char* arg) {
    if (arg && *arg) return arg;
    if (const char* env = std::getenv("GD_LEVEL_PATH")) return env;
    return kDefaultLevel;
}

// Versioned snapshot blob (contracts.hpp): a foreign blob (older layout,
// different sim version) is rejected by set_state instead of being
// reinterpreted.
struct StateBlob {
    uint32_t magic = gdenv::kSnapshotMagic;
    uint32_t version = static_cast<uint32_t>(gdenv::kSnapshotVersion);
    uint32_t simVersion = static_cast<uint32_t>(gdenv::kSimVersion);
    uint32_t reserved = 0;
    gdcore::PlayerState s;
    int32_t tick = 0;
    int32_t pad = 0;
};

bool blobValid(const StateBlob& b) {
    return b.magic == gdenv::kSnapshotMagic &&
           b.version == static_cast<uint32_t>(gdenv::kSnapshotVersion) &&
           b.simVersion == static_cast<uint32_t>(gdenv::kSimVersion);
}

} // namespace

struct GdEnv {
    gdcore::LevelGeometry geom;
    gdcore::PhysicsConfig cfg = gdcore::stereoMadnessCube();
    gdenv::EnvConfig ecfg;
    float endX = 0.0f;
    int obsVersion = gdenv::kObsVersionLegacy;
    int obsDim = gdenv::kObsDim;
    uint64_t levelFp = 0;
    std::vector<gdenv::EnvInstance> envs;
    std::vector<std::vector<uint8_t>> lastHolds;   // per env, last finished ep
    std::vector<std::vector<uint8_t>> lastActions; // per env, last finished ep
    std::unique_ptr<ThreadPool> pool;

    void obs(const gdcore::PlayerState& s, float* out) const {
        gdenv::buildObsVersioned(obsVersion, s, geom, cfg, endX, out);
    }
    int nActions() const { return gdenv::numActions(ecfg.actionMode); }
};

namespace {

// One replay loop shared by the holds and the decision-stream entry points:
// `next(e, cursor, len)` yields the action for the current decision (or -1
// to stop). cursorByTicks: the cursor indexes ticks and advances by the
// ticks the executor consumed (raw holds); otherwise it indexes decisions
// and advances by one.
template <class NextFn>
void replayOne(GdEnv* env, int i, NextFn next, bool cursorByTicks, int64_t len,
               int32_t max_steps, float* obs, uint8_t* acts, float* rews,
               float* ticks, int32_t* n_steps, const uint8_t* start_blobs,
               float* final_obs, uint8_t* ends, uint8_t* masks) {
    gdenv::EnvInstance& e = env->envs[static_cast<size_t>(i)];
    const size_t row0 = static_cast<size_t>(i) * static_cast<size_t>(max_steps);
    const int nA = env->nActions();
    e.reset();
    if (start_blobs) {
        StateBlob blob;
        std::memcpy(&blob, start_blobs + static_cast<size_t>(i) * sizeof(blob),
                    sizeof(blob));
        if (blobValid(blob)) {
            e.s = blob.s;
            e.tick = blob.tick;
        }
    }
    int32_t t = 0;
    int64_t cursor = 0;
    uint8_t end = 0;
    while (cursor < len) {
        const int a = next(e, cursor, len);
        if (a < 0) break;
        const bool rec = t < max_steps;
        const size_t row = row0 + static_cast<size_t>(t);
        if (rec) {
            if (obs) env->obs(e.s, obs + row * static_cast<size_t>(env->obsDim));
            if (masks)
                gdenv::actionMask(e.s, env->ecfg.actionMode,
                                  masks + row * static_cast<size_t>(nA));
            acts[row] = static_cast<uint8_t>(a);
        }
        gdenv::StepResult r = e.act(a, env->cfg, env->geom, env->ecfg, env->endX);
        if (rec) {
            rews[row] = r.reward;
            ticks[row] = static_cast<float>(r.ticks);
        }
        ++t;
        cursor += cursorByTicks ? r.ticks : 1;
        if (r.terminated) { end = 1; break; }
        if (r.truncated) { end = 2; break; }
    }
    if (final_obs)
        env->obs(e.s, final_obs + static_cast<size_t>(i) * static_cast<size_t>(env->obsDim));
    if (ends) ends[i] = end;
    n_steps[i] = t;
    e.reset();
}

} // namespace

extern "C" {

GDENV_API GdEnv* gdenv_create(const char* level_path, int32_t n_envs,
                              int32_t ticks_per_action, float end_x,
                              uint64_t /*seed*/, int32_t obs_version,
                              int32_t action_mode, double reward_gamma) {
    try {
        if (n_envs <= 0) {
            g_lastError = "n_envs must be positive";
            return nullptr;
        }
        if (obs_version <= 0) obs_version = gdenv::kObsVersionLegacy;
        if (obs_version != gdenv::kObsVersionLegacy &&
            obs_version != gdenv::kObsVersionV6) {
            g_lastError = "obs_version must be 5 or 6";
            return nullptr;
        }
        if (action_mode != gdenv::kActionModeMacro &&
            action_mode != gdenv::kActionModePrecision) {
            g_lastError = "action_mode must be 0 (macro) or 1 (precision)";
            return nullptr;
        }
        if (reward_gamma < 0.0 || reward_gamma >= 1.0) {
            g_lastError = "reward_gamma must be in [0, 1)";
            return nullptr;
        }
        auto env = std::make_unique<GdEnv>();
        const std::string path = resolveLevelPath(level_path);
        gd::Level level = gd::Level::loadFromFile(path);
        {
            std::ifstream f(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            env->levelFp = gdenv::fnv1a64(bytes.data(), bytes.size());
        }
        env->geom = gdcore::LevelGeometry::build(level);
        float maxObjX = 0.0f;
        for (const auto& o : level.objects) maxObjX = std::max(maxObjX, o.x);
        env->endX = end_x >= 0.0f ? end_x : maxObjX + 40.0f;
        if (ticks_per_action > 0) env->ecfg.ticksPerAction = ticks_per_action;
        env->ecfg.actionMode = action_mode;
        env->ecfg.rewardGamma = reward_gamma;
        env->obsVersion = obs_version;
        env->obsDim = gdenv::obsDimFor(obs_version);
        env->envs.resize(static_cast<size_t>(n_envs));
        env->lastHolds.resize(static_cast<size_t>(n_envs));
        env->lastActions.resize(static_cast<size_t>(n_envs));
        for (auto& e : env->envs) e.reset();
        env->pool = std::make_unique<ThreadPool>(
            std::min(resolveThreadCount(), n_envs));
        return env.release();
    } catch (const std::exception& e) {
        g_lastError = e.what();
        return nullptr;
    } catch (...) {
        g_lastError = "unknown error in gdenv_create";
        return nullptr;
    }
}

GDENV_API void gdenv_destroy(GdEnv* env) { delete env; }

GDENV_API const char* gdenv_last_error(void) { return g_lastError.c_str(); }

GDENV_API int32_t gdenv_obs_dim(const GdEnv* env) { return env->obsDim; }

GDENV_API int32_t gdenv_obs_version(void) { return gdenv::kObsVersion; }

GDENV_API int32_t gdenv_obs_version_of(const GdEnv* env) {
    return env->obsVersion;
}

GDENV_API int32_t gdenv_action_mode(const GdEnv* env) {
    return env->ecfg.actionMode;
}

GDENV_API int32_t gdenv_num_actions(const GdEnv* env) {
    return env->nActions();
}

GDENV_API int32_t gdenv_num_envs(const GdEnv* env) {
    return static_cast<int32_t>(env->envs.size());
}

GDENV_API float gdenv_level_end_x(const GdEnv* env) { return env->endX; }

GDENV_API int32_t gdenv_ticks_per_action(const GdEnv* env) {
    return env->ecfg.ticksPerAction;
}

GDENV_API void gdenv_contract_versions(int32_t* out) {
    out[0] = gdenv::kObsVersionLegacy;
    out[1] = gdenv::kObsVersionV6;
    out[2] = gdenv::kActionVersion;
    out[3] = gdenv::kTrajectoryVersion;
    out[4] = gdenv::kSnapshotVersion;
    out[5] = gdenv::kSimVersion;
}

GDENV_API uint64_t gdenv_level_fingerprint(const GdEnv* env) {
    return env->levelFp;
}

GDENV_API uint64_t gdenv_sim_fingerprint(const GdEnv* env) {
    // Canonical scripted run: three segments of LCG-driven holds with
    // press lengths of 1..8 ticks (so one-tick semantics are exercised),
    // raw per-tick stepping, hashing the full state trajectory. Restarts
    // from x=0 on death so a level's early wall does not hide the rest.
    uint64_t h = gdenv::fnv1a64(&gdenv::kSimVersion, sizeof(int));
    uint32_t rng = 0x9E3779B9u;
    gdcore::PlayerState s;
    int runLen = 0;
    bool hold = false;
    for (int t = 0; t < 12000; ++t) {
        if (runLen <= 0) {
            rng = rng * 1664525u + 1013904223u;
            hold = ((rng >> 16) & 3u) == 0u ? !hold : hold;
            runLen = 1 + static_cast<int>((rng >> 20) & 7u);
        }
        --runLen;
        gdcore::step(s, env->cfg, env->geom, hold);
        // Field by field: a packed struct would hash its padding bytes.
        const uint32_t f =
            (s.onGround ? 1u : 0u) | (s.ship ? 2u : 0u) | (s.ball ? 4u : 0u) |
            (s.bird ? 8u : 0u) | (s.upsideDown ? 16u : 0u) |
            (s.mini ? 32u : 0u) | (s.dead ? 64u : 0u) |
            (s.holding ? 128u : 0u) | (s.hasJustHeld ? 256u : 0u);
        h = gdenv::fnv1a64(&s.x, sizeof s.x, h);
        h = gdenv::fnv1a64(&s.y, sizeof s.y, h);
        h = gdenv::fnv1a64(&s.yvel, sizeof s.yvel, h);
        h = gdenv::fnv1a64(&f, sizeof f, h);
        if (s.dead || s.x >= env->endX) {
            s = gdcore::PlayerState{};
            hold = false;
            runLen = 0;
        }
    }
    return h;
}

GDENV_API void gdenv_reset_all(GdEnv* env, float* obs) {
    env->pool->run(static_cast<int>(env->envs.size()), [&](int i) {
        env->envs[i].reset();
        env->obs(env->envs[i].s, obs + static_cast<size_t>(i) * env->obsDim);
    });
}

GDENV_API void gdenv_action_masks(const GdEnv* env, uint8_t* out) {
    const int nA = env->nActions();
    for (size_t i = 0; i < env->envs.size(); ++i)
        gdenv::actionMask(env->envs[i].s, env->ecfg.actionMode,
                          out + i * static_cast<size_t>(nA));
}

GDENV_API void gdenv_step(GdEnv* env, const uint8_t* actions, float* obs,
                          float* rewards, float* durations,
                          uint8_t* terminated, uint8_t* truncated,
                          float* final_obs, float* final_info) {
    env->pool->run(static_cast<int>(env->envs.size()), [&](int i) {
        gdenv::EnvInstance& e = env->envs[i];
        gdenv::StepResult r =
            e.act(actions[i], env->cfg, env->geom, env->ecfg, env->endX);
        rewards[i] = r.reward;
        durations[i] = static_cast<float>(r.ticks);
        terminated[i] = r.terminated ? 1 : 0;
        truncated[i] = r.truncated ? 1 : 0;
        float* obsRow = obs + static_cast<size_t>(i) * env->obsDim;
        if (r.terminated || r.truncated) {
            if (final_obs)
                env->obs(e.s, final_obs + static_cast<size_t>(i) * env->obsDim);
            if (final_info) {
                final_info[i * 3 + 0] = e.s.x;
                final_info[i * 3 + 1] = r.won ? 1.0f : 0.0f;
                final_info[i * 3 + 2] = static_cast<float>(e.tick);
            }
            env->lastHolds[i].swap(e.holds);
            env->lastActions[i].swap(e.actions);
            e.reset();
        }
        env->obs(e.s, obsRow);
    });
}

GDENV_API int32_t gdenv_last_episode_holds(const GdEnv* env, int32_t i,
                                           uint8_t* out, int32_t cap) {
    const auto& h = env->lastHolds[static_cast<size_t>(i)];
    int32_t n = static_cast<int32_t>(h.size());
    int32_t m = n < cap ? n : cap;
    if (out && m > 0) std::memcpy(out, h.data(), static_cast<size_t>(m));
    return n;
}

GDENV_API int32_t gdenv_last_episode_actions(const GdEnv* env, int32_t i,
                                             uint8_t* out, int32_t cap) {
    const auto& a = env->lastActions[static_cast<size_t>(i)];
    int32_t n = static_cast<int32_t>(a.size());
    int32_t m = n < cap ? n : cap;
    if (out && m > 0) std::memcpy(out, a.data(), static_cast<size_t>(m));
    return n;
}

GDENV_API void gdenv_get_render_states(const GdEnv* env, float* out) {
    for (size_t i = 0; i < env->envs.size(); ++i) {
        const gdcore::PlayerState& s = env->envs[i].s;
        float* row = out + i * 5;
        row[0] = s.x;
        row[1] = s.y;
        row[2] = static_cast<float>(s.yvel);
        row[3] = static_cast<float>(gdenv::modeOf(s)); // 0 cube, 1 ship, 2 ball, 3 UFO
        row[4] = s.dead ? 1.0f : 0.0f;
    }
}

GDENV_API int32_t gdenv_state_size(void) {
    return static_cast<int32_t>(sizeof(StateBlob));
}

GDENV_API void gdenv_get_state(const GdEnv* env, int32_t i, uint8_t* buf) {
    StateBlob blob;
    blob.s = env->envs[i].s;
    blob.tick = env->envs[i].tick;
    std::memcpy(buf, &blob, sizeof(blob));
}

GDENV_API int32_t gdenv_set_state(GdEnv* env, int32_t i, const uint8_t* buf,
                                  float* obs_i) {
    StateBlob blob;
    std::memcpy(&blob, buf, sizeof(blob));
    if (!blobValid(blob)) {
        g_lastError = "gdenv_set_state: foreign snapshot blob (magic / "
                      "snapshot version / sim version mismatch)";
        return -1;
    }
    env->envs[i].s = blob.s;
    env->envs[i].tick = blob.tick;
    // A snapshot-spawned episode's holds don't start at tick 0, so they can
    // never form a valid from-start replay — keep the record empty.
    env->envs[i].holds.clear();
    env->envs[i].actions.clear();
    if (obs_i) env->obs(env->envs[i].s, obs_i);
    return 0;
}

GDENV_API void gdenv_reset_env(GdEnv* env, int32_t i, float* obs_i) {
    env->envs[i].reset();
    if (obs_i) env->obs(env->envs[i].s, obs_i);
}

GDENV_API uint64_t gdenv_state_hash(const uint8_t* buf) {
    StateBlob blob;
    std::memcpy(&blob, buf, sizeof(blob));
    if (!blobValid(blob)) return 0;
    return gdenv::exactStateHash(blob.s, blob.tick);
}

GDENV_API int32_t gdenv_replay(GdEnv* env, int32_t k, const uint8_t* holds,
                               const int32_t* lengths, int32_t max_steps,
                               float* obs, uint8_t* acts, float* rews,
                               float* ticks, int32_t* n_steps,
                               const uint8_t* start_blobs, float* final_obs,
                               uint8_t* ends, uint8_t* masks) {
    if (k < 0 || k > static_cast<int32_t>(env->envs.size()) || !holds ||
        !lengths || !n_steps || max_steps < 0 ||
        (max_steps > 0 && (!acts || !rews || !ticks))) {
        g_lastError = "gdenv_replay: bad arguments";
        return -1;
    }
    std::vector<int64_t> offsets(static_cast<size_t>(k) + 1, 0);
    for (int32_t i = 0; i < k; ++i)
        offsets[static_cast<size_t>(i) + 1] =
            offsets[static_cast<size_t>(i)] + lengths[i];
    const int mode = env->ecfg.actionMode;
    env->pool->run(k, [&](int i) {
        const uint8_t* h = holds + offsets[static_cast<size_t>(i)];
        replayOne(env, i,
                  [&](gdenv::EnvInstance& e, int64_t cursor, int64_t len) {
                      return gdenv::actionForHolds(e.s, mode, h, cursor, len);
                  },
                  true, lengths[i], max_steps, obs, acts, rews, ticks, n_steps,
                  start_blobs, final_obs, ends, masks);
    });
    return 0;
}

GDENV_API int32_t gdenv_replay_actions(GdEnv* env, int32_t k,
                                       const uint8_t* actions,
                                       const int32_t* lengths, int32_t max_steps,
                                       float* obs, uint8_t* acts, float* rews,
                                       float* ticks, int32_t* n_steps,
                                       const uint8_t* start_blobs,
                                       float* final_obs, uint8_t* ends,
                                       uint8_t* masks) {
    if (k < 0 || k > static_cast<int32_t>(env->envs.size()) || !actions ||
        !lengths || !n_steps || max_steps < 0 ||
        (max_steps > 0 && (!acts || !rews || !ticks))) {
        g_lastError = "gdenv_replay_actions: bad arguments";
        return -1;
    }
    std::vector<int64_t> offsets(static_cast<size_t>(k) + 1, 0);
    for (int32_t i = 0; i < k; ++i)
        offsets[static_cast<size_t>(i) + 1] =
            offsets[static_cast<size_t>(i)] + lengths[i];
    const int nA = env->nActions();
    env->pool->run(k, [&](int i) {
        const uint8_t* a = actions + offsets[static_cast<size_t>(i)];
        replayOne(env, i,
                  [&](gdenv::EnvInstance&, int64_t cursor, int64_t) {
                      const int id = a[cursor];
                      return id < nA ? id : -1;
                  },
                  false, lengths[i], max_steps, obs, acts, rews, ticks, n_steps,
                  start_blobs, final_obs, ends, masks);
    });
    return 0;
}

GDENV_API int32_t gdenv_solve(GdEnv* env, int32_t k, const uint8_t* start_blobs,
                              const float* target_x, int32_t max_nodes,
                              int32_t max_states, uint8_t* holds_out,
                              int32_t cap, int32_t* lens, uint8_t* acts_out,
                              uint16_t* durs_out, int32_t cap_acts,
                              int32_t* n_acts, int32_t* status, float* stats) {
    if (k < 0 || !start_blobs || !target_x || !holds_out || cap <= 0 || !lens ||
        !acts_out || !durs_out || cap_acts <= 0 || !n_acts || !status ||
        !stats) {
        g_lastError = "gdenv_solve: bad arguments";
        return -1;
    }
    env->pool->run(k, [&](int i) {
        StateBlob blob;
        std::memcpy(&blob, start_blobs + static_cast<size_t>(i) * sizeof(blob),
                    sizeof(blob));
        lens[i] = 0;
        n_acts[i] = 0;
        float* st = stats + static_cast<size_t>(i) * 5;
        if (!blobValid(blob)) {
            status[i] = gdenv::kSolveVerifyFailed;
            st[0] = st[1] = st[2] = st[3] = st[4] = 0.0f;
            return;
        }
        int stage = 0;
        gdenv::SolveResult r = gdenv::solveAny(
            blob.s, blob.tick, target_x[i], env->cfg, env->geom, env->ecfg,
            env->endX, max_nodes > 0 ? max_nodes : 200000,
            max_states > 0 ? max_states : 4096, stage);
        status[i] = r.status;
        st[0] = r.reachX;
        st[1] = static_cast<float>(r.nodes);
        st[2] = static_cast<float>(r.stored);
        st[3] = static_cast<float>(r.pruned);
        st[4] = static_cast<float>(stage);
        if (!r.found || static_cast<int32_t>(r.holds.size()) > cap ||
            static_cast<int32_t>(r.actions.size()) > cap_acts)
            return;
        lens[i] = static_cast<int32_t>(r.holds.size());
        std::memcpy(holds_out + static_cast<size_t>(i) * cap, r.holds.data(),
                    r.holds.size());
        n_acts[i] = static_cast<int32_t>(r.actions.size());
        std::memcpy(acts_out + static_cast<size_t>(i) * cap_acts,
                    r.actions.data(), r.actions.size());
        std::memcpy(durs_out + static_cast<size_t>(i) * cap_acts,
                    r.durations.data(), r.durations.size() * sizeof(uint16_t));
    });
    return 0;
}

GDENV_API int32_t gdenv_trace_states(GdEnv* env, const uint8_t* holds,
                                     int32_t len, int32_t n, const float* xs,
                                     uint8_t* blobs_out, uint8_t* found) {
    if (!holds || len < 0 || n < 0 || !xs || !blobs_out || !found ||
        env->envs.empty()) {
        g_lastError = "gdenv_trace_states: bad arguments";
        return -1;
    }
    gdenv::EnvInstance& e = env->envs[0];
    e.reset();
    std::memset(found, 0, static_cast<size_t>(n));
    int32_t j = 0;
    int64_t cursor = 0;
    for (;;) {
        while (j < n && e.s.x >= xs[j]) {
            StateBlob blob;
            blob.s = e.s;
            blob.tick = e.tick;
            std::memcpy(blobs_out + static_cast<size_t>(j) * sizeof(blob), &blob,
                        sizeof(blob));
            found[j] = 1;
            ++j;
        }
        if (j >= n || cursor >= len) break;
        const int a = gdenv::actionForHolds(e.s, env->ecfg.actionMode, holds,
                                            cursor, len);
        gdenv::StepResult r = e.act(a, env->cfg, env->geom, env->ecfg, env->endX);
        cursor += r.ticks;
        if (r.terminated || r.truncated) break;
    }
    e.reset();
    return 0;
}

GDENV_API int32_t gdenv_run_holds(const GdEnv* env, const uint8_t* holds,
                                  int32_t len, float* out) {
    gdcore::PlayerState s;
    int32_t t = 0;
    bool won = false;
    for (; t < len; ++t) {
        gdcore::step(s, env->cfg, env->geom, holds[t] != 0);
        if (s.dead) { ++t; break; }
        if (s.x >= env->endX) { won = true; ++t; break; }
    }
    out[0] = s.x;
    out[1] = won ? 1.0f : 0.0f;
    out[2] = static_cast<float>(t);
    return 0;
}

} // extern "C"
