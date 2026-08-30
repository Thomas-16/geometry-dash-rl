#include "env/gdenv_c.h"

#include "core/level.hpp"
#include "core/physics.hpp"
#include "env/episode.hpp"
#include "env/obs.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
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

struct StateBlob {
    gdcore::PlayerState s;
    int32_t tick;
};

} // namespace

struct GdEnv {
    gdcore::LevelGeometry geom;
    gdcore::PhysicsConfig cfg = gdcore::stereoMadnessCube();
    gdenv::EnvConfig ecfg;
    float endX = 0.0f;
    std::vector<gdenv::EnvInstance> envs;
    std::vector<std::vector<uint8_t>> lastHolds; // per env, last finished ep
    std::unique_ptr<ThreadPool> pool;
};

extern "C" {

GDENV_API GdEnv* gdenv_create(const char* level_path, int32_t n_envs,
                              int32_t ticks_per_action, float end_x,
                              uint64_t /*seed*/) {
    try {
        if (n_envs <= 0) {
            g_lastError = "n_envs must be positive";
            return nullptr;
        }
        auto env = std::make_unique<GdEnv>();
        gd::Level level = gd::Level::loadFromFile(resolveLevelPath(level_path));
        env->geom = gdcore::LevelGeometry::build(level);
        float maxObjX = 0.0f;
        for (const auto& o : level.objects) maxObjX = std::max(maxObjX, o.x);
        env->endX = end_x >= 0.0f ? end_x : maxObjX + 40.0f;
        if (ticks_per_action > 0) env->ecfg.ticksPerAction = ticks_per_action;
        env->envs.resize(static_cast<size_t>(n_envs));
        env->lastHolds.resize(static_cast<size_t>(n_envs));
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

GDENV_API int32_t gdenv_obs_dim(const GdEnv*) { return gdenv::kObsDim; }

GDENV_API int32_t gdenv_obs_version(void) { return gdenv::kObsVersion; }

GDENV_API int32_t gdenv_num_envs(const GdEnv* env) {
    return static_cast<int32_t>(env->envs.size());
}

GDENV_API float gdenv_level_end_x(const GdEnv* env) { return env->endX; }

GDENV_API int32_t gdenv_ticks_per_action(const GdEnv* env) {
    return env->ecfg.ticksPerAction;
}

GDENV_API void gdenv_reset_all(GdEnv* env, float* obs) {
    env->pool->run(static_cast<int>(env->envs.size()), [&](int i) {
        env->envs[i].reset();
        gdenv::buildObs(env->envs[i].s, env->geom, env->endX, env->cfg.dx,
                        obs + static_cast<size_t>(i) * gdenv::kObsDim);
    });
}

GDENV_API void gdenv_step(GdEnv* env, const uint8_t* actions, float* obs,
                          float* rewards, float* durations,
                          uint8_t* terminated, uint8_t* truncated,
                          float* final_obs, float* final_info) {
    env->pool->run(static_cast<int>(env->envs.size()), [&](int i) {
        gdenv::EnvInstance& e = env->envs[i];
        gdenv::StepResult r =
            e.step(actions[i] != 0, env->cfg, env->geom, env->ecfg, env->endX);
        rewards[i] = r.reward;
        durations[i] = static_cast<float>(r.ticks);
        terminated[i] = r.terminated ? 1 : 0;
        truncated[i] = r.truncated ? 1 : 0;
        float* obsRow = obs + static_cast<size_t>(i) * gdenv::kObsDim;
        if (r.terminated || r.truncated) {
            if (final_obs)
                gdenv::buildObs(e.s, env->geom, env->endX, env->cfg.dx,
                                final_obs +
                                    static_cast<size_t>(i) * gdenv::kObsDim);
            if (final_info) {
                final_info[i * 3 + 0] = e.s.x;
                final_info[i * 3 + 1] = r.won ? 1.0f : 0.0f;
                final_info[i * 3 + 2] = static_cast<float>(e.tick);
            }
            env->lastHolds[i].swap(e.holds);
            e.reset();
        }
        gdenv::buildObs(e.s, env->geom, env->endX, env->cfg.dx, obsRow);
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

GDENV_API void gdenv_get_render_states(const GdEnv* env, float* out) {
    for (size_t i = 0; i < env->envs.size(); ++i) {
        const gdcore::PlayerState& s = env->envs[i].s;
        float* row = out + i * 5;
        row[0] = s.x;
        row[1] = s.y;
        row[2] = static_cast<float>(s.yvel);
        row[3] = s.ship ? 1.0f : 0.0f;
        row[4] = s.dead ? 1.0f : 0.0f;
    }
}

GDENV_API int32_t gdenv_state_size(void) {
    return static_cast<int32_t>(sizeof(StateBlob));
}

GDENV_API void gdenv_get_state(const GdEnv* env, int32_t i, uint8_t* buf) {
    StateBlob blob{env->envs[i].s, env->envs[i].tick};
    std::memcpy(buf, &blob, sizeof(blob));
}

GDENV_API void gdenv_set_state(GdEnv* env, int32_t i, const uint8_t* buf,
                               float* obs_i) {
    StateBlob blob;
    std::memcpy(&blob, buf, sizeof(blob));
    env->envs[i].s = blob.s;
    env->envs[i].tick = blob.tick;
    // A snapshot-spawned episode's holds don't start at tick 0, so they can
    // never form a valid from-start replay — keep the record empty.
    env->envs[i].holds.clear();
    if (obs_i)
        gdenv::buildObs(env->envs[i].s, env->geom, env->endX, env->cfg.dx,
                        obs_i);
}

GDENV_API void gdenv_reset_env(GdEnv* env, int32_t i, float* obs_i) {
    env->envs[i].reset();
    if (obs_i)
        gdenv::buildObs(env->envs[i].s, env->geom, env->endX, env->cfg.dx,
                        obs_i);
}

GDENV_API int32_t gdenv_replay(GdEnv* env, int32_t k, const uint8_t* holds,
                               const int32_t* lengths, int32_t max_steps,
                               float* obs, uint8_t* acts, float* rews,
                               float* ticks, int32_t* n_steps) {
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
    env->pool->run(k, [&](int i) {
        gdenv::EnvInstance& e = env->envs[i];
        const uint8_t* h = holds + offsets[static_cast<size_t>(i)];
        const int32_t len = lengths[i];
        const size_t row0 = static_cast<size_t>(i) * max_steps;
        e.reset();
        int32_t t = 0;
        int64_t cursor = 0;
        while (cursor < len) {
            const bool hold = h[cursor] != 0;
            const bool rec = t < max_steps;
            const size_t row = row0 + t;
            if (rec) {
                if (obs)
                    gdenv::buildObs(e.s, env->geom, env->endX, env->cfg.dx,
                                    obs + row * gdenv::kObsDim);
                acts[row] = hold ? 1 : 0;
            }
            gdenv::StepResult r =
                e.step(hold, env->cfg, env->geom, env->ecfg, env->endX);
            if (rec) {
                rews[row] = r.reward;
                ticks[row] = static_cast<float>(r.ticks);
            }
            ++t;
            cursor += r.ticks;
            if (r.terminated || r.truncated) break;
        }
        n_steps[i] = t;
        e.reset();
    });
    return 0;
}

} // extern "C"
