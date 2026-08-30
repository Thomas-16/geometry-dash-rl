#include "env/gdenv_c.h"

#include "core/level.hpp"
#include "core/physics.hpp"
#include "env/episode.hpp"
#include "env/obs.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string g_lastError;

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

GDENV_API int32_t gdenv_num_envs(const GdEnv* env) {
    return static_cast<int32_t>(env->envs.size());
}

GDENV_API float gdenv_level_end_x(const GdEnv* env) { return env->endX; }

GDENV_API int32_t gdenv_ticks_per_action(const GdEnv* env) {
    return env->ecfg.ticksPerAction;
}

GDENV_API void gdenv_reset_all(GdEnv* env, float* obs) {
    for (size_t i = 0; i < env->envs.size(); ++i) {
        env->envs[i].reset();
        gdenv::buildObs(env->envs[i].s, env->geom, env->endX,
                        obs + i * gdenv::kObsDim);
    }
}

GDENV_API void gdenv_step(GdEnv* env, const uint8_t* actions, float* obs,
                          float* rewards, float* durations,
                          uint8_t* terminated, uint8_t* truncated,
                          float* final_obs, float* final_info) {
    const size_t n = env->envs.size();
    for (size_t i = 0; i < n; ++i) {
        gdenv::EnvInstance& e = env->envs[i];
        gdenv::StepResult r =
            e.step(actions[i] != 0, env->cfg, env->geom, env->ecfg, env->endX);
        rewards[i] = r.reward;
        durations[i] = static_cast<float>(r.ticks);
        terminated[i] = r.terminated ? 1 : 0;
        truncated[i] = r.truncated ? 1 : 0;
        float* obsRow = obs + i * gdenv::kObsDim;
        if (r.terminated || r.truncated) {
            if (final_obs)
                gdenv::buildObs(e.s, env->geom, env->endX,
                                final_obs + i * gdenv::kObsDim);
            if (final_info) {
                final_info[i * 3 + 0] = e.s.x;
                final_info[i * 3 + 1] = r.won ? 1.0f : 0.0f;
                final_info[i * 3 + 2] = static_cast<float>(e.tick);
            }
            env->lastHolds[i].swap(e.holds);
            e.reset();
        }
        gdenv::buildObs(e.s, env->geom, env->endX, obsRow);
    }
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
    if (obs_i) gdenv::buildObs(env->envs[i].s, env->geom, env->endX, obs_i);
}

GDENV_API void gdenv_reset_env(GdEnv* env, int32_t i, float* obs_i) {
    env->envs[i].reset();
    if (obs_i) gdenv::buildObs(env->envs[i].s, env->geom, env->endX, obs_i);
}

} // extern "C"
