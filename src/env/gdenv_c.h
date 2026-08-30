/* C ABI for the vectorized training env (gdenv.dll). Pure C header so any
 * FFI (Python ctypes) can consume it without C++ name mangling. No C++
 * exceptions cross this boundary: every export catches internally and
 * reports via gdenv_last_error(). */
#pragma once

#include <stdint.h>

#ifdef GDENV_BUILD
#define GDENV_API __declspec(dllexport)
#else
#define GDENV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GdEnv GdEnv; /* opaque batch of n_envs independent episodes */

/* Returns NULL on failure; gdenv_last_error() has the message.
 * level_path: NULL/"" -> $GD_LEVEL_PATH -> the default GD install path.
 * ticks_per_action: <=0 -> default 4.
 * end_x: win line; < 0 -> full level (maxObjX + 40). Lower values make
 *        truncated early-section curricula.
 * seed: reserved (the sim is deterministic); pass 0.
 * Env stepping parallelism: $GDENV_THREADS total threads (1 = serial),
 * default hardware_concurrency()-1. Bit-identical at any thread count. */
GDENV_API GdEnv* gdenv_create(const char* level_path, int32_t n_envs,
                              int32_t ticks_per_action, float end_x,
                              uint64_t seed);
GDENV_API void gdenv_destroy(GdEnv* env);
GDENV_API const char* gdenv_last_error(void);

GDENV_API int32_t gdenv_obs_dim(const GdEnv* env);  /* kObsDim */
GDENV_API int32_t gdenv_obs_version(void);          /* kObsVersion */
GDENV_API int32_t gdenv_num_envs(const GdEnv* env);
GDENV_API float gdenv_level_end_x(const GdEnv* env);
GDENV_API int32_t gdenv_ticks_per_action(const GdEnv* env);

/* Resets every episode and writes first observations. obs: [n][obs_dim]. */
GDENV_API void gdenv_reset_all(GdEnv* env, float* obs);

/* Standard auto-reset vectorized step.
 *   actions:    [n] u8, nonzero = hold/jump
 *   obs:        [n][obs_dim] next obs; where terminated|truncated it is the
 *               NEW episode's first obs (auto-reset)
 *   rewards:    [n] f32
 *   durations:  [n] f32, sim ticks this decision consumed (steps are
 *               variable-length: a ground jump rolls the whole hop) — use
 *               for SMDP discounting gamma^ticks
 *   terminated: [n] u8 (death or win — bootstrap 0)
 *   truncated:  [n] u8 (tick cap — bootstrap V(final_obs))
 *   final_obs:  [n][obs_dim] or NULL; the pre-reset terminal observation,
 *               written only where terminated|truncated
 *   final_info: [n][3] or NULL; {final_x, won, episode_ticks}, written only
 *               where terminated|truncated */
GDENV_API void gdenv_step(GdEnv* env, const uint8_t* actions, float* obs,
                          float* rewards, float* durations,
                          uint8_t* terminated, uint8_t* truncated,
                          float* final_obs, float* final_info);

/* Per-tick holds of env i's most recently FINISHED episode (call for envs
 * flagged terminated|truncated by the last gdenv_step). Copies up to cap
 * bytes of 0/1 into out and returns the episode's full tick count. */
GDENV_API int32_t gdenv_last_episode_holds(const GdEnv* env, int32_t i,
                                           uint8_t* out, int32_t cap);

/* Live visualization: out is [n][5] = {x, y, yvel, ship, dead}. */
GDENV_API void gdenv_get_render_states(const GdEnv* env, float* out);

/* Snapshot save/restore of one episode (debugging, curriculum). The blob is
 * an opaque fixed-size byte block; only valid within one build. set_state
 * and reset_env write the episode's current obs into obs_i ([obs_dim]),
 * which may be NULL. */
GDENV_API int32_t gdenv_state_size(void);
GDENV_API void gdenv_get_state(const GdEnv* env, int32_t i, uint8_t* buf);
GDENV_API void gdenv_set_state(GdEnv* env, int32_t i, const uint8_t* buf,
                               float* obs_i);
GDENV_API void gdenv_reset_env(GdEnv* env, int32_t i, float* obs_i);

#ifdef __cplusplus
}
#endif
