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

/* Batched deterministic replay of recorded episodes (SIL experience
 * regeneration). Episodes 0..k-1 are replayed on envs 0..k-1 (k <= n_envs),
 * each reset first (or restored from start_blobs[i] — [k][state_size], or
 * NULL for from-start — so a demonstration SEGMENT solved from a mid-level
 * state replays too); the per-tick hold bits are concatenated in `holds`,
 * `lengths[i]` ticks each. Every decision step t < max_steps of episode i
 * writes row i*max_steps + t of:
 *   obs   [k][max_steps][obs_dim]  observation BEFORE the decision (or NULL)
 *   acts  [k][max_steps] u8        the hold bit fed at that decision
 *   rews  [k][max_steps] f32       step reward
 *   ticks [k][max_steps] f32       sim ticks the decision consumed
 * and n_steps[i] receives the episode's total decision count (may exceed
 * max_steps; pass max_steps=0 for a count-only pass). final_obs
 * ([k][obs_dim] or NULL) receives the observation AFTER the last decision
 * and ends[i] ([k] u8 or NULL) how the stream ended: 0 = holds exhausted
 * with the player alive (bootstrap V(final_obs)), 1 = terminated (death or
 * win), 2 = tick cap. Replay uses the same macro-step code as gdenv_step,
 * so a stream is bit-identical to the episode that produced the holds. The
 * envs are left reset; the last-finished-episode records are untouched.
 * Returns 0, or -1 on bad arguments. */
GDENV_API int32_t gdenv_replay(GdEnv* env, int32_t k, const uint8_t* holds,
                               const int32_t* lengths, int32_t max_steps,
                               float* obs, uint8_t* acts, float* rews,
                               float* ticks, int32_t* n_steps,
                               const uint8_t* start_blobs, float* final_obs,
                               uint8_t* ends);

/* Exact planner in the policy's action space (env/solve.hpp): from each of
 * k start states (start_blobs [k][state_size]) search for a macro-decision
 * sequence reaching x >= target_x[i] alive. Solves run on the handle's
 * thread pool. holds_out ([k][cap] u8) receives the per-tick holds from the
 * start state, lens[i] their count (0 = no path found within the node
 * budget), reach[i] ([k] or NULL) the deepest x explored. Returns 0, or -1
 * on bad arguments. */
GDENV_API int32_t gdenv_solve(GdEnv* env, int32_t k, const uint8_t* start_blobs,
                              const float* target_x, int32_t max_nodes,
                              uint8_t* holds_out, int32_t cap, int32_t* lens,
                              float* reach);

/* Replays `holds` (len ticks) from a fresh episode on env slot 0 through
 * the macro layer and captures the state blob at the first decision
 * boundary where the player's x >= xs[j], for each of n sorted-ascending
 * xs, into blobs_out ([n][state_size]); found[j] = 1 where captured (0 if
 * the episode ended first). Used to take snapshots along a known path
 * (the deterministic policy's own run) as planner start states. */
GDENV_API int32_t gdenv_trace_states(GdEnv* env, const uint8_t* holds,
                                     int32_t len, int32_t n, const float* xs,
                                     uint8_t* blobs_out, uint8_t* found);

#ifdef __cplusplus
}
#endif
