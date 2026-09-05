#pragma once

#include <cstdint>

// Versioned contracts between the sim, the env DLL, the trainer and the
// deployment tools. Pure compile-time constants: the Geode mod may include
// this header. Every checkpoint records the values it was trained under
// (train.py `contracts`), and a resume / eval refuses a mismatch instead of
// silently mixing feature layouts, action spaces or snapshot blobs.
namespace gdenv {

// Observation layouts (obs.hpp). 5 = the frozen absolute-band layout every
// level 1-12 checkpoint was trained on; 6 = player-centered grid + appended
// scalars + exact geometry branch.
constexpr int kObsVersionLegacy = 5;
constexpr int kObsVersionV6 = 6;

// Action spaces (episode.hpp ActionMode).
constexpr int kActionModeMacro = 0;      // 2 actions: walk/release, hop/hold
constexpr int kActionModePrecision = 1;  // 7 actions: R1 R2 R4 H1 H2 H4 HOP
constexpr int kActionVersion = 1;

// Decision-stream (trajectory) format: per-decision action ids as u8, per
// episode, replayed from a snapshot blob or x=0 through the shared executor.
constexpr int kTrajectoryVersion = 1;

// Snapshot blob (gdenv_get_state / set_state): header + PlayerState + tick.
constexpr int kSnapshotVersion = 1;
constexpr uint32_t kSnapshotMagic = 0x42534447u; // 'GDSB'

// Simulator semantics. Bump whenever a physics change alters reachable
// states (a run trained before such a fix is poisoned: SIL buffer, respawn
// bank and certificate all assume the old dynamics). gdenv_sim_fingerprint
// additionally hashes a canonical scripted run on the loaded level, so an
// unversioned behavioral change is still caught.
constexpr int kSimVersion = 1;

// FNV-1a 64, the one hash every fingerprint in this project uses.
inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

} // namespace gdenv
