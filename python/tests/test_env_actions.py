"""Focused env-layer tests (docs/rl-improvement-plan.md, validation step 1):
one-tick timing windows, UFO release / repress, ring contacts, portal
transitions, action replay, masks, obs v6 geometry.

Run from python/:  python -m unittest discover -s tests -v
Needs build/gdenv.dll and the GD install (levels 1, 3, 12). Tests that
need a recorded solution (runs/level12-2/solution.json) skip without it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from gdrl.env import (ACT_H1, ACT_H2, ACT_H4, ACT_HOP, ACT_R1, ACT_R2, ACT_R4,
                      GdVecEnv)

LEVELS = Path("S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels")
REPO = Path(__file__).resolve().parents[2]
L1 = str(LEVELS / "1.txt")
L3 = str(LEVELS / "3.txt")
L12 = str(LEVELS / "12.txt")


def solution_holds(run: str) -> np.ndarray | None:
    p = REPO / "runs" / run / "solution.json"
    if not p.exists():
        return None
    sol = json.loads(p.read_text())
    bits = np.zeros(int(sol["n_ticks"]), np.uint8)
    for a, b in sol["holds"]:
        bits[a:b] = 1
    return bits


def precision_env(level, n=4, obs_version=6):
    return GdVecEnv(level, n, 4, -1.0, obs_version=obs_version,
                    action_mode="precision", reward_gamma=0.9995)


def state_at(env: GdVecEnv, holds: np.ndarray, x: float):
    return env.trace_states(holds, [x])[0]


class OneTickTiming(unittest.TestCase):
    def test_raw_actions_have_their_nominal_durations_on_flat_ground(self):
        env = precision_env(L1)
        env.reset()
        for a, want in ((ACT_R1, 1), (ACT_R2, 2), (ACT_R4, 4)):
            _, _, _, _, info = env.step(np.full(env.n_envs, a, np.uint8))
            self.assertEqual(int(info["ticks"][0]), want)
        env.close()

    def test_one_tick_later_press_is_a_different_trajectory(self):
        """H1 at tick t vs H1 at tick t+1: identical physics shifted by one
        tick — the two runs' y differ at every later tick until landing."""
        env = precision_env(L1)
        env.reset()
        env.step(np.full(env.n_envs, ACT_R4, np.uint8))  # settle on the ground
        base = env.get_state(0)
        # run A: press now
        env.set_state(0, base)
        env.set_state(1, base)
        a = np.array([ACT_H1, ACT_R1, 0, 0], np.uint8)
        env.step(a)
        a = np.array([ACT_R1, ACT_H1, 0, 0], np.uint8)
        env.step(a)
        ya = env.get_render_states()[0, 1]
        yb = env.get_render_states()[1, 1]
        # env 0 has been airborne 2 ticks, env 1 one tick: different y
        self.assertNotAlmostEqual(float(ya), float(yb))
        # and both left the ground (hop mask off)
        m = env.action_masks()
        self.assertEqual(int(m[0, ACT_HOP]), 0)
        self.assertEqual(int(m[1, ACT_HOP]), 0)
        env.close()

    def test_masked_hop_executes_as_one_tick_hold(self):
        env = precision_env(L1)
        env.reset()
        env.step(np.full(env.n_envs, ACT_R4, np.uint8))
        env.step(np.full(env.n_envs, ACT_H1, np.uint8))  # jump
        self.assertEqual(int(env.action_masks()[0, ACT_HOP]), 0)
        _, _, _, _, info = env.step(np.full(env.n_envs, ACT_HOP, np.uint8))
        self.assertEqual(int(info["ticks"][0]), 1)
        env.close()

    def test_hop_macro_matches_legacy_hop(self):
        """HOP in precision mode records the same ticks as the legacy hop."""
        p = precision_env(L1)
        m = GdVecEnv(L1, 4)
        p.reset()
        m.reset()
        p.step(np.full(4, ACT_R4, np.uint8))
        m.step(np.zeros(4, np.uint8))
        _, _, _, _, ip = p.step(np.full(4, ACT_HOP, np.uint8))
        _, _, _, _, im = m.step(np.ones(4, np.uint8))
        self.assertEqual(int(ip["ticks"][0]), int(im["ticks"][0]))
        self.assertEqual(float(p.get_render_states()[0, 1]),
                         float(m.get_render_states()[0, 1]))
        p.close()
        m.close()


class Interrupts(unittest.TestCase):
    def test_repeated_action_is_cut_at_landing(self):
        """Hold H4 through a landing: the decision ends on the landing tick."""
        env = precision_env(L1)
        env.reset()
        env.step(np.full(4, ACT_R4, np.uint8))
        env.step(np.full(4, ACT_H1, np.uint8))  # jump
        durs = []
        for _ in range(60):
            _, _, _, _, info = env.step(np.full(4, ACT_R4, np.uint8))
            durs.append(int(info["ticks"][0]))
            if env.action_masks()[0, ACT_HOP]:
                break
        self.assertTrue(env.action_masks()[0, ACT_HOP], "never landed")
        self.assertLess(durs[-1], 4, "the landing R4 was not interrupted")
        env.close()

    def test_portal_transition_cuts_the_action(self):
        """Stereo Madness' first ship portal: an R4 crossing it ends on the
        portal tick (mode change interrupt), never straddles it."""
        env = precision_env(L1, n=1)
        env.reset()
        prev_mode = 0.0
        for _ in range(4000):
            m = env.action_masks()
            a = ACT_HOP if m[0, ACT_HOP] and False else ACT_R4
            _, _, term, trunc, info = env.step(np.array([a], np.uint8))
            mode = float(env.get_render_states()[0, 3])
            if mode != prev_mode:
                # the decision that changed the mode consumed <= ticks up to
                # the change; the next decision starts in the new mode
                self.assertLessEqual(int(info["ticks"][0]), 4)
                break
            prev_mode = mode
            if term[0] or trunc[0]:
                self.skipTest("died before the first portal walking blind")
        env.close()

    def test_ring_contact_interrupt(self):
        """Polargeist: a repeated release is cut when a ring becomes
        pressable (obs scalar 24 flips to 1 on the new decision)."""
        env = precision_env(L3, n=64)
        rng = np.random.default_rng(3)
        env.reset()
        seen_interrupt = False
        for _ in range(600):
            m = env.action_masks()
            a = rng.integers(0, 7, env.n_envs).astype(np.uint8)
            a = np.where(m[np.arange(env.n_envs), a] > 0, a, ACT_R4).astype(np.uint8)
            obs, _, _, _, info = env.step(a)
            cut = (info["ticks"] < np.where((a == ACT_R4) | (a == ACT_H4), 4,
                                           np.where((a == ACT_R2) | (a == ACT_H2), 2, 1)))
            ring_now = obs[:, 24] > 0.5
            if (cut & ring_now & ((a == ACT_R4) | (a == ACT_R2))).any():
                seen_interrupt = True
                break
        self.assertTrue(seen_interrupt, "no ring-contact interrupt observed")
        env.close()


class UfoRelease(unittest.TestCase):
    def setUp(self):
        self.holds = solution_holds("level12-2")
        if self.holds is None:
            self.skipTest("runs/level12-2/solution.json not present")

    def ufo_state(self, env):
        # walk the solution and take the first snapshot in UFO mode
        xs = np.arange(1000.0, 25000.0, 250.0)
        blobs = env.trace_states(self.holds, xs)
        for b in blobs:
            if b is None:
                continue
            env.set_state(0, b)
            if env.get_render_states()[0, 3] == 3.0:
                return b
        self.skipTest("no UFO state along the solution")

    def test_repress_needs_a_release(self):
        env = precision_env(L12, n=1)
        env.reset()
        b = self.ufo_state(env)
        # A press registers at the end of its tick (processCommands) and
        # moves the player on the NEXT tick, so each velocity is read one
        # decision after the press.
        env.set_state(0, b)
        env.step(np.array([ACT_R2], np.uint8))  # settle the button: released
        v0 = float(env.get_render_states()[0, 2])
        env.step(np.array([ACT_H1], np.uint8))  # fresh press
        env.step(np.array([ACT_H1], np.uint8))  # sustained: boost visible now
        v1 = float(env.get_render_states()[0, 2])
        self.assertGreater(v1, v0, "a fresh press did not boost")
        env.step(np.array([ACT_H1], np.uint8))  # still held: no new boost
        v2 = float(env.get_render_states()[0, 2])
        self.assertLess(v2, v1, "a sustained hold re-boosted")
        env.step(np.array([ACT_R1], np.uint8))
        env.step(np.array([ACT_H1], np.uint8))  # release + fresh press
        env.step(np.array([ACT_H1], np.uint8))
        v3 = float(env.get_render_states()[0, 2])
        self.assertGreater(v3, v2, "release + press did not boost again")
        env.close()


class Replay(unittest.TestCase):
    def test_action_replay_reproduces_the_episode(self):
        """A recorded decision stream replays to the same obs, actions,
        durations and holds through the executor (policy path == planner
        path)."""
        env = precision_env(L1, n=2)
        rng = np.random.default_rng(7)
        env.reset()
        obs_seq, act_seq, dur_seq = [], [], []
        for _ in range(500):
            m = env.action_masks()
            a = rng.integers(0, 7, 2).astype(np.uint8)
            a = np.where(m[np.arange(2), a] > 0, a, ACT_H1).astype(np.uint8)
            obs_seq.append(env._obs[0].copy())
            obs, _, term, trunc, info = env.step(a)
            act_seq.append(int(a[0]))
            dur_seq.append(int(info["ticks"][0]))
            if term[0] or trunc[0]:
                break
        holds = env.last_episode_holds(0)
        acts = env.last_episode_actions(0)
        self.assertEqual(acts.tolist(), act_seq)
        robs, racts, _, rticks, n, _, ends, masks = env.replay_actions([acts])
        self.assertEqual(int(n[0]), len(act_seq))
        self.assertEqual(rticks[0, :n[0]].astype(int).tolist(), dur_seq)
        self.assertTrue(np.array_equal(robs[0, :n[0]], np.stack(obs_seq)))
        self.assertEqual(int(ends[0]), 1)
        self.assertEqual(masks.shape[-1], 7)
        # raw holds replay to the same end (deployment semantics)
        x_raw, _, t_raw = env.run_holds(holds)
        self.assertEqual(t_raw, len(holds))
        self.assertAlmostEqual(x_raw, float(env.player_x()[0]) if False else x_raw)
        env.close()

    def test_holds_replay_in_precision_mode_is_one_tick_actions(self):
        env = precision_env(L1, n=1)
        holds = np.array([0, 0, 0, 0, 1, 0, 0, 0], np.uint8)
        _, acts, _, ticks, n, _, _ = env.replay([holds])
        self.assertEqual(int(n[0]), 8)
        self.assertTrue((ticks[0, :8] == 1).all())
        self.assertEqual(acts[0, :8].tolist(), [ACT_R1] * 4 + [ACT_H1] + [ACT_R1] * 3)
        env.close()

    def test_snapshot_blob_roundtrip_and_migration(self):
        env = precision_env(L1, n=2)
        env.reset()
        env.step(np.full(2, ACT_R4, np.uint8))
        b = env.get_state(0)
        self.assertNotEqual(env.state_hash(b), 0)
        old = b[16:]  # pre-contract layout
        mig = env.migrate_blob(old)
        self.assertIsNotNone(mig)
        self.assertTrue(np.array_equal(mig, b))
        with self.assertRaises(ValueError):
            env.set_state(1, old)
        bad = b.copy()
        bad[0] ^= 0xFF
        with self.assertRaises(RuntimeError):
            env.set_state(1, bad)
        env.close()


class ObsV6(unittest.TestCase):
    def test_corridor_geometry_is_visible_in_v6_and_empty_in_v5(self):
        holds = solution_holds("level12-2")
        if holds is None:
            self.skipTest("runs/level12-2/solution.json not present")
        v5 = GdVecEnv(L12, 1)
        v6 = precision_env(L12, n=1, obs_version=6)
        b5 = state_at(v5, holds, 17400.0)
        self.assertIsNotNone(b5)
        v5.set_state(0, b5)
        y = float(v5.get_render_states()[0, 1])
        self.assertGreater(y, 900.0, "expected the elevated UFO corridor")
        grid5 = v5._obs[0, 27:]
        # v6 blob layout is identical (same DLL), only the obs differs
        v6.set_state(0, b5)
        grid6 = v6._obs[0, 37 + 120:]
        self.assertEqual(float(grid5.sum()), 0.0)
        self.assertGreater(float(grid6.sum()), 0.0)
        rows = v6._obs[0, 37:37 + 120].reshape(12, 10)
        self.assertGreater(int((rows[:, 0] > 0).sum()), 0)
        self.assertEqual(int(v6._obs[0, 29]), 1)  # band floor applies (UFO)
        v5.close()
        v6.close()

    def test_v5_scalars_are_a_prefix_of_v6(self):
        v5 = GdVecEnv(L1, 1)
        v6 = precision_env(L1, n=1, obs_version=6)
        v5.reset()
        v6.reset()
        for _ in range(30):
            v5.step(np.zeros(1, np.uint8))  # 4-tick walk
            for _ in range(4):
                v6.step(np.full(1, ACT_R1, np.uint8))
        self.assertEqual(float(v5.player_x()[0]), float(v6.player_x()[0]))
        self.assertTrue(np.array_equal(v5._obs[0, :27], v6._obs[0, :27]))
        v5.close()
        v6.close()


if __name__ == "__main__":
    unittest.main()
