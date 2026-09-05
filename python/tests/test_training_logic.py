"""Trainer-side logic tests: duration-aware GAE / truncation bootstrapping,
certificate replacement bookkeeping, planner state distinctions and the
categorical masked head.

Run from python/:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from gdrl.certificate import (Attachments, Certificate, common_prefix_ticks,
                              widen_schedule)
from gdrl.env import ACT_H1, ACT_R1, ACT_R4, GdVecEnv
from gdrl.model import (categorical_entropy, categorical_logprob, make_model,
                        masked_logits)
from gdrl.ppo import RolloutBuffer

LEVELS = Path("S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels")
L1 = str(LEVELS / "1.txt")


class Gae(unittest.TestCase):
    def _buf(self, rewards, values, durs, terminated, n_actions=7):
        T = len(rewards)
        buf = RolloutBuffer(T, 1, 3, torch.device("cpu"), n_actions)
        for t in range(T):
            alv = np.array([[0.0], [0.0], [values[t]]], np.float32)
            buf.add(alv, np.array([rewards[t]], np.float32),
                    np.array([0.9995 ** durs[t]], np.float32),
                    np.array([terminated[t]], np.float32),
                    durations=np.array([durs[t]], np.float32))
        return buf

    def test_duration_lambda_matches_hand_computation(self):
        gamma, lam = 0.9995, 0.95
        rewards = [1.0, 0.5, 2.0]
        values = [0.3, 0.2, 0.1]
        durs = [1, 4, 100]
        buf = self._buf(rewards, values, durs, [0, 0, 0])
        last_v = np.array([0.4], np.float32)
        adv, ret = buf.compute_advantages(last_v, lam, duration_lambda=True)
        # hand scan
        gae = 0.0
        want = []
        nxt = [values[1], values[2], last_v[0]]
        for t in reversed(range(3)):
            g = gamma ** durs[t]
            lt = lam ** (durs[t] / 4.0)
            delta = rewards[t] + g * nxt[t] - values[t]
            gae = delta + g * lt * gae
            want.append(gae)
        want = want[::-1]
        for t in range(3):
            self.assertAlmostEqual(float(adv[t, 0]), want[t], places=5)
            self.assertAlmostEqual(float(ret[t, 0]), want[t] + values[t], places=5)

    def test_plain_lambda_is_the_legacy_scan(self):
        buf = self._buf([1.0, 1.0], [0.0, 0.0], [100, 4], [0, 0], n_actions=2)
        adv_d, _ = buf.compute_advantages(np.zeros(1, np.float32), 0.95, True)
        adv_p, _ = buf.compute_advantages(np.zeros(1, np.float32), 0.95, False)
        # the last step has no carry -> identical; the first step's carry
        # uses lambda^(100/4) vs lambda -> different
        self.assertAlmostEqual(float(adv_d[1, 0]), float(adv_p[1, 0]))
        self.assertNotAlmostEqual(float(adv_d[0, 0]), float(adv_p[0, 0]))

    def test_termination_stops_bootstrap_and_carry(self):
        buf = self._buf([1.0, 5.0], [0.5, 0.5], [4, 4], [1, 0])
        adv, _ = buf.compute_advantages(np.array([9.0], np.float32), 0.95, True)
        # step 0 terminated: advantage = r - V, no next value, no carry
        self.assertAlmostEqual(float(adv[0, 0]), 1.0 - 0.5, places=6)

    def test_truncation_bootstrap_is_folded_into_reward(self):
        """The trainer adds gamma^dur * V(final) to a truncated step's
        reward and flags it terminated; GAE then treats it as terminal."""
        g = 0.9995 ** 4
        v_final = 2.0
        buf = self._buf([1.0 + g * v_final], [0.5], [4], [1])
        adv, ret = buf.compute_advantages(np.zeros(1, np.float32), 0.95, True)
        self.assertAlmostEqual(float(ret[0, 0]), 1.0 + g * v_final, places=6)


class CertificateTests(unittest.TestCase):
    def test_common_prefix(self):
        a = np.array([0, 1, 1, 0], np.uint8)
        self.assertEqual(common_prefix_ticks(a, np.array([0, 1, 0, 0], np.uint8)), 2)
        self.assertEqual(common_prefix_ticks(a, a), 4)
        self.assertEqual(common_prefix_ticks(a, np.array([0, 1, 1, 0, 1], np.uint8)), 4)
        self.assertEqual(common_prefix_ticks(np.zeros(0, np.uint8), a), 0)

    def test_replace_ids_and_orphans(self):
        c = Certificate()
        att = Attachments(4)
        cid, common = c.replace(np.array([0, 0, 1, 1, 0, 0], np.uint8), 600.0)
        self.assertEqual((cid, common), (1, 0))
        att.attach(0, cid, 4)  # cut at tick 4
        att.attach(1, cid, 2)  # cut at tick 2 (widened frontier)
        att.attach(2, cid, 0)  # from x=0
        att.detach(3)          # a respawn practice episode
        # extension: same prefix, longer -> nobody is orphaned
        cid2, common = c.replace(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.uint8), 900.0)
        self.assertEqual(cid2, 2)
        self.assertEqual(common, 6)
        self.assertEqual(att.reattach(cid2, common).tolist(), [])
        self.assertTrue((att.cert_id[:2] == cid2).all())
        self.assertFalse(att.stitched[2])  # from x=0: no prefix to attach
        # an earlier section is replaced at tick 3 -> env 0 (prefix 4) is
        # orphaned, env 1 (prefix 2) and env 2 (prefix 0) attach
        cid3, common = c.replace(np.array([0, 0, 1, 0, 1, 1, 0, 0, 1], np.uint8), 950.0)
        self.assertEqual(common, 3)
        self.assertEqual(att.reattach(cid3, common).tolist(), [0])
        self.assertEqual(int(att.cert_id[1]), cid3)
        self.assertEqual(int(att.cert_id[0]), cid2)
        # identical holds keep the id
        cid4, _ = c.replace(c.holds.copy(), 951.0)
        self.assertEqual(cid4, cid3)

    def test_widen_schedule(self):
        self.assertEqual(widen_schedule(600.0, 0), 600.0)
        self.assertEqual(widen_schedule(600.0, 2), 600.0)
        self.assertEqual(widen_schedule(600.0, 3), 1200.0)
        self.assertEqual(widen_schedule(600.0, 6), 2400.0)
        self.assertEqual(widen_schedule(600.0, 9), 4800.0)
        self.assertEqual(widen_schedule(600.0, 30), 4800.0)


class SolverStates(unittest.TestCase):
    def test_exact_key_distinguishes_input_flags(self):
        """Two airborne states with identical physics but different button
        history (held vs released) are different planner states."""
        env = GdVecEnv(L1, 2, action_mode="precision", obs_version=6)
        env.reset()
        env.step(np.full(2, ACT_R4, np.uint8))
        env.step(np.full(2, ACT_H1, np.uint8))  # both jump
        env.step(np.array([ACT_H1, ACT_R1], np.uint8))  # one keeps holding
        rs = env.get_render_states()
        self.assertEqual(float(rs[0, 1]), float(rs[1, 1]))  # same y
        b0, b1 = env.get_state(0), env.get_state(1)
        self.assertNotEqual(env.state_hash(b0), env.state_hash(b1))
        self.assertEqual(env.state_hash(b0), env.state_hash(b0.copy()))
        env.close()

    def test_budget_is_reported_apart_from_exhaustion(self):
        env = GdVecEnv(L1, 1, action_mode="precision", obs_version=6)
        s = env.solve([env.fresh_blob()], 800.0, max_nodes=200000)[0]
        self.assertEqual(s["status"], "found")
        self.assertEqual(s["stage"], 1)
        tiny = env.solve([env.fresh_blob()], 5000.0, max_nodes=50)[0]
        self.assertEqual(tiny["status"], "node_budget")
        self.assertIsNone(tiny["holds"])
        env.close()

    def test_found_path_replays_to_target(self):
        env = GdVecEnv(L1, 1, action_mode="precision", obs_version=6)
        s = env.solve([env.fresh_blob()], 1500.0)[0]
        self.assertEqual(s["status"], "found")
        self.assertEqual(int(s["durations"].sum()), len(s["holds"]))
        _, _, _, ticks, n, fo, ends, _ = env.replay_actions([s["actions"]])
        self.assertEqual(int(n[0]), len(s["actions"]))
        self.assertGreaterEqual(float(fo[0, 5] * env.end_x), 1500.0)
        self.assertEqual(int(ends[0]), 0)
        x_raw, _, _ = env.run_holds(s["holds"])
        self.assertGreaterEqual(x_raw, 1500.0)
        env.close()


class CategoricalHead(unittest.TestCase):
    def test_masked_logprob_and_entropy(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        lp = categorical_logprob(logits, torch.tensor([2]), mask)
        self.assertLess(float(lp), -1000.0)  # masked action has ~zero prob
        lp1 = categorical_logprob(logits, torch.tensor([1]), mask)
        want = torch.log_softmax(torch.tensor([1.0, 2.0, 0.0]), 0)[1]
        self.assertAlmostEqual(float(lp1), float(want), places=5)
        ent = categorical_entropy(logits, mask)
        self.assertGreater(float(ent), 0.0)
        self.assertTrue(torch.isfinite(ent).all())

    def test_model_masks_and_bounds(self):
        m = make_model(37 + 120 + 1536, "cnn", 4.0, 0.1, n_actions=7,
                       obs_version=6)
        obs = torch.randn(5, 37 + 120 + 1536)
        mask = torch.ones(5, 7)
        mask[:, 6] = 0.0
        logit, value = m(obs, mask=mask)
        self.assertEqual(tuple(logit.shape), (5, 7))
        self.assertTrue((logit[:, 6] < -1000).all())
        self.assertTrue((logit[:, :6].abs() <= 4.0 + 1e-5).all())
        out = m.act(obs, mask=mask)
        self.assertEqual(tuple(out.shape), (3, 5))
        self.assertTrue((out[0] != 6).all())
        lp, ent, v = m.evaluate(obs, out[0], torch.full((5,), 4.0), mask)
        self.assertTrue(torch.allclose(lp, out[1], atol=1e-5))
        det = m.act(obs, deterministic=True, mask=mask)
        self.assertTrue((det[0] != 6).all())


if __name__ == "__main__":
    unittest.main()
