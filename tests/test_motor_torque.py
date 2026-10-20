"""CPU tests of the shared ``motor_torque`` summary block (``dropbear_wbc.isaac.motor_torque``) used by
``scripts/play_locomotion.py`` and ``scripts/play.py`` (item 54: tracking summaries get the block so
``tools/scoreboard.py`` can torque-gate tracking skills). Fake duck-typed actuators; no Isaac."""
from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

torch = pytest.importorskip("torch")

from dropbear_wbc.isaac.motor_torque import KEYS, MotorTorqueSampler, motor_columns, tracking_sample_mask  # noqa: E402
from dropbear_wbc.motion.names import MOTOR_NAMES  # noqa: E402
from tools import scoreboard as sb  # noqa: E402

ROOT = sdk_test_paths.ROOT
PROFILE = "hw_v1"


def _robot(num_envs: int, peak: dict[str, float], seed: int = 0):
    """Fake ``robot.actuators``: motors split over 3 groups in a scrambled order (column order != MOTOR_NAMES)."""
    rng = np.random.default_rng(seed)
    names = list(MOTOR_NAMES)
    rng.shuffle(names)
    groups = {}
    for g, chunk in enumerate((names[:7], names[7:15], names[15:])):
        chunk = list(chunk) + ([f"neck_{g}"] if g == 1 else [])  # a non-motor joint in a group is ignored
        lim = torch.tensor([[peak.get(n, 5.0) for n in chunk]] * num_envs)
        groups[f"g{g}"] = SimpleNamespace(joint_names=chunk, effort_limit=lim,
                                          applied_effort=torch.zeros(num_envs, len(chunk)),
                                          computed_effort=torch.zeros(num_envs, len(chunk)))
    return groups


def _write(acts, applied: torch.Tensor, computed: torch.Tensor) -> None:
    """Write [E, M] (MOTOR_NAMES order) into the fake actuators' columns."""
    for act in acts.values():
        for k, n in enumerate(act.joint_names):
            if n in MOTOR_NAMES:
                i = MOTOR_NAMES.index(n)
                act.applied_effort[:, k] = applied[:, i]
                act.computed_effort[:, k] = computed[:, i]


def _old_play_locomotion_block(tq_samples, clip_samples, vel_samples, peak_lim):
    """Verbatim reduction of scripts/play_locomotion.py before item 54 (the refactor must not change numbers)."""
    tq, cl, vl = torch.cat(tq_samples), torch.cat(clip_samples).float(), torch.cat(vel_samples)
    return {name: {"peak_limit_Nm": round(float(peak_lim[i]), 2), "max_Nm": round(float(tq[:, i].max()), 2),
                   "p95_Nm": round(float(torch.quantile(tq[:, i], 0.95)), 2),
                   "rms_Nm": round(float(tq[:, i].pow(2).mean().sqrt()), 2),
                   "clip_frac": round(float(cl[:, i].mean()), 4),
                   "max_speed_rad_s": round(float(vl[:, i].max()), 2),
                   "p95_speed_rad_s": round(float(torch.quantile(vl[:, i], 0.95)), 2)}
            for i, name in enumerate(MOTOR_NAMES)}


def _rollout(sampler, acts, steps, num_envs, peak, rated, no_load, seed=1, clip_motor=None, clip_every=0,
             mask_fn=None, groups=None):
    """Random torques (|tau| mostly ~0.5 rated, rare spikes to 0.8 peak), speeds <= 0.6 no-load. Returns the
    old-style sample lists for cross-checking."""
    g = torch.Generator().manual_seed(seed)
    M = len(MOTOR_NAMES)
    pk = torch.tensor([peak[n] for n in MOTOR_NAMES])
    rt = torch.tensor([rated[n] for n in MOTOR_NAMES])
    nl = torch.tensor([no_load[n] for n in MOTOR_NAMES])
    old_t, old_c, old_v = [], [], []
    for s in range(steps):
        sign = torch.where(torch.rand(num_envs, M, generator=g) < 0.5, -1.0, 1.0)
        tau = sign * rt * 0.5 * torch.rand(num_envs, M, generator=g) * 1.6
        spike = torch.rand(num_envs, M, generator=g) < 0.01
        tau = torch.where(spike, sign * 0.8 * pk, tau)
        comp = tau.clone()
        if clip_motor is not None and clip_every and s % clip_every == 0:
            i = MOTOR_NAMES.index(clip_motor)
            comp[:, i] = tau[:, i] + 0.5 * pk[i]  # PD asked for more than the model delivered
        vel = (torch.rand(num_envs, M, generator=g) * 2 - 1) * 0.6 * nl
        _write(acts, tau, comp)
        mask = mask_fn(s) if mask_fn else torch.ones(num_envs, dtype=torch.bool)
        sampler.add(vel, mask, groups=groups)
        if mask.any():
            old_t.append(tau[mask].abs())
            old_c.append((comp[mask] - tau[mask]).abs() > 0.01 * pk)
            old_v.append(vel[mask].abs())
    return old_t, old_c, old_v, pk


@pytest.fixture(scope="module")
def specs():
    sp = sb.motor_specs(PROFILE)
    return ({n: v["peak"] for n, v in sp.items()}, {n: v["rated"] for n, v in sp.items()},
            {n: v["no_load"] for n, v in sp.items()})


def _tracking_summary(block, per_clip=None, profile=PROFILE):
    s = {"tool": "synthetic", "num_envs": 8, "actuator_profile": profile, "created": "2026-10-14T10:30:00+05:30",
         "finite": True, "falls": {"total": 0, "envs_with_a_fall": 0},
         "eval_design": {"perturbed": True, "randomization": "training-level: push_robot"}}
    if block is not None:
        s["motor_torque"] = block
    if per_clip is not None:
        s["per_clip"] = per_clip
    return s


def test_motor_columns_order_and_coverage(specs):
    acts = _robot(2, specs[0])
    cols = motor_columns(acts, MOTOR_NAMES)
    assert sorted(i for i, _, _ in cols) == list(range(len(MOTOR_NAMES)))
    for i, act, k in cols:
        assert act.joint_names[k] == MOTOR_NAMES[i]
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 2, "cpu")
    assert [round(float(x), 2) for x in s.peak_lim] == [round(specs[0][n], 2) for n in MOTOR_NAMES]


def test_missing_or_duplicate_motor_raises(specs):
    acts = _robot(2, specs[0])
    acts["g0"].joint_names = acts["g0"].joint_names[1:] + ["other"]  # drop one motor
    with pytest.raises(ValueError, match="no actuator group"):
        motor_columns(acts, MOTOR_NAMES)
    acts = _robot(2, specs[0])
    acts["g2"].joint_names = acts["g2"].joint_names[:-1] + [acts["g0"].joint_names[0]]
    with pytest.raises(ValueError):
        motor_columns(acts, MOTOR_NAMES)


def test_empty_block_is_none(specs):
    acts = _robot(3, specs[0])
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 3, "cpu")
    assert s.add(torch.zeros(3, len(MOTOR_NAMES)), torch.zeros(3, dtype=torch.bool)) == 0
    assert s.block("n") is None and s.group_blocks() == {}


def test_matches_old_play_locomotion_reduction(specs):
    peak, rated, nl = specs
    acts = _robot(6, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 6, "cpu")
    alive = lambda st: torch.tensor([True, True, st < 20, True, st % 3 != 0, False])  # noqa: E731
    old = _rollout(s, acts, 60, 6, peak, rated, nl, clip_motor="LL_knee_actuator_joint", clip_every=7, mask_fn=alive)
    blk = s.block("note", env_steps=360)
    assert blk["per_motor"] == _old_play_locomotion_block(*old)
    assert blk["samples"] == sum(int(alive(st).sum()) for st in range(60)) and blk["env_steps"] == 360
    assert list(blk["per_motor"]) == list(MOTOR_NAMES)
    assert all(tuple(v) == KEYS for v in blk["per_motor"].values())


def test_stats_against_numpy(specs):
    peak, rated, nl = specs
    acts = _robot(4, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 4, "cpu")
    t, c, v, _ = _rollout(s, acts, 50, 4, peak, rated, nl, clip_motor=MOTOR_NAMES[0], clip_every=5)
    tq, cl, vl = (torch.cat(x).double().numpy() for x in (t, c, v))
    blk = s.block("n")["per_motor"]
    for i, n in enumerate(MOTOR_NAMES):
        m = blk[n]
        assert m["max_Nm"] == pytest.approx(tq[:, i].max(), abs=0.006)
        assert m["rms_Nm"] == pytest.approx(np.sqrt((tq[:, i] ** 2).mean()), abs=0.006)
        assert m["p95_Nm"] == pytest.approx(np.quantile(tq[:, i], 0.95), abs=0.006)
        assert m["clip_frac"] == pytest.approx(cl[:, i].mean(), abs=6e-5)
        assert m["max_speed_rad_s"] == pytest.approx(vl[:, i].max(), abs=0.006)
        assert m["p95_speed_rad_s"] == pytest.approx(np.quantile(vl[:, i], 0.95), abs=0.006)
    clipped = [n for n in MOTOR_NAMES if blk[n]["clip_frac"] > 0]
    assert len(clipped) == 1 and blk[clipped[0]]["clip_frac"] == pytest.approx(0.2, abs=1e-4)


def test_clip_threshold_is_one_percent_of_peak(specs):
    peak = specs[0]
    acts = _robot(1, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 1, "cpu")
    pk = torch.tensor([[peak[n] for n in MOTOR_NAMES]])
    _write(acts, 0.5 * pk, 0.5 * pk + 0.009 * pk)  # under 1 %: not clipped
    s.add(torch.zeros(1, len(MOTOR_NAMES)), torch.ones(1, dtype=torch.bool))
    _write(acts, -0.5 * pk, -0.5 * pk - 0.02 * pk)  # 2 % (negative side): clipped
    s.add(torch.zeros(1, len(MOTOR_NAMES)), torch.ones(1, dtype=torch.bool))
    assert {v["clip_frac"] for v in s.block("n")["per_motor"].values()} == {0.5}


def test_tracking_mask_resets_and_wraps():
    """Single clip T=5, 3 envs; env 1 falls at step 6; reset_at_wrap rewrites at the clip end."""
    T, E = 5, 3
    ts = torch.zeros(E, dtype=torch.long)
    got_wrap, got_cont = [], []
    for step in range(12):
        prev = ts.clone()
        done = torch.zeros(E, dtype=torch.bool)
        ts = ts + 1
        if step == 6:
            done[1] = True
            ts[1] = 1  # Isaac Lab: reset -> _resample (0) -> command update (+1)
        ts = torch.where(ts >= T, torch.zeros_like(ts), ts)
        got_wrap.append(tracking_sample_mask(done, prev, ts, True).tolist())
        got_cont.append(tracking_sample_mask(done, prev, ts, False).tolist())
    wrap_steps = [i for i, m in enumerate(got_wrap) if not m[0]]
    assert wrap_steps == [4, 9]  # env 0 wraps every T steps -> excluded
    assert got_wrap[6] == [True, False, True]  # the fallen env is excluded on its reset step only
    assert [i for i, m in enumerate(got_wrap) if not m[1]] == [4, 6, 10]  # env 1: wrap, reset, wrap 5 steps later
    assert all(m[0] for m in got_cont) and got_cont[6] == [True, False, True]


def test_group_blocks_per_clip(specs):
    peak, rated, nl = specs
    acts = _robot(4, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 4, "cpu")
    clip = torch.tensor([0, 1, 0, 1])
    _rollout(s, acts, 30, 4, peak, rated, nl, groups=clip)
    gb = s.group_blocks()
    assert set(gb) == {0, 1} and gb[0]["samples"] == gb[1]["samples"] == 60
    pooled = s.block("n")["per_motor"]
    for n in MOTOR_NAMES:
        assert pooled[n]["max_Nm"] == max(gb[0]["per_motor"][n]["max_Nm"], gb[1]["per_motor"][n]["max_Nm"])


def test_scoreboard_tracking_pass_with_block_fail_without(specs):
    """(a) clean hw_v1 numbers through the sampler -> tracking PASS; (b) same summary without the block -> FAIL."""
    peak, rated, nl = specs
    acts = _robot(8, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 8, "cpu")
    _rollout(s, acts, 100, 8, peak, rated, nl)
    blk = s.block("synthetic")
    r = sb.evaluate("tracking", _tracking_summary(blk))
    assert r["pass"], r["why"]
    assert 0.79 <= r["torque"]["max_over_peak"]["value"] <= 0.81
    assert r["torque"]["rms_over_rated"]["value"] < 1.0
    r = sb.evaluate("tracking", _tracking_summary(None))
    assert not r["pass"] and r["why"] == ["torque UNKNOWN (no motor_torque block)"]


def test_scoreboard_tracking_library_shape_and_clip_fail(specs):
    peak, rated, nl = specs
    acts = _robot(4, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 4, "cpu")
    clip = torch.tensor([0, 1, 0, 1])
    _rollout(s, acts, 50, 4, peak, rated, nl, clip_motor=MOTOR_NAMES[3], clip_every=10, groups=clip)
    gb = s.group_blocks()
    per_clip = {name: {"envs": [e for e in range(4) if int(clip[e]) == c], "falls": 0, "motor_torque": gb[c]}
                for c, name in enumerate(("walk", "wave"))}
    r = sb.evaluate("tracking", _tracking_summary(s.block("n"), per_clip=per_clip))
    assert r["scenarios"] == 2 and r["envs"] == 4
    assert not r["pass"] and any(w.startswith("clip 10.0%") and MOTOR_NAMES[3] in w for w in r["why"])


def test_scoreboard_rejects_legacy_profile_even_with_block(specs):
    peak, rated, nl = specs
    acts = _robot(2, peak)
    s = MotorTorqueSampler(acts, MOTOR_NAMES, 2, "cpu")
    _rollout(s, acts, 10, 2, peak, rated, nl)
    r = sb.evaluate("tracking", _tracking_summary(s.block("n"), profile="legacy"))
    assert not r["pass"] and any("no motor map" in w for w in r["why"])


def _calls(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            for n in ast.walk(tree) if isinstance(n, ast.Call)}


@pytest.mark.parametrize("script", ["scripts/play.py", "scripts/play_locomotion.py"])
def test_play_scripts_use_shared_sampler(script):
    src = (ROOT / script).read_text(encoding="utf-8")
    calls = _calls(script)
    assert {"MotorTorqueSampler", "add", "block"} <= calls
    assert 'summary["motor_torque"] = mt' in src
    assert '"actuator_profile": profile' in src


def test_play_py_tracking_mask_and_per_clip():
    src = (ROOT / "scripts/play.py").read_text(encoding="utf-8")
    assert "tracking_sample_mask(d, ts_prev, command.time_steps, wrap_rewrites)" in src
    assert "torque.add(robot.data.joint_vel[:, motor_ids], tq_ok, groups=env_clip if is_library else None)" in src
    assert "ts_prev = command.time_steps.clone()" in src
    assert 'per_clip[name]["motor_torque"] = clip_torque[c]' in src
    assert '"perturbed": bool(args.perturbed)' in src
