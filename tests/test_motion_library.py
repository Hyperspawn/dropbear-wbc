"""CPU tests of the motion library (``tasks/tracking/motion_library.py``): manifest, fail-closed consistency checks,
concatenation offsets, and the (clip, time-bin) sampler -- including exact equivalence with the single-clip
BeyondMimic sampler of ``mdp/commands.py`` when the library has one clip. Needs numpy + torch (system Python)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

torch = pytest.importorskip("torch")

from dropbear_wbc.robots import dropbear_names as N  # noqa: E402
from dropbear_wbc.tasks.tracking import motion_library as ML  # noqa: E402
from dropbear_wbc.tasks.tracking.motion_npz import load_motion_npz, save_motion_npz  # noqa: E402

ROOT = sdk_test_paths.ROOT
SYN = ROOT / "data" / "motions" / "synthetic"
MANIFEST = ROOT / "data" / "motions" / "libraries" / "accepted_v0.json"
CLIPS = ("stand", "wave_right", "arm_swing", "weight_shift", "squat_lite")
REJECTED = ROOT / "data" / "motions" / "kimodo_g1" / "output_wave.npz"


def _need(*paths: Path):
    for p in paths:
        if not p.is_file():
            pytest.skip(f"{p} not present")


def _manifest(tmp: Path, entries: list, name: str = "t") -> Path:
    p = tmp / f"{name}.json"
    p.write_text(json.dumps({"schema": ML.LIBRARY_SCHEMA, "name": name, "clips": entries}), encoding="utf-8")
    return p


def _copy_clip(src: Path, dst_dir: Path, stem: str, with_verdict: bool = True) -> Path:
    dst = dst_dir / f"{stem}.npz"
    shutil.copyfile(src, dst)
    v = src.with_name(src.stem + ".validation.json")
    if with_verdict and v.is_file():
        shutil.copyfile(v, dst_dir / f"{stem}.validation.json")
    return dst


# ------------------------------------------------------------------------------------------------ manifest
def test_manifest_parsing_and_errors(tmp_path):
    m = ML.parse_manifest({"schema": ML.LIBRARY_SCHEMA, "clips": ["a/x.npz", {"npz": "b/y.npz", "weight": 2, "name": "yy"}]},
                          base_dir=tmp_path)
    assert [c.name for c in m.clips] == ["x", "yy"] and m.clips[1].weight == 2.0
    assert m.clips[0].npz == (tmp_path / "a" / "x.npz").resolve()
    bad = [
        {"schema": "nope", "clips": ["x.npz"]},
        {"schema": ML.LIBRARY_SCHEMA, "clips": []},
        {"schema": ML.LIBRARY_SCHEMA, "clips": [{"npz": "x.npz", "weight": 0}]},
        {"schema": ML.LIBRARY_SCHEMA, "clips": [{"npz": "x.npz", "weight": float("nan")}]},
        {"schema": ML.LIBRARY_SCHEMA, "clips": [{"npz": "a/x.npz"}, {"npz": "b/x.npz"}]},  # duplicate names
        {"schema": ML.LIBRARY_SCHEMA, "clips": [{"npz": "x.npz", "name": "a"}, {"npz": "x.npz", "name": "b"}]},
    ]
    for d in bad:
        with pytest.raises(ML.LibraryFormatError):
            ML.parse_manifest(d, base_dir=tmp_path)
    with pytest.raises(ML.LibraryFormatError):
        ML.load_manifest(tmp_path / "missing.json")


# ------------------------------------------------------------------------------------------------ building
def test_accepted_manifest_concatenation_and_offsets():
    _need(MANIFEST, *(SYN / f"{c}.npz" for c in CLIPS))
    data = ML.build_library(MANIFEST, keep_body_names=list(N.TRACKED_BODIES), expected_fps=50.0,
                            expected_usd_sha256=N.USD_SHA256, expected_authored_ankle=False)
    assert [c.name for c in data.clips] == list(CLIPS)
    lengths = [load_motion_npz(SYN / f"{c}.npz").num_frames for c in CLIPS]
    assert data.lengths.tolist() == lengths
    assert data.starts.tolist() == [0] + np.cumsum(lengths)[:-1].tolist()
    assert data.num_frames_total == sum(lengths)
    assert data.kept_body_names == [N.ROOT_BODY] + list(N.TRACKED_BODIES)
    for c, name in enumerate(CLIPS):  # every clip's rows are exactly its NPZ (joints: all 91; bodies: root + tracked)
        m = load_motion_npz(SYN / f"{name}.npz")
        sl = data.clip_slice(c)
        np.testing.assert_array_equal(data.joint_pos[sl], m.joint_pos)
        np.testing.assert_array_equal(data.joint_vel[sl], m.joint_vel)
        bi = [m.body_names.index(b) for b in data.kept_body_names]
        np.testing.assert_array_equal(data.body_pos_w[sl], m.body_pos_w[:, bi])
        np.testing.assert_array_equal(data.body_quat_w[sl], m.body_quat_w[:, bi])
        np.testing.assert_array_equal(data.body_lin_vel_w[sl], m.body_lin_vel_w[:, bi])
    rep = ML.library_report(data)
    assert rep["all_accepted"] and rep["num_clips"] == 5 and rep["authored_ankle_tierods"] is False
    assert len(rep["sha256"]) == 64 and rep["usd_sha256"] == N.USD_SHA256
    # the fingerprint depends on content, not on file location
    again = ML.build_library(MANIFEST, keep_body_names=list(N.TRACKED_BODIES))
    assert again.fingerprint["sha256"] == data.fingerprint["sha256"]


def test_torch_library_global_index_and_tensors():
    _need(MANIFEST)
    data = ML.build_library(MANIFEST, keep_body_names=list(N.TRACKED_BODIES))
    lib = ML.MotionLibrary(data, N.MOTOR_NAMES, N.TRACKED_BODIES)
    assert lib.num_clips == 5 and lib.joint_pos.shape == (data.num_frames_total, 22)
    assert lib.body_pos_w.shape == (data.num_frames_total, len(N.TRACKED_BODIES), 3)
    clips = torch.tensor([0, 1, 4, 4, 2])
    t = torch.tensor([0, 7, 900, 5000, -3])  # 5000 and -3 clamp to the clip's last / first frame
    g = lib.global_index(clips, t)
    starts, lengths = data.starts, data.lengths
    assert g.tolist() == [0, starts[1] + 7, starts[4] + 900, starts[4] + lengths[4] - 1, starts[2]]
    motor_cols = [data.joint_names.index(n) for n in N.MOTOR_NAMES]
    np.testing.assert_array_equal(lib.joint_pos[g].numpy(), data.joint_pos[g.numpy()][:, motor_cols])
    np.testing.assert_array_equal(lib.root_pos_w[g].numpy(), data.body_pos_w[g.numpy(), 0])
    assert lib.validation["verdict"] == "accepted" and lib.arrays_meta["library"]["num_clips"] == 5
    fut = ML.future_indices(lib, clips[:3], torch.tensor([0, 495, 10]), (5, 10))
    assert fut.tolist()[1] == [starts[1] + 500, starts[1] + 500]  # clamped to wave_right's last frame (501 frames)
    assert fut.tolist()[0] == [5, 10]


def test_rejected_and_unvalidated_clips_fail_closed(tmp_path):
    _need(SYN / "stand.npz", REJECTED)
    stand = _copy_clip(SYN / "stand.npz", tmp_path, "stand")
    rej = _copy_clip(REJECTED, tmp_path, "kimodo_wave")
    with pytest.raises(ML.LibraryFormatError, match="REJECTED"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(stand)}, {"npz": str(rej)}]))
    # exploratory override accepts it (and the report says not all accepted)
    data = ML.build_library(_manifest(tmp_path, [{"npz": str(stand)}, {"npz": str(rej)}], "x"), allow_rejected=True)
    assert not ML.library_report(data)["all_accepted"]
    # no verdict file at all -> refused by default
    nov = _copy_clip(SYN / "wave_right.npz", tmp_path, "noverdict", with_verdict=False)
    with pytest.raises(ML.LibraryFormatError, match="validation.json"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(nov)}], "y"))
    ML.build_library(_manifest(tmp_path, [{"npz": str(nov)}], "y2"), require_accepted=False)  # explicit opt-out ok
    # stale verdict (bytes changed after validation) -> refused
    stale = _copy_clip(SYN / "arm_swing.npz", tmp_path, "stale")
    m = load_motion_npz(stale)
    m.joint_pos = m.joint_pos + np.float32(1e-3)
    save_motion_npz(stale, m)
    with pytest.raises(ML.LibraryFormatError, match="STALE"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(stale)}], "z"))


def _variant(tmp_path: Path, stem: str, **meta_updates) -> Path:
    """Copy of stand.npz with modified meta, verdict disabled (require_accepted=False in the caller)."""
    m = load_motion_npz(SYN / "stand.npz")
    m.meta = {**m.meta, **meta_updates}
    return save_motion_npz(tmp_path / f"{stem}.npz", m)


def test_cross_clip_consistency_checks(tmp_path):
    _need(SYN / "stand.npz", SYN / "wave_right.npz")
    wave = _copy_clip(SYN / "wave_right.npz", tmp_path, "wave")
    kw = dict(require_accepted=False)
    # ankle variant differs
    authored = _variant(tmp_path, "authored", authored_ankle_tierods=True)
    with pytest.raises(ML.LibraryFormatError, match="authored_ankle_tierods"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}, {"npz": str(authored)}], "a"), **kw)
    # and vs the env's plant variant (single clip)
    with pytest.raises(ML.LibraryFormatError, match="ankle"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(authored)}], "a2"), expected_authored_ankle=False, **kw)
    # different USD
    other_usd = _variant(tmp_path, "usd", usd_sha256="0" * 64)
    with pytest.raises(ML.LibraryFormatError, match="usd_sha256"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}, {"npz": str(other_usd)}], "b"), **kw)
    # different calibrations (unknown is allowed, two different known ones are not)
    cal_a = _variant(tmp_path, "cal_a", source_calibration={"calibration_sha256": "a" * 64})
    cal_b = _variant(tmp_path, "cal_b", source_calibration={"calibration_sha256": "b" * 64})
    ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}, {"npz": str(cal_a)}], "c1"), **kw)
    with pytest.raises(ML.LibraryFormatError, match="calibration_sha256"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(cal_a)}, {"npz": str(cal_b)}], "c2"), **kw)
    # joint order differs
    m = load_motion_npz(SYN / "stand.npz")
    perm = list(range(len(m.joint_names)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    m.joint_names = [m.joint_names[i] for i in perm]
    m.joint_pos, m.joint_vel = m.joint_pos[:, perm], m.joint_vel[:, perm]
    swapped = save_motion_npz(tmp_path / "swapped.npz", m)
    with pytest.raises(ML.LibraryFormatError, match="joint_names"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}, {"npz": str(swapped)}], "d"), **kw)
    # names vs the live articulation (order matters) and fps vs the policy rate
    ref = load_motion_npz(SYN / "wave_right.npz")
    with pytest.raises(ML.LibraryFormatError, match="joint_names mismatch"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}], "e"), joint_names=list(reversed(ref.joint_names)),
                         body_names=ref.body_names, motor_names=ref.motor_names, **kw)
    with pytest.raises(ML.LibraryFormatError, match="fps"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}], "f"), expected_fps=60.0, **kw)
    with pytest.raises(ML.LibraryFormatError, match="not in the clips"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(wave)}], "g"), keep_body_names=["no_such_body"], **kw)


# ------------------------------------------------------------------------------------------------ sampler
def _upstream_probs(failed: torch.Tensor, kernel_size: int, lam: float, uniform_ratio: float) -> torch.Tensor:
    """mdp/commands.py MotionCommand._adaptive_sampling, verbatim math."""
    bin_count = failed.numel()
    kernel = torch.tensor([lam ** i for i in range(kernel_size)])
    kernel = kernel / kernel.sum()
    probs = failed + uniform_ratio / float(bin_count)
    probs = torch.nn.functional.pad(probs.unsqueeze(0).unsqueeze(0), (0, kernel_size - 1), mode="replicate")
    probs = torch.nn.functional.conv1d(probs, kernel.view(1, 1, -1)).view(-1)
    return probs / probs.sum()


@pytest.mark.parametrize("kernel_size", [1, 3])
def test_single_clip_sampler_equals_beyondmimic(kernel_size):
    T, fps = 901, 50.0
    s = ML.LibrarySampler([T], fps, adaptive_kernel_size=kernel_size, adaptive_lambda=0.8, adaptive_uniform_ratio=0.1)
    assert s.bin_count == int(T // fps) + 1 == 19
    g = torch.Generator().manual_seed(0)
    s.bin_failed_count[:] = torch.rand(s.bin_count, generator=g) * torch.tensor([0.0, 1.0]).repeat(10)[: s.bin_count]
    ref = _upstream_probs(s.bin_failed_count.clone(), kernel_size, 0.8, 0.1)
    torch.testing.assert_close(s.joint_probs(), ref, rtol=1e-6, atol=1e-7)
    # identical draws with the same generator state: upstream multinomial -> bins, then U(0,1) inside the bin
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    clips, t = s.sample(4096, generator=g1)
    bins = torch.multinomial(ref, 4096, replacement=True, generator=g2)
    t_ref = ((bins + torch.rand(4096, generator=g2)) / s.bin_count * (T - 1)).long()
    assert bool((clips == 0).all()) and torch.equal(t, t_ref)
    # failure bookkeeping bins like upstream: (t * bin_count) // T
    tt = torch.tensor([0, 49, 50, 450, 900])
    assert s.time_bin(torch.zeros(5, dtype=torch.long), tt).tolist() == ((tt * 19) // T).clamp(max=18).tolist()


def test_multi_clip_bins_offsets_and_sampling_support():
    lengths, fps = [501, 501, 701, 901, 60], 50.0
    s = ML.LibrarySampler(lengths, fps, clip_weighting="duration")
    nb = [int(T // fps) + 1 for T in lengths]
    assert s.bins_per_clip.tolist() == nb
    assert s.bin_offsets.tolist() == [0] + np.cumsum(nb).tolist()
    assert s.bin_clip.tolist() == sum(([c] * k for c, k in enumerate(nb)), [])
    assert s.bin_local.tolist() == sum((list(range(k)) for k in nb), [])
    within = s.bin_probs_within_clip()
    sums = torch.zeros(5, dtype=within.dtype).index_add_(0, s.bin_clip, within)
    torch.testing.assert_close(sums, torch.ones(5, dtype=within.dtype))
    torch.testing.assert_close(s.joint_probs().sum(), torch.tensor(1.0))
    # no failures yet: duration prior -> time-uniform over the whole library
    torch.testing.assert_close(s.clip_probs(), torch.tensor(lengths, dtype=torch.float) / sum(lengths))
    g = torch.Generator().manual_seed(1)
    clips, t = s.sample(200000, generator=g)
    lens = torch.tensor(lengths)
    assert bool((t >= 0).all()) and bool((t <= lens[clips] - 1).all())
    freq = torch.bincount(clips, minlength=5).float() / clips.numel()
    torch.testing.assert_close(freq, s.clip_probs(), atol=5e-3, rtol=0)
    # uniform clip weighting with per-clip weights
    su = ML.LibrarySampler(lengths, fps, weights=[1, 1, 1, 1, 4], clip_weighting="uniform")
    torch.testing.assert_close(su.clip_probs(), torch.tensor([1, 1, 1, 1, 4.0]) / 8)
    with pytest.raises(ValueError):
        ML.LibrarySampler(lengths, fps, clip_weighting="nope")
    with pytest.raises(ValueError):
        ML.LibrarySampler([1], fps)


def test_failures_steer_bins_and_clips():
    lengths, fps = [501, 501, 501], 50.0
    s = ML.LibrarySampler(lengths, fps, clip_adaptive_ratio=0.5, adaptive_alpha=0.5, clip_alpha=0.5)
    envs_clip = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    # clip 1 fails at t ~ 3.1 s (bin 3) on every episode; the others never fail
    t = torch.full((30,), 155)
    failed = envs_clip == 1
    for _ in range(20):
        s.record_failures(envs_clip, t, failed)
        s.record_exposure(envs_clip)
        s.step()
    haz = s.clip_hazard()
    assert float(haz[0]) == 0.0 and float(haz[2]) == 0.0 and float(haz[1]) == pytest.approx(1.0, rel=1e-3)
    p = s.clip_probs()
    torch.testing.assert_close(p, torch.tensor([0.5 / 3, 0.5 / 3 + 0.5, 0.5 / 3]), atol=1e-6, rtol=0)
    within = s.bin_probs_within_clip()
    b1 = s.bin_offsets[1] + 3
    assert int(torch.argmax(within[s.bin_offsets[1]:s.bin_offsets[2]])) == 3 and float(within[b1]) > 0.9
    # the other clips stay uniform over their bins
    torch.testing.assert_close(within[: s.bin_offsets[1]], torch.full((11,), 1 / 11))
    # clip_adaptive_ratio 0 -> prior only, whatever the failures
    s.clip_adaptive_ratio = 0.0
    torch.testing.assert_close(s.clip_probs(), torch.full((3,), 1 / 3))
    # stats are finite and name the failing clip as the most likely
    s.clip_adaptive_ratio = 0.5
    st = s.stats()
    assert st["top1_clip"] == 1 and st["top1_bin_local"] == 3 and 0 < st["entropy"] < 1


def test_sampler_state_roundtrip():
    s = ML.LibrarySampler([501, 701], 50.0)
    s.bin_failed_count[:] = torch.arange(s.bin_count, dtype=torch.float)
    s.clip_fail_ema[:] = torch.tensor([0.1, 0.2])
    s.clip_active_ema[:] = torch.tensor([3.0, 4.0])
    state = json.loads(json.dumps(s.state_dict()))  # survives JSON (checkpoint infos)
    s2 = ML.LibrarySampler([501, 701], 50.0)
    assert s2.load_state_dict(state)
    torch.testing.assert_close(s2.bin_failed_count, s.bin_failed_count)
    torch.testing.assert_close(s2.clip_active_ema, s.clip_active_ema)
    assert not ML.LibrarySampler([501, 501], 50.0).load_state_dict(state)  # other bin layout -> not restored


def test_play_assignment():
    names = ["a", "b", "c"]
    assert ML.play_assignment(7, 3, names).tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert ML.play_assignment(4, 3, names, ["c", "a"]).tolist() == [2, 0, 2, 0]
    with pytest.raises(ValueError):
        ML.play_assignment(4, 3, names, ["zz"])


# ------------------------------------------------------------------------------------------------ chunked-training state
def test_runner_persists_library_sampler_state(monkeypatch):
    """runner.DropbearOnPolicyRunner saves/restores the library command's clip-level sampler state (``command_extra``)
    next to ``bin_failed_count``; another library (sha) is not restored."""
    import types

    pytest.importorskip("rsl_rl")
    from rsl_rl.runners import OnPolicyRunner

    from dropbear_wbc.tasks.tracking.runner import TRAIN_STATE_KEY, DropbearOnPolicyRunner

    class FakeTerm:  # the MotionLibraryCommand persistence surface
        def __init__(self, sha):
            self.sampler = ML.LibrarySampler([501, 701], 50.0)
            self.sha = sha

        bin_count = property(lambda self: self.sampler.bin_count)
        bin_failed_count = property(lambda self: self.sampler.bin_failed_count)

        def extra_train_state(self):
            return {"library_sha256": self.sha, "sampler": self.sampler.state_dict()}

        def load_extra_train_state(self, state):
            return state.get("library_sha256") == self.sha and self.sampler.load_state_dict(state["sampler"])

    def fake_runner(term):
        r = DropbearOnPolicyRunner.__new__(DropbearOnPolicyRunner)
        r.alg = types.SimpleNamespace(learning_rate=3e-4, optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]))
        r.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(
            command_manager=types.SimpleNamespace(get_term=lambda name: term)))
        r.restore_train_state = True
        return r

    src = FakeTerm("a" * 64)
    src.sampler.bin_failed_count[:] = torch.linspace(0, 1, src.sampler.bin_count)
    src.sampler.clip_fail_ema[:] = torch.tensor([0.2, 0.0])
    src.sampler.clip_active_ema[:] = torch.tensor([5.0, 7.0])
    state = json.loads(json.dumps(fake_runner(src).train_state()))
    assert state["bin_count"] == src.sampler.bin_count and "command_extra" in state
    monkeypatch.setattr(OnPolicyRunner, "load", lambda self, path, load_optimizer=True: {TRAIN_STATE_KEY: state})
    dst = FakeTerm("a" * 64)
    r = fake_runner(dst)
    r.load("ckpt.pt")
    assert set(r.train_state_report["restored"]) == {"learning_rate", "bin_failed_count", "command_extra"}
    torch.testing.assert_close(dst.sampler.clip_active_ema, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(dst.sampler.bin_failed_count, src.sampler.bin_failed_count)
    other = FakeTerm("b" * 64)
    r2 = fake_runner(other)
    r2.load("ckpt.pt")
    assert r2.train_state_report.get("command_extra_mismatch") and float(other.sampler.clip_active_ema.sum()) == 0.0
    # ... and its per-bin failure EMA is not seeded from the other library either (same bin count, other clips)
    assert "bin_failed_count" not in r2.train_state_report["restored"]
    assert r2.train_state_report.get("bin_failed_count_skipped") and float(other.sampler.bin_failed_count.sum()) == 0.0


def test_manifest_sha256_pins(tmp_path):
    """A manifest clip entry may pin the NPZ bytes (tools/build_motion_library_manifest.py writes the pins)."""
    _need(SYN / "stand.npz")
    stand = _copy_clip(SYN / "stand.npz", tmp_path, "stand")
    good = ML._sha256_file(stand)
    data = ML.build_library(_manifest(tmp_path, [{"npz": str(stand), "sha256": good.upper()}], "p1"))
    assert data.clips[0].sha256 == good
    assert ML.manifest_fingerprint(_manifest(tmp_path, [{"npz": str(stand), "sha256": good}], "p2"))["sha256"] == \
        ML.manifest_fingerprint(_manifest(tmp_path, [{"npz": str(stand)}], "p3"))["sha256"]  # pins do not change identity
    with pytest.raises(ML.LibraryFormatError, match="manifest pin"):
        ML.build_library(_manifest(tmp_path, [{"npz": str(stand), "sha256": "0" * 64}], "p4"))
    with pytest.raises(ML.LibraryFormatError, match="sha256 pins"):
        ML.manifest_fingerprint(_manifest(tmp_path, [{"npz": str(stand), "sha256": "0" * 64}], "p5"))
    with pytest.raises(ML.LibraryFormatError, match="not a hex"):
        ML.parse_manifest({"schema": ML.LIBRARY_SCHEMA, "clips": [{"npz": "x.npz", "sha256": "abc"}]}, base_dir=tmp_path)


def test_pack_brev_bundle_ships_manifest_clips_and_verdicts(tmp_path):
    """tools/pack_brev_bundle.py: the manifest, every clip NPZ + verdict and the calibration, repo-relative, with a
    BUNDLE.json whose hashes match the files (data-only mode; the code dirs are just more files)."""
    import importlib.util
    import tarfile

    manifest = ROOT / "data" / "motions" / "libraries" / "accepted_v1.json"
    _need(manifest)
    spec = importlib.util.spec_from_file_location("pbb", ROOT / "tools" / "pack_brev_bundle.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = tmp_path / "b.tgz"
    assert tool.main(["--manifest", str(manifest), "--out", str(out), "--no-code"]) == 0
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
        info = json.loads(tar.extractfile("BUNDLE.json").read())
    man = json.loads(manifest.read_text(encoding="utf-8"))
    for c in man["clips"]:
        rel = (Path("data/motions/libraries") / c["npz"]).as_posix().replace("libraries/../", "")
        assert rel in names and rel.replace(".npz", ".validation.json") in names
        assert info["files"][rel]["sha256"] == c["sha256"]
    assert "data/motions/libraries/accepted_v1.json" in names
    assert "data/calibration/dropbear_semantic_calibration.json" in names
    assert not any(n.startswith("source/") for n in names)


def test_manifest_builder_tool_selects_accepted_clips(tmp_path):
    """tools/build_motion_library_manifest.py: accepted + consistent clips in, rejected/stale/unverdicted/other-variant
    out (with reasons), pins written, the result loads with build_library, an unchanged selection writes nothing."""
    _need(SYN / "stand.npz", SYN / "wave_right.npz", SYN / "arm_swing.npz", REJECTED)
    import importlib.util

    spec = importlib.util.spec_from_file_location("bmlm", ROOT / "tools" / "build_motion_library_manifest.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    root = tmp_path / "motions"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "smoke").mkdir()
    _copy_clip(SYN / "stand.npz", root / "a", "stand")
    _copy_clip(SYN / "wave_right.npz", root / "b", "wave")
    _copy_clip(SYN / "stand.npz", root / "smoke", "dup_stand")          # excluded by the default smoke/** glob
    _copy_clip(REJECTED, root / "b", "kimodo_wave")                      # rejected verdict
    _copy_clip(SYN / "arm_swing.npz", root / "b", "noverdict", with_verdict=False)  # no verdict -> not a candidate
    stale = _copy_clip(SYN / "arm_swing.npz", root / "a", "stale")
    m = load_motion_npz(stale)
    m.joint_pos = m.joint_pos + np.float32(1e-3)
    save_motion_npz(stale, m)                                            # verdict now stale
    authored = _variant(root / "b", "authored", authored_ankle_tierods=True)
    shutil.copyfile(SYN / "stand.validation.json", root / "b" / "authored.validation.json")  # stale AND other variant
    out_dir = root / "libraries"
    rc = tool.main(["--root", str(root), "--out-dir", str(out_dir), "--weight", "wave=2"])
    assert rc == 0
    man = json.loads((out_dir / "accepted_v0.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in man["clips"]] == ["stand", "wave"] and man["clips"][1]["weight"] == 2.0
    assert all(len(c["sha256"]) == 64 for c in man["clips"])
    reasons = {x["npz"]: x["reason"] for x in man["selection"]["excluded"]}
    assert set(reasons) == {"b/kimodo_wave.npz", "a/stale.npz", "b/authored.npz"}
    assert "rejected" in reasons["b/kimodo_wave.npz"] and "STALE" in reasons["a/stale.npz"]
    data = ML.build_library(out_dir / "accepted_v0.json", keep_body_names=list(N.TRACKED_BODIES))
    assert data.fingerprint["sha256"] == man["library"]["sha256"]
    # same selection again -> unchanged, nothing new written; a new accepted clip -> accepted_v1, base order kept
    assert tool.main(["--root", str(root), "--out-dir", str(out_dir), "--weight", "wave=2"]) == 0
    assert not (out_dir / "accepted_v1.json").exists()
    _copy_clip(SYN / "arm_swing.npz", root / "a", "arm")
    assert tool.main(["--root", str(root), "--out-dir", str(out_dir), "--weight", "wave=2"]) == 0
    man1 = json.loads((out_dir / "accepted_v1.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in man1["clips"]] == ["stand", "wave", "arm"]
