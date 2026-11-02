"""ONNX policy wrapper (onnxruntime), compatible with rsl_rl and BeyondMimic exports.

* rsl_rl / unitree_rl_lab export: one input ``obs`` [1, N] -> ``actions`` [1, A].
* BeyondMimic export (``_OnnxMotionPolicyExporter``): inputs ``obs`` [1, N] and
  ``time_step`` [1, 1]; outputs ``actions`` plus the reference ``joint_pos``,
  ``joint_vel``, ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``,
  ``body_ang_vel_w`` at that time step. Custom metadata (``joint_names``,
  ``default_joint_pos``, ``action_scale``, ``observation_names``, ...) is exposed
  as :attr:`OnnxPolicy.metadata`.

Inference defaults to the CPU provider: the MLPs are tiny, CPU latency is lower
than a GPU round trip, and it keeps the runner off the GPU lock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def onnx_external_data_files(path: str | Path) -> list[str]:
    """Relative paths of the external-data files an ONNX model references (empty if all weights are inline)."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    files = []
    for t in model.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            loc = next((e.value for e in t.external_data if e.key == "location"), None)
            if loc and loc not in files:
                files.append(loc)
    return files


def make_onnx_self_contained(path: str | Path) -> bool:
    """Inline external weights into ``path`` and delete the side files; returns True if anything changed.

    torch >= 2.9 exports through the dynamo exporter by default, which writes the weights to
    ``<name>.onnx.data`` next to the graph. Deploy bundles ship ``policy.onnx`` + ``policy.json`` only, and
    onnxruntime refuses a graph whose external data is missing, so every export is made single-file here.
    """
    import onnx

    path = Path(path)
    files = onnx_external_data_files(path)
    if not files:
        return False
    model = onnx.load(str(path))  # resolves external data relative to path.parent
    onnx.save(model, str(path))  # default: all tensors inline
    for f in files:
        side = (path.parent / f).resolve()
        if side.parent == path.parent.resolve() and side.exists():
            side.unlink()
    return True


class OnnxPolicy:
    """Thin onnxruntime session wrapper.

    Args:
        path: ONNX file.
        obs_input: name of the observation input (default: first input).
        time_input: name of the time-step input (default: ``"time_step"`` if present).
        providers: onnxruntime providers (default CPU).
    """

    def __init__(self, path: Path, obs_input: str | None = None, time_input: str | None = None,
                 providers: list[str] | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=opts,
                                            providers=providers or ["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.obs_input = obs_input or next(iter(inputs))
        if self.obs_input not in inputs:
            raise KeyError(f"{path}: no input {self.obs_input!r}; inputs {list(inputs)}")
        self.time_input = time_input if time_input else ("time_step" if "time_step" in inputs else None)
        if self.time_input is not None and self.time_input not in inputs:
            raise KeyError(f"{path}: no input {self.time_input!r}")
        extra = set(inputs) - {self.obs_input} - ({self.time_input} if self.time_input else set())
        if extra:
            raise ValueError(f"{path}: unsupported extra inputs {sorted(extra)}")
        shape = inputs[self.obs_input].shape
        self.obs_dim = shape[-1] if isinstance(shape[-1], int) else None
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.action_output = "actions" if "actions" in self.output_names else self.output_names[0]
        out_shape = self.session.get_outputs()[self.output_names.index(self.action_output)].shape
        self.action_dim = out_shape[-1] if isinstance(out_shape[-1], int) else None
        self.metadata: dict[str, str] = dict(self.session.get_modelmeta().custom_metadata_map)

    @property
    def has_reference(self) -> bool:
        """True when the export embeds the motion reference (BeyondMimic style)."""
        return self.time_input is not None and {"joint_pos", "body_pos_w", "body_quat_w"} <= set(self.output_names)

    def _feeds(self, obs: np.ndarray, time_step: int) -> dict:
        feeds = {self.obs_input: np.asarray(obs, np.float32).reshape(1, -1)}
        if self.time_input is not None:
            feeds[self.time_input] = np.array([[float(time_step)]], dtype=np.float32)
        return feeds

    def act(self, obs: np.ndarray, time_step: int = 0) -> np.ndarray:
        """Raw action (A,) for one observation vector."""
        if self.obs_dim is not None and obs.size != self.obs_dim:
            raise ValueError(f"observation has {obs.size} values, policy expects {self.obs_dim}")
        (act,) = self.session.run([self.action_output], self._feeds(obs, time_step))
        return np.asarray(act, np.float64).reshape(-1)

    def reference(self, time_step: int) -> dict[str, np.ndarray]:
        """Embedded reference outputs at ``time_step`` (obs input is zero; outputs don't depend on it)."""
        names = [n for n in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w") if n in self.output_names]
        outs = self.session.run(names, self._feeds(np.zeros(self.obs_dim or 1, np.float32), time_step))
        return {n: np.asarray(v, np.float64).reshape(np.asarray(v).shape[1:]) for n, v in zip(names, outs)}
