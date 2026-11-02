"""wxyz-semantics view of the ``isaaclab.utils.math`` quaternion functions dropbear_wbc uses.

Every quaternion inside dropbear_wbc is wxyz (``compat.root_quat_wxyz`` / ``body_quat_wxyz``, motion
NPZs, offsets such as ``up_offset_wxyz``). Isaac Lab 2.x math is wxyz too; Isaac Lab 3.0 math is xyzw.
Import the functions from here instead of ``isaaclab.utils.math``:

    from dropbear_wbc.isaac.qmath import quat_apply, quat_mul
    from dropbear_wbc.isaac import qmath as math_utils

On 2.x (``compat.isaac_quat_order() == 'wxyz'``) each name resolves to the isaaclab function object
itself, so behaviour is unchanged. On 3.x each name is wrapped: wxyz quaternion arguments are reordered
to xyzw on the way in and quaternion results back to wxyz on the way out. Names resolve on first
attribute access (module ``__getattr__``), so importing this module never imports isaaclab.
``tests/test_isaac_quat_boundary.py`` fails on any raw ``isaaclab.utils.math`` import in the package.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from dropbear_wbc.isaac.compat import WXYZ, _check4, _roll, as_tensor, isaac_quat_order

# name -> (indices of quaternion positional args, keyword names of quaternion args, result spec)
# result spec: "q" quaternion, "v" non-quaternion, ("v", "q") tuple with a quaternion second element.
_SPEC: dict[str, tuple[tuple[int, ...], tuple[str, ...], Any]] = {
    "quat_apply": ((0,), ("quat",), "v"),
    "quat_apply_inverse": ((0,), ("quat",), "v"),
    "quat_conjugate": ((0,), ("q",), "q"),
    "quat_inv": ((0,), ("q",), "q"),
    "quat_mul": ((0, 1), ("q1", "q2"), "q"),
    "yaw_quat": ((0,), ("quat",), "q"),
    "quat_from_euler_xyz": ((), (), "q"),
    "quat_error_magnitude": ((0, 1), ("q1", "q2"), "v"),
    "matrix_from_quat": ((0,), ("quaternions",), "v"),
    "subtract_frame_transforms": ((1, 3), ("q01", "q02"), ("v", "q")),
}
# order-free helpers re-exported unchanged
_PASSTHROUGH = ("sample_uniform",)

__all__ = sorted((*_SPEC, *_PASSTHROUGH))


def _isaac_math():
    import isaaclab.utils.math as m

    return m


# The wrapper exists only when Isaac is xyzw, so the reorder is unconditional here.
def _to_isaac(x: Any) -> Any:
    if x is None:
        return None
    _check4(x)
    return _roll(x, -1)  # wxyz -> xyzw


def _from_isaac(x: Any) -> Any:
    x = as_tensor(x)
    _check4(x)
    return _roll(x, 1)  # xyzw -> wxyz


def wrap(name: str, fn: Callable, order: str) -> Callable:
    """Return ``fn`` itself when Isaac is wxyz, else a wrapper giving it wxyz semantics."""
    if name in _PASSTHROUGH or order == WXYZ:
        return fn
    qpos, qkw, res = _SPEC[name]

    @functools.wraps(fn)
    def _wxyz(*args, **kwargs):
        args = tuple(_to_isaac(a) if i in qpos else a for i, a in enumerate(args))
        kwargs = {k: (_to_isaac(v) if k in qkw else v) for k, v in kwargs.items()}
        out = fn(*args, **kwargs)
        if res == "q":
            return _from_isaac(out)
        if isinstance(res, tuple):
            return tuple(_from_isaac(o) if r == "q" and o is not None else o for r, o in zip(res, out))
        return out

    _wxyz.__wrapped_order__ = order
    return _wxyz


def resolve(name: str, math_module: Any = None, order: str | None = None) -> Callable:
    if name not in _SPEC and name not in _PASSTHROUGH:
        raise AttributeError(f"dropbear_wbc.isaac.qmath has no {name!r}; add it to _SPEC with its quaternion args")
    m = _isaac_math() if math_module is None else math_module
    return wrap(name, getattr(m, name), isaac_quat_order() if order is None else order)


def __getattr__(name: str) -> Callable:
    if name.startswith("__"):
        raise AttributeError(name)
    fn = resolve(name)
    globals()[name] = fn  # order is fixed per process; cache after first resolve
    return fn
