"""Item 390: ``dropbear_wbc.deploy.config`` (hence ``tools/hw_stand_test.py --motor_pose_json``) must import in a
Python without pyzmq, like Isaac Sim 5.0's; the sdk transport names stay available, loaded on first use."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import sdk_test_paths  # noqa: F401  (adds source/)

ROOT = sdk_test_paths.ROOT
STAND = ROOT / "data/calibration/dropbear_loaded_stand.json"

_NO_ZMQ = textwrap.dedent("""
    import sys
    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "zmq" or name.startswith("zmq."):
                raise ModuleNotFoundError("No module named 'zmq'", name="zmq")
            return None
    sys.meta_path.insert(0, _Block())
    sys.path.insert(0, {src!r})
""")


def _run(body: str) -> subprocess.CompletedProcess:
    code = _NO_ZMQ.format(src=str(ROOT / "source")) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=str(ROOT))


def test_deploy_config_imports_and_loads_stand_without_zmq():
    r = _run(f"""
        from dropbear_wbc.deploy.config import load_motor_pose_json, resolve_default_pose
        import dropbear_wbc.deploy
        from dropbear_wbc.sdk import motors, types
        assert "zmq" not in sys.modules and "dropbear_wbc.sdk.transport" not in sys.modules
        q, info = load_motor_pose_json({str(STAND)!r})
        print("OK", len(q))
    """)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().startswith("OK 22")


def test_transport_name_without_zmq_raises_module_not_found():
    r = _run("""
        import dropbear_wbc.sdk as sdk
        try:
            sdk.ZmqPublisher
        except ModuleNotFoundError as e:
            print("MNF", e.name)
    """)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "MNF zmq"


def test_transport_names_still_exported_with_zmq():
    import dropbear_wbc.sdk as sdk
    from dropbear_wbc.sdk import transport
    for n in ("LOWCMD", "LOWSTATE", "ChannelFactoryInitialize", "ChannelPublisher", "ChannelSubscriber", "Endpoints",
              "ZmqPublisher", "ZmqSubscriber"):
        assert n in sdk.__all__
        assert getattr(sdk, n) is getattr(transport, n)
    ns: dict = {}
    exec("from dropbear_wbc.sdk import *", ns)
    assert ns["ZmqSubscriber"] is transport.ZmqSubscriber and ns["LOWSTATE"] == transport.LOWSTATE
    try:
        sdk.NoSuchName
    except AttributeError as e:
        assert "NoSuchName" in str(e)
    else:
        raise AssertionError("unknown attribute must raise AttributeError")


def test_shipped_stand_schema_is_what_the_isaac_path_reads():
    m = json.loads(STAND.read_text(encoding="utf-8"))
    assert len(m["loaded_standing_motor_pos"]) == 22
