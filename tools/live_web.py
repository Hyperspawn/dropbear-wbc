"""Browser front end of the live text -> Dropbear session: type a motion, watch the motor-twin robot do it.

Serves ``tools/live_web/index.html`` (three.js), the decimated robot ``site/assets/dropbear.glb``
(``tools/export_dropbear_glb.py``), the newest physics state (``/state``: every link pose of env 0 as published by
``scripts/play.py --state_out``) and a prompt endpoint (``POST /prompt`` writes ``logs/live/prompts/<name>.txt``, the
Kimodo service's input). ``/events`` reports each prompt's progress: sent -> clip ready (inbox) -> spliced (physics log).

Stdlib + numpy only, ~50 MB; it replaces a second Isaac process (~13 GB committed) or the Newton window (~5 GB) as the
viewer, which is what lets text encoder + Kimodo + real-time physics + viewer fit on the 31.6 GB laptop.

  python tools/live_web.py            # then open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "logs" / "live"


class StateFile:
    """Reads ``state_share`` files with plain file I/O (no mapping held, so the physics side can recreate them)."""

    def __init__(self, path: Path):
        self.path = path
        self.meta_mtime = None
        self.names: list[str] = []
        self.n = self.nb = 0

    def read(self) -> dict | None:
        side = self.path.with_name(self.path.name + ".json")
        try:
            mt = side.stat().st_mtime
            if mt != self.meta_mtime:
                meta = json.loads(side.read_text(encoding="utf-8"))
                self.names, self.n = meta.get("body_names", []), len(meta["joint_names"])
                self.nb, self.meta_mtime = len(self.names), mt
            for _ in range(20):
                raw = self.path.read_bytes()
                row = np.frombuffer(raw, dtype=np.float64)
                if len(row) < 9 + self.n + 7 * self.nb or int(row[0]) % 2 or row[0] <= 0:
                    time.sleep(0.001)
                    continue
                with open(self.path, "rb") as fh:
                    again = np.frombuffer(fh.read(8), dtype=np.float64)[0]
                if again != row[0]:
                    continue
                j1 = 9 + self.n
                return {"seq": int(row[0]), "t": float(row[1]),
                        "p": np.round(row[j1:j1 + 3 * self.nb], 5).tolist(),
                        "q": np.round(row[j1 + 3 * self.nb:j1 + 7 * self.nb], 5).tolist()}
        except (OSError, ValueError, KeyError):
            return None
        return None


class Prompts:
    """Prompt bookkeeping for /events (sent -> clip in the inbox -> spliced by the physics process)."""

    def __init__(self):
        self.items: list[dict] = []
        self.lock = threading.Lock()
        self.n = 0

    def send(self, text: str) -> dict:
        with self.lock:
            self.n += 1
            name = f"web{time.strftime('%H%M%S')}_{self.n}"
            (LIVE / "prompts").mkdir(parents=True, exist_ok=True)
            part = LIVE / "prompts" / f"{name}.txt.part"
            part.write_text(text.strip(), encoding="utf-8")
            os.replace(part, LIVE / "prompts" / f"{name}.txt")
            it = {"name": name, "text": text.strip(), "sent": time.time(), "clip": None, "clip_s": None,
                  "spliced": None, "splice_s": None, "gate": None, "refused": False}
            self.items.append(it)
            return it

    def refresh(self) -> list[dict]:
        service = {}  # prompt-file name -> the service's record (gate decision, timings)
        try:
            for line in (LIVE / "live_service.jsonl").read_text(encoding="utf-8").splitlines()[-200:]:
                try:
                    r = json.loads(line)
                    service[r.get("name")] = r
                except ValueError:
                    continue
        except OSError:
            pass
        splices = {}
        try:
            for line in (LIVE / "physics.log").read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.search(r"\[live\] t=\s*([\d.]+)s splice (\S+)", line)
                if m:
                    splices[Path(m.group(2)).name] = float(m.group(1))
        except OSError:
            pass
        with self.lock:
            for it in self.items:
                r = service.get(it["name"])
                if r and it.get("gate") is None and "speed_gate" in r:
                    g = r["speed_gate"]
                    it["gate"] = (f"refused: {g['worst_motor']} would need {g['p99_over_noload']:.2f}x its no-load speed"
                                  if r.get("refused") else f"slowed x{g['stretch']:.2f} for the motors"
                                  if g["stretch"] > 1.0 else "within motor speeds")
                    it["refused"] = bool(r.get("refused"))
                if it["clip"] is None:
                    hits = sorted((LIVE / "inbox").glob(f"*_{it['name']}.npz"))
                    if hits:
                        it["clip"], it["clip_s"] = hits[-1].name, round(time.time() - it["sent"], 1)
                if it["clip"] and it["spliced"] is None:
                    for k, t in splices.items():
                        if it["clip"] in k:
                            it["spliced"], it["splice_s"] = t, round(time.time() - it["sent"], 1)
            return [{k: v for k, v in it.items() if k != "sent"} | {"age_s": round(time.time() - it["sent"], 1)}
                    for it in self.items[-8:]]


def make_handler(state: StateFile, prompts: Prompts, glb: Path, page: Path):
    glb_bytes = {"data": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif path == "/dropbear.glb":
                if glb_bytes["data"] is None:
                    glb_bytes["data"] = glb.read_bytes()
                self._send(200, glb_bytes["data"], "model/gltf-binary")
            elif path == "/state":
                st = state.read()
                body = json.dumps(st | {"names": state.names} if st and "names" in self.path else st or {}).encode()
                self._send(200, body, "application/json")
            elif path == "/events":
                self._send(200, json.dumps({"prompts": prompts.refresh(), "physics": _physics_status()}).encode(),
                           "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/prompt":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", "0") or 0)
            try:
                text = str(json.loads(self.rfile.read(n) or b"{}").get("text", "")).strip()
            except ValueError:
                text = ""
            if not text or len(text) > 300:
                return self._send(400, b'{"error": "text (1-300 chars) required"}', "application/json")
            it = prompts.send(text)
            self._send(200, json.dumps({"name": it["name"]}).encode(), "application/json")

    return H


def _physics_status() -> dict:
    try:
        txt = (LIVE / "physics.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"running": False}
    return {"running": "[live] armed" in txt and '"timing"' not in txt, "armed": "[live] armed" in txt,
            "finished": '"timing"' in txt}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--state", type=Path, default=LIVE / "state.bin")
    ap.add_argument("--glb", type=Path, default=REPO / "site" / "assets" / "dropbear.glb")
    args = ap.parse_args()
    page = Path(__file__).resolve().parent / "live_web" / "index.html"
    if not args.glb.is_file():
        raise SystemExit(f"missing {args.glb} (it ships with the repository; tools/export_dropbear_glb.py rebuilds it)")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(StateFile(args.state), Prompts(), args.glb, page))
    print(f"[web] http://127.0.0.1:{args.port}  (state {args.state})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
