"""Type text prompts into a running live text -> motion session (``tools/live_session.sh``).

Each line becomes ``logs/live/prompts/<HHMMSS>_<n>.txt`` (atomic), which ``tools/kimodo_live_service.py`` turns into
a clip in ``logs/live/inbox/``, which the running ``scripts/play.py --live_dir`` splices into the motor-twin robot's
reference. After each prompt this waits for the clip and prints the prompt -> clip latency.

  python tools/live_prompt.py                          # interactive: 'wave hello', '6|walk forward slowly', 'quit'
  python tools/live_prompt.py --script "0:a person waves hello;12:a person walks forward slowly"

A line ``<seconds>|<text>`` sets the clip duration (default: the service's 4 s).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def send(text: str, prompts: Path, n: int) -> str:
    name = f"{time.strftime('%H%M%S')}_{n}"
    prompts.mkdir(parents=True, exist_ok=True)
    part = prompts / f"{name}.txt.part"
    part.write_text(text.strip(), encoding="utf-8")
    os.replace(part, prompts / f"{name}.txt")
    return name


def wait_clip(name: str, inbox: Path, timeout_s: float) -> Path | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hits = sorted(inbox.glob(f"*_{name}.npz"))
        if hits:
            return hits[-1]
        time.sleep(0.1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--prompts", type=Path, default=REPO / "logs/live/prompts")
    ap.add_argument("--inbox", type=Path, default=REPO / "logs/live/inbox")
    ap.add_argument("--script", default=None, help="'t_s:text;t_s:text' sent at wall-clock offsets, then exit")
    ap.add_argument("--timeout_s", type=float, default=120.0, help="how long to wait for each clip")
    ap.add_argument("--log", type=Path, default=REPO / "logs/live/prompt_log.jsonl")
    args = ap.parse_args()

    def one(text: str, n: int) -> None:
        t0 = time.time()
        name = send(text, args.prompts, n)
        clip = wait_clip(name, args.inbox, args.timeout_s)
        rec = {"prompt": text, "name": name, "clip": clip.name if clip else None,
               "prompt_to_clip_s": round(time.time() - t0, 2) if clip else None, "time": time.strftime("%H:%M:%S")}
        print(json.dumps(rec), flush=True)
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    if args.script:
        items = [(float(t), p) for t, p in (x.split(":", 1) for x in args.script.split(";") if x.strip())]
        t0 = time.time()
        for n, (t, p) in enumerate(items, 1):
            time.sleep(max(0.0, t - (time.time() - t0)))
            one(p, n)
        return 0
    print("type a motion ('wave hello', '6|walk forward slowly'); 'quit' to stop", flush=True)
    n = 0
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return 0
        if line.lower() in ("quit", "exit", "q"):
            return 0
        if line:
            n += 1
            one(line, n)


if __name__ == "__main__":
    raise SystemExit(main())
