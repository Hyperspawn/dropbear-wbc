"""Text -> Dropbear motion, today by semantic RETRIEVAL from the motion library (generation later: Kimodo / ARDY).

A prompt ("wave hello", "do a golf swing", "walk forward") is embedded with the local sentence model
``sentence-transformers/all-MiniLM-L6-v2`` (HF cache, no download) and matched against a readable description of every
library clip. Only clips the real-motor tracker held without a fall are eligible (``--safe_eval``: a
``logs/hw_twin/libeval_*.json`` per-clip evaluation); if the best overall match is an unsafe clip, the tool says so and
plays the best SAFE match instead (or nothing, if none is close enough).

The chosen contract NPZ is written atomically into ``--live_dir``, which ``scripts/play.py --live_dir`` watches: the
running simulation splices it into the tracking reference (``dropbear_wbc.isaac.live_reference``). Swapping retrieval
for a generator keeps this interface: write a settled contract NPZ into the same folder.

usage:
  python tools/text_to_motion.py --prompt "wave hello"                 # one prompt -> one clip into --live_dir
  python tools/text_to_motion.py --interactive                         # type prompts; 'stand' / 'quit'
  python tools/text_to_motion.py --prompt "golf" --dry_run             # show the ranking only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "data/motions_v6ts/libraries/accepted_v6ts.json"
SAFE_EVAL = REPO / "logs/hw_twin/libeval_kneeX10S2_on_v6ts.json"
LIVE_DIR = REPO / "logs/live/inbox"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GLOSSARY = {  # clip-name tokens -> words people would type
    "CR7": "Cristiano Ronaldo siuu goal celebration jump spin",
    "APT": "APT kpop dance",
    "Bolt": "Usain Bolt lightning bolt victory pose sprint",
    "Kobe": "Kobe Bryant basketball fadeaway jump shot",
    "SpiderMan": "Spider-Man superhero crouch web shooting pose",
    "TigerWoods": "Tiger Woods golf swing",
    "lebron1": "LeBron James basketball dunk celebration",
    "lebron2": "LeBron James basketball chalk toss celebration",
    "shoot": "shoot a basketball jump shot",
    "jump degree": "jump and turn in the air spin jump",
    "jump forward": "jump forward broad jump",
    "side jump": "jump sideways",
    "kick": "kick with the leg football kick",
    "single foot balance": "balance on one leg stand on one foot",
    "single foot jump": "hop on one foot",
    "squat": "squat down and stand up",
    "step forward back": "step forward and back",
    "step forward forward": "step forward twice",
    "walk": "walk forward walking",
    "armraise dance2": "raise arms and dance",
    "dance2": "dance",
    "run1": "run jog running",
    "output dance": "dance",
    "high5": "high five greeting",
    "jumpjack": "jumping jacks exercise",
    "output wave": "wave hello greeting",
    "throw ball": "throw a ball",
    "big light one hand pick up front low": "bend down and pick up a big light object with one hand",
    "small light one hand pick up front low": "bend down and pick up a small object with one hand",
    "item pick up standing": "pick up an item while standing",
    "body stretch": "stretch the body warm up",
    "dance basic chaines": "ballet chaines turns spin dance",
    "dance hiphop shuffle square": "hip hop shuffle dance",
    "high jump": "high jump athletics",
    "wave R": "wave hello with the right hand",
    "arm swing": "swing the arms",
    "squat lite": "small squat knee bend",
    "stand": "stand still idle stop",
    "wave right": "wave the right hand hello",
    "weight shift": "shift weight from foot to foot sway",
    "Take 102": "expressive full body dance take",
    "gangnam style": "Gangnam Style dance psy",
}


def describe(name: str) -> str:
    """Readable description of a library clip from its name (+ GLOSSARY)."""
    s = re.sub(r"(motions_raw_tairantestbed_smpl_video_|TairanTestbed_TairanTestbed_|_filter|_amass|_ts[\d.]+|_v\d+$)",
               "", name)
    s = re.sub(r"_(R|L)?_?\d{3}__A\d+", "", s)
    s = re.sub(r"_subject\d+", "", s)
    s = s.replace("G1_", "").replace("Neutral_", "").replace("output_", "output ").replace("_", " ").strip()
    level = re.search(r"level(\d)", s)
    words = [s]
    for key, gloss in GLOSSARY.items():
        if key.lower() in s.lower() or key.lower() in name.lower().replace("_", " "):
            words.append(gloss)
    if level:
        words.append({"1": "easy gentle", "2": "moderate", "3": "energetic", "4": "hard fast", "5": "very hard fast"}
                     .get(level.group(1), ""))
    return re.sub(r"\s+", " ", " ".join(words).replace(f"level{level.group(1)}" if level else "\0", "")).strip()


class Retriever:
    def __init__(self, library: Path = LIBRARY, safe_eval: Path | None = SAFE_EVAL):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        man = json.loads(library.read_text(encoding="utf-8"))
        root = library.parent
        self.clips = [(c["name"], (root / c["npz"]).resolve()) for c in man["clips"]]
        self.desc = [describe(n) for n, _ in self.clips]
        self.safe = None
        if safe_eval is not None and Path(safe_eval).is_file():
            rows = json.loads(Path(safe_eval).read_text(encoding="utf-8"))["rows"]
            self.safe = {r[0] for r in rows if r[1] == 0}
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True).eval()
        self.emb = self._embed(self.desc)

    def _embed(self, texts: list[str]):
        b = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            h = self.model(**b).last_hidden_state
        a = b["attention_mask"].unsqueeze(-1).float()
        return self.torch.nn.functional.normalize((h * a).sum(1) / a.sum(1), dim=-1)

    def rank(self, prompt: str, k: int = 5) -> list[dict]:
        sims = (self._embed([prompt]) @ self.emb.T)[0]
        order = self.torch.argsort(sims, descending=True).tolist()
        return [{"name": self.clips[i][0], "npz": str(self.clips[i][1]), "score": round(float(sims[i]), 3),
                 "desc": self.desc[i], "safe": None if self.safe is None else self.clips[i][0] in self.safe}
                for i in order[:k]]

    STOP_WORDS = {"stop", "stand", "idle", "halt", "freeze", "rest", "stand still", "stop moving"}

    def choose(self, prompt: str, min_score: float = 0.25) -> tuple[dict | None, list[dict]]:
        if prompt.strip().lower().rstrip("!. ") in self.STOP_WORDS:  # explicit stop -> the stand clip
            stand = next((i for i, (n, _) in enumerate(self.clips) if n == "stand"), None)
            if stand is not None:
                r = {"name": "stand", "npz": str(self.clips[stand][1]), "score": 1.0, "desc": self.desc[stand],
                     "safe": True}
                return r, [r]
        ranked = self.rank(prompt, k=len(self.clips))
        safe = [r for r in ranked if r["safe"] is not False]
        best = safe[0] if safe and safe[0]["score"] >= min_score else None
        return best, ranked[:5]


def send(npz: str | Path, live_dir: Path, tag: str) -> Path:
    """Atomic drop into the live folder (the simulation never sees a half-written file)."""
    live_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{tag}.npz"
    tmp = live_dir / (name + ".part")
    shutil.copyfile(npz, tmp)
    os.replace(tmp, live_dir / name)
    return live_dir / name


def handle(r: Retriever, prompt: str, live_dir: Path, dry_run: bool) -> dict:
    t0 = time.perf_counter()
    best, top = r.choose(prompt)
    out = {"prompt": prompt, "top": top, "chosen": best["name"] if best else None,
           "retrieval_ms": round(1e3 * (time.perf_counter() - t0), 1)}
    if top and top[0]["safe"] is False:
        out["note"] = (f"closest clip '{top[0]['name']}' falls on the real-motor twin; "
                       + (f"playing the closest safe clip '{best['name']}'" if best else "nothing safe is close enough"))
    if best and not dry_run:
        out["sent"] = str(send(best["npz"], live_dir, re.sub(r"[^A-Za-z0-9]+", "_", prompt)[:40]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--live_dir", type=Path, default=LIVE_DIR)
    ap.add_argument("--library", type=Path, default=LIBRARY)
    ap.add_argument("--safe_eval", type=Path, default=SAFE_EVAL, help="per-clip eval JSON; 'none' = no safety filter")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--script", default=None,
                    help="'t:prompt,t:prompt,...' -> print a scripts/play.py --live_script string (sim-time schedule; for "
                         "recorded demos, since the Isaac sim runs slower than real time)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    r = Retriever(args.library, None if str(args.safe_eval).lower() == "none" else args.safe_eval)
    print(f"[text2motion] {len(r.clips)} clips ({'all' if r.safe is None else len(r.safe)} safe), model loaded in "
          f"{time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    if args.script:
        items = []
        for item in [s.strip() for s in args.script.split(",") if s.strip()]:
            ts, prompt = item.split(":", 1)
            best, top = r.choose(prompt)
            print(f"[text2motion] {float(ts):5.1f} s  {prompt!r} -> {best['name'] if best else 'nothing safe'}"
                  + ("" if not top or top[0]["safe"] is not False else f"  (closest '{top[0]['name']}' is unsafe)"),
                  file=sys.stderr)
            if best:
                items.append(f"{float(ts):g}:{Path(best['npz']).as_posix()}")
        print(",".join(items))
    if args.prompt:
        print(json.dumps(handle(r, args.prompt, args.live_dir, args.dry_run), indent=1))
    if args.interactive:
        while True:
            try:
                prompt = input("dropbear> ").strip()
            except EOFError:
                break
            if prompt in ("quit", "exit"):
                break
            if prompt:
                res = handle(r, prompt, args.live_dir, args.dry_run)
                print(f"  -> {res['chosen']}  ({res['retrieval_ms']} ms)" + (f"  [{res['note']}]" if "note" in res else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
