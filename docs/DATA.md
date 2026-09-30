# Data: what is published, where it comes from, how to rebuild the rest

Motion data is the one part of this project that cannot simply be copied: several sources forbid redistribution. The
rule here is the one the pipeline already enforces: every clip's sidecar records its source license and a
`redistributable` flag (`source/dropbear_wbc/motion/g1_sources.py`, `SOURCES`).

## What you get

| what | where it lives | how to get it |
|---|---|---|
| calibration (`e51033d4`), robot model data (serial MJCF, sole hulls, actuator datasheet) | `data/calibration/`, `data/robot/` | in git |
| synthetic clips made here (stand, wave, squat, arm swing, weight shift...) | `data/motions/synthetic/`, `data/motions_v6ts/synthetic/`, `data/motions/smoke/` | in git |
| library manifests (`accepted_v6ts`, `accepted_v7gen`) and their public variants | `data/motions_v6ts/libraries/` | in git |
| Kimodo prompt list for the generated clips | `data/motions_gen/prompts/everyday_v1.json` | in git |
| 60 Kimodo-generated everyday clips (Froude-timed, settled, validated) | `data/motions_v6ts/kimodo_gen/` | `python tools/fetch_assets.py --groups motions` |
| Kimodo G1 example motions, retargeted | `data/motions_v6ts/kimodo_g1/` | same |
| ASAP clips (MIT), retargeted | `data/motions_v6ts/asap_g1/` | same |
| cyclic Kimodo walks | `data/motions_cyclic/kimodo_walk*.npz` | same |
| raw Kimodo G1 outputs (CSV) behind the generated clips | `data/motions_gen/raw/`, `data/motions_gen/kimodo_g1/` | same |
| GR00T tabletop push dataset (LeRobot format, sim) | `data/groot/dropbear_tabletop_push_v1/` | `--groups datasets` |

`tools/fetch_assets.py` verifies every file against the SHA-256 in `assets_manifest.json`.

## Libraries you can use immediately

| manifest | clips | notes |
|---|---|---|
| `data/motions_v6ts/libraries/accepted_v7gen_public.json` | 122 of 138 | everything in `accepted_v7gen` except the 16 non-redistributable clips |
| `data/motions_v6ts/libraries/accepted_v6ts_public.json` | 62 of 78 | same for `accepted_v6ts` |

Both were made by `tools/make_public_library.py` and keep the per-clip SHA-256 pins. Train or evaluate on them exactly
like the full libraries (`--motion_library data/motions_v6ts/libraries/accepted_v7gen_public.json`).

## What is not published, and why

| source | license | clips in `accepted_v7gen` |
|---|---|---|
| LAFAN1 (Ubisoft La Forge), via `lvhaidong/LAFAN1_Retargeting_Dataset` and GMR | CC BY-NC-ND 4.0: non-commercial, **no derivatives** | 4 (`gmr_lafan1`) + the cyclic LAFAN walk |
| NVIDIA BONES-SEED sample motions (bundled with soma-retargeter) | NVIDIA sample-data evaluation license | 10 (`soma_retargeter_g1`) |
| unitree_rl_lab mimic dance clips | repository Apache-2.0, upstream mocap provenance not documented | 2 (`unitree_rl_lab_mimic`) |
| GR00T-WholeBodyControl (SONIC) reference motions | see the repository's dual-license notice | used only for checks |

Policies trained on the full libraries were published anyway, for noncommercial research use (like everything here), with this provenance
listed (see [HF_MODEL_CARD.md](HF_MODEL_CARD.md)). If you need a clean-lineage library policy, train on
`accepted_v7gen_public.json`.

## Rebuilding the full libraries from your own downloads

Every clip goes through the same pipeline (docs/SERIAL_MODEL_AND_GMR.md, docs/DEMOS.md): G1 motion -> Dropbear
retarget (`tools/retarget_g1.py`, Froude time scale x1.19, ISSUES #27) -> physics settle in Isaac
(`tools/settle_motion.py`) -> validation (`tools/validate_motion_npz.py`) -> manifest
(`tools/build_motion_library_manifest.py`).

1. Put the upstream sources under `$DROPBEAR_UPSTREAM` (default `../upstream`) or `../data`:
   - LAFAN1 G1 retargets: `huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset
     --local-dir ../data/LAFAN1_Retargeting_Dataset` (accept its license).
   - `git clone https://github.com/NVIDIA/soma-retargeter ../upstream/soma-retargeter`
   - `git clone https://github.com/unitreerobotics/unitree_rl_lab ../upstream/unitree_rl_lab`
   - LAFAN1 BVH for the GMR route: `../upstream/lafan1/bvh/` (from Ubisoft La Forge), GMR via `python tools/setup_gmr.py`.
2. Retarget the missing G1-route clips with the source key of their manifest folder (`soma_retargeter_g1`,
   `unitree_rl_lab_mimic`; `lafan1_g1` for the LAFAN1 G1 retargets). The `gmr_lafan1` clips come from the GMR route
   (BVH -> Dropbear directly, below). `tools/retarget_g1.py --list-sources` shows the discovered files; for example:

   ```bash
   python tools/retarget_g1.py ../upstream/soma-retargeter/assets/motions/<clip>.csv --source soma_retargeter_g1 \
       --out-root data/motions_v6ts --time-scale 1.19
   ```
3. Settle and validate (Isaac Sim), exactly as the library build did:

   ```bash
   <isaac-python> -u tools/settle_motion.py --headless --keep-going --out-suffix _v6 --csv data/motions_v6ts/<source>/*_ts1.19.csv
   python tools/validate_motion_npz.py data/motions_v6ts/<source>/*_v6.npz --write-verdicts
   ```
4. The full manifests then resolve: their per-clip SHA-256 pins tell you whether your rebuild is byte-identical. A
   different upstream version gives a different NPZ; regenerate the manifest with
   `python tools/build_motion_library_manifest.py --root data/motions_v6ts --name accepted_v7gen_local ...`.

GMR LAFAN1 clips follow docs/SERIAL_MODEL_AND_GMR.md (`tools/gmr_lafan1_batch.py`, `.venv-gmr`).

## Provenance records

Calibration files and clip sidecars keep the absolute paths of the machine they were built on (for example the USD's
original location) inside their provenance blocks. They are records, not configuration: the code resolves every path
through `source/dropbear_wbc/paths.py`, and the calibration files are pinned by SHA-256, so they are left untouched.
