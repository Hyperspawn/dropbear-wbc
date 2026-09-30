# Third-party code, models and data

This project's own code, data, robot model and trained policies are licensed for **noncommercial use only** under the
PolyForm Noncommercial License 1.0.0 (`LICENSE`); for commercial use, contact Hyperspawn ([hyperspawn.co](https://hyperspawn.co), priyanshu@hyperspawn.co). The project
builds on, vendors, or downloads the following; their own licenses apply to those parts.

## Vendored / adapted code (in this repository)

| component | where | license | notes |
|---|---|---|---|
| rsl-rl-lib 2.3.3 (ETH Zurich RSL) | `third_party/pydeps/rsl_rl/` | BSD-3-Clause (`third_party/pydeps/rsl_rl_lib-2.3.3.dist-info/licenses/LICENSE`) | vendored unmodified; Isaac Sim's kit python ships an incompatible rsl-rl 5.x |
| Kimodo overlay (NVIDIA Kimodo) | `third_party/kimodo_overlay/` | Apache-2.0 (`third_party/kimodo_overlay/LICENSE`) | 3 modified/added files; `tools/setup_kimodo.sh` clones upstream at the pinned commit (`NOTICE.md`) |
| GMR patch (Yanjie Ze et al.) | `third_party/gmr_dropbear/` | MIT | a patch only; `tools/setup_gmr.py` applies it to a clone (`NOTICE.txt`) |
| televuer (Unitree) | `source/dropbear_wbc/teleop/` | MIT | adapted constants / transforms (`third_party/televuer_NOTICE.md`) |

## Installed separately (not in this repository)

NVIDIA Isaac Sim 5.0 and Isaac Lab 2.2 (their own licenses), NVIDIA Newton / Warp / MuJoCo Warp (Apache-2.0),
MuJoCo (Apache-2.0), PyTorch (BSD-3-Clause), Hugging Face transformers / peft (Apache-2.0).

## Models (downloaded by the user; not redistributed here)

| model | license | used for |
|---|---|---|
| NVIDIA Kimodo-G1-RP-v1 | NVIDIA Open Model License | text -> G1 motion generation |
| Meta Llama-3-8B-Instruct (gated) | Meta Llama 3 Community License (accept on Hugging Face) | base of the LLM2Vec text encoder |
| McGill-NLP LLM2Vec MNTP + supervised adapters | MIT | text encoder adapters |

`tools/build_llm2vec_w8.py` builds an int8 copy of the encoder locally from your own gated download; that copy is
not published.

## Motion data

See [docs/DATA.md](docs/DATA.md). In short: only clips whose source license allows redistribution are published
(synthetic clips made here, Kimodo-generated clips, ASAP clips); LAFAN1 (CC BY-NC-ND 4.0), NVIDIA BONES-SEED sample
data (evaluation license), GR00T/SONIC reference motions and the unitree_rl_lab dance clips must be rebuilt from your
own download with the documented scripts.

## Trained policies (Hugging Face)

Some published policies were trained on clip libraries that include LAFAN1 and BONES-SEED sample clips. They are
released for non-commercial research use; each policy's training data is listed in
[docs/HF_MODEL_CARD.md](docs/HF_MODEL_CARD.md).
