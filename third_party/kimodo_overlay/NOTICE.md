# Kimodo overlay (Apache-2.0)

Files here replace or add to NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) (Apache License 2.0, see
`LICENSE`), pinned at commit `54257dd8ff18aa764d620919427ce4dd29c111d0` (2026-05-03). `tools/setup_kimodo.sh`
clones that commit into `third_party/kimodo/` (untracked) and copies these files over it.

Modifications (dropbear-wbc, 2026-09-26):

| file | change |
|---|---|
| `kimodo/model/text_encoder_file.py` | **new.** `TextEncoderFile`: the text encoder as a client of a shared folder. Kimodo (WSL, GPU) writes `requests/<id>.txt`; `tools/kimodo_text_embed_server.py` (Windows, CPU) answers `responses/<id>.npy`. Folder: `$KIMODO_EMBED_BRIDGE`. |
| `kimodo/model/load_model.py` | `TEXT_ENCODER_MODE=file` selects `TextEncoderFile` (2 lines). |
| `kimodo/model/llm2vec/llm2vec_wrapper.py` | optional `TEXT_ENCODER_QUANT=4bit|8bit` (bitsandbytes) load path. Unused in practice: with transformers 5.1, PEFT adapters fail to load onto a quantized model. The working low-memory path is the int8 checkpoint of `tools/build_llm2vec_w8.py`. |

Why: the Llama-3-8B text encoder (16 GB in bf16) does not fit next to Kimodo on a 12 GB laptop GPU, so it runs on the
Windows CPU and Kimodo runs on the GPU in WSL (docs/TEXT_TO_MOTION.md).
