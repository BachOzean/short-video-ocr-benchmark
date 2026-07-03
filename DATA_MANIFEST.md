# Data manifest

This repository packages the code and data used for the short-video OCR benchmark run.

Included:

- `data/formal_testset/`: 25 formal test videos with non-empty OCR ground truth, subtitle TXT/JSON files, metadata, and manifest.
- `data/frame_predictions/`: external 0.5s/frame raw prediction CSVs for RapidOCR, Paddle/PPOCR, and GLM-OCR.
- `data/gomatching_pp_bov/`: GoMatching++ raw XML/JSON outputs used by the unified evaluator.
- `results/`: raw predictions, unified subtitle-level merge outputs, per-video CSVs, metrics JSON, status files, and analysis documents.
- `configs/mmocr_sar_cn/`: MMOCR 1.x SAR_CN config and Chinese dictionary used in the local run.
- `scripts/`: preprocessing, OCR run, conversion, metric, and unified evaluation scripts.
- `docs/`: original evaluation-standard PDF and Feishu-ready result analysis.
- `notebooks/`: original download/inspection notebook.

Not included:

- Extracted 0.5s image frames. They are regenerable from `data/formal_testset/videos/` and were omitted to avoid duplicating video data.
- MMOCR model checkpoints and local `mmcv` build products. These are too large/noisy for a normal GitHub repository. Use `scripts/download_mmocr_weights.sh` and the notes in `README.md`.
