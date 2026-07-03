# Short-video OCR benchmark

This repository contains the code, data, raw OCR outputs, unified post-processing outputs, and result analysis for the short-video full-frame OCR benchmark.

## Evaluation scope

- Formal test set: 25 videos with non-empty OCR ground truth.
- Frame sampling: 0.5s/frame.
- Ground truth: manually labeled subtitle text.
- Character accounting: each Chinese character, ASCII letter, or digit counts as one character.
- Metrics: CER, Precision, Recall, F1.

## Unified post-processing

All methods in the final table are evaluated after the same subtitle-level merge:

- merge type: adjacent conservative fuzzy merge
- frame interval: 0.5s
- similarity threshold: 0.88
- short-text threshold: 0.94
- max blank gap: 1.0s

Main evaluator:

```bash
python scripts/unified_subtitle_merge_eval.py
```

The repository copy of the script uses repo-relative default paths.

## Final micro-average results

| Method | CER | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| GoMatching++ | 0.5626 | 0.6409 | 0.7202 | 0.6782 |
| RapidOCR | 0.7862 | 0.5520 | 0.8839 | 0.6796 |
| PPOCR v4 | 0.9758 | 0.4986 | 0.9014 | 0.6420 |
| PPOCR v5 | 1.0832 | 0.4686 | 0.8944 | 0.6150 |
| EasyOCR | 1.0891 | 0.4418 | 0.8051 | 0.5705 |
| Paddle-VL1.5 / PPOCR result | 1.1087 | 0.4624 | 0.8787 | 0.6060 |
| PPOCR v6 | 1.1087 | 0.4624 | 0.8787 | 0.6060 |
| GLM-OCR | 1.4825 | 0.3555 | 0.7363 | 0.4795 |
| MMOCR PANet_IC15 + SAR_CN | 19.8879 | 0.0319 | 0.6560 | 0.0609 |
| MMOCR DBNet++ / DBPP_r50 + SAR_CN | 25.3159 | 0.0231 | 0.5977 | 0.0444 |

The Feishu-ready analysis document is available at `results/unified_subtitle_merge/ocr_result_analysis_feishu.md`.

## MMOCR notes

MMOCR was initially missing scores due to local environment and compatibility blockers:

- `mmcv-lite` lacked `mmcv._ext`; full `mmcv==2.0.1` with CUDA ops was required.
- SAR_CN needed an MMOCR 1.x-compatible config and dictionary.
- NumPy 2.x and Torch 2.6+ checkpoint loading required runtime compatibility shims.
- PANet occasionally produced invalid empty crops; the runner falls back to per-frame inference and emits empty OCR for failed frames.

The MMOCR final scores are poor because full-frame detection over-generates massive non-subtitle/garbage text. This is a modeling/post-filtering issue, not a scoring pipeline issue.

## Large artifacts

Model checkpoints are intentionally excluded from git. To download MMOCR weights:

```bash
bash scripts/download_mmocr_weights.sh ./weights/mmocr
```

The extracted 0.5s frames are also excluded because they can be regenerated from the included videos.

To regenerate frames:

```bash
python scripts/extract_frames_0.5s.py
```

To prepare the SAR_CN compatibility checkpoint after downloading the official weights:

```bash
python scripts/create_sar_cn_compat_checkpoint.py
```

## Repository layout

```text
configs/                 MMOCR SAR_CN config and dictionary
data/                    formal test set, external raw predictions, GoMatching outputs
docs/                    evaluation standard and report assets
environment/             exact pip freeze from the local OCR venv
notebooks/               original notebook
results/                 raw outputs, merged outputs, metrics, diagnostics, analysis
scripts/                 runnable evaluation and conversion scripts
```
