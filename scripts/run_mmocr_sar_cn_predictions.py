#!/usr/bin/env python3
"""Run MMOCR DBPP/PANet + SAR_CN on 0.5s frames and write JSONL outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def patch_runtime_compat() -> None:
    """Compatibility shims for this local Python 3.12 / NumPy 2 / Torch 2 setup."""
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.bytes_, np.str_],
        }

    original_torch_load = torch.load

    def torch_load_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat


REPO_ROOT = Path(__file__).resolve().parents[1]

FORMAL_ROOT = REPO_ROOT / "data/formal_testset"
FRAMES_ROOT = REPO_ROOT / "frames_0.5s"
OUT_DIR = REPO_ROOT / "results"

SAR_CN_CONFIG = REPO_ROOT / "configs/mmocr_sar_cn/sar_cn_mmocr1.py"
SAR_CN_WEIGHTS = Path(
    REPO_ROOT
    / "weights/mmocr/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214_mmocr1_compat.pth"
)
DBPP_WEIGHTS = Path(
    REPO_ROOT
    / "weights/mmocr/dbnetpp_resnet50_fpnc_1200e_icdar2015_20221025_185550-013730aa.pth"
)
PANET_WEIGHTS = Path(
    REPO_ROOT
    / "weights/mmocr/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth"
)

METHODS = {
    "quality": {
        "engine": "mmocr_quality_dbpp_sar_cn",
        "det": "DBPP_r50",
        "det_weights": DBPP_WEIGHTS,
        "out": OUT_DIR / "mmocr_quality_dbpp_sar_cn_predictions.jsonl",
    },
    "engineering": {
        "engine": "mmocr_engineering_panet_sar_cn",
        "det": "PANet_IC15",
        "det_weights": PANET_WEIGHTS,
        "out": OUT_DIR / "mmocr_engineering_panet_sar_cn_predictions.jsonl",
    },
}


def formal_video_ids() -> list[str]:
    gt_dir = FORMAL_ROOT / "subtitles_txt"
    return sorted(path.stem for path in gt_dir.glob("*.txt") if path.read_text(encoding="utf-8").strip())


def frame_paths(video_id: str) -> list[Path]:
    return sorted((FRAMES_ROOT / video_id).glob("*.jpg"))


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def prediction_texts(pred: dict[str, Any]) -> list[str]:
    texts = pred.get("rec_texts") or []
    polygons = pred.get("det_polygons") or []
    items: list[tuple[float, float, int, str]] = []
    for idx, text in enumerate(texts):
        text = clean_text(text)
        if not text:
            continue
        polygon = polygons[idx] if idx < len(polygons) else []
        if polygon:
            xs = [float(x) for x in polygon[0::2]]
            ys = [float(y) for y in polygon[1::2]]
            y = sum(ys) / len(ys)
            x = sum(xs) / len(xs)
        else:
            y = float(idx)
            x = 0.0
        items.append((y, x, idx, text))
    return [text for _, _, _, text in sorted(items)]


def existing_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = str(row.get("video_id") or "")
            if video_id:
                done.add(video_id)
    return done


def build_inferencer(method_cfg: dict[str, Any], device: str):
    from mmocr.apis import MMOCRInferencer

    return MMOCRInferencer(
        det=method_cfg["det"],
        det_weights=str(method_cfg["det_weights"]),
        rec=str(SAR_CN_CONFIG),
        rec_weights=str(SAR_CN_WEIGHTS),
        device=device,
    )


def call_inferencer(inferencer: Any, inputs: Any) -> list[dict[str, Any]]:
    result = inferencer(
        inputs,
        return_vis=False,
        show=False,
        print_result=False,
    )
    return result.get("predictions", [])


def infer_video_predictions(inferencer: Any, paths: list[Path]) -> list[dict[str, Any]]:
    try:
        return call_inferencer(inferencer, [str(path) for path in paths])
    except Exception as exc:
        print(
            f"batch inference failed; falling back to frame-by-frame: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    predictions: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        try:
            preds = call_inferencer(inferencer, str(path))
            predictions.append(preds[0] if preds else {})
        except Exception as exc:
            print(
                f"frame inference failed idx={idx} path={path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            predictions.append({})
    return predictions


def run_method(method_key: str, device: str, limit: int | None) -> None:
    method_cfg = METHODS[method_key]
    engine = method_cfg["engine"]
    out_path = method_cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_ids = formal_video_ids()
    if limit is not None:
        video_ids = video_ids[:limit]

    done_ids = existing_video_ids(out_path)
    pending_video_ids = [video_id for video_id in video_ids if video_id not in done_ids]
    if not pending_video_ids:
        print(f"{engine} already complete out={out_path}", flush=True)
        return

    inferencer = build_inferencer(method_cfg, device)
    start_all = time.time()
    mode = "a" if done_ids else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for video_idx, video_id in enumerate(pending_video_ids, start=1):
            paths = frame_paths(video_id)
            start = time.time()
            predictions = infer_video_predictions(inferencer, paths)
            frame_texts = []
            for frame_idx in range(len(paths)):
                pred = predictions[frame_idx] if frame_idx < len(predictions) else {}
                frame_texts.append(
                    {
                        "t": round(frame_idx * 0.5, 3),
                        "texts": prediction_texts(pred),
                    }
                )
            row = {
                "engine": engine,
                "video_id": video_id,
                "frames": len(paths),
                "frame_texts": frame_texts,
                "seconds": round(time.time() - start, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"{engine} {len(done_ids) + video_idx}/{len(video_ids)} {video_id} "
                f"frames={len(paths)} seconds={row['seconds']}",
                flush=True,
            )
    print(f"{engine} done total_seconds={time.time() - start_all:.3f} out={out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["quality", "engineering", "both"],
        default="both",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    patch_runtime_compat()
    method_keys = ["quality", "engineering"] if args.method == "both" else [args.method]
    for method_key in method_keys:
        run_method(method_key, args.device, args.limit)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
