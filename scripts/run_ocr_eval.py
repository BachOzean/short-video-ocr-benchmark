#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


def normalize_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[\u200b\ufeff]", "", s)
    punct = "，。！？、；：‘’“”（）()[]【】{}<>《》,.!?;:'\"`~·|\\/+-_=*&#@$%^"
    return "".join(ch for ch in s if ch not in punct).lower()


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accuracy(gt: str, pred: str) -> float:
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)
    denom = max(len(gt_n), len(pred_n), 1)
    return max(0.0, 1.0 - edit_distance(gt_n, pred_n) / denom)


def ordered_unique(items: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        n = normalize_text(item)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(str(item).strip())
    return out


def parse_gt(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return [str(value).strip()]
    except Exception:
        return [raw]


def load_dataset(dataset: Path) -> List[Dict[str, str]]:
    meta = dataset / "metadata.csv"
    rows = []
    with meta.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt = parse_gt(row.get("ocr_gt", ""))
            if not gt:
                continue
            row["_gt_lines"] = gt
            row["_video_path"] = str(dataset / "videos" / f"{row['video_id']}.mp4")
            rows.append(row)
    return rows


def sample_frames(video_path: Path, interval: float, crop_bottom_ratio: float) -> Iterable[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps * interval)))
    frame_idx = 0
    next_sample = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx >= next_sample:
            if 0.0 < crop_bottom_ratio < 1.0:
                h = frame.shape[0]
                y0 = int(h * (1.0 - crop_bottom_ratio))
                frame = frame[y0:h, :]
            yield frame_idx / fps, frame
            next_sample += step
        frame_idx += 1
    cap.release()


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, device: str):
        import torch
        import easyocr
        gpu = device.startswith("cuda") and torch.cuda.is_available()
        self.reader = easyocr.Reader(["ch_sim", "en"], gpu=gpu)

    def recognize(self, frame: np.ndarray) -> List[str]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.reader.readtext(rgb, detail=0, paragraph=False)
        return [str(x).strip() for x in result if str(x).strip()]


class MMOcrEngine:
    name = "mmocr"

    def __init__(self, device: str):
        from mmocr.apis import MMOCRInferencer
        self.ocr = MMOCRInferencer(det="DBNet", rec="CRNN", device=device)

    def recognize(self, frame: np.ndarray) -> List[str]:
        result = self.ocr(frame, return_vis=False, print_result=False, show=False)
        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        if predictions and isinstance(predictions[0], dict):
            pred = predictions[0]
            for key in ("rec_texts", "texts"):
                if key in pred and isinstance(pred[key], list):
                    return [str(x).strip() for x in pred[key] if str(x).strip()]
        return []


class VimTSEngine:
    name = "vimts"

    def __init__(self, device: str):
        raise RuntimeError(
            "VimTS is not deployed in this environment. Official VimTS requires "
            "Python 3.8, PyTorch 1.10.0, CUDA 11.3, Detectron2 0.2.1, custom CUDA ops, "
            "and an external checkpoint. That stack is not compatible with the current "
            "Python 3.12 + RTX 5070 sm_120 + CUDA 13 runtime without porting."
        )

    def recognize(self, frame: np.ndarray) -> List[str]:
        return []


def make_engine(name: str, device: str):
    if name == "easyocr":
        return EasyOCREngine(device)
    if name == "mmocr":
        return MMOcrEngine(device)
    if name == "vimts":
        return VimTSEngine(device)
    raise ValueError(name)


def read_done(path: Path) -> Dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done[item["video_id"]] = item
    return done


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def evaluate_records(records: List[dict]) -> dict:
    per_video = []
    total_gt = []
    total_pred = []
    for r in records:
        gt_lines = r.get("gt_lines", [])
        pred_lines = r.get("pred_lines_unique", [])
        gt_text = "".join(gt_lines)
        pred_text = "".join(pred_lines)
        gt_norm = normalize_text(gt_text)
        pred_norm = normalize_text(pred_text)
        acc = accuracy(gt_text, pred_text)
        gt_line_hits = 0
        for line in gt_lines:
            if normalize_text(line) and normalize_text(line) in pred_norm:
                gt_line_hits += 1
        line_recall = gt_line_hits / len(gt_lines) if gt_lines else 0.0
        per_video.append({
            "video_id": r["video_id"],
            "acc_norm_edit": acc,
            "gt_line_recall": line_recall,
            "gt_chars": len(gt_norm),
            "pred_chars": len(pred_norm),
            "frames": r.get("frames", 0),
        })
        total_gt.append(gt_text)
        total_pred.append(pred_text)
    macro_acc = sum(x["acc_norm_edit"] for x in per_video) / len(per_video) if per_video else 0.0
    macro_line_recall = sum(x["gt_line_recall"] for x in per_video) / len(per_video) if per_video else 0.0
    corpus_acc = accuracy("".join(total_gt), "".join(total_pred)) if per_video else 0.0
    return {
        "macro_acc_norm_edit": macro_acc,
        "corpus_acc_norm_edit": corpus_acc,
        "macro_gt_line_recall": macro_line_recall,
        "videos": len(per_video),
        "per_video": per_video,
    }


def run_engine(engine_name: str, rows: List[Dict[str, str]], args) -> dict:
    out_dir = Path(args.out)
    pred_path = out_dir / f"{engine_name}_predictions.jsonl"
    status_path = out_dir / f"{engine_name}_status.json"
    done = read_done(pred_path)
    try:
        engine = make_engine(engine_name, args.device)
    except Exception as e:
        status = {"engine": engine_name, "status": "unavailable", "error": f"{type(e).__name__}: {e}"}
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    started = time.time()
    for idx, row in enumerate(rows, 1):
        video_id = row["video_id"]
        if video_id in done:
            print(f"[{engine_name}] [{idx}/{len(rows)}] skip {video_id}", flush=True)
            continue
        frame_texts = []
        frame_count = 0
        video_path = Path(row["_video_path"])
        t0 = time.time()
        try:
            for ts, frame in sample_frames(video_path, args.interval, args.crop_bottom_ratio):
                texts = engine.recognize(frame)
                frame_texts.append({"t": round(ts, 3), "texts": texts})
                frame_count += 1
            all_lines = []
            for frame_item in frame_texts:
                all_lines.extend(frame_item["texts"])
            item = {
                "engine": engine_name,
                "video_id": video_id,
                "video_path": str(video_path),
                "gt_lines": row["_gt_lines"],
                "frames": frame_count,
                "frame_texts": frame_texts,
                "pred_lines_unique": ordered_unique(all_lines),
                "seconds": round(time.time() - t0, 3),
            }
        except Exception as e:
            item = {
                "engine": engine_name,
                "video_id": video_id,
                "video_path": str(video_path),
                "gt_lines": row["_gt_lines"],
                "frames": frame_count,
                "frame_texts": frame_texts,
                "pred_lines_unique": [],
                "seconds": round(time.time() - t0, 3),
                "error": f"{type(e).__name__}: {e}",
            }
        append_jsonl(pred_path, item)
        done[video_id] = item
        state = "error" if "error" in item else "done"
        print(f"[{engine_name}] [{idx}/{len(rows)}] {state} {video_id} frames={frame_count} sec={item['seconds']}", flush=True)

    records = list(read_done(pred_path).values())
    metrics = evaluate_records(records)
    status = {
        "engine": engine_name,
        "status": "complete",
        "prediction_file": str(pred_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "metrics": metrics,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def write_summary(out_dir: Path, statuses: List[dict]) -> None:
    summary = {"statuses": statuses}
    rows = []
    for s in statuses:
        m = s.get("metrics") or {}
        rows.append({
            "engine": s.get("engine"),
            "status": s.get("status"),
            "videos": m.get("videos", 0),
            "macro_acc_norm_edit": m.get("macro_acc_norm_edit", ""),
            "corpus_acc_norm_edit": m.get("corpus_acc_norm_edit", ""),
            "macro_gt_line_recall": m.get("macro_gt_line_recall", ""),
            "error": s.get("error", ""),
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["engine"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/formal_testset")
    parser.add_argument("--out", default="results")
    parser.add_argument("--engine", choices=["easyocr", "mmocr", "vimts", "all"], default="all")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-bottom-ratio", type=float, default=1.0)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(dataset)
    if args.max_videos > 0:
        rows = rows[: args.max_videos]
    engines = ["easyocr", "mmocr", "vimts"] if args.engine == "all" else [args.engine]
    statuses = []
    for engine_name in engines:
        statuses.append(run_engine(engine_name, rows, args))
    write_summary(out_dir, statuses)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "metrics": str(out_dir / "metrics.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
