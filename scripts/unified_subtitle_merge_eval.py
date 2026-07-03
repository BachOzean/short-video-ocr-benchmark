#!/usr/bin/env python3
"""Unified subtitle-level merge and OCR metric evaluation.

The input side accepts several raw 0.5s/frame result formats and converts all
of them to the same representation:

    video_id -> ordered frame texts -> adjacent conservative fuzzy merge

The evaluation side compares the merged per-video prediction text with the
formal test-set subtitle ground truth using a character-level edit distance.
Only CJK unified ideographs, ASCII letters and digits are counted. ASCII
letters are lowercased.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FORMAL_ROOT = REPO_ROOT / "data/formal_testset"
DEFAULT_OUT_DIR = REPO_ROOT / "results/unified_subtitle_merge"

DEFAULT_SOURCES = [
    {
        "method": "easyocr",
        "kind": "jsonl_frames",
        "path": str(REPO_ROOT / "results/easyocr_predictions.jsonl"),
    },
    {
        "method": "gomatching_pp",
        "kind": "gomatching_xml_dir",
        "path": str(REPO_ROOT / "data/gomatching_pp_bov/preds"),
    },
    {
        "method": "rapidocr",
        "kind": "csv_pred_text",
        "path": str(REPO_ROOT / "data/frame_predictions/frame_predictions_rapidocr.csv"),
    },
    {
        "method": "paddle_vl15",
        "kind": "csv_pred_text",
        "path": str(REPO_ROOT / "data/frame_predictions/frame_predictions_ppocr.csv"),
    },
    {
        "method": "glm_ocr",
        "kind": "csv_pred_text",
        "path": str(REPO_ROOT / "data/frame_predictions/frame_predictions_glmocr_clean.csv"),
    },
    {
        "method": "mmocr_quality_dbpp_sar_cn",
        "kind": "jsonl_frames",
        "path": str(REPO_ROOT / "results/mmocr_quality_dbpp_sar_cn_predictions.jsonl"),
        "optional": True,
    },
    {
        "method": "mmocr_engineering_panet_sar_cn",
        "kind": "jsonl_frames",
        "path": str(REPO_ROOT / "results/mmocr_engineering_panet_sar_cn_predictions.jsonl"),
        "optional": True,
    },
    {
        "method": "vimts_original",
        "kind": "jsonl_frames",
        "path": str(REPO_ROOT / "results/vimts_original_predictions.jsonl"),
        "optional": True,
    },
]


@dataclass
class FrameText:
    t: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    frames: int


@dataclass
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    gt_chars: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def correct(self) -> int:
        return max(0, self.gt_chars - self.substitutions - self.deletions)

    @property
    def pred_chars(self) -> int:
        return self.correct + self.substitutions + self.insertions


def normalize_counted(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            chars.append(ch)
        elif "A" <= ch <= "Z" or "a" <= ch <= "z" or "0" <= ch <= "9":
            chars.append(ch.lower())
    return "".join(chars)


def clean_raw_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def join_texts(texts: Iterable[Any]) -> str:
    parts = [clean_raw_text(str(x)) for x in texts if clean_raw_text(str(x))]
    return " ".join(parts)


def parse_gt(formal_root: Path) -> dict[str, list[str]]:
    gt_dir = formal_root / "subtitles_txt"
    gt: dict[str, list[str]] = {}
    for path in sorted(gt_dir.glob("*.txt")):
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            gt[path.stem] = lines
    return gt


def load_jsonl_frames(path: Path, allowed_video_ids: set[str]) -> dict[str, list[FrameText]]:
    videos: dict[str, list[FrameText]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id") or row.get("id") or "")
            if video_id not in allowed_video_ids:
                continue
            frames: list[FrameText] = []
            for item in row.get("frame_texts", []) or []:
                t = safe_float(item.get("t", item.get("timestamp_sec", 0.0)))
                if "text" in item:
                    text = clean_raw_text(str(item.get("text") or ""))
                else:
                    text = join_texts(item.get("texts", []) or [])
                frames.append(FrameText(t=t, text=text))
            videos[video_id] = sorted(frames, key=lambda x: x.t)
    return videos


def load_csv_pred_text(path: Path, allowed_video_ids: set[str]) -> dict[str, list[FrameText]]:
    videos: dict[str, list[FrameText]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = str(row.get("video_id") or row.get("sample_id") or "")
            if video_id not in allowed_video_ids:
                continue
            text = clean_raw_text(row.get("pred_text") or "")
            t = safe_float(row.get("timestamp_sec", row.get("t", 0.0)))
            videos[video_id].append(FrameText(t=t, text=text))
    return {video_id: sorted(frames, key=lambda x: x.t) for video_id, frames in videos.items()}


def load_gomatching_xml_dir(path: Path, allowed_video_ids: set[str], frame_interval: float) -> dict[str, list[FrameText]]:
    videos: dict[str, list[FrameText]] = {}
    for xml_path in sorted(path.glob("res_*.xml")):
        video_id = xml_path.stem
        if video_id.startswith("res_"):
            video_id = video_id[4:]
        if video_id not in allowed_video_ids:
            continue
        tree = ET.parse(xml_path)
        frames: list[FrameText] = []
        for frame in tree.findall(".//frame"):
            frame_id = int(frame.attrib.get("ID") or frame.attrib.get("id") or len(frames) + 1)
            objects: list[tuple[float, float, str]] = []
            for obj in frame.findall("./object"):
                text = clean_raw_text(obj.attrib.get("Transcription") or obj.attrib.get("transcription") or "")
                if not text:
                    continue
                xs: list[float] = []
                ys: list[float] = []
                for point in obj.findall("./Point"):
                    xs.append(safe_float(point.attrib.get("x", 0.0)))
                    ys.append(safe_float(point.attrib.get("y", 0.0)))
                min_x = min(xs) if xs else 0.0
                min_y = min(ys) if ys else 0.0
                objects.append((min_y, min_x, text))
            objects.sort(key=lambda item: (item[0], item[1]))
            frames.append(FrameText(t=(frame_id - 1) * frame_interval, text=join_texts(item[2] for item in objects)))
        videos[video_id] = sorted(frames, key=lambda x: x.t)
    return videos


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def should_merge(prev_norm: str, cur_norm: str, sim_threshold: float, short_threshold: float) -> bool:
    if not prev_norm or not cur_norm:
        return False
    if prev_norm == cur_norm:
        return True
    min_len = min(len(prev_norm), len(cur_norm))
    max_len = max(len(prev_norm), len(cur_norm))
    if min_len <= 4:
        return False
    length_ratio = min_len / max_len if max_len else 0.0
    if length_ratio < 0.72:
        return False
    threshold = short_threshold if max_len <= 10 else sim_threshold
    return SequenceMatcher(None, prev_norm, cur_norm).ratio() >= threshold


def choose_segment_text(texts: list[str]) -> str:
    non_empty = [clean_raw_text(text) for text in texts if normalize_counted(text)]
    if not non_empty:
        return ""
    by_norm: dict[str, Counter[str]] = defaultdict(Counter)
    for text in non_empty:
        by_norm[normalize_counted(text)][text] += 1
    best_norm = max(by_norm, key=lambda n: (sum(by_norm[n].values()), len(n)))
    raw_counter = by_norm[best_norm]
    return max(raw_counter, key=lambda t: (raw_counter[t], len(normalize_counted(t))))


def merge_frames(
    frames: list[FrameText],
    sim_threshold: float,
    short_threshold: float,
    max_blank_gap: float,
) -> list[Segment]:
    segments: list[Segment] = []
    active_texts: list[str] = []
    active_start = 0.0
    active_end = 0.0
    active_norm = ""

    def flush() -> None:
        nonlocal active_texts, active_start, active_end, active_norm
        text = choose_segment_text(active_texts)
        if text:
            segments.append(Segment(start=active_start, end=active_end, text=text, frames=len(active_texts)))
        active_texts = []
        active_norm = ""

    for frame in frames:
        raw = clean_raw_text(frame.text)
        norm = normalize_counted(raw)
        if not norm:
            if active_texts and frame.t - active_end <= max_blank_gap:
                continue
            flush()
            continue
        if not active_texts:
            active_start = frame.t
            active_end = frame.t
            active_texts = [raw]
            active_norm = norm
            continue
        if frame.t - active_end > max_blank_gap:
            flush()
            active_start = frame.t
            active_end = frame.t
            active_texts = [raw]
            active_norm = norm
            continue
        if should_merge(active_norm, norm, sim_threshold=sim_threshold, short_threshold=short_threshold):
            active_texts.append(raw)
            active_end = frame.t
            chosen = choose_segment_text(active_texts)
            active_norm = normalize_counted(chosen) or active_norm
        else:
            flush()
            active_start = frame.t
            active_end = frame.t
            active_texts = [raw]
            active_norm = norm
    flush()
    return segments


def edit_counts(gt: str, pred: str) -> EditCounts:
    n = len(gt)
    m = len(pred)
    prev: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        gt_ch = gt[i - 1]
        for j in range(1, m + 1):
            pred_ch = pred[j - 1]
            if gt_ch == pred_ch:
                diag = prev[j - 1]
            else:
                base = prev[j - 1]
                diag = (base[0] + 1, base[1] + 1, base[2], base[3])
            base = prev[j]
            delete = (base[0] + 1, base[1], base[2] + 1, base[3])
            base = cur[j - 1]
            insert = (base[0] + 1, base[1], base[2], base[3] + 1)
            cur.append(min(diag, delete, insert, key=lambda x: (x[0], x[1] + x[2] + x[3], x[3])))
        prev = cur
    dist, substitutions, deletions, insertions = prev[m]
    if dist != substitutions + deletions + insertions:
        raise RuntimeError("invalid edit accounting")
    return EditCounts(substitutions=substitutions, deletions=deletions, insertions=insertions, gt_chars=n)


def metrics_from_counts(counts: EditCounts) -> dict[str, float | int]:
    n = counts.gt_chars
    errors = counts.errors
    correct = counts.correct
    pred_chars = counts.pred_chars
    cer = errors / n if n else math.nan
    precision = correct / pred_chars if pred_chars else (1.0 if n == 0 else 0.0)
    recall = correct / n if n else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "cer": cer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": errors,
        "correct": correct,
        "gt_chars": n,
        "pred_chars": pred_chars,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "substitutions": sum(int(row["substitutions"]) for row in rows),
        "deletions": sum(int(row["deletions"]) for row in rows),
        "insertions": sum(int(row["insertions"]) for row in rows),
        "gt_chars": sum(int(row["gt_chars"]) for row in rows),
    }
    counts = EditCounts(
        substitutions=totals["substitutions"],
        deletions=totals["deletions"],
        insertions=totals["insertions"],
        gt_chars=totals["gt_chars"],
    )
    micro = metrics_from_counts(counts)
    macro = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("cer", "precision", "recall", "f1")
    }
    return {
        "videos": len(rows),
        "micro": micro,
        "macro": macro,
    }


def load_source(source: dict[str, Any], allowed_video_ids: set[str], frame_interval: float) -> dict[str, list[FrameText]]:
    path = Path(source["path"])
    kind = source["kind"]
    if kind == "jsonl_frames":
        return load_jsonl_frames(path, allowed_video_ids)
    if kind == "csv_pred_text":
        return load_csv_pred_text(path, allowed_video_ids)
    if kind == "gomatching_xml_dir":
        return load_gomatching_xml_dir(path, allowed_video_ids, frame_interval)
    raise ValueError(f"unsupported source kind: {kind}")


def evaluate_source(
    source: dict[str, Any],
    gt: dict[str, list[str]],
    out_dir: Path,
    sim_threshold: float,
    short_threshold: float,
    frame_interval: float,
    max_blank_gap: float,
) -> dict[str, Any]:
    method = source["method"]
    path = Path(source["path"])
    if not path.exists():
        status = {
            "method": method,
            "status": "missing",
            "path": str(path),
            "optional": bool(source.get("optional")),
        }
        return status

    by_video = load_source(source, set(gt), frame_interval=frame_interval)
    pred_jsonl = out_dir / f"{method}_merged_predictions.jsonl"
    per_video_csv = out_dir / f"{method}_per_video.csv"
    metrics_json = out_dir / f"{method}_metrics.json"
    rows: list[dict[str, Any]] = []

    with pred_jsonl.open("w", encoding="utf-8") as pred_f:
        for video_id in sorted(gt):
            frames = by_video.get(video_id, [])
            segments = merge_frames(
                frames,
                sim_threshold=sim_threshold,
                short_threshold=short_threshold,
                max_blank_gap=max_blank_gap,
            )
            gt_lines = gt[video_id]
            gt_text = "".join(gt_lines)
            pred_lines = [seg.text for seg in segments]
            pred_text = "".join(pred_lines)
            gt_norm = normalize_counted(gt_text)
            pred_norm = normalize_counted(pred_text)
            counts = edit_counts(gt_norm, pred_norm)
            row = {
                "method": method,
                "video_id": video_id,
                "frames": len(frames),
                "segments": len(segments),
                "gt_lines": len(gt_lines),
                "pred_lines": len(pred_lines),
                **metrics_from_counts(counts),
            }
            rows.append(row)
            pred_f.write(
                json.dumps(
                    {
                        "method": method,
                        "video_id": video_id,
                        "gt_lines": gt_lines,
                        "pred_lines": pred_lines,
                        "pred_text": pred_text,
                        "segments": [seg.__dict__ for seg in segments],
                        "frame_count": len(frames),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    fieldnames = [
        "method",
        "video_id",
        "frames",
        "segments",
        "gt_lines",
        "pred_lines",
        "cer",
        "precision",
        "recall",
        "f1",
        "substitutions",
        "deletions",
        "insertions",
        "errors",
        "correct",
        "gt_chars",
        "pred_chars",
    ]
    with per_video_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": method,
        "status": "ok",
        "source": str(path),
        "merge": {
            "type": "adjacent_conservative_fuzzy",
            "sim_threshold": sim_threshold,
            "short_threshold": short_threshold,
            "frame_interval": frame_interval,
            "max_blank_gap": max_blank_gap,
        },
        **aggregate(rows),
        "files": {
            "merged_predictions": str(pred_jsonl),
            "per_video": str(per_video_csv),
            "metrics": str(metrics_json),
        },
    }
    metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_summary(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    summary_json = out_dir / "summary_metrics.json"
    summary_csv = out_dir / "summary_metrics.csv"
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "method",
        "status",
        "videos",
        "micro_cer",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_cer",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "source",
        "path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            micro = summary.get("micro") or {}
            macro = summary.get("macro") or {}
            writer.writerow(
                {
                    "method": summary.get("method"),
                    "status": summary.get("status"),
                    "videos": summary.get("videos", ""),
                    "micro_cer": micro.get("cer", ""),
                    "micro_precision": micro.get("precision", ""),
                    "micro_recall": micro.get("recall", ""),
                    "micro_f1": micro.get("f1", ""),
                    "macro_cer": macro.get("cer", ""),
                    "macro_precision": macro.get("precision", ""),
                    "macro_recall": macro.get("recall", ""),
                    "macro_f1": macro.get("f1", ""),
                    "source": summary.get("source", ""),
                    "path": summary.get("path", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sim-threshold", type=float, default=0.88)
    parser.add_argument("--short-threshold", type=float, default=0.94)
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--max-blank-gap", type=float, default=1.0)
    parser.add_argument(
        "--source-json",
        type=Path,
        default=None,
        help="Optional JSON file with a list of {method, kind, path, optional}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = DEFAULT_SOURCES
    if args.source_json:
        sources = json.loads(args.source_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = parse_gt(args.formal_root)
    summaries = []
    for source in sources:
        summaries.append(
            evaluate_source(
                source=source,
                gt=gt,
                out_dir=args.out_dir,
                sim_threshold=args.sim_threshold,
                short_threshold=args.short_threshold,
                frame_interval=args.frame_interval,
                max_blank_gap=args.max_blank_gap,
            )
        )
    write_summary(summaries, args.out_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
