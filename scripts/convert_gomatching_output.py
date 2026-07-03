#!/usr/bin/env python3
"""Convert GoMatching/GoMatching++ predictions into OCR benchmark JSONL.

The official eval script writes:

    output/preds/res_<video_id>.xml
    output/preds/res_<video_id>.txt

The TXT contains one majority-vote transcription per track. This converter
turns those track texts into the same JSONL shape used by the OCR metric code.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_METADATA = Path("/home/derbach/code/douyin_ocr_formal_testset/metadata.csv")
DEFAULT_PREDS_DIR = Path("/home/derbach/code/gomatching_work/output/gomatching_pp_bov/preds")
DEFAULT_OUTPUT = Path("/home/derbach/code/ocr_eval/results/gomatching_pp_predictions.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", default="gomatching_pp_bov")
    parser.add_argument("--fuzzy", choices=("none", "conservative"), default="conservative")
    parser.add_argument(
        "--source",
        choices=("auto", "txt", "xml"),
        default="auto",
        help="Use official track-majority txt, frame-level xml, or auto preference. Use xml for video-level OCR text coverage.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if (row.get("ocr_gt") or "").strip()]


def get_video_id(row: Dict[str, str]) -> str:
    for key in ("video_id", "id", "aweme_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"metadata row has no video id: {row}")


def normalize_for_cluster(text: str) -> str:
    chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            chars.append(ch)
        elif "0" <= ch <= "9" or "a" <= ch.lower() <= "z":
            chars.append(ch.lower())
    return "".join(chars)


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def conservative_threshold(norm_text: str) -> float:
    n = len(norm_text)
    if n <= 4:
        return 0.95
    if n <= 10:
        return 0.90
    return 0.86


def exact_unique(lines: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def conservative_fuzzy_cluster(lines: Iterable[str]) -> List[str]:
    clusters: List[Dict[str, object]] = []

    for raw in exact_unique(lines):
        norm = normalize_for_cluster(raw)
        if not norm:
            continue

        best_idx: Optional[int] = None
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            rep_norm = str(cluster["norm"])
            score = similarity(norm, rep_norm)
            threshold = max(conservative_threshold(norm), conservative_threshold(rep_norm))
            length_ratio = min(len(norm), len(rep_norm)) / max(len(norm), len(rep_norm))
            if score >= threshold and length_ratio >= 0.70 and score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            clusters.append({"norm": norm, "lines": [raw]})
        else:
            clusters[best_idx]["lines"].append(raw)

    representatives: List[str] = []
    for cluster in clusters:
        raw_lines = list(cluster["lines"])
        representatives.append(max(raw_lines, key=lambda x: (len(normalize_for_cluster(x)), len(x))))
    return representatives


def parse_txt(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                lines.append(row[1].strip())
            elif row:
                lines.append(row[0].strip())
    return lines


def parse_xml(path: Path) -> List[str]:
    root = ET.parse(path).getroot()
    lines: List[str] = []
    for frame in root.findall(".//frame"):
        for obj in frame.findall("object"):
            text = obj.attrib.get("Transcription", "").strip()
            if text:
                lines.append(text)
    return lines


def parse_xml_track_majority(path: Path) -> List[str]:
    root = ET.parse(path).getroot()
    track_texts: Dict[str, List[str]] = defaultdict(list)
    order: List[str] = []

    for obj in root.findall(".//object"):
        track_id = obj.attrib.get("ID", "").strip()
        text = obj.attrib.get("Transcription", "").strip()
        if not track_id or not text:
            continue
        if track_id not in track_texts:
            order.append(track_id)
        track_texts[track_id].append(text)

    def sort_key(track_id: str) -> tuple[int, str]:
        return (int(track_id), track_id) if track_id.isdigit() else (10**9, track_id)

    lines: List[str] = []
    for track_id in sorted(order, key=sort_key):
        counts = Counter(track_texts[track_id])
        lines.append(counts.most_common(1)[0][0])
    return lines


def find_prediction_file(preds_dir: Path, video_id: str, suffix: str) -> Optional[Path]:
    direct = preds_dir / f"res_{video_id}{suffix}"
    if direct.is_file():
        return direct
    matches = sorted(preds_dir.glob(f"*{video_id}*{suffix}"))
    return matches[0] if matches else None


def load_prediction_lines(preds_dir: Path, video_id: str, source: str) -> tuple[List[str], Optional[Path]]:
    if source in ("xml", "auto"):
        xml_path = find_prediction_file(preds_dir, video_id, ".xml")
        if xml_path:
            return parse_xml(xml_path), xml_path

    if source in ("txt", "auto"):
        txt_path = find_prediction_file(preds_dir, video_id, ".txt")
        if txt_path:
            return parse_txt(txt_path), txt_path

    return [], None


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(args.metadata)
    missing: List[str] = []
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            video_id = get_video_id(row)
            lines, source_file = load_prediction_lines(args.preds_dir, video_id, args.source)
            if source_file is None:
                missing.append(video_id)

            pred_lines = (
                conservative_fuzzy_cluster(lines)
                if args.fuzzy == "conservative"
                else exact_unique(lines)
            )

            record = {
                "method": args.method,
                "video_id": video_id,
                "ocr_gt": row.get("ocr_gt", ""),
                "pred_lines": pred_lines,
                "pred_lines_unique": pred_lines,
                "pred_text": "".join(pred_lines),
                "source_file": str(source_file) if source_file else None,
                "source": args.source,
                "fuzzy": args.fuzzy,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": str(args.metadata),
                "preds_dir": str(args.preds_dir),
                "output": str(args.output),
                "method": args.method,
                "fuzzy": args.fuzzy,
                "source": args.source,
                "video_count": len(rows),
                "missing_count": len(missing),
                "missing": missing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"wrote predictions: {args.output}")
    print(f"wrote summary: {summary_path}")
    if missing:
        print(f"missing prediction files: {len(missing)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
