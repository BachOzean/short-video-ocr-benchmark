#!/usr/bin/env python3
"""Prepare frame-folder input for GoMatching/GoMatching++ evaluation.

GoMatching expects an input directory shaped as:

    input_root/
      video_id_1/
        000001.jpg
        000002.jpg
      video_id_2/
        000001.jpg

This script samples the formal OCR test videos at a fixed time interval and
writes that layout without changing the original videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2


DEFAULT_METADATA = Path("data/formal_testset/metadata.csv")
DEFAULT_VIDEOS_DIR = Path("data/formal_testset/videos")
DEFAULT_OUT_DIR = Path("frames_0.5s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="Do not filter rows with empty ocr_gt. Formal set normally keeps only non-empty GT.",
    )
    return parser.parse_args()


def read_metadata(path: Path, include_empty_gt: bool) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if include_empty_gt:
        return rows
    return [row for row in rows if (row.get("ocr_gt") or "").strip()]


def get_video_id(row: Dict[str, str]) -> str:
    for key in ("video_id", "id", "aweme_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"metadata row has no video id: {row}")


def resolve_video_path(row: Dict[str, str], videos_dir: Path) -> Optional[Path]:
    for key in ("video_path", "local_video_path", "path", "file_path", "filename", "file_name"):
        value = (row.get(key) or "").strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = videos_dir / candidate
        if candidate.is_file():
            return candidate

    video_id = get_video_id(row)
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"):
        candidate = videos_dir / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate

    matches = sorted(p for p in videos_dir.glob(f"{video_id}.*") if p.is_file())
    return matches[0] if matches else None


def iter_sampled_frames(cap: cv2.VideoCapture, interval_sec: float) -> Iterable[tuple[int, object]]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = 0
    next_t = 0.0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if fps > 0:
            t = frame_idx / fps
        else:
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

        if t + 1e-9 >= next_t:
            saved_idx += 1
            yield saved_idx, frame
            next_t += interval_sec

        frame_idx += 1


def sample_video(
    video_path: Path,
    out_video_dir: Path,
    interval_sec: float,
    jpg_quality: int,
    overwrite: bool,
) -> Dict[str, object]:
    if out_video_dir.exists() and overwrite:
        shutil.rmtree(out_video_dir)
    out_video_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_video_dir.glob("*.jpg"))
    if existing and not overwrite:
        return {
            "source": str(video_path),
            "out_dir": str(out_video_dir),
            "frames_written": len(existing),
            "skipped_existing": True,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "source": str(video_path),
            "out_dir": str(out_video_dir),
            "frames_written": 0,
            "error": "failed_to_open_video",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    quality_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    frames_written = 0
    for saved_idx, frame in iter_sampled_frames(cap, interval_sec):
        frame_path = out_video_dir / f"{saved_idx:06d}.jpg"
        if cv2.imwrite(str(frame_path), frame, quality_param):
            frames_written += 1

    cap.release()
    duration_sec = total_frames / fps if fps > 0 else None
    return {
        "source": str(video_path),
        "out_dir": str(out_video_dir),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "frames_written": frames_written,
        "skipped_existing": False,
    }


def main() -> int:
    args = parse_args()
    rows = read_metadata(args.metadata, args.include_empty_gt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "metadata": str(args.metadata),
        "videos_dir": str(args.videos_dir),
        "out_dir": str(args.out_dir),
        "interval_sec": args.interval_sec,
        "video_count": len(rows),
        "videos": [],
        "missing": [],
    }

    for row in rows:
        video_id = get_video_id(row)
        video_path = resolve_video_path(row, args.videos_dir)
        if video_path is None:
            manifest["missing"].append(video_id)
            continue
        stats = sample_video(
            video_path=video_path,
            out_video_dir=args.out_dir / video_id,
            interval_sec=args.interval_sec,
            jpg_quality=args.jpg_quality,
            overwrite=args.overwrite,
        )
        stats["video_id"] = video_id
        manifest["videos"].append(stats)
        print(f"{video_id}\t{stats.get('frames_written', 0)} frames")

    manifest_path = args.out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"wrote manifest: {manifest_path}")
    if manifest["missing"]:
        print(f"missing videos: {len(manifest['missing'])}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
