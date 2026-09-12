"""Build correctly labelled GRU sequences from WLASL metadata.

Numeric MP4 names are video IDs, not labels. This script resolves every ID
through ``nslt_100.json`` and ``wlasl_class_list.txt`` and preserves the
official train/val/test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from Shared_lib.sequence_utils import DEFAULT_SEQUENCE_LENGTH, landmarks_to_vector


def load_class_list(path: Path) -> dict[int, str]:
    classes: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                class_id, gloss = line.split("\t", 1)
                classes[int(class_id)] = gloss.strip()
            except ValueError as exc:
                raise ValueError(f"Invalid class-list line {line_number}: {line!r}") from exc
    return classes


def load_wlasl_records(metadata_path: Path, class_list_path: Path, videos_dir: Path):
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    class_names = load_class_list(class_list_path)
    records, invalid = [], []
    for video_id, item in metadata.items():
        try:
            class_id, frame_start, frame_end = item["action"]
            gloss = class_names[int(class_id)]
            split = item["subset"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unknown split {split!r}")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append((video_id, str(exc)))
            continue
        video_path = videos_dir / f"{video_id}.mp4"
        records.append({
            "video_id": video_id,
            "video_path": video_path,
            "gloss": gloss,
            "class_id": int(class_id),
            "split": split,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "available": video_path.is_file(),
        })
    return records, invalid


def print_audit(records, invalid, metadata_path: Path) -> dict:
    available = [record for record in records if record["available"]]
    split_counts = Counter(record["split"] for record in available)
    class_split_counts = defaultdict(Counter)
    for record in available:
        class_split_counts[record["gloss"]][record["split"]] += 1
    summary = {
        "metadata": str(metadata_path),
        "metadata_records": len(records) + len(invalid),
        "valid_records": len(records),
        "available_videos": len(available),
        "missing_videos": sum(not record["available"] for record in records),
        "available_classes": len(class_split_counts),
        "split_counts": dict(split_counts),
        "classes_without_train": sorted(
            name for name, counts in class_split_counts.items() if not counts["train"]
        ),
        "classes_without_val": sorted(
            name for name, counts in class_split_counts.items() if not counts["val"]
        ),
        "invalid_records": invalid,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def frame_indices(total_frames: int, frame_start: int, frame_end: int, length: int):
    """Return evenly spaced zero-based indices inside the annotated range."""
    if total_frames <= 0:
        return []
    start = max(0, frame_start - 1)
    end = total_frames - 1 if frame_end <= 0 else min(total_frames - 1, frame_end - 1)
    if start > end:
        # Some processed mirrors are already trimmed and re-encoded.
        start, end = 0, total_frames - 1
    return np.linspace(start, end, length).round().astype(int).tolist()


def extract_keypoints(video_path: Path, hands, frame_start: int, frame_end: int,
                      sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    indices = frame_indices(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_start, frame_end, sequence_length
    )
    if not indices:
        cap.release()
        return None

    # Decode once from left to right. Repeated random seeks are very slow for
    # inter-frame-compressed MP4 files, especially near the end of long clips.
    wanted = set(indices)
    extracted = {}
    frame_number = 0
    while frame_number <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number in wanted:
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            extracted[frame_number] = landmarks_to_vector(
                result.multi_hand_landmarks, result.multi_handedness
            )
        frame_number += 1
    cap.release()
    zero = np.zeros(126, dtype=np.float32)
    sequence = [extracted.get(index, zero) for index in indices]
    if not sequence or sum(np.any(frame) for frame in sequence) < 5:
        return None
    return np.asarray(sequence, dtype=np.float32)


def build_dataset(records, output_dir: Path, limit: int | None = None,
                  overwrite: bool = False) -> dict:
    from mediapipe.python.solutions import hands as mp_hands

    selected = [record for record in records if record["available"]]
    if limit is not None:
        selected = selected[:limit]
    stats, failures = Counter(), []
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
    )
    try:
        for record in tqdm(selected, desc="Extracting WLASL keypoints"):
            destination = (
                output_dir / record["split"] / record["gloss"] /
                f'{record["video_id"]}.npy'
            )
            if destination.exists() and not overwrite:
                try:
                    existing = np.load(destination, allow_pickle=False)
                    detected = int(np.count_nonzero(np.any(existing != 0, axis=1)))
                    if existing.shape == (DEFAULT_SEQUENCE_LENGTH, 126) and detected >= 5:
                        stats["skipped_existing"] += 1
                        continue
                    quarantine = destination.with_suffix('.npy.sparse')
                    suffix = 1
                    while quarantine.exists():
                        quarantine = destination.with_suffix(f'.npy.sparse.{suffix}')
                        suffix += 1
                    destination.rename(quarantine)
                    stats["quarantined_sparse"] += 1
                except (OSError, ValueError):
                    quarantine = destination.with_suffix('.npy.invalid')
                    suffix = 1
                    while quarantine.exists():
                        quarantine = destination.with_suffix(f'.npy.invalid.{suffix}')
                        suffix += 1
                    destination.rename(quarantine)
                    stats["quarantined_invalid"] += 1
            sequence = extract_keypoints(
                record["video_path"], hands, record["frame_start"], record["frame_end"]
            )
            if sequence is None:
                stats["failed_no_hands"] += 1
                failures.append(record["video_id"])
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, sequence)
            stats[f'written_{record["split"]}'] += 1
    finally:
        hands.close()
    result = dict(stats)
    result["selected"] = len(selected)
    result["failed_video_ids"] = failures
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(config.WLASL_METADATA_PATH))
    parser.add_argument("--class-list", type=Path, default=Path(config.WLASL_CLASS_LIST_PATH))
    parser.add_argument("--videos", type=Path, default=Path(config.WLASL_VIDEOS_DIR))
    parser.add_argument("--output", type=Path, default=Path(config.SEQUENCES_PROCESSED_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write no NPY files")
    parser.add_argument("--limit", type=int, help="Process only the first N available videos")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for required in (args.metadata, args.class_list, args.videos):
        if not required.exists():
            raise FileNotFoundError(f"Required WLASL input not found: {required}")
    records, invalid = load_wlasl_records(args.metadata, args.class_list, args.videos)
    summary = print_audit(records, invalid, args.metadata)
    if invalid:
        raise ValueError(f"Metadata contains {len(invalid)} invalid records")
    if args.dry_run:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    build = build_dataset(records, args.output, args.limit, args.overwrite)
    manifest_path = args.output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump({"audit": summary, "build": build}, stream, ensure_ascii=False, indent=2)
    print(json.dumps(build, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
