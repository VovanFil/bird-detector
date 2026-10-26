"""Download a "Bird" subset of Open Images V7 and export it in YOLO format (single class `bird`).

Cleaning: group boxes (`IsGroupOf`) and drawings/illustrations (`IsDepiction`) are dropped, as are
images with more than --max-boxes birds (dense colonies and patterns with unreliable boxes).
The result is subsampled to --num-images, and the train/val split is stratified by number of boxes
per image so both splits get a similar mix of simple and crowded scenes.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

CLASS_NAME = "bird"
BUCKETS = ((1, 1), (2, 3), (4, 10), (11, 10**9))  # boxes per image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("datasets/birds"))
    p.add_argument("--max-samples", type=int, default=5000, help="images to download before cleaning")
    p.add_argument("--num-images", type=int, default=2000,
                   help="final dataset size after cleaning (sampled proportionally per box-count bucket)")
    p.add_argument("--max-boxes", type=int, default=50, help="drop images with more birds than this")
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-groups", action="store_true", help="keep 'IsGroupOf' boxes")
    p.add_argument("--keep-depictions", action="store_true", help="keep drawings/illustrations")
    return p.parse_args()


def bucket(n_boxes):
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= n_boxes <= hi)


def main():
    args = parse_args()

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Bird"],
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=True,
        dataset_name="open-images-birds",
        drop_existing_dataset=True,
    )

    F = fo.ViewField
    expr = F("label") == "Bird"
    if not args.keep_groups:
        expr = expr & (F("IsGroupOf") == False)  # noqa: E712
    if not args.keep_depictions:
        expr = expr & (F("IsDepiction") == False)  # noqa: E712
    n_boxes = F("ground_truth.detections").length()
    view = (
        dataset.filter_labels("ground_truth", expr)
        .match(n_boxes > 0)
        .match(n_boxes <= args.max_boxes)
    )

    ids, counts = view.values("id"), view.values(F("ground_truth.detections").length())
    groups = defaultdict(list)
    for sample_id, n in zip(ids, counts):
        groups[bucket(n)].append(sample_id)

    rng = random.Random(args.seed)
    keep_frac = min(1.0, args.num_images / len(ids))
    splits = {"train": [], "val": []}
    for group in groups.values():
        rng.shuffle(group)
        group = group[: round(len(group) * keep_frac)]
        n_val = round(len(group) * args.val_ratio)
        splits["val"] += group[:n_val]
        splits["train"] += group[n_val:]

    if args.out.exists():
        shutil.rmtree(args.out)
    for split, split_ids in splits.items():
        sub = view.select(split_ids)
        sub.export(
            export_dir=str(args.out),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=["Bird"],
            split=split,
        )
        print(f"{split}: {len(split_ids)} images, {sub.count('ground_truth.detections')} boxes")

    # The exporter names the class "Bird"; overwrite so the single class is `bird`.
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )


if __name__ == "__main__":
    main()
