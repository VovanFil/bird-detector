"""Compare fine-tuned weights with the COCO-pretrained baseline on the validation set
and report the confidence threshold that maximises F1."""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

COCO_BIRD = 14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/birds/data.yaml")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--baseline", default="yolo26n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    return p.parse_args()


def make_coco_view(data_yaml: Path, out: Path) -> Path:
    """Copy of the val split whose class id is COCO's `bird` (14), so the baseline can be scored."""
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    shutil.rmtree(out, ignore_errors=True)
    (out / "labels").mkdir(parents=True)
    (out / "images").mkdir()
    # Hard links, not a directory symlink: Ultralytics resolves symlinks and would read the original labels.
    for img in (root / cfg["val"]).iterdir():
        os.link(img, out / "images" / img.name)
    for f in (root / "labels" / "val").glob("*.txt"):
        rows = [f"{COCO_BIRD} {l.split(maxsplit=1)[1]}" for l in f.read_text().splitlines() if l.strip()]
        (out / "labels" / f.name).write_text("\n".join(rows) + "\n")
    names = {i: "other" for i in range(80)}
    names[COCO_BIRD] = "bird"
    coco_yaml = out / "data.yaml"
    coco_yaml.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images", "val": "images", "names": names}))
    return coco_yaml


def make_subset(data_yaml: Path, out: Path, keep) -> tuple:
    """Val images whose box count satisfies `keep`, as a standalone dataset. Returns (yaml, n_images)."""
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    n = 0
    for lbl in (root / "labels" / "val").glob("*.txt"):
        if keep(sum(1 for l in lbl.read_text().splitlines() if l.strip())):
            os.link(next((root / cfg["val"]).glob(lbl.stem + ".*")), out / "images" / (lbl.stem + ".jpg"))
            shutil.copy(lbl, out / "labels" / lbl.name)
            n += 1
    sub_yaml = out / "data.yaml"
    sub_yaml.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images", "val": "images",
                                        "names": {0: "bird"}}))
    return sub_yaml, n


def row(name, m):
    return f"{name:<28}{m.box.mp:>8.3f}{m.box.mr:>8.3f}{m.box.map50:>8.3f}{m.box.map:>10.3f}"


def main():
    args = parse_args()
    data = Path(args.data)

    tuned = YOLO(args.weights).val(data=str(data), imgsz=args.imgsz, device=args.device, plots=False)
    coco_yaml = make_coco_view(data, data.parent / "_coco_val")
    base = YOLO(args.baseline).val(data=str(coco_yaml), imgsz=args.imgsz, device=args.device,
                                   classes=[COCO_BIRD], plots=False)

    print(f"\n{'model':<28}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    print(row(f"{args.baseline} (COCO, bird)", base))
    print(row("fine-tuned", tuned))

    print(f"\n{'fine-tuned, val subset':<28}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for label, keep in (("typical (<=10 boxes)", lambda n: n <= 10), ("crowded (>10 boxes)", lambda n: n > 10)):
        sub_yaml, n_img = make_subset(data, data.parent / "_subset", keep)
        m = YOLO(args.weights).val(data=str(sub_yaml), imgsz=args.imgsz, device=args.device, plots=False)
        print(row(f"{label} [{n_img} img]", m))
    shutil.rmtree(data.parent / "_subset", ignore_errors=True)

    f1 = np.asarray(tuned.box.f1_curve)[0]
    conf = np.asarray(tuned.box.px)
    best = int(f1.argmax())
    print(f"\nF1-optimal conf for fine-tuned model: {conf[best]:.2f} (F1={f1[best]:.3f})")
    shutil.rmtree(data.parent / "_coco_val", ignore_errors=True)


if __name__ == "__main__":
    main()
