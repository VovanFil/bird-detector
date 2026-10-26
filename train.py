"""Fine-tune YOLO26n on the single-class bird dataset and copy the best weights to weights/best.pt."""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/birds/data.yaml")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20,
                   help="early stopping: epochs without val fitness improvement before stopping")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    p.add_argument("--name", default="yolo26n_bird")
    return p.parse_args()


def default_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = args.device if args.device is not None else default_device()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    best = Path(model.trainer.best)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/best.pt")

    metrics = YOLO("weights/best.pt").val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"Validation: P={metrics.box.mp:.3f} R={metrics.box.mr:.3f} "
          f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
