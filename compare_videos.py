"""Compare the COCO-pretrained baseline with the fine-tuned model on a video.

Videos have no ground truth, so this reports detection statistics per frame and saves
side-by-side frames (baseline | fine-tuned) for visual inspection.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COCO_BIRD = 14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out-prefix", required=True, help="side-by-side frames are saved as <prefix>_<n>.jpg")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--baseline", default="yolo26n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--frames", type=int, default=3, help="number of side-by-side frames to save")
    return p.parse_args()


def detect(model, frames, conf, classes=None):
    return model.predict(frames, conf=conf, classes=classes, verbose=False)


def main():
    args = parse_args()
    tuned, base = YOLO(args.weights), YOLO(args.baseline)

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_at = set(np.linspace(total * 0.15, total * 0.85, args.frames).astype(int).tolist())

    counts = {"baseline": [], "fine-tuned": []}
    confs = {"baseline": [], "fine-tuned": []}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rb = detect(base, frame, args.conf, [COCO_BIRD])[0]
        rt = detect(tuned, frame, args.conf)[0]
        for name, r in (("baseline", rb), ("fine-tuned", rt)):
            counts[name].append(len(r.boxes))
            confs[name] += r.boxes.conf.tolist()
        if idx in save_at:
            pair = np.hstack([rb.plot(), rt.plot()])
            cv2.imwrite(f"{args.out_prefix}_{idx}.jpg", cv2.resize(pair, None, fx=0.6, fy=0.6))
        idx += 1
    cap.release()

    print(f"{Path(args.input).name}: {idx} frames, conf >= {args.conf}")
    for name in counts:
        c = np.array(counts[name])
        print(f"  {name:<11} birds/frame mean {c.mean():5.2f}  max {c.max():3d}  "
              f"frames with 0 detections {np.mean(c == 0) * 100:4.1f}%  mean conf {np.mean(confs[name]):.3f}")


if __name__ == "__main__":
    main()
