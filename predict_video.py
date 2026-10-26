"""Run the bird detector on every frame of a video and save the annotated result."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="path to the input video")
    p.add_argument("--output", required=True, help="path to the output video (.mp4)")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="frames per inference call")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument("--device", default=None)
    return p.parse_args()


class VideoSink:
    """Writes frames as H.264 through ffmpeg when available (plays in browsers, small files),
    otherwise falls back to OpenCV's mp4v."""

    def __init__(self, path, fps, width, height):
        self.proc = self.writer = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
                 "-movflags", "+faststart", path],
                stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()


def main():
    args = parse_args()
    if not Path(args.input).is_file():
        sys.exit(f"Input video not found: {args.input}")
    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("--output must differ from --input")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(args.output, fps, width, height)

    frame_idx = detections = 0
    start = time.perf_counter()

    def flush(frames):
        nonlocal frame_idx, detections
        results = model.predict(frames, conf=args.conf, imgsz=args.imgsz, half=args.half,
                                device=args.device, verbose=False)
        for result in results:
            detections += len(result.boxes)
            sink.write(result.plot())  # box + class name + confidence
            frame_idx += 1
        print(f"{frame_idx}/{total} frames, {frame_idx / (time.perf_counter() - start):.1f} fps",
              end="\r", flush=True)

    try:
        batch = []
        while True:
            ok, frame = cap.read()
            if ok:
                batch.append(frame)
            if batch and (len(batch) == args.batch or not ok):
                flush(batch)
                batch = []
            if not ok:
                break
    finally:
        cap.release()
        sink.close()
    print(f"\nProcessed {frame_idx} frames, {detections} detections in "
          f"{time.perf_counter() - start:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
