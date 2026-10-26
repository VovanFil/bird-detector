# bird-detector

Fine-tuning of a pretrained **YOLO26n** ([Ultralytics](https://docs.ultralytics.com/)) for a single class, `bird`, plus a script that runs the detector on a video and saves an annotated copy.

## Project layout

```
bird-detector/
├── README.md
├── requirements.txt
├── prepare_dataset.py   # downloads Open Images "Bird" subset, cleans it, converts to YOLO, splits train/val
├── train.py             # fine-tunes yolo26n.pt, copies best weights to weights/best.pt
├── predict_video.py     # runs the model on every frame of a video, saves annotated video
├── evaluate.py          # fine-tuned vs COCO baseline on val, F1-optimal confidence
├── compare_videos.py    # baseline vs fine-tuned on a video (stats + side-by-side frames)
├── weights/best.pt      # trained weights (5.4 MB)
└── examples/
    ├── input.mp4, output.mp4      # easy clip: seagulls on a boardwalk
    ├── input2.mp4, output2.mp4    # hard clip: small flying silhouettes at sunset
    ├── compare_*.jpg              # baseline (left) vs fine-tuned (right)
    └── training_curves.png, pr_curve.png, val_predictions.jpg
```

## Installation

Python 3.9+ is required. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`fiftyone` in `requirements.txt` is only used by `prepare_dataset.py`. `train.py` picks CUDA, then Apple MPS, then CPU automatically. `ffmpeg` is optional (see below).

## Training

```bash
python prepare_dataset.py   # creates datasets/birds (2000 images, downloaded from Open Images)
python train.py             # up to 100 epochs, early stopping; writes weights/best.pt
```

Useful options: `python train.py --epochs 150 --batch 32 --device 0` (see `--help`). Logs and plots are written to `runs/yolo26n_bird/`.

## Running on a video

```bash
python predict_video.py --input examples/input.mp4 --output examples/output.mp4
```

Options: `--weights` (default `weights/best.pt`), `--conf` (default `0.25`), `--imgsz` (default `640`), `--batch` (frames per inference call, default `8`), `--half` (FP16, GPU only), `--device`.

Every frame is processed; boxes are drawn with the class name and confidence, and the result keeps the original resolution and FPS. If `ffmpeg` is installed the output is encoded directly as H.264 (small file, plays in browsers and on GitHub); otherwise OpenCV's `mp4v` is used.

Speed on an Apple-silicon GPU (MPS), including drawing and encoding: the 1128-frame 720p example takes 17.7 s (~64 fps); the 765-frame example takes 11.2 s.

## Dataset

- **Source:** [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), class `Bird`, bounding-box annotations, `train` split, downloaded through `fiftyone` (5000 images sampled with a fixed seed).
- **Licence:** annotations CC BY 4.0, images are listed by Open Images under CC BY 2.0.
- **Cleaning** (4502 images remain): boxes flagged `IsGroupOf` (one box around a flock) and `IsDepiction` (drawings, engravings, patterns) are dropped, and so are images with more than 50 birds (dense colonies with unreliable boxes).
- **Subsampling and split:** 2000 images are sampled and split 70/30 into train/val with seed 0, stratified by the number of birds per image (1, 2-3, 4-10, more than 10), so both splits have the same mix of simple and crowded scenes.
- **Format:** YOLO, one class (`0: bird`).

| Split | Images | Boxes |
|-------|-------:|------:|
| train | 1400   | 3263  |
| val   | 600    | 1355  |

Leakage check: no file name or file content (md5) is shared between train and val, and there are no duplicates inside either split. Near-duplicates (for example, frames of the same photo series) are not excluded.

## Training setup

`yolo26n.pt` (COCO-pretrained), image size 640, batch 16, default Ultralytics augmentations and optimizer, seed 0. Up to 100 epochs with early stopping (`patience=20`): the best epoch was 78 and training stopped at epoch 98 (4.0 h on Apple MPS). `best.pt` is the checkpoint of epoch 78.

## Results on the validation set

Evaluated with `python evaluate.py` on the 600 validation images (1355 boxes):

| Model                    | Precision | Recall | mAP50 | mAP50-95 |
|--------------------------|----------:|-------:|------:|---------:|
| `yolo26n.pt` (COCO bird) | 0.724     | 0.647  | 0.719 | 0.418    |
| fine-tuned (`best.pt`)   | 0.737     | 0.672  | 0.713 | 0.408    |

The baseline is the original `yolo26n.pt` scored only on its COCO `bird` class (id 14).

**Fine-tuning did not improve mAP on this validation set.** Precision and recall are 1-2.5 points higher, mAP is 0.5-1 point lower; all within the noise of a 600-image validation set. COCO already has a `bird` class and far more data than this subset, so the baseline is strong, and Open Images boxes are loose and inconsistent, which limits what fine-tuning on 1400 images can add. One caveat: Open Images and COCO both draw on Flickr photos, so some overlap between the baseline's training data and this validation set cannot be ruled out.

Breakdown of the fine-tuned model by scene complexity:

| Validation subset            | Images | P     | R     | mAP50 | mAP50-95 |
|------------------------------|-------:|------:|------:|------:|---------:|
| typical (up to 10 birds)     | 576    | 0.845 | 0.700 | 0.803 | 0.492    |
| crowded (more than 10 birds) | 24     | 0.742 | 0.540 | 0.595 | 0.268    |

The overall numbers are pulled down by a few crowded images that contain a large share of all boxes. The F1-optimal confidence threshold is 0.29 (F1 = 0.704); the examples use the default `--conf 0.25`.

Curves: `examples/training_curves.png`, `examples/pr_curve.png`. Sample predictions on validation images: `examples/val_predictions.jpg`.

`val` is used both to pick the best epoch and to report metrics (there is no separate test set), so the fine-tuned numbers are slightly optimistic.

## Examples

Both clips are from [Mixkit](https://mixkit.co/free-stock-video/) (Mixkit Stock Video Free License), 1280x720. Outputs use the default `--conf 0.25`.

| Clip | Input | Frames |
|------|-------|-------:|
| Easy: seagulls on a boardwalk with the sea in the background | `examples/input.mp4`, 50 fps, 22.5 s | 1128 |
| Hard: seagulls flying over the sea at sunset | `examples/input2.mp4`, 25 fps, 30.6 s | 765 |

### Baseline vs fine-tuned on video

Videos have no ground truth, so `python compare_videos.py` reports detection statistics per frame (not accuracy) and saves side-by-side frames, baseline on the left, fine-tuned on the right (`examples/compare_*.jpg`):

| Clip | Model      | Birds per frame (mean) | Mean confidence |
|------|------------|-----------------------:|----------------:|
| easy | baseline   | 10.2                   | 0.46            |
| easy | fine-tuned | 26.8                   | 0.55            |
| hard | baseline   | 6.2                    | 0.57            |
| hard | fine-tuned | 11.6                   | 0.51            |

By visual inspection, the fine-tuned model finds clearly more birds: most of the flock standing on the ground in the easy clip, and many more small distant birds in the hard clip, where the baseline detects only the largest ones. In dense groups the fine-tuned model also produces overlapping boxes around the same bird, which the baseline does not. This matches the validation result (higher recall, similar mAP), and suggests that the benefit of fine-tuning shows up as recall on small and crowded birds rather than as a higher mAP on Open Images.

## Problems and possible improvements

**What I observed**

- Recall (0.67) is lower than precision (0.74): the model misses more birds than it invents. Small, distant, flying birds are the weakest case at 640 px with the nano model.
- In dense groups there are duplicate overlapping boxes; the second clip also shows two close birds merged into one box.
- mAP50-95 (0.41) is much lower than mAP50 (0.71): boxes are found but not tight. Open Images boxes are annotated loosely and inconsistently, which caps the achievable score.
- Fine-tuning gave no mAP gain over the COCO baseline (see above), so the conclusion is limited to "higher recall on video", not "better model".
- Detection runs per frame independently, so boxes flicker between frames.
- Training on a laptop GPU is slow (4 h for one run), which limited the experiments.
- The early-stopping counter reset several times: the validation fitness plateaued around epoch 60 and then improved again at epoch 78, so a smaller patience would have stopped too early.

**Improvements**

- Larger `imgsz` (960-1280) or tiled inference (SAHI) for small birds; a larger model (`yolo26s/m`).
- Cleaner and more consistent labels, and more data with scenes closer to the target video; hard-negative images to cut false positives.
- Tune NMS (`iou`) and `conf` thresholds to reduce duplicate boxes in groups.
- A separate test set so metrics are not used for model selection.
- Tracking (`model.track`, ByteTrack) and temporal smoothing to stabilise boxes between frames.
