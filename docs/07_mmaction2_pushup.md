# 07 MMAction2: Push-up Only (No Manual Labeling)

This pipeline builds a **pseudo-labeled** skeleton dataset from your existing YOLO pose output, then trains an MMAction2 STGCN classifier.

## 1) Build pseudo dataset from raw videos

```bash
source .venv/bin/activate
python scripts/build_mmaction2_pushup_dataset.py \
  --video-dir data/raw_videos \
  --out-dir datasets/mmaction2_pushup \
  --weights yolo11n-pose.pt \
  --clip-len 48 \
  --stride 12 \
  --pos-angle-range 28 \
  --neg-angle-range 10 \
  --neg-ratio 1.0
```

Outputs:
- `datasets/mmaction2_pushup/train.pkl`
- `datasets/mmaction2_pushup/val.pkl`
- `datasets/mmaction2_pushup/label_map.txt`
- `datasets/mmaction2_pushup/summary.json`

## 2) Train in MMAction2

1. Clone and install MMAction2 in another directory.
2. Copy `configs/mmaction2/pushup_stgcn_pseudo.py` into MMAction2 `configs/`.
3. Edit absolute paths in config:
   - `ann_file_train`
   - `ann_file_val`

Run training:

```bash
cd /path/to/mmaction2
python tools/train.py /abs/path/to/pushup_stgcn_pseudo.py
```

## Notes
- This is **pseudo-label** training (no manual labeling). Expect noise.
- Improve quality by:
  - better source videos
  - tightening `pos-angle-range`
  - adding a small manually-checked validation set later
- Class IDs are:
  - `0: other`
  - `1: push_up`
