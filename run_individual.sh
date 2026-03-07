#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SOURCE="${1:-data/raw_videos/1.MOV}"
OUTPUT="${2:-outputs/1_ready_latch.mp4}"

if [[ ! -x .venv/bin/python ]]; then
  echo "[ERROR] .venv/bin/python not found."
  echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

.venv/bin/python -m pose_counter.infer_video \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --weights "yolo11n-pose.pt" \
  --min-hand-contacts 1 \
  --contact-hold-frames 1 \
  --ready-hold-s 2.0 \
  --hand-drop-fraction-start 0.90 \
  --ground-contact-tol 24 \
  --ground-model-alpha 0.18 \
  --ground-model-min-points 14 \
  --down-th 118 \
  --up-th 157 \
  --min-rep-duration 1.0 \
  --head-drop-fraction-or 0.45 \
  --head-recover-ratio-or 0.85
