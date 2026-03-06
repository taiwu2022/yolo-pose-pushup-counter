from __future__ import annotations

import argparse
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export Ultralytics YOLO Pose model to ONNX (or other formats).")
    p.add_argument("--weights", type=str, required=True, help="Path to *.pt weights.")
    p.add_argument("--format", type=str, default="onnx", help="Export format: onnx/openvino/tflite/engine/...")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--out-dir", type=str, default="weights/exported")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device if args.device else None,
    )
    # Ultralytics returns path(s)
    print("Export result:", exported)
    print("Tip: exported files are usually written next to your weights or into runs/ by Ultralytics; check output logs.")


if __name__ == "__main__":
    main()
