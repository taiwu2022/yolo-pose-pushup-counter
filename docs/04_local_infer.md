# 04 本地部署推理（VS Code）

## 1) 安装与环境
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 预训练权重直接跑
```bash
python -m pose_counter.infer_video \
  --source data/raw_videos/pushup.mp4 \
  --weights yolo11n-pose.pt \
  --output outputs/pushup_counted.mp4
```

## 3) 用自训练权重跑
```bash
python -m pose_counter.infer_video \
  --source data/raw_videos/pushup.mp4 \
  --weights weights/pushup_best.pt \
  --output outputs/pushup_custom.mp4
```

## 4) 常用调参（非常影响计数）
- `--down-th`：下去的肘角阈值（角度越小越“下”）
- `--up-th`：上来的肘角阈值（角度越大越“上”）
- `--min-interval`：两次计数最小间隔（秒），防止抖动重复计数
- `--kpt-conf`：关键点置信度门槛
