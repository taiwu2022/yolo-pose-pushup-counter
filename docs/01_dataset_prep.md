# 01 数据准备（从视频到训练帧）

## 1) 建议的数据目录（在本仓库内）
```
data/
  raw_videos/        # 你自拍视频（mp4）
  frames/            # 抽帧输出（jpg）
  annotations/       # 标注工程（CVAT 导出前）
  exports/           # 导出的 YOLO Pose labels（images/ + labels/）
```

> `data/` 默认被 `.gitignore` 忽略，避免把大文件提交到 GitHub。

---

## 2) 从视频抽帧

### 抽帧原则（非常重要）
- 不要抽每一帧（冗余巨大）
- 训练 pose 时更需要“姿态多样性”
- 推荐：**每秒 5~10 帧**（比如 30fps 视频抽每 3~6 帧一张）

### 命令
```bash
python scripts/extract_frames.py \
  --video-dir data/raw_videos \
  --out-dir data/frames \
  --every-n 3
```

---

## 3) 切分 train/val/test（按视频分组，避免泄漏）

```bash
python scripts/split_dataset.py \
  --frames-dir data/frames \
  --labels-dir data/exports/labels_all \
  --out-dataset-dir datasets/pushup_pose \
  --train 0.7 --val 0.2 --test 0.1 \
  --seed 42
```

输出结构会变成：
```
datasets/pushup_pose/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```
