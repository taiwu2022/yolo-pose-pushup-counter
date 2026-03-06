# 03 用 Colab 训练/微调 YOLO Pose（推荐流程）

> 说明：Ultralytics 支持直接从 `*.pt` 预训练权重开始训练（推荐）。COCO8-pose 示例 YAML 中也这么做：`model = YOLO("...-pose.pt"); model.train(...)`。

---

## 方案 A（最省事）：把数据集 zip 上传到 Google Drive
把 `datasets/pushup_pose/` 打包成 zip，上传到 Drive，比如：
- `MyDrive/datasets/pushup_pose.zip`

目录内应包含：
```
pushup_pose/
  images/train
  images/val
  labels/train
  labels/val
```

---

## Colab Notebook（复制运行）

### 1) 安装依赖
```python
!pip -q install ultralytics opencv-python
```

### 2) 挂载 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3) 解压数据集
```python
!mkdir -p /content/datasets
!unzip -q "/content/drive/MyDrive/datasets/pushup_pose.zip" -d /content/datasets
!ls -R /content/datasets/pushup_pose | head -n 50
```

### 4) 写 data.yaml（指向你的数据路径）
```python
data_yaml = r'''
path: /content/datasets/pushup_pose
train: images/train
val: images/val
test: images/test

# Keypoints
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person

kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
    - left_ear
    - right_ear
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle
'''
open("/content/pushup_pose.yaml","w").write(data_yaml)
print(open("/content/pushup_pose.yaml").read())
```

### 5) 训练（从预训练 pose 权重开始微调）
你可以用 YOLO11 / YOLOv8 / YOLO26 的 pose 权重之一，例如：
- `yolo11n-pose.pt`（轻量）
- `yolov8n-pose.pt`
- `yolo26n-pose.pt`（Ultralytics 新一代）

```python
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # 也可换成 yolov8n-pose.pt / yolo26n-pose.pt
results = model.train(
    data="/content/pushup_pose.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

### 6) 把最优权重拷回 Drive
```python
!ls runs/pose/train/weights
!cp runs/pose/train/weights/best.pt /content/drive/MyDrive/weights/pushup_best.pt
```

---

## 下一步
把 `pushup_best.pt` 下载到本地仓库的 `weights/` 下，然后跑：

```bash
python -m pose_counter.infer_video --source your.mp4 --weights weights/pushup_best.pt --output outputs/out.mp4
```
