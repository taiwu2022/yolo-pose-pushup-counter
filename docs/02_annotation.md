# 02 标注（CVAT -> YOLO Pose）

YOLO Pose 训练需要每张图对应的 label（TXT），每行代表一个人：

```
class x_center y_center width height  x1 y1 v1  x2 y2 v2 ... x17 y17 v17
```

- 坐标均为 **归一化到 0~1**
- v 通常表示可见性/置信度（第三维），COCO Pose 常用 0/1/2

Ultralytics 的 COCO8-pose 数据集 YAML 就是 `kpt_shape: [17, 3]`（17 点，每点 3 维）。你后面写 data.yaml 也要匹配。  

---

## 推荐标注方式（省时间）
1) 先抽一小批帧（300~500 张）认真标好  
2) 训一个初版 pose  
3) 用初版给更多帧自动预测，然后你只“改错”（半自动标注），效率会高很多

---

## CVAT 操作建议
- 新建 Task -> 上传 images
- 标注 skeleton/keypoints（按 COCO 17 点）
- 导出格式选择：**Ultralytics YOLO Pose**（CVAT 支持 Ultralytics YOLO Pose 格式族）
- 导出后把内容整理到：
  - `data/exports/images_all/`
  - `data/exports/labels_all/`

然后跑一次校验：
```bash
python scripts/validate_labels.py --images-dir data/exports/images_all --labels-dir data/exports/labels_all --k 17
```
