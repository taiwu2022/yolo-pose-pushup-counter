from __future__ import annotations

# COCO 17 keypoints order used by Ultralytics YOLO Pose (COCO keypoints)
# 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
# 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
# 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
# 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

KPT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# A simple skeleton for visualization (COCO-like)
SKELETON_EDGES = [
    (KPT["left_shoulder"], KPT["right_shoulder"]),
    (KPT["left_shoulder"], KPT["left_elbow"]),
    (KPT["left_elbow"], KPT["left_wrist"]),
    (KPT["right_shoulder"], KPT["right_elbow"]),
    (KPT["right_elbow"], KPT["right_wrist"]),
    (KPT["left_shoulder"], KPT["left_hip"]),
    (KPT["right_shoulder"], KPT["right_hip"]),
    (KPT["left_hip"], KPT["right_hip"]),
    (KPT["left_hip"], KPT["left_knee"]),
    (KPT["left_knee"], KPT["left_ankle"]),
    (KPT["right_hip"], KPT["right_knee"]),
    (KPT["right_knee"], KPT["right_ankle"]),
    (KPT["nose"], KPT["left_eye"]),
    (KPT["nose"], KPT["right_eye"]),
    (KPT["left_eye"], KPT["left_ear"]),
    (KPT["right_eye"], KPT["right_ear"]),
    (KPT["nose"], KPT["left_shoulder"]),
    (KPT["nose"], KPT["right_shoulder"]),
]
