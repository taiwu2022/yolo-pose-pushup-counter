# 06 之后如何加引体向上

原则：
- YOLO Pose 模型通常只需要一个（输出人体关键点即可）
- 引体向上只需要新增一个 `PullUpCounter`（状态机逻辑与俯卧撑不同）

落地步骤：
1) 先复用现有推理管线（`infer_video.py`）
2) 新增 `pose_counter/counters/pullup.py`
3) 在 `infer_video.py` 里通过 `--counter pullup` 选择不同 counter
4) 若引体场景与俯卧撑差异很大导致关键点崩：
   - 先用同一个模型在引体数据上继续微调（finetune）
   - 不建议一开始维护两套 pose 权重
