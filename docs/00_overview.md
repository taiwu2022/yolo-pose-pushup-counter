# 00 项目全览

你要实现的最小闭环（MVP）是：

1. YOLO Pose（预训练 or 自训）对视频逐帧输出人体关键点（COCO 17）
2. 对关键点做平滑与置信度门控
3. 俯卧撑计数状态机：up/down 两阈值（hysteresis）+ 最小时间间隔（去抖）
4. 输出叠加视频 + rep 时间戳 CSV

建议先只做俯卧撑。引体向上之后用同一个 pose 模型，另写一个 counter 类即可（见 `06_add_pullup_later.md`）。
