# 牧濑红莉栖试标 v0

正式数据文件：`kurisu_trial_v0_compact.jsonl`

一条记录是一段完整序列。每个事件都包含 `historical_emotion`，表示该事件发生前角色已有的情绪状态。

事件字段：

- `text`：事件摘要。
- `historical_emotion`：事件发生前的情绪状态，顺序为 `joy, trust, fear, surprise, sadness, disgust, anger, anticipation`。
- `current_stimulus`：这个事件带来的即时情绪刺激，顺序同上。
- `emotion_target`：处理这个事件后的情绪状态，顺序同上。
- `behavior_target`：事件后的行为倾向，顺序为 `engagement, avoidance, expression, exploration, initiative, activation`。
- `delta_seconds`：距离上一个事件的时间间隔。

本试标中：

- 第一个事件的 `historical_emotion` 根据选段前文估计：红莉栖此前因包装工作不耐烦，随后因帮助真帆取书和研究话题转为较高的信任与期待。
- 后续事件的 `historical_emotion` 使用上一个事件的 `emotion_target`，因为它代表连续状态的自然继承。
- 如果原文明确描述了时间经过中的恢复、累积或状态改变，应以文本证据修正该值。

没有空值。当前情绪向量使用 `0.0` 表示没有可观察信号；行为向量使用 `0.5` 表示中性倾向。

训练时应注意：这些历史情绪可用于标注、审计和对照实验；第一版不要把每一步的真实 `historical_emotion` 直接作为模型输入，否则会把上一时刻的目标答案泄漏给模型。CfC 的隐状态应负责记忆序列历史。
