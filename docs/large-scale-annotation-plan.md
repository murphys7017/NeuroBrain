# NeuroBrain 大规模小说标注计划

## 目标

把小说按“完整场景 → 连续事件序列”整理成训练数据。核心训练文件保持简洁；来源行号、证据和人工复核信息放在单独的审计文件中。

## 核心数据格式

一条 JSONL 记录对应一段完整序列：

```json
{
  "sequence_id": "kurisu_ch01_scene_001",
  "target_character": "牧濑红莉栖",
  "events": [
    {
      "text": "事件摘要",
      "historical_emotion": {
        "joy": 0.0,
        "trust": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "disgust": 0.0,
        "anger": 0.0,
        "anticipation": 0.0
      },
      "current_stimulus": {
        "joy": 0.0,
        "trust": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "disgust": 0.0,
        "anger": 0.0,
        "anticipation": 0.0
      },
      "emotion_target": {
        "joy": 0.0,
        "trust": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "sadness": 0.0,
        "disgust": 0.0,
        "anger": 0.0,
        "anticipation": 0.0
      },
      "behavior_target": {
        "engagement": 0.5,
        "avoidance": 0.5,
        "expression": 0.5,
        "exploration": 0.5,
        "initiative": 0.5,
        "activation": 0.5
      },
      "delta_seconds": 1.0
    }
  ]
}
```

固定维度顺序：

- 情绪：`joy, trust, fear, surprise, sadness, disgust, anger, anticipation`
- 行为：`engagement, avoidance, expression, exploration, initiative, activation`

## 标注规则

### 1. 事件切分

- 一个事件是对目标角色产生可识别影响的叙事变化、对话、行动或信息揭示。
- 只保留目标角色相关事件；纯背景描写可以合并到相邻事件。
- 一个序列应保持同一场景、同一目标角色和连续时间关系。
- 不把同一故事的相邻事件拆到训练集和验证集。

### 2. 三类情绪字段

- `historical_emotion`：事件发生前角色已有的情绪状态。
- `current_stimulus`：这一个事件造成的即时刺激。
- `emotion_target`：角色处理该事件后的状态。

三者不能直接复制。事件可能带来高恐惧刺激，但目标状态只小幅上升；也可能刺激结束后目标状态仍然持续，这正是 CfC 要学习的动态。

### 3. 历史情绪的上下文使用

- 每个事件都必须有 `historical_emotion`。
- 序列第一个事件：使用选段前文能支持的状态估计；没有上下文证据时使用全零默认。
- 后续事件：默认继承上一事件的 `emotion_target`。
- 只有在文本明确表现出恢复、累积、时间流逝或新的背景变化时，才修正继承值。
- 这些值是标注估计，不是客观心理测量。

### 4. 没有证据时的中性值

- 情绪刺激、情绪状态、历史情绪：没有可观察信号或上下文证据时填 `0.0`。
- 行为倾向：填 `0.5`。
- 不使用 `null`，不使用 mask。

### 5. 角色实体与记忆副本

- 仅标注目标角色实际出场、直接说话，或其情绪状态能从叙事证据可靠判断的场景。
- 其他角色单方面回忆、谈论目标角色的场景，不把回忆者的情绪转写成目标角色标签。
- 原始红莉栖统一写为 `target_character: "牧濑红莉栖"`；Amadeus 记忆副本单独写为 `target_character: "Amadeus（牧濑红莉栖记忆副本）"`，避免与原角色的连续状态混为一谈。

### 6. 时间间隔

优先级如下：

1. 文本明确时间；
2. 场景内可推断的时间；
3. 同一连续对话但无时间线索时使用 `1.0` 测试占位。

正式数据的时间估计和证据写入审计文件。场景切换不能机械地继续使用 `1.0`，需要按文本估计较大的间隔。

## 批量流程

### 阶段 A：小规模冻结

先从不同类型场景选 10–20 个序列，覆盖：冲突、惊讶、恐惧、喜悦、探索、恢复、持续压力和近中性场景。若当前语料不足 10 个合格序列，则覆盖所有符合规则的场景，并注明样本量限制；检查字段语义和历史情绪继承规则。

### 阶段 B：正式标注

按章节或完整场景分批处理，每批约 10–30 个序列。每批生成：

- 训练核心 JSONL；
- 审计 JSONL：来源范围、证据句、置信度、歧义说明；
- 自动校验报告。

### 阶段 C：质量抽检

- 100% 自动检查结构、空值、范围、时间和维度；
- 100% 检查低置信度或歧义标注；
- 普通样本至少抽样 10–20% 人工复核；
- 保留一小批人工确认数据作为独立验证集。

## 数据划分

按完整章节、故事段或场景划分，不按事件随机切分。验证集和测试集不能包含训练集中同一故事的相邻事件；审计中标为同一底层冲突的不同世界线分支也应放在同一数据分区。

## 目录建议

```text
data/annotations/
  kurisu_two_volumes_v0.jsonl        # 当前合并核心数据
  kurisu_two_volumes_v0_audit.jsonl  # 来源、证据与排除审计
  train/*.jsonl                      # 后续按完整场景划分
  validation/*.jsonl
  test_human_verified/*.jsonl
```

当前合并标注文件 `data/annotations/kurisu_two_volumes_v0.jsonl` 含 36 段序列、1025 个事件步；来源、逐事件证据、覆盖审查和排除理由统一保存在 `data/annotations/kurisu_two_volumes_v0_audit.jsonl`。本批次包含自动批量初标草稿，标签未经过独立人工确认，不应当作客观心理真值或独立评估集。
