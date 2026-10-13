# NeuroBrain 训练说明

当前训练实现用于验证 CfC-512 的监督学习流程，并生成可供离线推理的 checkpoint。训练数据是 LLM 辅助的初始标注，不是人工确认的心理真值；训练指标不能单独证明真实角色泛化。

## 训练入口

训练脚本：

```powershell
uv run --locked python scripts/train.py --help
```

CPU smoke training：

```powershell
uv run --locked python scripts/train.py `
  --data data/annotations/kurisu_two_volumes_v0.jsonl `
  --epochs 1 `
  --batch-size 1 `
  --val-ratio 0 `
  --history-mode zeros `
  --device cpu `
  --output artifacts/smoke-cfc512.pt
```

GPU 训练：

```powershell
uv run --locked python scripts/train.py `
  --data data/annotations/kurisu_two_volumes_v0.jsonl `
  --epochs 20 `
  --batch-size 4 `
  --val-ratio 0.2 `
  --history-mode zeros `
  --device cuda `
  --output artifacts/kurisu-cfc512.pt
```

训练输出保存到 `artifacts/`，该目录默认被 Git 忽略。训练期间不会修改公开 Python API，也不会在线更新已经运行的 session。

## 数据输入

训练脚本读取 JSONL，每一行是一条完整序列。每条记录至少包含：

- `sequence_id`：唯一、非空字符串；
- `target_character`：可选角色名；
- `events`：按时间顺序排列的非空事件数组。

每个 event 包含：

- `text`：原始事件文本，可为空字符串；
- `current_stimulus`：8 维当前刺激；
- `historical_emotion`：8 维历史条件，可在记录顶层提供并沿序列复用，也可逐事件提供；
- `emotion_target`：8 维事件后的情绪目标；
- `behavior_target`：6 维行为目标；
- `delta_seconds`：有限、非负的时间间隔。

8 个情绪维度顺序是：

```text
joy, trust, fear, surprise,
sadness, disgust, anger, anticipation
```

6 个行为维度顺序是：

```text
engagement, avoidance, expression,
exploration, initiative, activation
```

加载器会校验 JSON、维度、范围、有限性、非空序列和重复 `sequence_id`，然后转换为 `AnnotationSequence`。

## 校验数据

```powershell
uv run --locked python scripts/validate_annotations.py `
  data/annotations/kurisu_two_volumes_v0.jsonl
```

校验失败时应先修复数据，不要在训练脚本中绕过协议检查。当前训练语料的来源、覆盖范围和审核状态见仓库中的 annotation audit 文件。

## 历史模式

训练脚本的 `--history-mode` 有三种：

| 模式 | 行为 | 适用范围 |
|---|---|---|
| `zeros` | 每个时间步把历史条件置零 | 默认训练路径；与公开 `NeuroBrainSession` 兼容 |
| `provided` | 使用 JSONL 中提供的历史条件 | 历史通道专项实验；会形成 facade 不兼容的 checkpoint |
| `fixed` | 为一条序列重复首个历史条件 | 对照实验；会形成 facade 不兼容的 checkpoint |

默认使用 `zeros`，因为事件级历史标签通常来自前一事件的目标状态，直接反馈会造成目标信息泄漏或过强捷径。需要研究历史字段时，应单独记录实验假设和评估结果。

训练 checkpoint 中会写入 `history_mode`。`NeuroBrainSession.from_checkpoint()` 只接受 `zeros` 或缺失该字段的旧 checkpoint；`provided`/`fixed` 模型应直接交给 `NeuroBrainModel`，由调用方显式传入历史条件。

## 训练流程

每个 epoch 的基本流程：

```text
JSONL → 协议校验 → 变长 batch 补零 → mask loss
     → CfC 前向 → Smooth L1 → 反向传播 → 保存最佳 checkpoint
```

损失由情绪头和行为头的 masked Smooth L1 组成。补零时间步不参与损失。存在验证集时，以验证总损失选择并保存最佳 epoch；`--val-ratio 0` 时使用训练损失，只适合流程 smoke test。

## checkpoint 内容

训练脚本保存：

- `model_state_dict`；
- `optimizer_state_dict`；
- `model_config`：编码器、隐状态和 CfC mode；
- `protocol_version`；
- `epoch`、`best_loss` 和训练历史；
- 数据路径、随机种子和 `history_mode`。

公开 facade 加载时还会检查协议和历史模式。模型文件是本地部署产物，不应作为客户端可上传的请求参数。

## 推理验证

训练完成后先用底层 checkpoint 做最小推理：

```powershell
uv run --locked python -c "from neurobrain import NeuroBrainSession; s=NeuroBrainSession.from_checkpoint('artifacts/kurisu-cfc512.pt'); print(s.update([0.0]*8).as_dict())"
```

再运行完整测试：

```powershell
uv run --locked pytest -q
```

如果需要验证连续性、恢复和前史影响，运行：

```powershell
uv run --locked python scripts/synthetic_benchmark.py --device cpu
```

合成基准和真实标注训练回答的是不同问题：前者验证已知动力学的可学习性，后者只能在当前标注分布上提供训练流程和有限的 held-out 证据。
