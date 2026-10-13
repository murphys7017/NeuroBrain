# NeuroBrain 输入输出协议 v0.2

**协议状态：当前实现基线**
**核心模型：CfC-512**
**公开 facade：`NeuroBrainSession`**

## 协议分层

当前代码有两层接口，含义必须区分：

```text
业务系统
  └─ NeuroBrainSession.update(8 维当前刺激)
       ├─ historical_condition 固定为全零
       ├─ session 内部保存 CfC hidden_state
       └─ StateSnapshot：8 维 emotion + 6 维 behavior

训练/实验
  └─ NeuroBrainModel.forward/step(当前刺激 + 历史条件 + 时间间隔 + 可选隐状态)
```

- **公开 facade** 面向应用集成，调用简单，只接受一个当前刺激，自动维护隐状态。
- **底层模型** 面向训练、批量序列和历史条件实验，调用方需要自行管理输入形状、设备和隐状态。
- 当前仓库没有 HTTP server；网络协议由宿主应用包装 Python facade 后自行定义。

对业务调用而言，每次事件只有一个 stimulus 输入；模型隐状态由 session 内部维护，历史条件由 facade 固定为零，时间间隔使用 session 默认配置（可在创建 session 时一次性设置）。

## 外部维度

### 当前刺激

公开 `NeuroBrainSession.update()` 接收 8 个维度：

```text
joy, trust, fear, surprise,
sadness, disgust, anger, anticipation
```

每个值表示当前事件新增的刺激强度，不直接指定最终情绪。值必须是有限数字，并位于 `[0.0, 1.0]`。

### 情绪状态

输出情绪仍使用上述 8 个维度。它是模型结合当前刺激和连续隐状态后的估计，不使用 Softmax，不要求 8 个值之和为 1。

### 行为倾向

```text
engagement, avoidance, expression,
exploration, initiative, activation
```

行为值表示倾向强度，不是具体动作指令。多个维度可以同时较高，最终动作由外部决策系统选择。

## 公开 facade 协议

### 创建 session

```python
NeuroBrainSession.from_checkpoint(
    path,
    device="cpu",
)
```

- `path`：模型 checkpoint。
- `device`：默认 `cpu`，可显式传 `cuda`。
- checkpoint 的 `protocol_version` 必须为 `io-v0.2`。
- checkpoint 的 `history_mode` 必须为 `zeros`，或缺少该字段；`provided`/`fixed` checkpoint 不兼容该 facade。

每次事件默认使用 `1.0` 秒间隔。宿主可以在创建 session 时一次性设置另一个固定间隔，但 `update()` 每次仍只有一个 stimulus 输入。

### 更新输入

命名 mapping：

```json
{
  "joy": 0.2,
  "trust": 0.0,
  "fear": 0.0,
  "surprise": 0.1,
  "sadness": 0.0,
  "disgust": 0.0,
  "anger": 0.0,
  "anticipation": 0.4
}
```

或按固定顺序的 8 项序列：

```text
[joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
```

输入约束：

- mapping 必须恰好包含 8 个已知名称。
- 序列必须恰好有 8 项。
- 不接受 `bool`、字符串、`NaN` 或无穷大。
- 所有值必须位于 `[0, 1]`。
- 一次 `update()` 表示一个事件，成功后 `step` 加 1。
- 每次更新只提交当前刺激；不要同时提交历史状态、输出目标或 CfC 隐状态。

### 更新输出

`StateSnapshot.as_dict()` 返回：

```json
{
  "protocol_version": "io-v0.2",
  "step": 1,
  "emotion": {
    "joy": 0.31,
    "trust": 0.28,
    "fear": 0.12,
    "surprise": 0.19,
    "sadness": 0.11,
    "disgust": 0.09,
    "anger": 0.14,
    "anticipation": 0.27
  },
  "behavior": {
    "engagement": 0.41,
    "avoidance": 0.18,
    "expression": 0.36,
    "exploration": 0.39,
    "initiative": 0.32,
    "activation": 0.44
  }
}
```

数值仅为结构示例。`read()` 返回最近快照，`read("emotion")`/`read("behavior")` 返回整组输出，`read(name)` 可读取任意一个已定义维度；第一次更新前这些读取均为 `None`。读取不推进状态。`reset()` 清除隐状态、快照和步数。

## 底层模型协议

`NeuroBrainModel.forward()` 的序列输入：

```text
current_stimulus:     [B, T, 8]
historical_condition: [B, T, 8]
delta_seconds:        [B, T] 或 [B, T, 1]
hidden_state:         [B, 512]（可选）
```

`current_stimulus` 和 `historical_condition` 的最后一维都必须是 8，值在 `[0, 1]`。`delta_seconds` 必须有限且非负。

序列输出：

```text
emotion:      [B, T, 8]
behavior:     [B, T, 6]
final_hidden: [B, 512]
```

`NeuroBrainModel.step()` 支持单个样本 `[8]` 或批量 `[B, 8]`，返回相应的情绪、行为和下一步 `final_hidden`。它适合训练循环、显式历史通道和需要每步不同 `delta_seconds` 的调用方。

## 时间和状态语义

- `delta_seconds` 表示该事件距离上一次状态更新经过的时间。
- 公开 facade 在 session 创建时固定时间间隔；底层模型可以每步传不同时间间隔。
- `historical_condition` 是输入背景，不会替换 CfC 隐状态。
- CfC 隐状态是连续计算状态，不是可查询的文本记忆。
- 具体历史经历、长期故事和检索由外部 Memory 系统管理。

## 版本和兼容性

改变维度顺序、字段含义、范围或张量形状时，必须升级协议版本并同步：

1. `src/neurobrain/protocol.py`；
2. 模型和 facade 的校验；
3. checkpoint 加载检查；
4. 测试和文档；
5. 外部适配层的请求/响应 schema。

业务服务不要把 checkpoint 文件格式当成公开网络协议；对外应只暴露结构化刺激和 `StateSnapshot`。
