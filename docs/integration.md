# 外部集成

本文说明外部服务如何把 NeuroBrain 当作一个持续状态组件调用。除了 Python API，仓库现在提供一个可选的 JSON WebSocket 服务；服务仍由宿主负责启动、鉴权和进程生命周期。

## 集成边界

推荐把系统拆成四层：

```text
用户文本/事件
    ↓
宿主应用或 LLM：提取 8 维当前刺激
    ↓
NeuroBrainSession：维护隐状态并输出 emotion/behavior
    ↓
宿主应用：把状态交给回复生成、行为决策或表现层
```

NeuroBrain 接收的是结构化数字，不读取自然语言。8 个当前刺激维度为：

```text
joy, trust, fear, surprise,
sadness, disgust, anger, anticipation
```

每个值必须是 `[0, 1]` 范围内的有限数。输出包含 8 个 `emotion` 维度和 6 个 `behavior` 维度：

```text
engagement, avoidance, expression,
exploration, initiative, activation
```

行为值表示倾向强度，不是动作命令；由宿主应用决定是否转成台词、动作或工具调用。

## AstrBot 适配器

AstrBot 的 Persona 输出由 AstrBot 自己拥有，使用大小写敏感的八个维度名，
每项为 `0..10` 的整数。NeuroBrain 不依赖 AstrBot 包；需要接入时使用仓库内的
转换器，把这个宿主输入协议转换成 NeuroBrain 的小写维度和 `[0, 1]` 浮点刺激：

```python
from neurobrain import convert_astrbot_tendency_to_stimulus

stimulus = convert_astrbot_tendency_to_stimulus({
    "Joy": 8,
    "Trust": 4,
    "Fear": 0,
    "Surprise": 2,
    "Sadness": 1,
    "Disgust": 0,
    "Anger": 3,
    "Anticipation": 7,
})
state = session.update(stimulus)
```

转换器只负责字段、类型校验、边界裁剪和线性归一化；整数负值按 `0` 处理，
大于 `10` 的整数按 `10` 处理。它不读取文本，也不解释
`tendency` 的心理语义。宿主应用仍负责决定何时把 Persona 结果作为一次新的
NeuroBrain 输入，以及如何把 NeuroBrain 输出交给其他插件或表现层。

## 推荐的进程内调用

模型只加载一次，每个角色或独立会话创建一个 session：

```python
from neurobrain import NeuroBrainSession

CHECKPOINT = "artifacts/kurisu-two-volumes-baseline-cfc512.pt"
sessions: dict[str, NeuroBrainSession] = {}


def get_session(session_id: str) -> NeuroBrainSession:
    session = sessions.get(session_id)
    if session is None:
        session = NeuroBrainSession.from_checkpoint(
            CHECKPOINT,
            device="cpu",
        )
        sessions[session_id] = session
    return session


def update_state(session_id: str, stimulus: dict[str, float]) -> dict:
    session = get_session(session_id)
    return session.update(stimulus).as_dict()


def read_state(session_id: str) -> dict | None:
    session = sessions.get(session_id)
    snapshot = session.read() if session is not None else None
    return snapshot.as_dict() if snapshot is not None else None


def reset_state(session_id: str) -> None:
    session = sessions.get(session_id)
    if session is not None:
        session.reset()
```

生产服务通常会把 `sessions` 替换成带过期时间的会话仓库。checkpoint 应在进程启动时验证路径并以只读方式加载；不要让客户端提交任意模型路径。

## 一次事件的调用语义

`update()` 是唯一会推进状态的方法：

```python
state = get_session("character-42").update({
    "joy": 0.7,
    "trust": 0.3,
    "fear": 0.0,
    "surprise": 0.1,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.4,
})
```

它会：

1. 校验输入维度和范围；
2. 把历史条件固定为全零；
3. 用 session 保存的 CfC 隐状态推进一个时间步；
4. 保存新的隐状态和最近快照；
5. 返回不可变的 `StateSnapshot`。

`read()` 和 `read(name)` 都是只读操作，不会增加 `step`，不会重复计算：

```python
latest = get_session("character-42").read()
joy = get_session("character-42").read("joy")
emotions = get_session("character-42").read("emotion")
behaviors = get_session("character-42").read("behavior")
```

`reset()` 会同时清空隐状态、最近快照和步数。下一次 `update()` 的 `step` 会重新从 1 开始。

每次事件的时间间隔默认是 `1.0` 秒。宿主可以在创建 session 时一次性覆盖这个固定配置，但不需要把时间或历史条件随每个事件传入。

## 推荐的网络契约

如果不希望把 PyTorch/CUDA 安装到 AstrBot 环境，可以直接启动仓库自带的 WebSocket 服务：

```powershell
$env:NEUROBRAIN_WRITE_TOKEN = "replace-with-a-long-random-secret"
uv run --locked python scripts/neurobrain_server.py `
  --checkpoint artifacts/kurisu-two-volumes-baseline-cfc512.pt `
  --device cpu `
  --host 127.0.0.1 `
  --port 8765 `
  --write-token $env:NEUROBRAIN_WRITE_TOKEN
```

服务使用 `neurobrain-ws-v1` JSON 协议。writer 连接发送 `stimulus`，reader/writer 连接都可以发送 `get_state` 和 `subscribe`；只有 writer 可以 `reset`。`stimulus` 支持一个 `tendency` 或按顺序提交多个 `segments`，每个事件必须带唯一 `event_id`，重复事件只返回缓存快照而不会再次推进状态。服务端将每个 `state_key` 串行处理，并把状态更新广播给订阅者。

服务必须配置非空 `--write-token` 或 `NEUROBRAIN_WRITE_TOKEN`；AstrBot 桥接插件的 `write_token` 必须使用同一值。reader 默认可免令牌查询和订阅；设置 `--read-token` 后，reader 必须使用读令牌或写令牌。读令牌不会获得写权限。

AstrBot 桥接插件是 writer；其他插件通过桥接插件的只读 API 读取缓存状态。AG99live 可以作为同一 AstrBot 进程中的 reader 使用该 API，不需要修改既有 `output.segment.v5` 协议。

如果宿主应用要提供 HTTP，也可以把 Python facade 包装成以下资源模型。下面的 HTTP 路由仍是推荐契约，不是仓库内置路由。

### 推进一个事件

```text
POST /sessions/{session_id}/events
Content-Type: application/json
```

请求体：

```json
{
  "stimulus": {
    "joy": 0.7,
    "trust": 0.3,
    "fear": 0.0,
    "surprise": 0.1,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.4
  }
}
```

响应体直接对应 `StateSnapshot.as_dict()`：

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

上面的数值只是响应形状示例，不是固定输出。

### 读取最近状态

```text
GET /sessions/{session_id}/state
```

- session 不存在或从未更新：返回 `null`，或由宿主 API 统一返回 `404`。
- session 已更新：返回最近一次 `StateSnapshot.as_dict()`。
- 该请求不推进状态。

### 重置状态

```text
POST /sessions/{session_id}/reset
```

建议响应：

```json
{"ok": true, "step": 0}
```

重置应只影响指定 `session_id`，不能影响其他角色或会话。

## 错误映射

`NeuroBrainSession` 会主动拒绝无效输入，宿主服务可以按下表映射 HTTP 状态：

| Python 异常 | 常见原因 | 建议 HTTP 状态 |
|---|---|---:|
| `TypeError` | 输入不是 mapping/序列，或维度值不是数字 | 400 |
| `ValueError` | 缺失/多余维度、数量不是 8、值不在 `[0,1]` | 400 |
| `KeyError` | 读取不存在的输出维度 | 400 |
| `FileNotFoundError` | 服务启动时 checkpoint 路径错误 | 500 或启动失败 |
| `RuntimeError` | 模型或快照内部状态不一致 | 500 |

不要把完整 Python traceback 返回给外部客户端；记录服务端日志，向客户端返回稳定的错误码和简短消息。

## 会话、并发和扩展

- 一个 `NeuroBrainSession` 对应一个独立角色或独立对话；不要跨角色共享。
- `NeuroBrainSession` 内部使用锁保护 `update()` 和 `read()`，同一实例可以被多个线程访问，但宿主仍应限制会话数量和请求队列。
- 多进程部署时，每个进程都有自己的 session 内存；需要粘性路由、共享状态或显式状态序列化时，宿主必须补充相应机制。
- 当前没有隐状态序列化协议。进程重启后只能从零隐状态开始，除非宿主自行保存和恢复更底层的模型状态。
- 在线请求只更新隐状态，不更新模型权重。训练、评估和模型替换应在独立离线流程完成。
- `delta_seconds` 是 session 配置，不是每个 `update()` 的公开字段；如果不同事件需要不同时间间隔，请使用底层 `NeuroBrainModel.step()`，或扩展 facade 的版本化接口。

## 文本到刺激的职责分工

推荐让外部 LLM 或规则层输出严格的结构化对象，再调用 NeuroBrain：

```json
{
  "stimulus": {
    "joy": 0.0,
    "trust": 0.2,
    "fear": 0.6,
    "surprise": 0.4,
    "sadness": 0.1,
    "disgust": 0.0,
    "anger": 0.3,
    "anticipation": 0.2
  }
}
```

NeuroBrain 不负责判断文本中“应该有多少恐惧”或“角色应该说什么”。刺激提取器的提示词、规则、人工审核和重试策略属于宿主系统；应记录原始文本、提取结果和模型版本，方便回放和排查。

## 需要更底层控制时

训练、批量序列或历史条件实验使用 `NeuroBrainModel`：

```python
import torch
from neurobrain import NeuroBrainModel

model = NeuroBrainModel.load_checkpoint(
    "artifacts/kurisu-two-volumes-baseline-cfc512.pt",
    map_location="cpu",
)
model.eval()

output = model.step(
    torch.zeros(8),
    torch.zeros(8),
    delta_seconds=1.0,
)
print(output.emotion.shape)      # torch.Size([8])
print(output.behavior.shape)     # torch.Size([6])
print(output.final_hidden.shape) # torch.Size([512])
```

底层调用方必须自己管理 `historical_condition`、`hidden_state`、设备、批次和生命周期。普通业务接口应优先使用 `NeuroBrainSession`，避免把这些内部细节暴露给客户端。

## 部署前检查清单

- checkpoint 与代码的 `protocol_version` 匹配。
- 对外使用的 checkpoint 训练时 `history_mode=zeros`。
- 每个角色/会话有独立 `session_id`。
- 请求只允许 8 个已知维度和值范围 `[0,1]`。
- 服务端不接受客户端提供的 checkpoint 路径。
- 记录模型版本、协议版本、刺激提取结果和 `step`。
- 设计会话过期、进程重启和多副本部署后的状态策略。
- 使用独立测试和人工复核解释状态输出，不把情绪数值当作心理真值或动作命令。
