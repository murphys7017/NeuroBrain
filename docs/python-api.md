# Python 推理接口

NeuroBrain 可以直接嵌入其他 Python 程序。推荐使用 `NeuroBrainSession`，而不是让业务代码直接管理 CfC 隐状态。公开契约刻意保持为“一个输入、任意读取”：每次更新只提交一个当前事件的 8 维刺激对象；不需要传历史条件、隐状态或时间间隔。session 自动保存连续状态，输出可按整组或任意维度读取。

当前 facade 是进程内 Python API：它不启动 HTTP server，不读取文本，也不依赖 Ollama 或 MiniMax。

## 安装和导入

在宿主项目中使用同一个 uv 环境：

```powershell
uv sync --locked
```

然后导入：

```python
from neurobrain import NeuroBrainSession, StateSnapshot
```

如果宿主项目不在这个仓库中，可以把 NeuroBrain 安装为本地包，或把仓库加入该项目的工作区；具体依赖仍以 `pyproject.toml` 和 `uv.lock` 为准。

## 从 checkpoint 创建 session

```python
from neurobrain import NeuroBrainSession

brain = NeuroBrainSession.from_checkpoint(
    "artifacts/kurisu-two-volumes-baseline-cfc512.pt",
    device="cpu",
)
```

参数：

| 参数 | 说明 |
|---|---|
| `path` | checkpoint 路径；服务端应固定或白名单化，不要直接接受客户端路径 |
| `device` | 默认 `"cpu"`；有可用 GPU 时可以传 `"cuda"` |

每次事件使用默认 `1.0` 秒的时间间隔；它不是 `update()` 的输入。公开调用只传当前刺激。`device` 和 checkpoint 路径是初始化模型的部署配置，不是逐事件状态输入。

如果宿主明确需要不同的固定采样间隔，可以在创建 session 时设置 `delta_seconds`；对外每次更新仍然只传一个 stimulus。常规集成建议保留默认值。

`from_checkpoint()` 会检查协议版本，并要求 checkpoint 的 `history_mode` 是 `zeros`；没有 `history_mode` 字段的旧 checkpoint 也可以加载。`provided` 或 `fixed` 历史模式需要使用底层 `NeuroBrainModel`。

## 推进一个事件

`update()` 每次只接收一个位置参数 `stimulus`。推荐传入一个命名 mapping，这样不会依赖调用方记忆维度顺序：

```python
state = brain.update({
    "joy": 0.2,
    "trust": 0.0,
    "fear": 0.0,
    "surprise": 0.1,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.4,
})
```

需要紧凑的进程内调用时，也可以把同一个刺激对象表示为按协议顺序排列的 8 项序列：

```python
state = brain.update([
    0.2,  # joy
    0.0,  # trust
    0.0,  # fear
    0.1,  # surprise
    0.0,  # sadness
    0.0,  # disgust
    0.0,  # anger
    0.4,  # anticipation
])
```

输入规则：

- 必须恰好包含 8 个值。
- mapping 必须恰好包含这 8 个名称：`joy`、`trust`、`fear`、`surprise`、`sadness`、`disgust`、`anger`、`anticipation`。
- 每个值必须是 `int` 或 `float`，不能是 `bool`、`NaN` 或无穷大。
- 每个值必须位于 `[0.0, 1.0]`。
- 一次 `update()` 只表示一个新事件，并使 `step` 加 1。

公开 facade 将底层的 `historical_condition` 固定为全零；历史影响由 session 保存的 CfC 隐状态承载。调用方无需向公开 facade 提供历史状态或时间步参数。

## 读取状态

`update()` 返回一个不可变的 `StateSnapshot`：

```python
state = brain.update([0.0] * 8)

print(state.step)
print(state.emotion["joy"])
print(state.behavior["engagement"])
```

快照字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `protocol_version` | `str` | 当前为 `io-v0.2` |
| `step` | `int` | session 处理过的事件数 |
| `emotion` | `Mapping[str, float]` | 8 个情绪输出 |
| `behavior` | `Mapping[str, float]` | 6 个行为倾向输出 |

要转成 JSON 请求的响应体：

```python
payload = state.as_dict()
```

返回值形状如下：

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

上面的数值只是结构示例，不是固定结果。

`read()` 只读最近快照，不推进状态：

```python
latest = brain.read()       # StateSnapshot | None
emotions = brain.read("emotion")   # Mapping[str, float] | None
behaviors = brain.read("behavior") # Mapping[str, float] | None
joy = brain.read("joy")     # float | None
avoidance = brain.read("avoidance")
```

第一次 `update()` 之前：

```python
assert brain.read() is None
assert brain.read("joy") is None
```

可以读取完整快照、整组 `emotion`、整组 `behavior`，或 8 个情绪与 6 个行为中的任意维度；其他名称会抛出 `KeyError`。所有读取都是只读操作，不推进 step。

## 重置和 session 隔离

```python
brain.reset()

assert brain.read() is None
assert brain.update([0.0] * 8).step == 1
```

每个 session 都有自己的隐状态：

```python
alice = NeuroBrainSession.from_checkpoint("artifacts/model.pt")
bob = NeuroBrainSession.from_checkpoint("artifacts/model.pt")

alice.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
assert bob.read() is None
```

不要让不同角色共用一个 session，否则它们的经历会混在同一个 CfC 隐状态里。

## 嵌入宿主程序

一个最小的宿主层可以这样写：

```python
from neurobrain import NeuroBrainSession

CHECKPOINT = "artifacts/kurisu-two-volumes-baseline-cfc512.pt"
sessions: dict[str, NeuroBrainSession] = {}


def handle_event(session_id: str, stimulus: dict[str, float]) -> dict:
    session = sessions.get(session_id)
    if session is None:
        session = NeuroBrainSession.from_checkpoint(CHECKPOINT, device="cpu")
        sessions[session_id] = session
    return session.update(stimulus).as_dict()
```

这段函数可以被 Python Web 框架、Agent tool、机器人事件循环、游戏逻辑或桌面应用调用。外部系统负责：

1. 把文本、事件或传感器输入转换为 8 维刺激；
2. 为每个角色/对话维护稳定的 `session_id`；
3. 决定何时调用 `reset()`；
4. 把返回的状态交给 LLM、行为决策或表现层。

NeuroBrain 负责：

1. 校验 8 维输入；
2. 持续保存并更新 CfC 隐状态；
3. 输出统一的情绪和行为数值；
4. 提供只读查询和 JSON-friendly 快照。

## 非 Python 程序

如果调用方是 JavaScript、C#、Java、Go 或其他语言，当前推荐增加一个很薄的 Python 适配服务：

```text
其他程序 → HTTP/WebSocket/gRPC/stdin → Python 适配层 → NeuroBrainSession
```

适配层只需要把请求转成 `session.update(stimulus).as_dict()`，不需要改动模型。推荐的 HTTP 路由和错误映射见 [外部集成](integration.md)。仓库目前没有提供该网络服务，也没有固定端口。

## 线程和进程

`NeuroBrainSession` 内部用锁保护更新和读取，同一个实例可以在多线程宿主中安全地串行推进。但是：

- 多进程部署时，每个进程的 session 内存彼此独立。
- 进程重启后，隐状态不会自动恢复。
- 当前没有公开的隐状态序列化协议。
- 需要跨进程一致性时，宿主必须实现粘性路由、单写者队列或自己的状态存储。
- 训练和模型替换不能在在线请求线程中进行。

## 错误处理

常见异常：

```python
try:
    state = brain.update(stimulus)
except TypeError:
    # 输入容器或值类型错误
    ...
except ValueError:
    # 维度、范围、有限性或 delta_seconds 错误
    ...
except KeyError:
    # read() 查询了未知输出维度
    ...
```

HTTP 或 RPC 宿主通常可以把 `TypeError`、`ValueError` 和 `KeyError` 映射为客户端 400，把 checkpoint、设备或模型内部错误记录为服务端 500；不要把 Python traceback 直接返回给客户端。

## 底层模型接口

需要批量序列、显式历史条件或自行管理隐状态时，使用 `NeuroBrainModel`：

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

底层接口的输入输出形状和范围见 [输入输出协议 v0.2](io-protocol-v0.2.md)。普通业务调用应优先使用 `NeuroBrainSession`，避免把隐状态、历史通道和 checkpoint 细节暴露给客户端。

## 当前边界

- `NeuroBrainSession` 不接收文本，只接收结构化数字。
- 它不提供 HTTP server，也不负责鉴权、限流、持久化或多租户隔离。
- 它不会在线更新模型参数。
- 输出是实验模型估计，不是心理真值，不等于动作指令。
- checkpoint 训练时的历史模式必须与调用方式匹配；公开 facade 使用全零历史条件。
