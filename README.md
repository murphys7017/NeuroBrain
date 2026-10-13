# NeuroBrain

NeuroBrain 是一个面向虚拟角色、对话系统和 Agent 的实验性连续状态引擎。它接收结构化的当前刺激，维护每个角色或会话独立的 CfC 隐状态，并输出 8 维情绪状态和 6 维行为倾向。

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

语言模型负责理解文本、提取刺激和组织回复；NeuroBrain 负责持续状态演化。它不直接读取文本，不直接生成台词，也不执行动作。

> 当前项目仍处于原型和数据验证阶段。合成基准可以验证模型是否学会指定的状态动力学，但不能证明它模拟了真实心理，也不能替代人工评估。

## 当前能力

- CfC-512 连续时间动力核心：32 维输入编码、512 维隐状态、8+6 维输出。
- 单步和整段序列推理：底层 `NeuroBrainModel` 支持显式历史条件、时间间隔和隐状态传递。
- 面向应用的 Python facade：`NeuroBrainSession` 每次只接收一个 8 维当前刺激，自动保留隐状态，随时可读最近结果。
- 面向宿主的状态仓库：`NeuroBrainStateStore` 管理多个状态流，支持按事件 ID 幂等更新和只读快照。
- 训练和 checkpoint：支持 JSONL 序列、变长批处理、掩码损失和带元数据的模型保存。
- 合成状态动力学基准：独立生成训练、验证和测试序列，报告 held-out 误差、方向命中率和恢复探针。
- 角色试玩 CLI：可连接本地 Ollama 或 MiniMax，展示角色回复、NeuroBrain 状态和诊断信息。

当前调用边界包括 Python API、进程内 `NeuroBrainStateStore` 和可选的 JSON WebSocket 服务。AstrBot 插件在 AstrBot 进程内加载状态仓库并直接运行推理，因此 AstrBot 环境需要安装 NeuroBrain 与 PyTorch 等运行依赖；若要隔离模型进程，可改用独立的 WebSocket 服务。具体适配方式见 [外部集成](docs/integration.md)。

## 快速开始

### 1. 安装环境

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。项目为 Windows/Linux 锁定了 PyTorch 2.14.1 的 CUDA 13.0 wheel；没有 NVIDIA GPU 也可以在这些平台上使用 CPU 推理，只是安装包仍然较大。macOS 需要单独配置 CPU-only PyTorch，当前锁文件不覆盖该环境。

在 PowerShell 中执行：

```powershell
uv sync --locked
```

检查 PyTorch、CUDA 和 CfC：

```powershell
uv run --locked python scripts/check_environment.py
```

运行完整测试：

```powershell
uv run --locked pytest -q
```

只做包安装 smoke check：

```powershell
uv run --locked neurobrain
```

这条命令只打印包已就绪，不会启动常驻服务，也不会监听端口。

### 2. 运行合成基准

合成基准不依赖 LLM，也不会写入 checkpoint：

```powershell
uv run --locked python scripts/synthetic_benchmark.py --device cpu
```

有可用 GPU 时可以改成：

```powershell
uv run --locked python scripts/synthetic_benchmark.py --device cuda
```

默认配置会训练 128 条序列、验证 32 条序列、测试 64 条序列，共 60 轮。较快的边界 smoke test：

```powershell
uv run --locked python scripts/synthetic_benchmark.py `
  --device cpu `
  --train-episodes 2 `
  --validation-episodes 1 `
  --test-episodes 1 `
  --steps 1 `
  --epochs 1 `
  --batch-size 1
```

结果解释和已记录基线见 [合成状态动力学基准](docs/synthetic-benchmark.md)。

### 3. 直接调用 Python API

应用集成最小示例：

```python
from neurobrain import NeuroBrainSession

brain = NeuroBrainSession.from_checkpoint(
    "artifacts/kurisu-two-volumes-baseline-cfc512.pt",
    device="cpu",
)

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

payload = state.as_dict()
print(payload["emotion"]["joy"])
print(payload["behavior"]["engagement"])

# read() 只读最近结果，不推进模型。
latest = brain.read()
emotions = brain.read("emotion")
behaviors = brain.read("behavior")
joy = brain.read("joy")

# 新角色或新会话开始时清空隐状态。
brain.reset()
```

每次状态更新只有一个输入：当前事件的 8 维刺激对象。调用方不需要传历史状态、模型隐状态、情绪目标或行为目标；时间间隔采用 session 默认配置。输入可以是 8 项序列或上例中的命名 mapping，mapping 更适合跨服务传输。每个值必须是有限数，并位于 `[0, 1]`。

输出可以随时读取：`read()` 返回完整快照，`read("emotion")`/`read("behavior")` 返回整组结果，`read("joy")` 或 `read("avoidance")` 返回任意一个已定义维度。读取不推进状态。

返回的快照包含：

| 字段 | 内容 |
|---|---|
| `protocol_version` | 当前为 `io-v0.2` |
| `step` | 当前 session 已处理的事件数，从 1 开始 |
| `emotion` | 8 个情绪维度，值在 `[0, 1]` |
| `behavior` | 6 个行为倾向维度，值在 `[0, 1]` |

完整签名、异常行为、checkpoint 兼容性和外部服务适配见 [Python 推理接口](docs/python-api.md) 与 [外部集成](docs/integration.md)。

AstrBot 集成可使用 `convert_astrbot_tendency_to_stimulus()` 将 AstrBot Persona 的
八维 `0..10` 强度映射为 NeuroBrain 的八维 `[0, 1]` 输入；该转换器位于宿主适配层，
不改变 NeuroBrain 核心模型的输入协议。

### 4. 运行角色试玩闭环

角色试玩会在每一轮调用两个外部语言模型阶段：先生成结构化角色草稿，再结合 NeuroBrain 状态生成最终回复。它不是纯 NeuroBrain 服务，需要先准备一个可访问的 LLM。

#### Ollama

确保 Ollama 已运行，并准备代码默认使用的模型或通过 `--model` 指定其他模型：

```powershell
ollama serve
ollama pull qwen3.5:2b
uv run --locked python scripts/character_test.py `
  --provider ollama `
  --device cpu `
  --message "今天的实验终于有进展了。"
```

省略 `--message` 会进入交互模式；输入 `/reset` 清空当前 CfC 隐状态和历史条件，输入 `/quit` 退出。`--show-json` 可以输出完整诊断结构，`--output result.json` 可以保存结果。

#### MiniMax

将凭据放入环境变量，不要写进代码或提交到仓库：

```powershell
$env:MINIMAX_API_KEY = "你的 Subscription Key"
uv run --locked python scripts/character_test.py `
  --provider minimax `
  --device cpu `
  --message "今天的实验终于有进展了。"
```

默认 endpoint、模型、超时、历史模式和参数见 `scripts/character_test.py --help`。外部 LLM 失败时，CLI 会返回错误；NeuroBrain 本身仍是独立的本地模型。

### 5. 校验数据并训练 checkpoint

先校验 JSONL：

```powershell
uv run --locked python scripts/validate_annotations.py `
  data/annotations/kurisu_two_volumes_v0.jsonl
```

运行一轮训练链路 smoke test：

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

正式实验时，将 `--device cpu` 换为 `--device cuda`，并设置合理的 `--epochs`、`--val-ratio` 和输出路径。默认 `--history-mode zeros` 与公开 `NeuroBrainSession` 兼容；`provided` 和 `fixed` checkpoint 不能直接交给该 facade。

## 对外调用方式

### 推荐：宿主进程内嵌

在 Python 服务、Agent、机器人或游戏进程中加载一次 checkpoint，并为每个独立角色/会话创建一个 `NeuroBrainSession`：

```python
from neurobrain import NeuroBrainSession

sessions: dict[str, NeuroBrainSession] = {}


def handle_event(session_id: str, stimulus: dict[str, float]) -> dict:
    session = sessions.get(session_id)
    if session is None:
        session = NeuroBrainSession.from_checkpoint(
            "artifacts/kurisu-two-volumes-baseline-cfc512.pt",
            device="cpu",
        )
        sessions[session_id] = session
    return session.update(stimulus).as_dict()
```

外部 LLM 或规则层负责把文本转成 8 维刺激；NeuroBrain 返回数值状态；宿主应用再把状态传给语言生成、行为决策或表现层。不要把不同角色共用一个 session，否则它们会共享隐状态。

AG99 的 AstrBot 环境使用 [进程内插件](https://github.com/murphys7017/astrbot_plugin_neurobrain)：插件在 AstrBot 启动时加载 `NeuroBrainStateStore`，Persona 结果推进状态，并把当前状态动态加入模型提示词；其他插件通过只读 API 获取快照。此模式直接在 AstrBot 进程中运行 PyTorch/CUDA，不使用 WebSocket。

### 需要进程隔离时

仓库还提供可选的 JSON WebSocket 服务。它在 NeuroBrain 自己的环境中加载 checkpoint，
外部宿主只需要发送结构化刺激或读取状态，不需要安装 PyTorch/CUDA：

```powershell
$env:NEUROBRAIN_WRITE_TOKEN = "replace-with-a-long-random-secret"
uv run --locked python scripts/neurobrain_server.py `
  --checkpoint artifacts/kurisu-two-volumes-baseline-cfc512.pt `
  --host 127.0.0.1 `
  --port 8765 `
  --write-token $env:NEUROBRAIN_WRITE_TOKEN
```

服务使用 `neurobrain-ws-v1`，供需要进程隔离的外部客户端提交结构化刺激或读取状态。
写入客户端必须配置非空 writer token；reader 未配置读令牌时可免令牌读取，写入令牌也可以读取。
当前 AstrBot 插件使用进程内状态仓库，不连接此 WebSocket 服务；AG99live 通过插件只读 API
在第三条动作链读取状态，不改变既有 `output.segment.v5` 消息。

如果宿主更适合 HTTP，也可以提供以下薄适配层：

```text
POST /sessions/{id}/events  {"stimulus": {8 个维度}}
GET  /sessions/{id}/state
POST /sessions/{id}/reset
```

`POST /events` 的响应直接使用 `StateSnapshot.as_dict()`；`GET /state` 对尚未更新的 session 返回 `null`；`reset` 清空当前 session。这个路由只是推荐的外部契约，不是仓库已经提供的监听端口。适配注意事项见 [外部集成](docs/integration.md)。

## 输入输出协议

### 公开 facade

| 方向 | 维度 | 说明 |
|---|---:|---|
| `update(stimulus)` | 8 | 当前事件新增刺激 |
| `emotion` | 8 | `joy`、`trust`、`fear`、`surprise`、`sadness`、`disgust`、`anger`、`anticipation` |
| `behavior` | 6 | `engagement`、`avoidance`、`expression`、`exploration`、`initiative`、`activation` |

公开 facade 把低层的 `historical_condition` 固定为全零；CfC 隐状态仍然跨 `update()` 持续传递。时间间隔是 session 创建时的 `delta_seconds`，默认 `1.0` 秒。

### 底层模型

`NeuroBrainModel` 面向训练和实验，输入为：

```text
current_stimulus:     [B, T, 8]
historical_condition: [B, T, 8]
delta_seconds:        [B, T] 或 [B, T, 1]
hidden_state:         [B, 512]（可选）
```

输出为：

```text
emotion:      [B, T, 8]
behavior:     [B, T, 6]
final_hidden: [B, 512]
```

底层接口适合批量序列训练、历史条件实验和需要自行管理隐状态的调用方，不建议把它直接暴露给普通业务请求。

## 目录和入口

| 路径 | 用途 |
|---|---|
| `src/neurobrain/api.py` | 对外有状态 Python facade |
| `src/neurobrain/model.py` | CfC 模型、单步/序列推理和 checkpoint |
| `src/neurobrain/data.py` | JSONL 加载、协议校验和变长 batch |
| `src/neurobrain/character_loop.py` | LLM + NeuroBrain 角色闭环 |
| `scripts/character_test.py` | 角色试玩 CLI |
| `scripts/train.py` | 监督训练入口 |
| `scripts/synthetic_benchmark.py` | 合成数据训练与 held-out 评估 |
| `scripts/check_environment.py` | PyTorch/CUDA/CfC 环境检查 |
| `tests/` | 模型、API、数据和基准回归测试 |
| `docs/` | 协议、架构、训练、接口和实验说明 |

## 文档索引

- [上手与运行](docs/getting-started.md)：安装、验证、启动、训练和排错。
- [Python 推理接口](docs/python-api.md)：`NeuroBrainSession` 的完整调用约定。
- [外部集成](docs/integration.md)：宿主进程、HTTP 适配、生命周期和并发边界。
- [输入输出协议 v0.2](docs/io-protocol-v0.2.md)：底层张量协议与公开 facade 的关系。
- [项目架构](docs/architecture.md)：模块职责和状态演化闭环。
- [合成状态动力学基准](docs/synthetic-benchmark.md)：随机老师数据、指标和已记录结果。
- [训练说明](docs/training-v0.1.md)：JSONL 格式、历史模式和训练入口。
- [核心模型技术方案](docs/core-model-technical-plan.md)：CfC-512 结构和训练方案。
- [项目路线图](docs/roadmap.md)：已完成工作和后续计划。

`docs/design-baseline.md` 和 `docs/cfc-v0.1-implementation-plan.md` 保留为历史设计记录；其中的早期维度和阶段描述不覆盖当前 v0.2 实现。

## 设计边界

NeuroBrain 维护连续内部状态和行为倾向，不负责文本理解、最终台词、长期记忆检索、Live2D、语音或外部动作执行。公开输出是实验模型估计，应结合独立测试和人工复核解释。训练、模型发布和正式运行应相互隔离，避免在线请求修改模型权重。
