# 上手与运行

本文面向第一次拿到 NeuroBrain 仓库的开发者，说明依赖安装、环境检查、测试、模型运行、角色试玩和训练入口。

## 先理解项目的运行形态

NeuroBrain 目前有四种运行方式：

1. **库内嵌模式**：宿主 Python 进程导入 `NeuroBrainSession`，这是对外集成的推荐方式。
2. **角色试玩模式**：`scripts/character_test.py` 调用 Ollama 或 MiniMax，再把结构化刺激交给 NeuroBrain。
3. **实验模式**：运行训练脚本或合成基准，验证模型和数据流程。
4. **WebSocket 服务模式**：独立进程加载 checkpoint，为 AstrBot 或其他宿主提供状态读写。

`uv run --locked neurobrain` 只是安装 smoke check；它不会监听端口。需要网络接口时，
可以启动仓库自带的可选 WebSocket 服务；HTTP 或其他 RPC 仍由宿主自行包装，详见 [外部集成](integration.md)。

## 系统要求

- Python 3.11 或更高版本。
- [uv](https://docs.astral.sh/uv/)。
- CPU 可以运行推理和测试；NVIDIA GPU + CUDA 可以加速训练和基准。
- Windows PowerShell、Linux shell 和 macOS shell 均可运行 Python 入口；本文命令使用 PowerShell 多行语法。
- 角色试玩额外需要一个可访问的 Ollama 或 MiniMax endpoint。

项目依赖锁定了 PyTorch 2.9.1 CUDA 12.6 wheel。CPU-only 主机也可以安装和运行，但 PyTorch 安装包会比较大。

## 安装

在仓库根目录执行：

```powershell
uv sync --locked
```

`--locked` 要求 `pyproject.toml` 和 `uv.lock` 一致。如果你有意修改依赖，先更新锁文件，再重新执行安装；日常启动不应自动改锁文件。

## 安装后验证

### 检查运行时

```powershell
uv run --locked python scripts/check_environment.py
```

该脚本会打印 PyTorch 版本、CUDA 构建版本、CUDA 是否可用以及 CfC 最小前向结果。CUDA 不可用时，只要 CfC 的 CPU 前向成功，环境仍可用于测试和 CPU 推理。

### 运行测试

```powershell
uv run --locked pytest -q
```

### 检查包入口

```powershell
uv run --locked neurobrain
```

预期输出类似：

```text
NeuroBrain CfC-512 package is ready.
```

这只是命令行入口 smoke check，不会加载 checkpoint，不会启动网络服务。

## 使用已有 checkpoint 做一次推理

仓库通常包含一个本地基线 checkpoint：

```text
artifacts/kurisu-two-volumes-baseline-cfc512.pt
```

它是开发和试玩产物，不代表经过充分验证的生产模型。最小调用：

```powershell
uv run --locked python -c "from neurobrain import NeuroBrainSession; s=NeuroBrainSession.from_checkpoint('artifacts/kurisu-two-volumes-baseline-cfc512.pt'); print(s.update([0.0]*8).as_dict())"
```

如果 checkpoint 不在本地，请先训练一个，或把 `from_checkpoint()` 的路径换成你的模型文件。公开 facade 要求 checkpoint 的 `history_mode` 是 `zeros`，或者旧 checkpoint 中没有这个字段；`provided` 和 `fixed` 模式需要使用底层 `NeuroBrainModel` 自行传入历史条件。

## 运行角色试玩

角色试玩把外部语言模型和 NeuroBrain 串成一个闭环。每一轮流程是：

```text
用户文本 → LLM 提取结构化角色草稿/刺激 → NeuroBrain 更新状态 → LLM 生成最终回复
```

### Ollama

启动 Ollama，并准备一个与参数匹配的模型：

```powershell
ollama serve
ollama pull qwen3.5:2b
```

然后运行单轮测试：

```powershell
uv run --locked python scripts/character_test.py `
  --provider ollama `
  --device cpu `
  --message "今天的实验终于有进展了。"
```

默认 endpoint 是 `http://127.0.0.1:11434/api/chat`，默认模型是 `qwen3.5:2b`。如果你的 Ollama 地址或模型不同：

```powershell
uv run --locked python scripts/character_test.py `
  --provider ollama `
  --endpoint http://127.0.0.1:11434/api/chat `
  --model 你的模型名 `
  --message "测试一轮"
```

### MiniMax

```powershell
$env:MINIMAX_API_KEY = "你的 Subscription Key"
uv run --locked python scripts/character_test.py `
  --provider minimax `
  --device cpu `
  --message "今天的实验终于有进展了。"
```

`--message` 省略时会进入交互模式：`/reset` 清除 CfC 隐状态和历史条件，`/quit` 退出。使用 `--show-json` 查看完整输入、输出和 timing，使用 `--output result.json` 保存一份结果。

### 试玩参数

```powershell
uv run --locked python scripts/character_test.py --help
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--checkpoint` | 指定模型 checkpoint |
| `--provider` | `ollama` 或 `minimax` |
| `--device` | `cpu`、`cuda` 或其他 PyTorch device |
| `--history-mode` | `zeros`（默认）或实验性的 `previous_output` |
| `--delta-seconds` | 每个角色草稿 segment 的时间间隔 |
| `--conversation-turns` | 保留给角色模型的最近对话轮数 |
| `--message` | 执行单轮后退出 |
| `--show-json` | 输出完整诊断 JSON |
| `--output` | 把完整结果写到文件 |

角色试玩的 LLM 请求失败不会改变 checkpoint；排查重点是 endpoint、模型名、API key 和返回 JSON 格式。

## 运行合成基准

基准不需要 LLM：

```powershell
uv run --locked python scripts/synthetic_benchmark.py --device cpu
```

默认会生成独立的训练、验证和测试序列，训练全新 CfC 模型并打印：

- 情绪和行为的 MAE/MSE；
- 相对训练目标均值基线的 MSE skill；
- 相邻状态变化的方向命中率；
- joy/fear 脉冲与无刺激恢复；
- 不同前史之后输入相同中性刺激时的状态差异。

它不会写 checkpoint，也不会修改 `data/`。完整指标和限制见 [合成状态动力学基准](synthetic-benchmark.md)。

## 校验数据和训练

校验 JSONL：

```powershell
uv run --locked python scripts/validate_annotations.py `
  data/annotations/kurisu_two_volumes_v0.jsonl
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

GPU 训练只需把 `--device cpu` 改成 `--device cuda`，并增加训练轮数。`--history-mode` 的含义：

- `zeros`：每一步历史条件置零，避免把目标标签直接反馈为输入；公开 `NeuroBrainSession` 使用这一模式。
- `provided`：使用 JSONL 中提供的历史条件，适合专门研究历史通道。
- `fixed`：为一条序列重复首个历史条件，属于实验模式。

checkpoint 会保存模型权重、优化器状态、模型配置、协议版本、训练历史、数据路径、随机种子和 history mode。训练产物位于 `artifacts/`，默认不会进入 Git。

## 常见问题

### `uv sync --locked` 失败

先确认 Python、uv 和网络环境，再检查是否有人修改了 `pyproject.toml` 但没有同步 `uv.lock`。如果只是首次安装，重新执行命令即可；不要为了绕过错误删除锁文件。

### CUDA 不可用

使用 `--device cpu` 仍可运行测试、Python facade、角色试玩和小规模训练。只有在需要 GPU 性能时才排查驱动、CUDA wheel 和 `torch.cuda.is_available()`。

### `from_checkpoint()` 报 history mode 错误

公开 facade 把低层 `historical_condition` 固定为全零，因此只接受 `history_mode=zeros` 的训练 checkpoint。需要 `provided` 或 `fixed` 历史条件时，直接使用 `NeuroBrainModel.load_checkpoint()` 和底层 `model.step()`/`model.forward()`。

### 角色试玩无法连接

先单独确认 Ollama/MiniMax endpoint 可用，再检查 `--provider`、`--model`、`--endpoint` 和环境变量。NeuroBrain 的 CfC 推理是本地调用，不依赖外部 LLM；只有角色试玩的文本闭环需要它。

### 进程重启后状态丢失

`NeuroBrainSession` 当前只在内存中保留隐状态，进程退出或 session 从字典中删除后状态会丢失。需要跨进程恢复时，应在宿主层实现会话生命周期和持久化方案；仓库目前没有隐状态序列化协议。

## 下一步阅读

- [Python 推理接口](python-api.md)
- [外部集成](integration.md)
- [输入输出协议 v0.2](io-protocol-v0.2.md)
- [训练说明](training-v0.1.md)
- [项目架构](architecture.md)
