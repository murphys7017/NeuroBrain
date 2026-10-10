# NeuroBrain

### 让角色拥有连续演化的内在状态

NeuroBrain 是一个面向虚拟角色、对话系统与 Agent 的实验性状态引擎。它把当前情绪刺激、历史背景和持续运行的隐状态结合起来，逐步演化出情绪状态与行为倾向，让每次回应都能受到此前经历的影响。

语言模型负责理解文本、扮演角色并组织表达；NeuroBrain 负责维护一条独立、可观察的状态演化过程。两者协作，让角色不只“说得像”，也能在系统内部保留情绪变化的连续性。

> 当前项目处于早期原型与数据验证阶段。它研究的是可计算的角色状态动力学，不宣称模拟真实人类心理或证明机器意识。

## 从对话到状态反馈

```text
用户文本
   ↓
角色模型提取本轮刺激
   ↓
CfC 状态核心结合刺激、历史条件与连续隐状态
   ↓
输出情绪状态与行为倾向
   ↓
角色模型参考状态生成回应
```

CfC 接收的是结构化数值，不直接读取文本。文本理解与语言生成由外部模型完成；NeuroBrain 输出的状态再作为角色模型的上下文信号，形成可检查的反馈闭环。

## 当前原型

- **连续状态演化**：基于 CfC-512，跨对话轮次保留网络隐状态。
- **清晰的输入输出协议**：16 维输入由当前刺激和历史情绪条件各 8 维组成；14 维输出包括 8 维情绪估计与 6 维行为倾向。当前刺激与历史条件保持独立语义，均使用 `0.0–1.0` 数值。
- **端到端角色试玩**：命令行程序连接本地 Ollama 或 MiniMax 模型，展示角色回复、情绪与行为估计、意图解读和分阶段耗时。
- **序列训练工具**：支持 JSONL 序列校验、变长序列批处理与有效时间步掩码，并可保存 checkpoint。
- **可追踪的实验入口**：保留协议、架构说明、训练计划与研究记录，便于复现实验和继续迭代。

目前的主要目标是验证数据协议、状态闭环和训练流程。现有小说标注是模型生成的初始标注，并非人工确认的心理真值；训练冒烟测试也不代表已验证泛化能力。

协议中的情绪维度为 `joy`、`trust`、`fear`、`surprise`、`sadness`、`disgust`、`anger`、`anticipation`；行为倾向为 `engagement`、`avoidance`、`expression`、`exploration`、`initiative`、`activation`。历史情绪条件提供情境背景，不覆盖 CfC 自身持续传递的隐状态。

## 快速开始

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --locked
uv run --locked pytest
uv run --locked python examples/interactive_demo.py
```

项目锁定了 CUDA 12.6 版 PyTorch wheel。首次同步会下载体积较大的 PyTorch 依赖；具备兼容 NVIDIA 驱动时可使用 CUDA，未检测到 CUDA 时可将模型运行在 CPU 上。

### 试用持续对话闭环

角色测试 CLI 需要一个 CfC checkpoint 和可用的语言模型。它默认连接本机 Ollama 的 `qwen3.5:2b`，并默认读取 `artifacts/kurisu-two-volumes-baseline-cfc512.pt`。该 checkpoint 属于本地生成产物，不随仓库分发；如需从头体验闭环，可先运行下方的训练流程冒烟命令，再使用生成的 checkpoint。

```powershell
uv run --locked python scripts/character_test.py --checkpoint artifacts/smoke-cfc512.pt
```

省略 `--message` 会进入持续对话；输入 `/reset` 重置 CfC 隐状态和历史条件，输入 `/quit` 退出。加上 `--show-json` 可以查看完整诊断数据。使用 MiniMax 时，将 Subscription Key 放入环境变量后运行：

```powershell
$env:MINIMAX_API_KEY = "你的 MiniMax Subscription Key"
uv run --locked python scripts/character_test.py --provider minimax --checkpoint artifacts/smoke-cfc512.pt
```

密钥不要写入代码或提交到仓库。MiniMax 默认使用中国站 OpenAI 兼容接口；国际站密钥可通过 `--endpoint https://api.minimax.io/v1/chat/completions` 指定对应接口。

### 校验与训练流程冒烟测试

训练数据采用 JSONL 序列格式。以下命令校验当前示例语料，并运行一个 epoch 来确认数据读取、反向传播和 checkpoint 保存链路可用：

```powershell
uv run --locked python scripts/validate_annotations.py data/annotations/kurisu_two_volumes_v0.jsonl
uv run --locked python scripts/train.py --data data/annotations/kurisu_two_volumes_v0.jsonl --epochs 1 --batch-size 1 --val-ratio 0 --history-mode zeros --output artifacts/smoke-cfc512.pt
```

`--val-ratio 0` 只适用于这次流程检查，不用于模型评估。训练数据包含 36 段序列、1025 个事件步，属于未经独立人工确认的初始标注。上面的命令会生成 `artifacts/smoke-cfc512.pt`，可用于试玩闭环，但单 epoch 冒烟训练不代表模型已经具备可靠效果。默认 `--history-mode zeros` 可避免把前一时刻的目标状态直接作为下一步输入；验证历史条件字段时可显式选择 `--history-mode provided`。

## 文档

- [项目架构](docs/architecture.md)：模块职责与状态演化闭环
- [核心模型技术方案](docs/core-model-technical-plan.md)：模型结构和技术决策
- [CfC-512 第一版建立计划](docs/cfc-v0.1-implementation-plan.md)：当前阶段目标与验收门槛
- [训练说明](docs/training-v0.1.md)：数据格式、历史条件和训练入口
- [项目路线图](docs/roadmap.md)：后续阶段规划

## 设计边界

NeuroBrain 负责连续内部状态与行为倾向，不替代语言模型，不直接决定最终台词，也不执行 Live2D、语音或其他外部动作。集成方可以把它的输出交给语言模型或行为决策模块，再由相应执行器完成动作。情绪和行为标签是建模目标与标注判断，应结合人工复核和独立评估解读。
