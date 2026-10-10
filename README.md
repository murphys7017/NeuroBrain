# NeuroBrain

NeuroBrain 是一个独立、可训练、持续运行的人工心理动力系统。它接收当前心理刺激和历史情绪条件，持续演化内部隐状态，并输出心理状态与行为倾向，为虚拟角色、对话系统和 Agent 提供可复用的内部状态层。

项目当前处于模型构建与初始实验阶段，本文档、[核心架构与设计规范](docs/architecture.md) 和 [核心模型技术方案](docs/core-model-technical-plan.md) 构成当前设计基线。

## 第一阶段基线

- 当前心理刺激：8 维，采用 `joy`、`trust`、`fear`、`surprise`、`sadness`、`disgust`、`anger`、`anticipation`
- 历史情绪条件：8 维，与心理状态维度相同；第一版使用固定预设值
- 心理状态输出：8 维
- 行为倾向输出：6 维，`engagement`、`avoidance`、`expression`、`exploration`、`initiative`、`activation`
- 输入合计 16 维，输出合计 14 维
- 状态和条件值暂定归一化到 `0.0` 至 `1.0`

历史条件是独立语义输入，不能覆盖内部隐状态，也不能被重复当作新的刺激事件。长期具体经历由外部 Memory 系统管理，第一阶段不接入真实 Memory。

## 仓库结构

```text
.
├── configs/              # 实验配置与协议示例
├── docs/                 # 项目定位、架构、路线图与研究记录
├── examples/             # 可运行示例
├── src/neurobrain/       # Python 包源码
├── tests/                # 自动化测试
├── pyproject.toml        # 项目元数据与 UV 配置
├── uv.lock               # UV 锁定文件
└── README.md
```

## 开始开发

需要 Python 3.11+ 和 UV。

UV 使用仓库指定的托管 Python 3.11。首次运行 `uv sync --locked` 会下载 CUDA 版 PyTorch wheel（约 2.4 GiB）；需要可用的 NVIDIA 驱动。PyTorch wheel 已包含 CUDA runtime，日常运行无需另装 CUDA Toolkit。

```powershell
uv sync --locked
uv run --locked pytest
uv run --locked neurobrain
uv run --locked python scripts/check_environment.py
uv run --locked python examples/cfc_smoke.py
uv run --locked python examples/interactive_demo.py
uv run --locked python scripts/validate_annotations.py data/annotations/kurisu_two_volumes_v0.jsonl
uv run --locked python scripts/train.py --data data/annotations/kurisu_two_volumes_v0.jsonl --epochs 1 --batch-size 1 --val-ratio 0 --history-mode zeros --output artifacts/smoke-cfc512.pt
uv run --locked python scripts/character_test.py --message "你好，今天想聊什么？"
```

当前源码包含第一阶段输入输出协议和 `NeuroBrainModel` CfC-512 实现。环境诊断可运行 `uv run python scripts/check_environment.py`；模型结构、阶段门槛和后续训练范围见 [CfC-512 第一版建立计划](docs/cfc-v0.1-implementation-plan.md)。

训练脚本当前用于训练流程冒烟测试。`kurisu_two_volumes_v0.jsonl` 是合并后的小说批量初标序列（当前 36 段、1025 个事件），命令使用 `--val-ratio 0` 只为检查数据读取、反向传播和 checkpoint 保存，不用于评估泛化能力。默认 `--history-mode zeros`，避免把标注中的前一时刻目标状态直接作为当前输入；如需专门测试历史字段通路，可显式使用 `--history-mode provided`。这些标注不是人工确认的心理真值。

`scripts/character_test.py` 是端到端试玩入口。每轮角色模型先为 NeuroBrain 生成本轮情绪刺激 `tendency`，按 0–10 归一化到 0–1；CfC 根据刺激和连续隐状态输出情绪与行为估计，角色模型再参考这些估计形成实际回复，并解释意图与心态。CLI 默认展示用户输入、角色回应、当前心理活动估计和分阶段耗时；`--show-json` 或 `--output` 可查看完整诊断数据，包括供 CfC 使用的模型候选、CfC 输入输出和定时信息。默认使用本地 Ollama 的 `qwen3.5:2b`；也可选择 MiniMax `MiniMax-M2.7`。`--history-mode previous_output` 可测试历史情绪条件通路。

直接省略 `--message` 就会进入持续对话模式。程序默认把最近 8 轮用户消息和最终角色回复交给小模型，同时保留 CfC 隐状态；输入 `/reset` 清空整段对话和脑状态，输入 `/quit` 退出。`--conversation-turns 0` 可关闭文字历史，只测试 NeuroBrain 的连续状态。

使用 MiniMax Token Plan 时，把订阅页提供的 Subscription Key（通常以 `sk-cp-` 开头）放在环境变量中；它和按量 API Key 不通用。不要写进代码或提交到仓库：

```powershell
$env:MINIMAX_API_KEY = "你的 MiniMax Subscription Key"
uv run python scripts/character_test.py --provider minimax
```

MiniMax 模式每轮会进行两次模型调用（生成 CfC 情绪输入、根据 CfC 生成角色回应）；可用 `--model`、`--endpoint`、`--api-key-env` 调整模型、接口地址和密钥变量名。Ollama 保持为默认 provider，也可通过 `--provider ollama` 显式选择。

MiniMax 默认走中国站 Token Plan 的 OpenAI 兼容入口 `https://api.minimax.cn/v1/chat/completions`。如果 Subscription Key 是从国际站获取的，可显式指定 `--endpoint https://api.minimax.io/v1/chat/completions`；密钥和接口区域需要匹配。

## 设计边界

NeuroBrain 负责内部状态和行为倾向，不直接替代 LLM，也不直接执行 Live2D、语音或其他外部动作。外部集成层负责将状态交给 LLM 或行为决策模块，再由具体执行器完成动作。
