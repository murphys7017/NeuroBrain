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
uv run --locked python scripts/validate_annotations.py data/annotations/kurisu_trial_v0_compact.jsonl
uv run --locked python scripts/train.py --data data/annotations/kurisu_trial_v0_compact.jsonl --epochs 20 --val-ratio 0 --history-mode zeros --output artifacts/smoke-cfc512.pt
```

当前源码包含第一阶段输入输出协议和 `NeuroBrainModel` CfC-512 实现。环境诊断可运行 `uv run python scripts/check_environment.py`；模型结构、阶段门槛和后续训练范围见 [CfC-512 第一版建立计划](docs/cfc-v0.1-implementation-plan.md)。

训练脚本当前用于训练流程冒烟测试。`kurisu_trial_v0_compact.jsonl` 只有一条序列，运行时应关闭验证集；它可以用于确认数据读取、反向传播、损失下降和 checkpoint 保存，不能用于评估泛化能力。默认 `--history-mode zeros`，避免把标注中的前一时刻目标状态直接作为当前输入；如需专门测试历史字段通路，可显式使用 `--history-mode provided`。

## 设计边界

NeuroBrain 负责内部状态和行为倾向，不直接替代 LLM，也不直接执行 Live2D、语音或其他外部动作。外部集成层负责将状态交给 LLM 或行为决策模块，再由具体执行器完成动作。
