# NeuroBrain

NeuroBrain 是一个独立、可训练、持续运行的人工心理动力系统。它接收当前心理刺激和历史情绪条件，持续演化内部隐状态，并输出心理状态与行为倾向，为虚拟角色、对话系统和 Agent 提供可复用的内部状态层。

项目当前处于模型构建与初始实验阶段，本文档、[核心架构与设计规范](docs/architecture.md) 和 [核心模型技术方案](docs/core-model-technical-plan.md) 构成当前设计基线。

## 第一阶段基线

- 当前心理刺激：5 维，`joy`、`sadness`、`tension`、`curiosity`、`surprise`
- 历史情绪条件：5 维，与心理状态维度相同；第一版使用固定预设值
- 心理状态输出：5 维
- 行为倾向输出：6 维，`engagement`、`avoidance`、`expression`、`exploration`、`initiative`、`activation`
- 输入合计 10 维，输出合计 11 维
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

```powershell
uv sync
uv run pytest
uv run neurobrain
```

当前源码包含第一阶段输入输出协议占位类型，CfC-512 模型尚未实现。下一步请阅读 [docs/core-model-technical-plan.md](docs/core-model-technical-plan.md) 和 [docs/roadmap.md](docs/roadmap.md)。

## 设计边界

NeuroBrain 负责内部状态和行为倾向，不直接替代 LLM，也不直接执行 Live2D、语音或其他外部动作。外部集成层负责将状态交给 LLM 或行为决策模块，再由具体执行器完成动作。
