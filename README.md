# NeuroBrain

NeuroBrain 是一个独立、可训练、持续运行的人工神经状态与行为引擎。它接收结构化外部刺激，持续演化内部状态，并输出心理状态与行为倾向，为虚拟角色、对话系统和 Agent 提供可复用的内部状态层。

项目当前处于**概念设计与初期研究阶段**。仓库先固定第一阶段的状态协议和实验边界，算法、网络结构与训练方法通过后续实验逐步确定。

## 第一阶段基线

- 输入：5 维情绪刺激：`joy`、`sadness`、`tension`、`curiosity`、`surprise`
- 心理状态输出：5 维，与输入维度对应
- 行为倾向输出：6 维：`engagement`、`avoidance`、`expression`、`exploration`、`initiative`、`activation`
- 所有状态值暂定归一化到 `0.0` 至 `1.0`
- 目标：验证状态持续演化、历史依赖、训练改善和标准接口调用

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

需要 Python 3.11+ 和 [UV](https://docs.astral.sh/uv/)。

```powershell
uv sync
uv run pytest
uv run neurobrain
```

当前源码只包含状态协议占位类型，不代表最终的神经动力学实现。请先阅读 [docs/architecture.md](docs/architecture.md) 和 [docs/roadmap.md](docs/roadmap.md)。

## 设计边界

NeuroBrain 负责内部状态和行为倾向，不直接替代 LLM，也不直接执行 Live2D、语音或其他外部动作。外部集成层负责把状态交给 LLM 或行为决策模块，再由具体执行器完成动作。

## 许可证

许可证尚未确定。项目早期研究材料和代码在许可证确定前按项目参与者约定使用。
