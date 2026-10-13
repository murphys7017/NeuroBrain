# CfC-512 第一版建立计划

> **状态说明：** 本计划记录 CfC-512 核心建立阶段，核心模型、协议、测试、checkpoint 和基础实验已经完成。当前的启动、训练和对外调用方式以 [上手与运行](getting-started.md)、[Python 推理接口](python-api.md) 和 [外部集成](integration.md) 为准。

**目标：** 建立一个可以在 CUDA 上运行、协议稳定、支持单步和序列推理的 CfC-512 心理动力核心。

**范围：** 安装运行环境、确定基础协议、编写 CfC 模型。
**不包含：** 真实人格训练、LLM 标注、AG99 集成、Memory 接入和在线学习。

## 总体结构

```text
阶段一：PyTorch + CUDA + ncps 环境
        ↓
阶段二：16 维输入 / 14 维输出协议冻结
        ↓
阶段三：32 维编码器 + CfC-512 + 双输出解码器
```

完成第三阶段后，才进入合成数据训练和模型对照实验。

## 阶段一：安装支持 CUDA 的 PyTorch 环境

### 目标

建立可复现的 UV 环境，确认 PyTorch 能调用 NVIDIA GPU，并锁定 PyTorch、CUDA wheel、`ncps` 和 Python 版本。当前开发机检测到 NVIDIA GeForce RTX 4070，显存约 12 GB；最终安装命令仍以 PyTorch 官方安装选择器生成的 CUDA 组合为准。

### 工作项

1. 使用仓库指定的 UV 托管 Python 3.11，并记录 Python、UV、NVIDIA 驱动和 GPU 信息。
2. 使用 PyTorch 官方安装选择器生成 Windows/Pip/CUDA 安装命令。
3. 将 PyTorch 和 `ncps` 加入项目依赖，执行 `uv sync --locked`。
4. 固定依赖版本并维护 `uv.lock`。
5. 编写环境诊断命令，检查：
   - `torch.__version__`
   - `torch.version.cuda`
   - `torch.cuda.is_available()`
   - GPU 名称
   - CUDA 张量创建与矩阵计算
6. 编写 `ncps.torch.CfC` 最小实例化 smoke test。

### 阶段门槛

以下检查已在当前开发机通过：

```text
torch.cuda.is_available() == True
CUDA 张量计算成功
CfC 可以实例化
CfC 可以完成一次前向传播
uv.lock 与项目配置一致，`uv lock --check` 通过
```

项目固定使用 PyTorch 官方 `cu130` wheel 索引中的 Windows wheel（当前锁定 `torch 2.14.1+cu130`）。实际 CUDA 可用性与设备状态以 `scripts/check_environment.py` 的输出为准。

## 阶段二：确定并冻结基础协议

### 输入

```text
current_stimulus:       [B, T, 8]
historical_condition:   [B, T, 8]
delta_seconds:          [B, T] 或 [B, T, 1]
initial_hidden:         [B, 512]（可选）
```

八维情绪顺序固定为：

```text
joy, trust, fear, surprise, sadness, disgust, anger, anticipation
```

### 输出

```text
emotion:                [B, T, 8]
behavior:               [B, T, 6]
final_hidden:           [B, 512]
```

行为顺序固定为：

```text
engagement, avoidance, expression,
exploration, initiative, activation
```

### 协议规则

- 当前刺激和历史条件必须是两个独立字段。
- 所有外部输入值范围为 `[0, 1]`。
- 情绪和行为输出使用逐维 `sigmoid`，不使用 Softmax。
- 输出各维可以同时为高值，总和不要求为 1。
- 历史条件不得覆盖或替换 CfC 隐状态。
- `delta_seconds` 必须参与 CfC 更新。
- 单步调用和整段序列调用必须使用同一套字段语义。

### 建议的数据类型

```python
@dataclass
class ModelOutput:
    emotion: Tensor
    behavior: Tensor
    final_hidden: Tensor
```

阶段二已完成：协议常量、张量形状、维度顺序和范围检查都有测试，任何维度变更都必须修改协议版本。

## 阶段三：编写 CfC-512 模型

### 模型结构

```text
concat(current_stimulus, historical_condition)  # 16
        ↓
Linear(16, 32) + activation                  # 32
        ↓
ncps.torch.CfC(input_size=32, units=512)     # hidden 512
        ↓
Linear(512, 8) + sigmoid                      # emotion
Linear(512, 6) + sigmoid                      # behavior
```

CfC 使用单层、512 个 units、`batch_first=True`。当前实现通过 `NeuroBrainModel` 暴露序列 `forward`、单步 `step`、零状态 `reset_state` 以及 checkpoint 保存/加载接口。CfC 的时间序列和隐状态接口先在小张量上验证，再接入训练循环。

### 必须提供的接口

```python
class NeuroBrainModel(nn.Module):
    def forward(
        self,
        current_stimulus: Tensor,
        historical_condition: Tensor,
        delta_seconds: Tensor,
        hidden_state: Tensor | None = None,
    ) -> ModelOutput: ...
```

另外提供：

- `step(...)`：处理一个时间步并返回新隐状态；
- `forward(...)`：处理完整序列；
- `reset_state(...)`：创建零状态或指定初始状态；
- `save_checkpoint(...)` / `load_checkpoint(...)`：保存模型参数和协议版本。

模型保存使用 PyTorch `state_dict`，并在 checkpoint 中保存模型配置和协议版本；训练元数据在训练脚本建立后补充。

### 阶段门槛（已完成）

1. 随机输入可以完成 CPU 前向传播。
2. CUDA 可用时模型可以迁移到 GPU 并完成前向传播。
3. 输出形状为 8 维情绪和 6 维行为。
4. 输出值全部位于 `[0, 1]`。
5. 隐状态可以从一次 `step` 传给下一次 `step`。
6. 相同输入在不同初始隐状态下可以产生不同输出。
7. `delta_seconds=0`、`1`、`10` 都能运行。
8. 保存后重新加载，输出在容差范围内一致。
9. CPU 与 CUDA 的结果在允许的浮点误差内一致。

当前实现已通过 CPU 测试、CUDA 前向与反向传播测试，并验证逐步调用和整段序列调用结果一致。

## 阶段三完成后的下一阶段

CfC-512 建立完成后，再开始：

1. 人工构造恢复、累积、顺序和初始状态数据；
2. 完成小数据集过拟合；
3. 加入简单衰减方程和 GRU 基线；
4. 比较 CfC、GRU 和基线的泛化与稳定性；
5. 再设计 LLM 辅助标注的真实训练数据；
6. 最后接入 AG99。

## 第一版交付物

- CUDA 环境诊断脚本；
- 锁定的 PyTorch、CUDA wheel 和 `ncps` 依赖；
- v0.2 输入输出协议实现；
- `NeuroBrainModel` CfC-512 实现；
- 单步与序列推理接口；
- checkpoint 保存与加载；
- 模型结构、设备、形状和状态传递测试；
- 一份可运行的最小示例。

## 参考资料

- [PyTorch 官方 Windows 安装说明](https://pytorch.org/get-started/locally/)
- [PyTorch 官方 CUDA wheel 安装说明](https://pytorch.org/get-started/previous-versions/)
- [`ncps` CfC PyTorch 实现](https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc.py)
- [`ncps` CfC 时间间隔批量广播问题](https://github.com/mlech26l/ncps/issues/83)
