# 初始架构

## 分层

### NeuroBrain Core

负责刺激接收、内部动力学更新、心理状态解码和行为倾向解码。心理与行为共享同一内部状态，允许使用不同的输出解码器。

### Learning System

负责训练案例、LLM 辅助标注、数据审核、训练、评估、版本管理和回滚。训练与运行时隔离，只有通过验证的模型才能发布。

### Evaluation & Observation

包含互动观察实验室和标准化对照测试。前者帮助发现自然性和异常反应，后者使用固定刺激序列衡量稳定性、历史依赖、恢复性和回归情况。

### Integration Layer

提供与对话系统、AstrBot、AG99、Live2D 或其他 Agent 的标准输入输出接口。Core 不依赖这些具体应用。

## 第一阶段接口草案

输入刺激是一个 5 维、范围为 `[0, 1]` 的向量；输出由 5 维心理状态和 6 维行为倾向组成。接口应逐步补充时间步长、模型版本、事件标识、随机种子和可选调试信息。

```text
Stimulus {
  joy: float
  sadness: float
  tension: float
  curiosity: float
  surprise: float
}

State {
  psychological: { joy, sadness, tension, curiosity, surprise }
  behavioral: { engagement, avoidance, expression,
                exploration, initiative, activation }
}
```

## 算法研究顺序

先实现可解释的状态方程作为基线，再比较简单递归网络、储备池计算和其他适合连续状态演化的模型。每种模型都必须通过相同的标准化测试，避免只凭互动观感判断改进。
