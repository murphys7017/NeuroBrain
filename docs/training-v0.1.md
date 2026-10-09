# NeuroBrain 训练代码 v0.1

当前训练实现用于验证 CfC-512 的监督学习流程，不代表已经进入正式数据训练阶段。

## 数据输入

训练脚本读取 JSONL。每一行是一条完整序列，事件字段包含：

- `current_stimulus`：8 维当前刺激；
- `historical_emotion`：可位于记录顶层，也可位于每个事件中；
- `emotion_target`：8 维事件后的情绪状态；
- `behavior_target`：6 维事件后的行为倾向；
- `delta_seconds`：当前事件的时间间隔。

加载器会校验字段、维度、数值范围、有限性和重复的 `sequence_id`，并将历史条件广播或保留为逐事件输入。

## 冒烟测试

```powershell
uv run --locked python scripts/validate_annotations.py data/annotations/kurisu_trial_v0_compact.jsonl
uv run --locked python scripts/train.py `
  --data data/annotations/kurisu_trial_v0_compact.jsonl `
  --epochs 20 `
  --val-ratio 0 `
  --history-mode zeros `
  --device cuda `
  --output artifacts/smoke-cfc512.pt
```

`kurisu_trial_v0_compact.jsonl` 只有一条序列，因此这个命令只检查训练流程是否正常运行、损失是否下降以及 checkpoint 是否可以保存。训练脚本会对变长序列补零，并使用有效时间步掩码计算 Smooth L1 损失。

默认使用 `--history-mode zeros`。当前试标文件中的事件级 `historical_emotion` 大多等于前一事件的目标状态；直接将它作为模型输入会把目标信息泄漏到下一步。需要测试历史字段输入通路时，再显式使用 `--history-mode provided`。

## 训练输出

checkpoint 保存模型参数、优化器状态、模型配置、协议版本、训练历史、数据路径、随机种子和历史输入模式。`artifacts/` 已加入 `.gitignore`，训练产物默认只保留在本地。
