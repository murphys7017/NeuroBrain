# 标注数据目录

- `kurisu_two_volumes_v0.jsonl`：当前训练用核心数据，36 条序列、1025 个事件。
- `kurisu_two_volumes_v0_audit.jsonl`：核心数据的来源行号、证据、覆盖记录和 Luna 复核元数据。
- `kurisu_two_volumes_v0_notes.md`：数据规则和本批次处理说明。

核心 JSONL 是训练输入；审计文件只用于追溯来源和检查标注。两者都不构成人工确认的心理真值或独立测试集。正式评估前，应按完整场景抽样人工复核。

本地复核批次、请求结果、历史备份和未合并候选由 `.gitignore` 排除，不作为正式训练数据，也不会混入核心 JSONL。
