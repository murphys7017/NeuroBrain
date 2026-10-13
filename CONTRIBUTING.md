# NeuroBrain 开发说明

## 环境

项目使用 Python 3.11+ 和 uv。首次进入仓库后执行：

```powershell
uv sync --locked
```

运行环境检查：

```powershell
uv run --locked python scripts/check_environment.py
```

## 开发验证

提交代码前至少运行：

```powershell
uv run --locked pytest -q
uv run --locked python -m compileall -q src scripts tests
uv run --locked python scripts/synthetic_benchmark.py `
  --device cpu `
  --train-episodes 2 `
  --validation-episodes 1 `
  --test-episodes 1 `
  --steps 1 `
  --epochs 1 `
  --batch-size 1
```

如果修改了数据协议、模型输入输出或 facade，应额外运行：

```powershell
uv run --locked python scripts/validate_annotations.py `
  data/annotations/kurisu_two_volumes_v0.jsonl
```

## 文档

运行方式、Python 嵌入、网络适配边界和错误映射分别维护在：

- [上手与运行](docs/getting-started.md)
- [Python 推理接口](docs/python-api.md)
- [外部集成](docs/integration.md)
- [输入输出协议 v0.2](docs/io-protocol-v0.2.md)

修改协议维度、字段、范围或 checkpoint 元数据时，必须同步更新代码、测试和协议文档，并说明协议版本是否需要升级。不要把训练产物、API key 或本地中间文件提交到仓库。

## 实验原则

- 使用固定随机种子和可复现命令。
- 保留旧 checkpoint 和基准结果，避免只报告最新一次运行。
- 区分合成数据结论、LLM 标注数据结论和人工评估结论。
- 在线推理不更新模型权重；训练和发布在离线流程完成。
