# Python 推理接口

`NeuroBrainSession` 是面向应用集成的有状态 Python 接口。每个角色或独立对话创建一个 session；每次只提交当前事件的 8 维刺激，CfC 隐状态由 session 持续保存。情绪和行为可以随时读取，读取不会推进状态。

```python
from neurobrain import NeuroBrainSession

brain = NeuroBrainSession.from_checkpoint("artifacts/my-cfc512.pt")
# 默认在 CPU 推理；需要 CUDA 时可传 device="cuda"。
# brain = NeuroBrainSession.from_checkpoint("artifacts/my-cfc512.pt", device="cuda")

# 每个值范围为 0.0 到 1.0；也可以传入按协议维度排序的 8 项 list/tuple。
brain.update({
    "joy": 0.2,
    "trust": 0.0,
    "fear": 0.0,
    "surprise": 0.1,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.4,
})

state = brain.read()
print(state.as_dict())
print("joy:", brain.read("joy"))
print("avoidance:", brain.read("avoidance"))

# 新角色或新对话开始时清空持续状态。
brain.reset()
```

输入维度名称与范围会严格校验。每次 `update` 恰好处理一个新事件；空闲时反复 `read` 只返回最近一次结果。第一次更新前 `read()` 返回 `None`。快照包含协议版本、步数、8 维 `emotion` 和 6 维 `behavior`，可用 `as_dict()` 转成 JSON 友好的字典。

为保持对外调用简单，当前 facade 将 `historical_condition` 固定为全零，符合项目默认的训练/推理路径；时间间隔作为 session 配置，默认每次更新 `1.0` 秒，可在构造时指定。底层 `NeuroBrainModel` 仍保留显式历史条件、时间间隔和隐状态接口，供序列训练、实验与需要该控制的调用方使用。该 Python facade 不读取文本，也不启动 HTTP 服务。

`from_checkpoint()` 默认将 checkpoint 加载到 CPU，适合 CPU-only 环境；可用 `device="cuda"` 显式选择 CUDA。该 facade 会拒绝标记为 `fixed` 或 `provided` 历史模式训练的 checkpoint，因为它只提供零历史条件。

每个 session 持有独立状态。应用需要按角色/会话隔离创建实例；不要让不同角色共用一个 session。当前输出仍是实验模型估计，需结合独立测试和人工复核解释。
