#!/usr/bin/env python3
"""Prepare JSONL batches and a strict review prompt for a Luna pass.

This script does not call a model. It creates review-ready inputs so a Luna
endpoint or an external batch runner can process them without changing the
training schema. Model output is expected to contain one JSON object per line
with the same sequence_id and revised events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT = r'''你是 NeuroBrain 的第二轮语义复核器。输入是一段小说连续场景和当前批量初标。

任务：
1. 识别红莉栖（牧濑红莉栖、牧濑、红莉栖、助手、克里斯蒂娜、Christina、Kurisu 等）的直接出场、直接台词、明确回忆和可由上下文确认的指代。
2. 在不把其他角色情绪转给红莉栖的前提下，把长段拆成有情绪变化意义的事件；只有名字重复、称呼重复或没有新刺激的句子应合并或删除。
3. 修正 historical_emotion、current_stimulus、emotion_target、behavior_target。所有值必须在 [0,1]；没有证据的情绪维度填 0.0，行为倾向填 0.5；禁止 null。
4. historical_emotion 是事件发生前状态；后续事件默认继承上一事件 emotion_target，只有明确恢复、时间间隔或状态转折才修改。
5. current_stimulus 不能直接复制 emotion_target；目标状态应反映累积、延迟和恢复。
6. 只保留红莉栖相关事件。若只是别人谈论红莉栖且无法判断她的状态，可以保留为间接事件，但全部无法证实的情绪维度必须为 0.0，并在 audit_note 说明 indirect。

输出严格为一个 JSON 对象，不要 Markdown：
{"sequence_id":"原ID","target_character":"牧濑红莉栖","events":[...],"review":{"changed":true,"confidence":"high|medium|low","audit_note":"...","source_lines":[起始行,结束行]}}

events 只能包含 text、historical_emotion、current_stimulus、emotion_target、behavior_target、delta_seconds；不得增加字段。'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('--out-dir', type=Path, default=Path('data/annotations/luna_review_batches'))
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'SYSTEM_PROMPT.txt').write_text(PROMPT, encoding='utf-8')
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        out = args.out_dir / f'batch_{start // args.batch_size + 1:03d}.jsonl'
        out.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in batch), encoding='utf-8')
    manifest = {
        'source': str(args.input),
        'sequences': len(rows),
        'batch_size': args.batch_size,
        'batches': (len(rows) + args.batch_size - 1) // args.batch_size,
        'expected_output': 'one reviewed JSON object per input sequence, preserving sequence_id',
        'merge_policy': 'review output replaces matching sequence_id only after validation; no automatic overwrite',
    }
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
