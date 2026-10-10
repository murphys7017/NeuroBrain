#!/usr/bin/env python3
"""Generate the incremental Kurisu annotations from the three Underling volumes.

The source novels stay outside the repository.  This script writes a compact
training JSONL file and a separate evidence/audit JSONL file so the draft can
be reviewed before it is merged into a larger corpus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "annotations"
DATA_PATH = OUT_DIR / "kurisu_extra_underling_bonus_draft.jsonl"
AUDIT_PATH = OUT_DIR / "kurisu_extra_underling_bonus_draft_audit.jsonl"

EMOTION = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)
BEHAVIOR = (
    "engagement", "avoidance", "expression",
    "exploration", "initiative", "activation",
)


def vec(names: Iterable[str], values: Iterable[float]) -> dict[str, float]:
    return {name: round(float(value), 2) for name, value in zip(names, values)}


def emotion(*values: float) -> dict[str, float]:
    return vec(EMOTION, values)


def behavior(*values: float) -> dict[str, float]:
    return vec(BEHAVIOR, values)


def make_events(specs: list[dict], initial_history: dict[str, float] | None = None) -> list[dict]:
    previous = initial_history or emotion(0, 0, 0, 0, 0, 0, 0, 0)
    events: list[dict] = []
    for index, spec in enumerate(specs):
        history = spec.pop("historical_emotion", None) or previous
        event = {
            "text": spec["text"],
            "historical_emotion": history,
            "current_stimulus": spec["current_stimulus"],
            "emotion_target": spec["emotion_target"],
            "behavior_target": spec["behavior_target"],
            "delta_seconds": float(spec.get("delta_seconds", 0.0 if index == 0 else 1.0)),
        }
        events.append(event)
        previous = event["emotion_target"]
    return events


def sequence(sequence_id: str, specs: list[dict], initial_history: dict[str, float] | None = None) -> dict:
    return {
        "sequence_id": sequence_id,
        "target_character": "牧濑红莉栖",
        "events": make_events(specs, initial_history),
    }


def audit(
    sequence_id: str,
    source_file: str,
    chapter: str,
    source_lines: list[int],
    events: list[tuple[list[int], str, str]],
    context_note: str,
    timing_note: str,
) -> dict:
    return {
        "record_type": "included_sequence",
        "sequence_id": sequence_id,
        "target_character": "牧濑红莉栖",
        "source_file": source_file,
        "chapter": chapter,
        "source_lines": source_lines,
        "context_note": context_note,
        "events": [
            {
                "event_index": index,
                "source_lines": lines,
                "evidence_summary": evidence,
                "uncertainty": uncertainty,
            }
            for index, (lines, evidence, uncertainty) in enumerate(events)
        ],
        "timing_note": timing_note,
    }


def main() -> None:
    zero = emotion(0, 0, 0, 0, 0, 0, 0, 0)
    records: list[dict] = []
    audits: list[dict] = []

    # 第一卷：实验室聚会和 D-Mail 异常。
    sid = "underling_v1_lab_party_calm_001"
    specs = [
        {
            "text": "红莉栖在实验室聚会中看到冈部失控，先用带讽刺的语气回应他的中二发言。",
            "current_stimulus": emotion(.05, .05, .10, .20, 0, .05, .15, .10),
            "emotion_target": emotion(.15, .25, .12, .25, .05, .05, .20, .25),
            "behavior_target": behavior(.65, .15, .70, .45, .55, .50),
        },
        {
            "text": "冈部扬言用道具破坏宴会，红莉栖递饮料并直接叫他冷静。",
            "current_stimulus": emotion(0, .05, .20, .25, 0, .05, .35, .05),
            "emotion_target": emotion(.10, .30, .25, .25, .05, .05, .35, .20),
            "behavior_target": behavior(.75, .10, .80, .40, .70, .65),
        },
        {
            "text": "她说明宴会是冈部提出的庆祝新道具完成，并追问他为何忘记。",
            "current_stimulus": emotion(.05, .10, .10, .35, .05, 0, .05, .25),
            "emotion_target": emotion(.15, .40, .20, .45, .05, 0, .15, .50),
            "behavior_target": behavior(.85, .05, .75, .75, .65, .60),
        },
        {
            "text": "红莉栖与真由理、琉华子谈论食物，短暂回到轻松的实验室交流。",
            "current_stimulus": emotion(.25, .25, 0, .05, 0, 0, 0, .25),
            "emotion_target": emotion(.35, .50, .10, .35, .02, 0, .05, .55),
            "behavior_target": behavior(.85, .05, .80, .60, .65, .60),
        },
    ]
    records.append(sequence(sid, specs, zero))
    audits.append(audit(
        sid, "第一卷.txt", "实验室聚会", [307, 439],
        [
            ([307, 313], "红莉栖出场并以助手、克莉丝汀娜等称呼与冈部斗嘴。", "直接台词支持轻微烦躁和熟悉的社交互动，具体强度为估计。"),
            ([419, 425], "冈部扬言用道具破坏聚会，红莉栖递上饮料并要求他冷静。", "焦躁和控制局面有直接台词支持；恐惧保持低值。"),
            ([427, 439], "她解释聚会由冈部为新道具完成而召集，并指出他似乎忘记了。", "惊讶、探究和期待由问答内容推断。"),
            ([537, 538], "她和真由理、琉华子讨论食物，场面恢复轻松。", "轻松气氛有对话支持，情绪数值仍属标注估计。"),
        ],
        "同一实验室聚会内只保留红莉栖直接说话或可观察反应；没有把冈部的世界线记忆差异直接当作红莉栖情绪。",
        "原文无逐事件时间戳；连续对话使用 1.0 秒占位，首步 0.0。",
    ))

    sid = "underling_v1_dmail_anomaly_001"
    specs = [
        {
            "text": "红莉栖与桥田尝试设置向遥远过去发送 D-Mail 的参数，专注验证装置可行性。",
            "current_stimulus": emotion(0, .20, .05, .10, 0, 0, 0, .55),
            "emotion_target": emotion(.05, .45, .10, .15, .02, 0, .02, .75),
            "behavior_target": behavior(.90, .05, .45, .95, .80, .70),
        },
        {
            "text": "电话微波炉突然放电，红莉栖和桥田慌张地试图阻止可能发送到远古的 D-Mail。",
            "current_stimulus": emotion(0, 0, .80, .75, .05, 0, .15, .10),
            "emotion_target": emotion(0, .25, .75, .70, .15, 0, .20, .20),
            "behavior_target": behavior(.80, .10, .70, .65, .70, .95),
        },
        {
            "text": "世界线变化后，红莉栖焦急询问冈部是否发送了 D-Mail，并质疑实验危险性。",
            "current_stimulus": emotion(0, 0, .45, .55, .10, .05, .35, .15),
            "emotion_target": emotion(.05, .30, .55, .55, .20, .05, .40, .30),
            "behavior_target": behavior(.85, .15, .90, .75, .70, .80),
        },
        {
            "text": "听完冈部对放电和世界变化的解释后，她抬头整理改变前后的共通点与差异。",
            "current_stimulus": emotion(.05, .20, .15, .35, 0, 0, 0, .45),
            "emotion_target": emotion(.10, .45, .35, .45, .10, 0, .10, .65),
            "behavior_target": behavior(.90, .05, .65, .95, .80, .70),
        },
        {
            "text": "她提醒改变过去很危险，继续追问冈部是否真的相信 Reading Steiner 的说法。",
            "current_stimulus": emotion(0, 0, .30, .20, .10, .05, .25, .10),
            "emotion_target": emotion(.05, .35, .50, .25, .20, .05, .30, .45),
            "behavior_target": behavior(.75, .20, .80, .80, .70, .60),
        },
        {
            "text": "确认新世界线仍保留部分共同事实后，红莉栖暂时接受 D-Mail 造成变化的可能。",
            "current_stimulus": emotion(.05, .35, .10, .20, 0, 0, 0, .40),
            "emotion_target": emotion(.12, .50, .25, .25, .08, 0, .05, .70),
            "behavior_target": behavior(.90, .05, .65, .90, .80, .70),
        },
        {
            "text": "她吐槽 Reading Steiner 的命名并短暂回避冈部的视线，关系恢复到熟悉的斗嘴。",
            "current_stimulus": emotion(.05, .10, .10, .15, 0, .05, .20, .05),
            "emotion_target": emotion(.15, .35, .20, .20, .10, .05, .25, .45),
            "behavior_target": behavior(.65, .15, .75, .70, .50, .50),
        },
    ]
    records.append(sequence(sid, specs, zero))
    audits.append(audit(
        sid, "第一卷.txt", "D-Mail 异常与世界线变化", [503, 509, 579, 629],
        [
            ([503, 505], "红莉栖与桥田为向遥远过去发送 D-Mail 进行设置。", "专注、探索和期待由实验动作推断。"),
            ([507, 509], "电话微波炉突然放电，两人慌张地试图停止装置。", "突发放电直接支持高惊讶和高恐惧刺激。"),
            ([577, 583], "红莉栖焦急追问实验内容及是否发送 D-Mail，并强调改变过去的危险。", "焦急和警惕有直接台词支持，强度为估计。"),
            ([601, 609], "她把改变前后的过去进行对照，整理共通点和差异。", "分析行为明确；情绪向理性探究回落。"),
            ([615, 621], "她质疑 Reading Steiner 的特殊性及命名，继续讨论可信度。", "怀疑、烦躁和探索由直接问句推断。"),
            ([621, 625], "在冈部坚持超自然解释后，她暂时接受需要继续分析的事实。", "接受程度和期待为谨慎估计。"),
            ([627, 629], "她以熟悉的网络用语吐槽并回避目光。", "轻微尴尬、烦躁和社交表达有直接互动支持。"),
        ],
        "同一场景中将即时放电、世界线解释和后续理论讨论拆成连续事件；不把冈部的眩晕直接赋予红莉栖。",
        "原文没有秒级间隔；首步 0.0，其后连续实验对话使用 1.0 秒占位。",
    ))

    sid = "underling_v1_timeleap_machine_001"
    specs = [
        {
            "text": "红莉栖独自在安静的 Lab 等待冈部，表示他差点迟到并暗示时间已经接近。",
            "historical_emotion": emotion(.10, .35, .10, .10, .25, 0, .05, .70),
            "current_stimulus": emotion(0, .05, .25, .15, .15, 0, .10, .55),
            "emotion_target": emotion(.05, .40, .35, .20, .35, 0, .10, .75),
            "behavior_target": behavior(.75, .25, .65, .55, .70, .55),
        },
        {
            "text": "她宣布十三号机已与电话微波炉联动，能够把记忆送回过去，时间跳跃机完成。",
            "current_stimulus": emotion(.25, .30, .05, .45, 0, 0, 0, .80),
            "emotion_target": emotion(.35, .60, .10, .55, .05, 0, .02, .90),
            "behavior_target": behavior(.95, .05, .85, .98, .90, .80),
        },
        {
            "text": "红莉栖解释把海马记忆压缩到极小容量、借助量子通讯线路完成时间跳跃的原理。",
            "current_stimulus": emotion(.10, .20, 0, .25, 0, 0, 0, .65),
            "emotion_target": emotion(.40, .65, .05, .35, .03, 0, 0, .85),
            "behavior_target": behavior(.98, .03, .80, .98, .90, .75),
        },
        {
            "text": "她邀请冈部试用机器，面对他拿旧道具故障找茬时仍保持耐心。",
            "current_stimulus": emotion(.05, .05, .10, .15, 0, .05, .15, .35),
            "emotion_target": emotion(.30, .55, .15, .25, .08, .05, .15, .70),
            "behavior_target": behavior(.85, .10, .70, .80, .75, .65),
        },
        {
            "text": "谈到十二号机和两人无法取下腕带时，红莉栖情绪动摇却强作微笑。",
            "current_stimulus": emotion(0, 0, .15, .35, .45, .05, .20, .15),
            "emotion_target": emotion(.20, .42, .25, .45, .55, .10, .25, .50),
            "behavior_target": behavior(.50, .30, .75, .50, .40, .55),
        },
        {
            "text": "时间逼近八点半，她绷紧表情催促冈部立即戴上耳机。",
            "current_stimulus": emotion(0, .05, .45, .35, .05, 0, .10, .65),
            "emotion_target": emotion(.08, .45, .55, .40, .25, 0, .15, .85),
            "behavior_target": behavior(.80, .10, .80, .60, .85, .90),
        },
        {
            "text": "她说明即使失败也会等待并一次次重来，要求冈部继承自己的意志。",
            "current_stimulus": emotion(0, .35, .25, .20, .20, 0, .05, .55),
            "emotion_target": emotion(.20, .75, .45, .20, .40, 0, .10, .85),
            "behavior_target": behavior(.90, .05, .90, .65, .98, .65),
        },
        {
            "text": "红莉栖温柔地给冈部戴上耳机并启动时间跳跃机。",
            "current_stimulus": emotion(0, .10, .65, .70, .05, 0, .10, .45),
            "emotion_target": emotion(.10, .65, .75, .65, .45, .05, .15, .65),
            "behavior_target": behavior(.75, .15, .70, .35, .95, .98),
        },
        {
            "text": "收到威胁邮件后，她没有解释，只微笑道别并承诺一直等待冈部回忆起来。",
            "current_stimulus": emotion(0, .25, .55, .40, .50, .05, .10, .35),
            "emotion_target": emotion(.12, .85, .70, .45, .65, .05, .12, .60),
            "behavior_target": behavior(.75, .20, .80, .50, .95, .70),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第一卷.txt", "时间跳跃机完成与送回冈部", [7097, 7291],
        [
            ([7105, 7113], "红莉栖独自等候冈部，表示时间已接近且他差点迟到。", "等待、疲惫和不安由环境及台词推断。"),
            ([7127, 7135], "她报告十三号机完成并明确说明时间跳跃机的用途。", "成就感、期待和紧张由报告行为推断。"),
            ([7165, 7180], "她解释记忆容量、量子通讯和 SERN 线路如何支撑时间跳跃。", "科学投入有大量直接说明支持，具体情绪强度仍为估计。"),
            ([7187, 7207], "她邀请冈部试用，耐心回应他对旧道具故障的质疑。", "耐心和主动性由连续对话支持。"),
            ([7215, 7223], "谈到两人无法取下腕带时，她首次显露悲伤并强作微笑。", "悲伤和动摇有直接叙述支持。"),
            ([7227, 7235], "时间将到，她绷紧表情催促冈部戴上耳机。", "紧迫、恐惧与行动意愿由动作和目光描写支持。"),
            ([7239, 7259], "她承诺失败后仍会等待和重来，要求冈部继承自己的意志。", "承诺和决心有直接台词支持；悲伤强度为估计。"),
            ([7259, 7261], "红莉栖温柔替冈部戴好耳机并启动机器。", "保护性行动有直接叙述支持。"),
            ([7283, 7291], "收到威胁邮件后，她微笑道别，承诺等待冈部回忆起来。", "威胁刺激明确；克制和承诺由台词及表情支持。"),
        ],
        "首步历史状态参考她独自等待和持续开发的前文；后续按上一事件目标状态继承。",
        "原文无精确秒数；连续说明和操作使用 1.0 秒占位，首步 0.0。",
    ))

    # 第二卷：雨夜、彻夜开发、电话追问和桥上救援。
    sid = "underling_v2_rain_and_overnight_development_001"
    specs = [
        {
            "text": "雨后楼梯上，红莉栖抱着熟睡的真由理，身体发抖却否认寒冷。",
            "historical_emotion": emotion(.10, .45, .15, .15, .25, .05, .15, .55),
            "current_stimulus": emotion(0, .15, .30, .20, .10, 0, .10, .10),
            "emotion_target": emotion(.05, .45, .40, .25, .35, .05, .15, .35),
            "behavior_target": behavior(.55, .35, .35, .25, .40, .45),
        },
        {
            "text": "她避开冈部视线，继续抚摸真由理的头发，语气生硬而不愿说明原因。",
            "current_stimulus": emotion(0, 0, .25, .15, .20, .05, .15, .05),
            "emotion_target": emotion(.03, .35, .45, .20, .50, .08, .25, .20),
            "behavior_target": behavior(.45, .55, .30, .25, .30, .40),
        },
        {
            "text": "红莉栖短暂注视冈部肩膀，吞咽后转开脸，并用平时做坏事的玩笑解释大雨。",
            "current_stimulus": emotion(.05, .15, .15, .30, .05, 0, .10, .15),
            "emotion_target": emotion(.10, .45, .35, .35, .35, .03, .20, .30),
            "behavior_target": behavior(.55, .40, .45, .45, .45, .45),
        },
        {
            "text": "彻夜开发十三号机后，红莉栖端着咖啡从开发室出来，脸色疲倦而阴沉。",
            "delta_seconds": 28800.0,
            "current_stimulus": emotion(0, 0, .10, .15, .35, .05, .10, .15),
            "emotion_target": emotion(.05, .30, .35, .30, .55, .10, .20, .35),
            "behavior_target": behavior(.60, .25, .45, .80, .55, .60),
        },
        {
            "text": "听到冈部想把 IBN5100 当交易工具，她用锋利的目光责备他的盘算。",
            "current_stimulus": emotion(0, 0, .05, .25, .05, .25, .60, .05),
            "emotion_target": emotion(.03, .22, .30, .40, .30, .35, .65, .20),
            "behavior_target": behavior(.70, .20, .85, .70, .65, .65),
        },
        {
            "text": "她指出铃羽只是需要情报而不会期待冈部，并对他的自信叹气。",
            "current_stimulus": emotion(0, .05, .05, .15, .10, .10, .25, .10),
            "emotion_target": emotion(.05, .30, .25, .30, .35, .15, .40, .30),
            "behavior_target": behavior(.65, .25, .65, .75, .55, .50),
        },
        {
            "text": "冈部询问她是否继续十三号机开发，红莉栖短暂沉默后说这是自己的战斗。",
            "current_stimulus": emotion(0, 0, .10, .35, .15, .05, .20, .25),
            "emotion_target": emotion(.05, .30, .30, .45, .40, .12, .35, .55),
            "behavior_target": behavior(.75, .20, .70, .80, .65, .55),
        },
        {
            "text": "她收紧眼睛，表示绝对要做点什么给冈部看，重新把注意力投入开发。",
            "current_stimulus": emotion(0, .10, .10, .10, .05, 0, .05, .55),
            "emotion_target": emotion(.10, .40, .25, .20, .25, .05, .30, .85),
            "behavior_target": behavior(.90, .10, .80, .85, .95, .75),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第二卷.txt", "雨夜与彻夜开发后的十三号机", [2020, 2209],
        [
            ([2020, 2026], "雨后楼梯上她身体发抖，却否认寒冷并照顾熟睡的真由理。", "疲惫、压抑和照顾行为有直接叙述支持。"),
            ([2028, 2048], "她避开冈部视线，观察后又用玩笑掩饰情绪。", "回避与掩饰有动作和台词支持，具体情绪为估计。"),
            ([2123, 2125], "彻夜开发后的红莉栖从开发室出来，脸色明显疲倦。", "疲劳和低落有直接外貌描写支持。"),
            ([2127, 2139], "她以锋利眼神责备冈部把 IBN5100 当交易工具。", "愤怒、厌恶和表达行为有直接对话支持。"),
            ([2141, 2155], "她解释铃羽真正需要的是情报，并认为冈部的交易计划是多余的。", "理性分析和轻度烦躁由台词支持。"),
            ([2175, 2195], "面对冈部提出帮忙开发的说法，她沉默并拒绝把工作交给他。", "沉默和拒绝有直接台词支持；悲伤和不安为估计。"),
            ([2199, 2203], "红莉栖明确说这是自己的战斗。", "自我承担和界限表达有直接台词支持。"),
            ([2205, 2209], "她表示绝对要做出成果，重新投入十三号机开发。", "决心和主动性由动作、目光及台词支持。"),
        ],
        "2020–2048 的雨夜与 2125 之后的次日清晨由文本章节时间切换分隔；中间使用约 8 小时占位，历史情绪保留疲惫与未完成任务的承接。",
        "同一条记录包含明确的过夜时间跳跃；雨夜连续事件使用 1.0 秒，清晨首个事件使用 28800 秒占位，其后使用 1.0 秒。",
    ))

    sid = "underling_v2_timeleap_phone_inquiry_001"
    specs = [
        {
            "text": "红莉栖在电话中告知冈部时间跳跃机完成，并先确认他所在的位置。",
            "historical_emotion": emotion(.05, .35, .20, .10, .20, .05, .10, .65),
            "current_stimulus": emotion(.10, .20, .05, .35, 0, 0, 0, .65),
            "emotion_target": emotion(.20, .50, .15, .40, .10, 0, .05, .80),
            "behavior_target": behavior(.85, .10, .75, .85, .80, .65),
        },
        {
            "text": "得知冈部正在跟踪真由理，她决定赶去会合，并用玩笑抱怨自己的彻夜努力。",
            "current_stimulus": emotion(.20, .25, .05, .25, .05, 0, .10, .45),
            "emotion_target": emotion(.30, .60, .10, .30, .12, 0, .15, .75),
            "behavior_target": behavior(.90, .05, .80, .75, .85, .65),
        },
        {
            "text": "红莉栖压低声音追问真由理究竟想去哪里，显出担忧和追踪意愿。",
            "current_stimulus": emotion(0, .05, .35, .25, .15, 0, .05, .50),
            "emotion_target": emotion(.15, .45, .45, .30, .25, 0, .10, .75),
            "behavior_target": behavior(.85, .20, .70, .65, .75, .70),
        },
        {
            "text": "她提出失败的 D-Mail 可能发生在电话微波炉只放电却没有产生结果的时间带。",
            "current_stimulus": emotion(.05, .15, .15, .45, .05, 0, 0, .60),
            "emotion_target": emotion(.20, .55, .30, .55, .12, 0, .05, .80),
            "behavior_target": behavior(.90, .05, .75, .95, .85, .75),
        },
        {
            "text": "红莉栖推算远距离 D-Mail 需要长时间放电，并认真接受高电费的解释。",
            "current_stimulus": emotion(.05, .20, .10, .25, 0, 0, 0, .55),
            "emotion_target": emotion(.15, .55, .25, .35, .10, 0, .03, .80),
            "behavior_target": behavior(.95, .05, .75, .95, .90, .70),
        },
        {
            "text": "她看到时间跳跃机后提出先在未来观察放电，再回到过去发送邮件的办法。",
            "current_stimulus": emotion(.10, .15, .05, .35, 0, 0, 0, .80),
            "emotion_target": emotion(.25, .60, .15, .45, .05, 0, .02, .90),
            "behavior_target": behavior(.98, .03, .80, .98, .95, .80),
        },
        {
            "text": "红莉栖叹气，认为偶然事件迟早会发生，不必追究所有未知原因。",
            "current_stimulus": emotion(0, 0, .10, .20, .10, 0, 0, .10),
            "emotion_target": emotion(.10, .45, .20, .25, .20, 0, .03, .55),
            "behavior_target": behavior(.80, .10, .65, .75, .65, .45),
        },
        {
            "text": "她说明琉华子母亲的 D-Mail 不是世界线改变的直接原因，但验证资料仍有价值。",
            "current_stimulus": emotion(.05, .15, .10, .20, 0, 0, 0, .50),
            "emotion_target": emotion(.15, .55, .20, .30, .08, 0, .02, .70),
            "behavior_target": behavior(.90, .05, .75, .95, .90, .65),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第二卷.txt", "时间跳跃机完成后的电话追问", [4455, 4556],
        [
            ([4455, 4469], "红莉栖在电话中报告时间跳跃机完成，并决定赶往冈部所在位置。", "电话内容和主动前往有直接台词支持。"),
            ([4471, 4475], "她以彻夜努力为由开玩笑抱怨冈部等人去泡温泉。", "熟悉的玩笑语气支持轻度喜悦和信任。"),
            ([4513, 4523], "她压低声音追问真由理的目的地。", "担忧和行动意愿有直接问句支持。"),
            ([4500, 4508], "她解释要验证过去 D-Mail 是否实际发送，并提出失败放电时间带的假设。", "科学探究和惊讶由直接说明支持。"),
            ([4518, 4526], "她推算远距离邮件可能需要连续放电，并认真解释电费异常。", "推理和专注有直接台词支持。"),
            ([4532, 4540], "她由时间跳跃机联想到在未来观察、回到过去定点发送邮件。", "探索和主动性有连续推理支持。"),
            ([4542, 4544], "她叹气并认为所有可能事件终会发生，无需追问不明原因。", "克制和轻度悲观来自直接叹气及台词。"),
            ([4550, 4556], "她区分直接原因和有价值的实验资料，继续让琉华子参与验证。", "理性收束和协作意愿有直接对话支持。"),
        ],
        "序列从同一通电话和实验讨论中抽取；远处场景仍以连续对话占位，不把冈部的主观回忆当作红莉栖状态。",
        "原文无逐事件时间戳；电话和实验室连续讨论使用 1.0 秒占位，首步 0.0。",
    ))

    sid = "underling_v2_bridge_rescue_001"
    specs = [
        {
            "text": "红莉栖在桥上找到沉沦的冈部，直接叫他跟自己走。",
            "historical_emotion": emotion(0, .15, .45, .30, .60, .10, .20, .45),
            "current_stimulus": emotion(0, .10, .55, .40, .15, 0, .15, .55),
            "emotion_target": emotion(.02, .40, .65, .45, .65, .05, .20, .65),
            "behavior_target": behavior(.80, .10, .80, .40, .85, .85),
        },
        {
            "text": "她告诉冈部真由理、桥田和时间跳跃机都在等，劝他立刻回去。",
            "current_stimulus": emotion(.05, .35, .30, .20, .05, 0, 0, .70),
            "emotion_target": emotion(.08, .65, .45, .25, .45, .02, .08, .80),
            "behavior_target": behavior(.90, .05, .80, .45, .90, .80),
        },
        {
            "text": "红莉栖抓住冈部的脸，带着可怕的笑容宣布这是始于他的战斗。",
            "current_stimulus": emotion(0, .05, .35, .45, .25, .10, .55, .55),
            "emotion_target": emotion(.02, .45, .60, .55, .55, .20, .65, .75),
            "behavior_target": behavior(.90, .05, .95, .40, .95, .90),
        },
        {
            "text": "她说即使杀了冈部也要把他带回去，强硬决心压过了温和语气。",
            "current_stimulus": emotion(0, 0, .45, .50, .20, .05, .70, .35),
            "emotion_target": emotion(0, .30, .70, .65, .50, .20, .80, .65),
            "behavior_target": behavior(.85, .05, .98, .30, .98, .98),
        },
        {
            "text": "红莉栖扶着冈部回到 Lab，确认桥田和铃羽已经把时间跳跃机设好。",
            "current_stimulus": emotion(.10, .30, .20, .20, 0, 0, 0, .65),
            "emotion_target": emotion(.10, .70, .35, .25, .25, .05, .10, .80),
            "behavior_target": behavior(.90, .05, .80, .55, .90, .75),
        },
        {
            "text": "听到车辆冲近，她在门口责骂冈部快走，推动他进入实验室。",
            "current_stimulus": emotion(0, 0, .60, .55, .10, 0, .35, .40),
            "emotion_target": emotion(.02, .45, .70, .60, .35, .08, .45, .55),
            "behavior_target": behavior(.85, .10, .95, .35, .95, .98),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第二卷.txt", "桥上劝回冈部", [4951, 5007],
        [
            ([4951, 4965], "红莉栖出声叫冈部跟她走，并直接劝他先放下复仇。", "劝阻和救援行动有直接台词支持。"),
            ([4965, 4971], "她指出真由理、桥田和时间跳跃机都在等，说明还没有失去任何东西。", "信任、希望和行动意愿由台词支持。"),
            ([4973, 4979], "她抓住冈部的脸，用恐怖笑容把事件定义为始于他的战斗。", "强烈表达和激活有直接叙述支持。"),
            ([4981, 4985], "她说即使杀了冈部也要把他带回去。", "强硬决心有直接台词支持；高 anger 是标注估计。"),
            ([4989, 4997], "红莉栖扶着冈部回到 Lab，并确认时间跳跃机已准备好。", "照护与推进动作有直接叙述支持。"),
            ([4999, 5007], "车辆逼近时她责骂冈部快走，迫使他进入实验室。", "突发危险和催促行动有直接叙述支持。"),
        ],
        "该序列是桥上分支的连续救援场景，与其他世界线的父女冲突和第三卷等待场景不共享历史状态。",
        "原文没有秒级时间；桥上对话和返回 Lab 的连续事件使用 1.0 秒占位，首步 0.0。",
    ))

    # 第三卷：广播馆屋顶、D-Mail 验证、等待并送回冈部。
    sid = "underling_v3_roof_future_revelation_001"
    specs = [
        {
            "text": "红莉栖听到铃羽来自未来且是桥田的女儿，先用电影桥段类比并保持冷静。",
            "historical_emotion": emotion(.15, .55, .10, .15, .15, 0, .05, .60),
            "current_stimulus": emotion(.05, .30, .25, .65, 0, 0, 0, .45),
            "emotion_target": emotion(.20, .65, .25, .55, .05, 0, .05, .70),
            "behavior_target": behavior(.80, .08, .75, .80, .75, .65),
        },
        {
            "text": "她说明自己有 D-Mail 和时间机器的切身经验，认为先听铃羽说明再验证更合理。",
            "current_stimulus": emotion(.05, .25, .05, .25, 0, 0, 0, .55),
            "emotion_target": emotion(.25, .72, .12, .35, .05, 0, .03, .75),
            "behavior_target": behavior(.90, .05, .75, .90, .85, .65),
        },
        {
            "text": "红莉栖接受保密要求并选择矿泉水，继续观察铃羽的任务和时间机器。",
            "current_stimulus": emotion(.05, .20, .10, .15, 0, 0, 0, .45),
            "emotion_target": emotion(.22, .75, .15, .25, .05, 0, .04, .65),
            "behavior_target": behavior(.80, .10, .65, .75, .75, .55),
        },
        {
            "text": "听到 D-Mail 使世界线不稳定，红莉栖皱眉思考潜在的未来干涉。",
            "current_stimulus": emotion(0, .05, .35, .35, .15, .05, .10, .55),
            "emotion_target": emotion(.12, .55, .45, .45, .20, .10, .15, .75),
            "behavior_target": behavior(.80, .15, .70, .90, .80, .70),
        },
        {
            "text": "得知琉华子母亲曾把传呼机给 SERN 研究员看过，红莉栖明显惊讶并追问。",
            "current_stimulus": emotion(0, 0, .55, .75, .10, .05, .10, .35),
            "emotion_target": emotion(.05, .45, .65, .75, .20, .10, .15, .65),
            "behavior_target": behavior(.80, .15, .75, .85, .75, .85),
        },
        {
            "text": "她追问除铃羽外是否还有其他时间旅行者，并分析 D-Mail 的蝴蝶效应难以撤销。",
            "current_stimulus": emotion(0, .10, .45, .35, .15, .05, .15, .65),
            "emotion_target": emotion(.08, .55, .55, .45, .20, .10, .20, .80),
            "behavior_target": behavior(.90, .10, .80, .95, .85, .80),
        },
        {
            "text": "红莉栖确认第二封 D-Mail 无法简单取消第一封造成的变化。",
            "current_stimulus": emotion(0, .05, .25, .20, .10, .05, .10, .45),
            "emotion_target": emotion(.08, .60, .40, .30, .18, .08, .15, .75),
            "behavior_target": behavior(.90, .08, .70, .95, .80, .70),
        },
        {
            "text": "面对铃羽关于未来是否注定的挑衅，红莉栖坚持未来终究会改变。",
            "current_stimulus": emotion(.15, .20, .10, .20, 0, 0, .05, .70),
            "emotion_target": emotion(.25, .65, .25, .30, .05, 0, .10, .90),
            "behavior_target": behavior(.90, .05, .90, .98, .90, .75),
        },
        {
            "text": "她继续解释收束范围、量子不确定性和世界线摇晃，进入高度专注的理论分析。",
            "current_stimulus": emotion(.10, .15, .05, .25, 0, 0, 0, .75),
            "emotion_target": emotion(.30, .70, .15, .35, .05, 0, .03, .90),
            "behavior_target": behavior(.98, .03, .80, .99, .95, .80),
        },
        {
            "text": "红莉栖承认未来可以改变但幅度受世界线限制，并支持谨慎遵守禁止事项。",
            "current_stimulus": emotion(.05, .10, .20, .15, .05, 0, .05, .50),
            "emotion_target": emotion(.20, .65, .35, .25, .10, 0, .08, .75),
            "behavior_target": behavior(.90, .10, .75, .95, .85, .65),
        },
        {
            "text": "铃羽用脱衣玩笑试探时，红莉栖目光降到冰点并斥责这种做法很差劲。",
            "current_stimulus": emotion(0, 0, .10, .35, 0, .55, .55, .05),
            "emotion_target": emotion(.05, .35, .25, .45, .05, .55, .50, .30),
            "behavior_target": behavior(.65, .25, .85, .60, .45, .70),
        },
        {
            "text": "铃羽暗示身体可能有拷问痕迹，红莉栖和桥田收起玩笑，冷静注视并感到不安。",
            "current_stimulus": emotion(0, .05, .60, .65, .35, .20, .15, .20),
            "emotion_target": emotion(.03, .40, .65, .65, .40, .25, .20, .35),
            "behavior_target": behavior(.70, .25, .65, .50, .50, .90),
        },
        {
            "text": "她把信息管制概括为避免制造世界线摇晃的禁止事项。",
            "current_stimulus": emotion(.05, .20, .15, .25, .05, 0, 0, .45),
            "emotion_target": emotion(.10, .60, .45, .30, .20, .05, .08, .70),
            "behavior_target": behavior(.85, .10, .70, .90, .80, .65),
        },
        {
            "text": "听到未来可能危险，红莉栖望向远方并追问铃羽真正想拜托的事情。",
            "current_stimulus": emotion(0, .10, .30, .35, .20, .05, .10, .40),
            "emotion_target": emotion(.05, .55, .55, .45, .30, .08, .15, .65),
            "behavior_target": behavior(.80, .15, .75, .85, .75, .70),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第三卷.txt", "广播馆屋顶的未来说明", [344, 940],
        [
            ([356, 412], "铃羽揭示时间机器和未来身份，红莉栖用常见电影桥段解释并保持镇定。", "平静接受和理性解释有直接台词支持。"),
            ([454, 494], "她说明应先听取铃羽叙述再验证，并回忆自己对 D-Mail 的经历。", "信任、探究和从容来自直接对话。"),
            ([514, 556], "红莉栖接受保密要求并继续配合屋顶讨论。", "合作和谨慎有直接行为支持。"),
            ([566, 590], "她听到 D-Mail 使世界线不稳定后继续追问。", "担忧和探索由世界线说明及问句推断。"),
            ([644, 672], "得知 SERN 与 Rounder 可能卷入，红莉栖参与判断未来敌对势力。", "惊讶和警惕由新信息引起，具体强度为估计。"),
            ([698, 730], "面对未来是否注定的提问，她坚持未来终将改变。", "台词直接支持坚定、信任科学和期待。"),
            ([734, 760], "她展开收束范围、量子不确定性和世界线摇晃的理论。", "高度探索和表达行为有长段解释支持。"),
            ([772, 776], "她认可信息管制是保留未来变化空间的理由之一。", "谨慎和行动判断由直接回应支持。"),
            ([788, 798], "铃羽开脱衣玩笑时，红莉栖冷眼斥责。", "厌恶和愤怒有直接反应支持。"),
            ([822, 844], "铃羽拉起衣服并提到拷问痕迹，红莉栖与桥田转为严肃注视。", "危险暗示和不安由场景动作支持；不把冈部的惊恐直接复制给红莉栖。"),
            ([884, 894], "她把避免制造世界线摇晃的信息管制概括为禁止事项。", "总结性台词直接支持理性收束。"),
            ([904, 940], "未来危险的说明结束后，红莉栖望向远方并询问铃羽的请求。", "沉思和行动准备有直接台词及动作支持。"),
        ],
        "屋顶对话中目标角色身份明确；将铃羽的未来经历和冈部的惊恐与红莉栖自己的反应分开标注。",
        "原文为连续长对话；首步 0.0，其后使用 1.0 秒占位，不把未来背景年数当作事件间隔。",
    ))

    sid = "underling_v3_dmail_verification_001"
    specs = [
        {
            "text": "红莉栖和琉华子启动电话微波炉，冈部起床时她以无奈表情叹气。",
            "historical_emotion": emotion(.05, .35, .20, .10, .20, .05, .10, .65),
            "current_stimulus": emotion(.05, .05, .05, .25, .05, .05, .10, .35),
            "emotion_target": emotion(.10, .45, .20, .35, .25, .10, .20, .60),
            "behavior_target": behavior(.75, .15, .65, .75, .70, .55),
        },
        {
            "text": "她澄清正在验证琉华子母亲是否收到过 D-Mail，并说明这是时间跳跃机完成后的新问题。",
            "current_stimulus": emotion(.05, .20, .05, .25, 0, 0, 0, .55),
            "emotion_target": emotion(.20, .60, .15, .35, .12, 0, .05, .75),
            "behavior_target": behavior(.90, .05, .75, .90, .85, .70),
        },
        {
            "text": "红莉栖提出失败 D-Mail 可能发生在电话微波炉放电却没有产生结果的时间带。",
            "current_stimulus": emotion(.05, .15, .15, .45, .05, 0, 0, .60),
            "emotion_target": emotion(.20, .55, .30, .55, .12, 0, .05, .80),
            "behavior_target": behavior(.90, .05, .75, .95, .85, .75),
        },
        {
            "text": "她把这种可能与最初的 D-Mail 和广播馆事件联系起来，认真追问时间条件。",
            "current_stimulus": emotion(0, .10, .35, .55, .15, .05, .05, .65),
            "emotion_target": emotion(.12, .50, .45, .65, .25, .08, .10, .82),
            "behavior_target": behavior(.95, .05, .75, .98, .90, .80),
        },
        {
            "text": "红莉栖推算远距离 D-Mail 可能需要连续两天放电，并认真解释高电费。",
            "current_stimulus": emotion(.05, .20, .10, .25, 0, 0, 0, .55),
            "emotion_target": emotion(.18, .60, .25, .35, .12, 0, .03, .82),
            "behavior_target": behavior(.95, .05, .75, .95, .90, .70),
        },
        {
            "text": "看到时间跳跃机后，她提出在未来观察放电再回到过去定点发送邮件的办法。",
            "current_stimulus": emotion(.10, .15, .05, .35, 0, 0, 0, .80),
            "emotion_target": emotion(.28, .65, .15, .45, .05, 0, .02, .92),
            "behavior_target": behavior(.98, .03, .80, .98, .95, .82),
        },
        {
            "text": "她盯着装置叹气，认为偶然事件迟早会发生，不必追究所有未知原因。",
            "current_stimulus": emotion(0, 0, .10, .20, .10, 0, 0, .10),
            "emotion_target": emotion(.12, .50, .20, .25, .22, 0, .03, .58),
            "behavior_target": behavior(.80, .10, .65, .75, .65, .45),
        },
        {
            "text": "她说明琉华子母亲的 D-Mail 不是世界线改变的直接原因，但相关发送精度资料仍需验证。",
            "current_stimulus": emotion(.05, .15, .10, .20, 0, 0, 0, .50),
            "emotion_target": emotion(.18, .60, .20, .30, .08, 0, .02, .72),
            "behavior_target": behavior(.90, .05, .75, .95, .90, .68),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第三卷.txt", "D-Mail 发送验证", [4472, 4556],
        [
            ([4472, 4494], "红莉栖与琉华子启动装置，她对冈部的误会露出无奈。", "无奈和熟悉的社交表达有直接动作及台词支持。"),
            ([4500, 4506], "她说明正在验证琉华子母亲是否收到 D-Mail。", "专注和探索由明确实验目标支持。"),
            ([4506, 4508], "她提出失败 D-Mail 可能只造成放电而没有改变结果。", "科学假设和期待有直接台词支持。"),
            ([4514, 4518], "她把这一情况与最初 D-Mail 和广播馆事件联系起来。", "惊讶、警惕和探索由直接推理支持。"),
            ([4518, 4526], "她推算长时间放电及电费异常，并认真点头。", "专注和理性分析有直接台词支持。"),
            ([4532, 4540], "她由时间跳跃机提出先观察后回溯发送的验证方案。", "探索、主动性和激活明显提高。"),
            ([4542, 4544], "她叹气，认为偶然发生的事情不必追问全部原因。", "克制和低落由动作及台词推断。"),
            ([4550, 4556], "她区分直接原因与实验资料价值，并继续验证。", "理性收束和协作行为有直接台词支持。"),
        ],
        "同一实验室验证场景内保持红莉栖身份明确；不把琉华子的困惑或冈部的回忆直接标为目标状态。",
        "原文无精确时间；连续实验对话使用 1.0 秒占位，首步 0.0。",
    ))

    sid = "underling_v3_wait_and_send_001"
    specs = [
        {
            "text": "红莉栖独自在 Lab 等待冈部，表面平静地转笔喝咖啡，却紧握着笔和杯子。",
            "historical_emotion": emotion(.10, .55, .20, .10, .35, .05, .10, .75),
            "current_stimulus": emotion(0, .05, .15, .20, .20, .05, .05, .55),
            "emotion_target": emotion(.08, .55, .35, .30, .45, .10, .12, .80),
            "behavior_target": behavior(.65, .25, .55, .55, .65, .55),
        },
        {
            "text": "她以听到接吻为由调侃真由理，努力维持平常的语气。",
            "current_stimulus": emotion(.10, .10, .15, .25, .10, .05, .15, .20),
            "emotion_target": emotion(.20, .50, .30, .35, .40, .10, .20, .55),
            "behavior_target": behavior(.75, .20, .80, .50, .55, .45),
        },
        {
            "text": "红莉栖说自己是在等冈部，因为时间差不多了，随后开始准备时间跳跃机。",
            "current_stimulus": emotion(0, .05, .35, .30, .10, 0, .05, .65),
            "emotion_target": emotion(.10, .55, .50, .35, .50, .05, .15, .85),
            "behavior_target": behavior(.80, .15, .75, .65, .80, .85),
        },
        {
            "text": "她设置时间跳跃机并开始倒计时，冷静地准备把冈部送回临界点前。",
            "current_stimulus": emotion(0, .10, .45, .35, .05, 0, .05, .80),
            "emotion_target": emotion(.08, .60, .60, .45, .55, .05, .18, .95),
            "behavior_target": behavior(.90, .10, .80, .55, .95, .95),
        },
        {
            "text": "她给出‘现实冷酷无情’的答案，指出这里是临界点，冈部不得不回去。",
            "current_stimulus": emotion(0, .05, .35, .25, .25, .05, .15, .45),
            "emotion_target": emotion(.05, .60, .65, .35, .70, .10, .25, .75),
            "behavior_target": behavior(.85, .10, .85, .55, .90, .80),
        },
        {
            "text": "面对真由理的祝福，红莉栖叹气道自己会好好期待下一次，接受这份安慰。",
            "current_stimulus": emotion(.10, .35, .10, .15, .20, 0, 0, .25),
            "emotion_target": emotion(.15, .70, .55, .20, .70, .05, .12, .45),
            "behavior_target": behavior(.75, .10, .75, .45, .75, .55),
        },
        {
            "text": "她给冈部戴上耳机，说明自己将不再进行时间跳跃，并计划把机器限制在四十八小时内。",
            "current_stimulus": emotion(0, .25, .25, .25, .10, 0, .05, .60),
            "emotion_target": emotion(.08, .80, .45, .25, .55, .05, .12, .80),
            "behavior_target": behavior(.90, .05, .85, .80, .98, .75),
        },
        {
            "text": "红莉栖说明条件会变得严酷，成功取决于冈部能否在反复失败中挺过去。",
            "current_stimulus": emotion(0, .10, .30, .20, .15, .05, .05, .35),
            "emotion_target": emotion(.05, .85, .50, .25, .65, .08, .10, .70),
            "behavior_target": behavior(.85, .05, .80, .75, .95, .65),
        },
        {
            "text": "门外传来敲门声，她停下说明并露出难以启齿的表情，然后重新回到发送准备。",
            "current_stimulus": emotion(0, 0, .20, .45, .10, .05, .05, .15),
            "emotion_target": emotion(.03, .70, .45, .45, .55, .10, .12, .55),
            "behavior_target": behavior(.70, .20, .55, .50, .80, .60),
        },
        {
            "text": "红莉栖按下发送键并道别，强调无论跨越世界线还是回到过去，他们永远是同伴。",
            "current_stimulus": emotion(.15, .40, .30, .25, .35, 0, 0, .30),
            "emotion_target": emotion(.25, .95, .60, .30, .75, .05, .08, .55),
            "behavior_target": behavior(.85, .10, .90, .55, .90, .75),
        },
    ]
    records.append(sequence(sid, specs))
    audits.append(audit(
        sid, "第三卷.txt", "临界点等待与送回冈部", [7422, 7552],
        [
            ([7422, 7458], "红莉栖独自等冈部，以听到接吻为由调侃真由理并维持平常语气。", "压抑、克制和社交表达有直接动作及台词支持。"),
            ([7466, 7474], "她说时间差不多，开始准备时间跳跃机。", "等待和紧迫行动有直接台词与动作支持。"),
            ([7478, 7486], "她启动机器并说明现实冷酷无情。", "高 activation、恐惧和悲伤由倒计时、外观和台词支持。"),
            ([7486, 7488], "她指出这里是临界点，冈部不得不回去，并向真由理道歉。", "悲伤和责任感有直接台词支持。"),
            ([7494, 7502], "真由理祝福她下一次会得到幸福，红莉栖勉强接受并说自己会期待。", "安慰和克制的期待由直接互动支持。"),
            ([7504, 7518], "她给冈部戴耳机，说明不再进行时间跳跃并设置四十八小时限制。", "主动性、承诺和理性规划有直接台词支持。"),
            ([7518, 7520], "她坦言条件会很严酷，成功取决于冈部能否挺过失败。", "担忧和支持由直接台词支持。"),
            ([7522, 7528], "门外敲门使她停顿，露出难以启齿的表情后继续准备。", "惊讶、回避和未言明的情绪有动作支持，强度保守估计。"),
            ([7538, 7552], "她按下发送键道别，并强调两人永远是同伴。", "承诺、悲伤和高表达有直接台词支持。"),
        ],
        "该序列的首步历史情绪参考她等待冈部、持续开发和已知临界点；真由理的情绪不作为红莉栖目标。",
        "原文未给出每句秒数；首步 0.0，其后用 1.0 秒连续事件占位。",
    ))

    # 剧场版特典只旁述红莉栖在美国，没有合格的直接红莉栖事件。
    audits.append({
        "record_type": "excluded_scope",
        "source_file": "剧场版附送特典小说.txt",
        "source_lines": [330, 409],
        "target_character": "牧濑红莉栖",
        "context_note": "本段由真由理和旁观者视角叙述，红莉栖只在分配炸鸡时被提到在美国。",
        "reason": "没有红莉栖本人出场、直接台词或可可靠归因于她的现场反应；不把旁人的思念和期待转写为红莉栖标签。",
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with AUDIT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in audits:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} sequences / {sum(len(r['events']) for r in records)} events")
    print(DATA_PATH)
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
