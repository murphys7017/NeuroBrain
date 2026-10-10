from __future__ import annotations
import json, math, re, hashlib
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path(os.environ.get("NEUROBRAIN_SOURCE_DIR", ROOT / "data" / "sources"))
DATA = ROOT / 'data' / 'annotations'
CORE = DATA / 'kurisu_two_volumes_v0.jsonl'
AUDIT = DATA / 'kurisu_two_volumes_v0_audit.jsonl'

SOURCES = [
    '第一卷_蝶翼的世界性变动率：Reverse.txt',
    '第二卷_形而上のネクローシス：Revers.txt',
    '第三卷_境界面上のシュタインズ_ゲート：Rebirth.txt',
    '第一卷.txt',
    '第二卷.txt',
    '第三卷.txt',
    '剧场版附送特典小说.txt',
]
EMOTION = ('joy','trust','fear','surprise','sadness','disgust','anger','anticipation')
BEHAVIOR = ('engagement','avoidance','expression','exploration','initiative','activation')

# Strong names plus the established nickname “助手”. This is intentionally broad
# because the user requested inclusion of indirect mentions too.
MENTION_RE = re.compile(r'(红莉栖|紅莉栖|牧濑|牧瀬|克莉丝|克里斯蒂娜|Christina|Kurisu|助手)', re.I)
HEADING_RE = re.compile(r'^(?:第\s*[0-9０-９一二三四五六七八九十百]+\s*[幕章回].*|序(?:幕|章).*|终幕.*|intermission.*|chapter\s*\d+.*)$', re.I)
HTML_RE = re.compile(r'<script|</?script|BAIDU_CLB|http://|https://', re.I)
META_RE = re.compile(r'^(?:文件发布时间|小说名称|本卷名称|网译版|转自|翻译[:：]|图源[:：]|校对[:：])')

LEX = {
    'joy': ('笑','微笑','开心','高兴','愉快','轻松','安心','幸福','快乐','兴奋','喜悦','满足','温和','温柔','喜欢'),
    'trust': ('相信','信任','依赖','朋友','伙伴','约定','感谢','拜托','帮助','理解','支持','可靠','亲密'),
    'fear': ('害怕','恐惧','恐慌','不安','危险','颤','发抖','惊恐','害怕','死','死亡','杀','威胁','绝望','恐怖','担心'),
    'surprise': ('突然','惊讶','吃惊','震惊','意外','没想到','愕然','诧异','惊','想不到'),
    'sadness': ('悲伤','难过','伤心','哭','眼泪','泪','孤独','失去','痛苦','悲惨','遗憾','悔恨','寂寞'),
    'disgust': ('讨厌','恶心','厌恶','嫌弃','反感','憎恨','烦','麻烦'),
    'anger': ('生气','愤怒','怒','恼火','吼','怒吼','反驳','抗议','不满','气愤','骂','责备'),
    'anticipation': ('期待','希望','准备','计划','想要','研究','探索','观察','寻找','调查','等待','未来','打算','决定','目标'),
}

BEHAV_LEX = {
    'engagement': ('询问','提问','回答','解释','讨论','对话','交流','听','看','关注','回应','参与','研究'),
    'avoidance': ('离开','拒绝','逃','躲','回避','沉默','不想','不愿','避开','断开','消失'),
    'expression': ('说','告诉','喊','叫','笑','哭','表情','点头','摇头','写','发言','承认'),
    'exploration': ('研究','观察','寻找','调查','确认','分析','实验','检查','探索','思考','推理'),
    'initiative': ('主动','提出','决定','尝试','行动','开始','邀请','请求','建议','保护','阻止','追问'),
    'activation': ('跑','冲','战斗','急','紧张','颤','挥','攻击','危险','哭喊','忙','努力','赶紧'),
}

def clip(x: float) -> float:
    return round(min(1.0, max(0.0, x)), 3)

def scores(text: str) -> tuple[dict[str,float], dict[str,float], str]:
    low = text.lower()
    emo = {k: 0.0 for k in EMOTION}
    for k, words in LEX.items():
        count = sum(low.count(w.lower()) for w in words)
        # one clear cue gives a moderate stimulus; repeated cues saturate gradually
        emo[k] = clip(0.18 * min(5, count))
    behavior = {k: 0.5 for k in BEHAVIOR}
    for k, words in BEHAV_LEX.items():
        count = sum(low.count(w.lower()) for w in words)
        delta = min(0.38, 0.07 * min(5, count))
        behavior[k] = clip(0.5 + delta)
    # Explicit withdrawal/action cues shift the opposing tendency.
    if any(w in text for w in ('离开','逃','躲','回避','拒绝','不愿','不想')):
        behavior['avoidance'] = clip(behavior['avoidance'] + 0.18)
        behavior['engagement'] = clip(behavior['engagement'] - 0.08)
    if any(w in text for w in ('主动','决定','提出','尝试','保护','阻止')):
        behavior['initiative'] = clip(behavior['initiative'] + 0.15)
    evidence = 'direct' if re.search(r'(红莉栖|紅莉栖|牧濑|牧瀬|克莉丝|克里斯蒂娜|Christina|Kurisu)', text, re.I) else 'indirect'
    return emo, behavior, evidence

def chapter_name(lines: list[str]) -> str:
    for line in lines:
        if HEADING_RE.match(line.strip()):
            return line.strip()
    return '未分章片段'

def read_blocks(path: Path):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    current = '前言'
    blocks: list[tuple[int,int,str,str]] = []
    buf: list[tuple[int,str]] = []
    def flush():
        nonlocal buf
        if not buf:
            return
        start = buf[0][0]; end = buf[-1][0]
        text = ' '.join(s.strip() for _,s in buf if s.strip())
        if text and not HTML_RE.search(text) and not META_RE.match(text):
            blocks.append((start,end,current,text))
        buf = []
    for no, raw in enumerate(lines, 1):
        s = raw.strip()
        if HEADING_RE.match(s):
            flush(); current = s; continue
        if not s or HTML_RE.search(s) or META_RE.match(s):
            flush(); continue
        buf.append((no,s))
    flush()
    return lines, blocks

def slug(s: str) -> str:
    s = re.sub(r'[^0-9A-Za-z一-龥]+', '_', s)
    return s.strip('_')[:48] or 'scene'

def make_event(text: str, history: dict[str,float], delta: float):
    stimulus, behavior, _ = scores(text)
    target = history.copy() if not any(stimulus.values()) else {k: clip(0.68*history[k] + 0.32*stimulus[k]) for k in EMOTION}
    return {
        'text': text[:600],
        'historical_emotion': history.copy(),
        'current_stimulus': stimulus,
        'emotion_target': target,
        'behavior_target': behavior,
        'delta_seconds': delta,
    }, target

def load_jsonl(path: Path):
    out=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip(): out.append(json.loads(line))
    return out

old_core = [
    r for r in load_jsonl(CORE)
    if not str(r.get('sequence_id', '')).startswith('expanded_')
]
old_audit = [
    r for r in load_jsonl(AUDIT)
    if r.get('record_type') != 'expanded_sequence'
]
# Preserve prior IDs; generated IDs are prefixed with “expanded_”.
seen_ids = {r.get('sequence_id') for r in old_core}
seen_text = set()
for r in old_core:
    for e in r.get('events',[]): seen_text.add((r.get('target_character'), e.get('text','').strip()))

new_core=[]; new_audit=[]; total_events=0; total_sequences=0
for fname in SOURCES:
    path=DOWNLOADS/fname
    if not path.exists():
        continue
    lines, blocks=read_blocks(path)
    # Group nearby mention blocks into chapter sequences, preserving order.
    by_chapter: dict[str,list[tuple[int,int,str]]] = {}
    for start,end,ch,text in blocks:
        if not MENTION_RE.search(text):
            continue
        # Ignore very short metadata-like mentions.
        if len(text.strip()) < 4:
            continue
        by_chapter.setdefault(ch,[]).append((start,end,text))
    for idx,(ch,items) in enumerate(by_chapter.items(),1):
        sid = 'expanded_' + slug(path.stem) + '_' + slug(ch) + f'_{idx:03d}'
        if sid in seen_ids:
            continue
        history={k:0.0 for k in EMOTION}
        events=[]; audit_events=[]
        for j,(start,end,text) in enumerate(items):
            key=('牧濑红莉栖', text.strip())
            if key in seen_text:
                continue
            ev,target=make_event(text, history, 0.0 if not events else 1.0)
            events.append(ev)
            _,_,evidence=scores(text)
            audit_events.append({
                'event_index': len(events)-1,
                'source_lines':[start,end],
                'evidence_summary': text[:240],
                'evidence_class': evidence,
                'uncertainty': '批量初标；情绪数值由词汇和叙事线索估计，间接提及时无依据维度保留中性值。',
            })
            history=target
            seen_text.add(key)
        if not events:
            continue
        record={'sequence_id':sid,'target_character':'牧濑红莉栖','events':events}
        new_core.append(record); total_sequences+=1; total_events+=len(events)
        new_audit.append({
            'record_type':'expanded_sequence',
            'sequence_id':sid,
            'target_character':'牧濑红莉栖',
            'source_file':fname,
            'chapter':ch,
            'source_lines':[events and audit_events[0]['source_lines'][0], events and audit_events[-1]['source_lines'][1]],
            'context_note':'用户要求将所有提及红莉栖的片段纳入；直接出场与间接提及共用核心序列，证据类别仅保留在审计字段。',
            'events':audit_events,
            'timing_note':'原文没有逐事件秒级时间戳；连续文本使用 1.0 秒占位，序列首步为 0.0。',
        })

all_core=old_core+new_core
all_audit=old_audit+new_audit
CORE.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in all_core),encoding='utf-8')
AUDIT.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in all_audit),encoding='utf-8')
print(json.dumps({'old_sequences':len(old_core),'new_sequences':total_sequences,'total_sequences':len(all_core),'new_events':total_events,'total_events':sum(len(x.get('events',[])) for x in all_core),'audit_records':len(all_audit)},ensure_ascii=False))
