#!/usr/bin/env python3
"""Validate NeuroBrain sequence annotation JSONL files.

The validator checks the compact, human-readable schema used for CfC data.
It rejects null values and malformed dimensions, and reports continuity
warnings for historical emotion transitions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EMOTION = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)
BEHAVIOR = (
    "engagement", "avoidance", "expression",
    "exploration", "initiative", "activation",
)
ROOT_KEYS = {"sequence_id", "target_character", "events"}
EVENT_KEYS = {
    "text", "historical_emotion", "current_stimulus",
    "emotion_target", "behavior_target", "delta_seconds",
}


def fail(errors: list[str], where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def check_vector(
    value: Any,
    expected: tuple[str, ...],
    where: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        fail(errors, where, "must be an object with named dimensions")
        return
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        fail(errors, where, f"dimensions mismatch; missing={missing}, extra={extra}")
        return
    for name in expected:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            fail(errors, f"{where}.{name}", "must be a number")
        elif not math.isfinite(float(item)):
            fail(errors, f"{where}.{name}", "must be finite")
        elif not 0.0 <= float(item) <= 1.0:
            fail(errors, f"{where}.{name}", "must be in [0, 1]")


def validate_record(record: Any, line_no: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    where = f"line {line_no}"
    if not isinstance(record, dict):
        return [f"{where}: record must be an object"], warnings
    if set(record) != ROOT_KEYS:
        fail(errors, where, f"root keys must be {sorted(ROOT_KEYS)}")
        return errors, warnings
    if not isinstance(record["sequence_id"], str) or not record["sequence_id"].strip():
        fail(errors, where, "sequence_id must be a non-empty string")
    if not isinstance(record["target_character"], str) or not record["target_character"].strip():
        fail(errors, where, "target_character must be a non-empty string")
    events = record["events"]
    if not isinstance(events, list) or not events:
        fail(errors, where, "events must be a non-empty array")
        return errors, warnings

    previous_target: dict[str, float] | None = None
    for index, event in enumerate(events):
        ewhere = f"{where} event {index}"
        if not isinstance(event, dict):
            fail(errors, ewhere, "event must be an object")
            continue
        if set(event) != EVENT_KEYS:
            fail(errors, ewhere, f"event keys must be {sorted(EVENT_KEYS)}")
            continue
        if not isinstance(event["text"], str) or not event["text"].strip():
            fail(errors, ewhere, "text must be a non-empty string")
        for field in ("historical_emotion", "current_stimulus", "emotion_target"):
            check_vector(event[field], EMOTION, f"{ewhere}.{field}", errors)
        check_vector(event["behavior_target"], BEHAVIOR, f"{ewhere}.behavior_target", errors)
        delta = event["delta_seconds"]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            fail(errors, f"{ewhere}.delta_seconds", "must be a number")
        elif not math.isfinite(float(delta)) or float(delta) < 0.0:
            fail(errors, f"{ewhere}.delta_seconds", "must be finite and non-negative")

        history = event["historical_emotion"]
        if previous_target is not None and isinstance(history, dict) and set(history) == set(EMOTION):
            distance = max(abs(float(history[k]) - previous_target[k]) for k in EMOTION)
            if distance > 0.20:
                warnings.append(
                    f"{ewhere}.historical_emotion differs from previous emotion_target by {distance:.2f}; "
                    "check for an explicit recovery, time gap, or context change"
                )
        target = event["emotion_target"]
        if isinstance(target, dict) and set(target) == set(EMOTION):
            previous_target = {k: float(target[k]) for k in EMOTION}
        else:
            previous_target = None
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    seen_sequences: set[str] = set()
    records = 0
    for path in args.paths:
        if not path.is_file():
            errors.append(f"{path}: file does not exist")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            errors.append(f"{path}: must be UTF-8 ({exc})")
            continue
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON ({exc.msg})")
                continue
            record_errors, record_warnings = validate_record(record, line_no)
            errors.extend(f"{path}: {item}" for item in record_errors)
            warnings.extend(f"{path}: {item}" for item in record_warnings)
            if isinstance(record, dict) and isinstance(record.get("sequence_id"), str):
                sequence_id = record["sequence_id"]
                if sequence_id in seen_sequences:
                    errors.append(f"{path}:{line_no}: duplicate sequence_id {sequence_id!r}")
                seen_sequences.add(sequence_id)
    for warning in warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s), {records} record(s)")
        return 1
    print(f"PASS: {records} record(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
