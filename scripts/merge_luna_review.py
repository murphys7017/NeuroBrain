#!/usr/bin/env python3
"""Validate and merge a completed Luna review into the canonical JSONL data."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT_KEYS = {"sequence_id", "target_character", "events"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temp.replace(path)


def validate_review_rows(
    rows: list[dict[str, Any]],
    expected_ids: set[str],
    expected_targets: dict[str, str],
) -> None:
    actual_ids = [row.get("sequence_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in actual_ids):
        raise ValueError("reviewed rows must have non-empty sequence_id")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("reviewed output contains duplicate sequence_id")
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        extra = sorted(set(actual_ids) - expected_ids)
        raise ValueError(f"sequence IDs differ; missing={missing}, extra={extra}")
    for row in rows:
        if set(row) != {"sequence_id", "target_character", "events", "review"}:
            raise ValueError(f"unexpected Luna root keys for {row.get('sequence_id')}: {sorted(row)}")
        expected_target = expected_targets[row["sequence_id"]]
        if row["target_character"] != expected_target:
            raise ValueError(
                f"unexpected target_character for {row['sequence_id']}: "
                f"expected {expected_target!r}, got {row['target_character']!r}"
            )
        events = row["events"]
        if not isinstance(events, list) or not events:
            raise ValueError(f"empty events for {row['sequence_id']}")
        review = row["review"]
        if not isinstance(review, dict):
            raise ValueError(f"missing review metadata for {row['sequence_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=Path("data/annotations/kurisu_two_volumes_v0.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/annotations/kurisu_two_volumes_v0_audit.jsonl"))
    parser.add_argument("--reviewed", type=Path, default=Path("data/annotations/luna_review_results/dense_complete.jsonl"))
    parser.add_argument("--backup-suffix", default=".pre_luna_20261010")
    parser.add_argument(
        "--preserve-events",
        action="store_true",
        help="require one-to-one event alignment and merge labels only",
    )
    args = parser.parse_args()

    core = load_jsonl(args.core)
    reviewed = load_jsonl(args.reviewed)
    core_by_id = {row.get("sequence_id"): row for row in core}
    if len(core_by_id) != len(core) or any(not isinstance(k, str) for k in core_by_id):
        raise ValueError("canonical core contains duplicate or invalid sequence IDs")
    expected_targets = {
        sequence_id: str(row.get("target_character", ""))
        for sequence_id, row in core_by_id.items()
    }
    validate_review_rows(reviewed, set(core_by_id), expected_targets)
    reviewed_by_id = {row["sequence_id"]: row for row in reviewed}

    if args.preserve_events:
        for original in core:
            candidate = reviewed_by_id[original["sequence_id"]]
            if candidate["target_character"] != original["target_character"]:
                raise ValueError(f"target_character changed for {original['sequence_id']}")
            if len(candidate["events"]) != len(original["events"]):
                raise ValueError(
                    f"event count changed for {original['sequence_id']}: "
                    f"{len(original['events'])} -> {len(candidate['events'])}"
                )
            for index, (old, new) in enumerate(zip(original["events"], candidate["events"])):
                if old["text"] != new["text"] or old["delta_seconds"] != new["delta_seconds"]:
                    raise ValueError(f"event identity changed for {original['sequence_id']} event {index}")

    merged = []
    for row in core:
        candidate = reviewed_by_id[row["sequence_id"]]
        if args.preserve_events:
            label_fields = (
                "historical_emotion", "current_stimulus", "emotion_target", "behavior_target",
            )
            events = []
            for original_event, reviewed_event in zip(row["events"], candidate["events"]):
                event = dict(original_event)
                for field in label_fields:
                    event[field] = reviewed_event[field]
                events.append(event)
        else:
            events = candidate["events"]
        merged.append(
            {
                "sequence_id": row["sequence_id"],
                "target_character": row["target_character"],
                "events": events,
            }
        )

    audit_rows = load_jsonl(args.audit)
    audit_by_id = {
        row.get("sequence_id"): row
        for row in audit_rows
        if row.get("record_type") in {"included_sequence", "expanded_sequence"}
    }
    if set(audit_by_id) != set(core_by_id):
        raise ValueError("audit sequence IDs do not match canonical core")

    confidence = Counter()
    changed = Counter()
    original_events = 0
    reviewed_events = 0
    updated_audit: list[dict[str, Any]] = []
    for row in audit_rows:
        sequence_id = row.get("sequence_id")
        if row.get("record_type") not in {"included_sequence", "expanded_sequence"}:
            updated_audit.append(row)
            continue
        review = reviewed_by_id[sequence_id].get("review", {})
        current = dict(row)
        old_count = len(row.get("events", []))
        new_count = len(reviewed_by_id[sequence_id]["events"])
        current["original_event_count"] = old_count
        current["reviewed_event_count"] = new_count
        current["luna_review"] = review
        current["review_model"] = "gpt-6-luna"
        current["review_date"] = date.today().isoformat()
        current["review_source_context"] = "event-level source excerpts with local line context"
        current["review_granularity"] = "one_to_one_labels_only" if args.preserve_events else "semantic_event_rewrite"
        updated_audit.append(current)
        confidence[review.get("confidence", "unknown")] += 1
        changed[bool(review.get("changed"))] += 1
        original_events += old_count
        reviewed_events += new_count

    updated_audit.append(
        {
            "record_type": "luna_review_summary",
            "review_model": "gpt-6-luna",
            "review_date": date.today().isoformat(),
            "sequence_count": len(merged),
            "original_event_count": original_events,
            "reviewed_event_count": reviewed_events,
            "preserved_event_alignment": args.preserve_events,
            "confidence_counts": dict(confidence),
            "changed_counts": {str(key).lower(): value for key, value in changed.items()},
            "canonical_merge": True,
        }
    )

    core_backup = args.core.with_name(args.core.name + args.backup_suffix)
    audit_backup = args.audit.with_name(args.audit.name + args.backup_suffix)
    if not core_backup.exists():
        shutil.copy2(args.core, core_backup)
    if not audit_backup.exists():
        shutil.copy2(args.audit, audit_backup)
    write_jsonl(args.core, merged)
    write_jsonl(args.audit, updated_audit)
    print(
        f"merged sequences={len(merged)} original_events={original_events} "
        f"reviewed_events={reviewed_events} confidence={dict(confidence)}"
    )
    print(f"backups: {core_backup} and {audit_backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
