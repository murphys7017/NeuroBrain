#!/usr/bin/env python3
"""Run the prepared annotation sequences through the configured Luna route.

The script enriches each sequence with source-line evidence before calling the
OpenAI-compatible Responses endpoint. It writes one reviewed JSON object per
sequence and keeps request failures separate so a partial run cannot overwrite
the canonical dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
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
EVENT_KEYS = {
    "text", "historical_emotion", "current_stimulus",
    "emotion_target", "behavior_target", "delta_seconds",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def source_paths(source_dir: Path) -> dict[str, Path]:
    return {path.name: path for path in source_dir.glob("*.txt")}


def load_source_lines(source_dir: Path, name: str) -> list[str]:
    path = source_paths(source_dir).get(name)
    if path is None:
        raise FileNotFoundError(f"source file not found in {source_dir}: {name}")
    return path.read_text(encoding="utf-8-sig").splitlines()


def event_context(lines: list[str], start: int, end: int, radius: int = 3) -> str:
    # Audit line numbers are one-based. Keep a small local window so Luna can
    # resolve pronouns and dialogue turns without sending a whole chapter.
    lo = max(1, start - radius)
    hi = min(len(lines), end + radius)
    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1))


def build_audit_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sequence_id = record.get("sequence_id")
        if record.get("record_type") in {"included_sequence", "expanded_sequence"} and isinstance(sequence_id, str):
            result[sequence_id] = record
    return result


def build_request_input(
    sequence: dict[str, Any],
    audit: dict[str, Any],
    source_dir: Path,
) -> str:
    source_file = audit.get("source_file")
    if not isinstance(source_file, str):
        raise ValueError(f"{sequence.get('sequence_id')}: audit record has no source_file")
    lines = load_source_lines(source_dir, source_file)
    evidence: list[dict[str, Any]] = []
    for event in audit.get("events", []):
        source_lines = event.get("source_lines")
        if (
            not isinstance(source_lines, list)
            or len(source_lines) != 2
            or not all(isinstance(item, int) for item in source_lines)
        ):
            continue
        start, end = source_lines
        evidence.append(
            {
                "event_index": event.get("event_index"),
                "source_lines": source_lines,
                "evidence_class": event.get("evidence_class", "unknown"),
                "evidence_summary": event.get("evidence_summary", ""),
                "source_context": event_context(lines, start, end),
            }
        )

    payload = {
        "sequence": sequence,
        "source": {
            "file": source_file,
            "chapter": audit.get("chapter", ""),
            "sequence_lines": audit.get("source_lines", []),
            "event_evidence": evidence,
        },
        "instruction": (
            "请以 source.event_evidence 中的原文行作为事实依据，重新复核 sequence。"
            "不要只改写已有数值；若原文无法支持某种情绪，按系统规则填中性值。"
            "只返回系统提示词规定的一个 JSON 对象。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("model response did not contain a JSON object")
    if not isinstance(value, dict):
        raise ValueError("model response JSON root is not an object")
    return value


def validate_review(
    value: dict[str, Any],
    expected_id: str,
    expected_target: str,
    expected_sequence: dict[str, Any] | None = None,
) -> None:
    if value.get("sequence_id") != expected_id:
        raise ValueError(f"sequence_id mismatch: expected {expected_id!r}, got {value.get('sequence_id')!r}")
    if value.get("target_character") != expected_target:
        raise ValueError(f"target_character mismatch: expected {expected_target!r}")
    events = value.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a non-empty array")
    if expected_sequence is not None:
        expected_events = expected_sequence.get("events")
        if not isinstance(expected_events, list) or len(events) != len(expected_events):
            raise ValueError(
                f"event count changed: expected {len(expected_events or [])}, got {len(events)}"
            )
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != EVENT_KEYS:
            raise ValueError(f"event {index}: keys must be {sorted(EVENT_KEYS)}")
        if not isinstance(event["text"], str) or not event["text"].strip():
            raise ValueError(f"event {index}: text must be non-empty")
        if expected_sequence is not None:
            expected_event = expected_events[index]
            if event["text"] != expected_event["text"]:
                raise ValueError(f"event {index}: text changed")
            if event["delta_seconds"] != expected_event["delta_seconds"]:
                raise ValueError(f"event {index}: delta_seconds changed")
        for field in ("historical_emotion", "current_stimulus", "emotion_target"):
            vector = event[field]
            if not isinstance(vector, dict) or set(vector) != set(EMOTION):
                raise ValueError(f"event {index}: {field} dimensions mismatch")
            for name in EMOTION:
                item = vector[name]
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0.0 <= float(item) <= 1.0:
                    raise ValueError(f"event {index}: {field}.{name} must be in [0,1]")
        vector = event["behavior_target"]
        if not isinstance(vector, dict) or set(vector) != set(BEHAVIOR):
            raise ValueError(f"event {index}: behavior_target dimensions mismatch")
        for name in BEHAVIOR:
            item = vector[name]
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not 0.0 <= float(item) <= 1.0:
                raise ValueError(f"event {index}: behavior_target.{name} must be in [0,1]")
        delta = event["delta_seconds"]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or float(delta) < 0.0:
            raise ValueError(f"event {index}: delta_seconds must be non-negative")
    review = value.get("review")
    if review is not None and not isinstance(review, dict):
        raise ValueError("review must be an object when present")


def response_text(value: dict[str, Any]) -> str:
    chunks: list[str] = []
    for output in value.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if not chunks and isinstance(value.get("output_text"), str):
        chunks.append(value["output_text"])
    return "\n".join(chunks)


def call_luna_single(
    sequence: dict[str, Any],
    audit: dict[str, Any],
    source_dir: Path,
    prompt: str,
    endpoint: str,
    key: str,
    timeout: float,
    attempts: int,
    preserve_events: bool = False,
) -> tuple[str, dict[str, Any] | None, str | None]:
    sequence_id = str(sequence.get("sequence_id"))
    input_text = build_request_input(sequence, audit, source_dir)
    max_tokens = min(50000, max(5000, 3000 + len(sequence.get("events", [])) * 300))
    body = {
        "model": "gpt-6-luna",
        "instructions": prompt,
        "input": input_text,
        "store": False,
        "stream": False,
        "temperature": 0,
        "max_output_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/responses",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            parsed = extract_json(response_text(raw))
            validate_review(
                parsed,
                sequence_id,
                str(sequence.get("target_character", "")),
                expected_sequence=sequence if preserve_events else None,
            )
            return sequence_id, parsed, None
        except Exception as exc:  # noqa: BLE001 - retain per-sequence failure details
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    return sequence_id, None, last_error


def call_luna(
    sequence: dict[str, Any],
    audit: dict[str, Any],
    source_dir: Path,
    prompt: str,
    endpoint: str,
    key: str,
    timeout: float,
    attempts: int,
    preserve_events: bool = False,
    chunk_size: int = 24,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Review oversized sequences in order-preserving event chunks."""
    events = sequence.get("events", [])
    audit_events = audit.get("events", [])
    if not preserve_events or len(events) <= chunk_size:
        return call_luna_single(
            sequence, audit, source_dir, prompt, endpoint, key,
            timeout, attempts, preserve_events,
        )
    if len(audit_events) != len(events):
        return str(sequence.get("sequence_id")), None, (
            f"cannot chunk: audit event count {len(audit_events)} "
            f"does not match sequence event count {len(events)}"
        )

    reviewed_events: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for start in range(0, len(events), chunk_size):
        end = min(len(events), start + chunk_size)
        part_sequence = dict(sequence)
        part_sequence["events"] = events[start:end]
        part_audit = dict(audit)
        part_audit["events"] = audit_events[start:end]
        sid, result, error = call_luna_single(
            part_sequence, part_audit, source_dir, prompt, endpoint, key,
            timeout, attempts, True,
        )
        if result is None:
            return sid, None, f"chunk {start}-{end - 1} failed: {error}"
        reviewed_events.extend(result["events"])
        reviews.append(result.get("review", {}))

    confidences = [review.get("confidence", "low") for review in reviews]
    confidence = "low" if "low" in confidences else "medium" if "medium" in confidences else "high"
    combined = {
        "sequence_id": sequence["sequence_id"],
        "target_character": sequence["target_character"],
        "events": reviewed_events,
        "review": {
            "changed": any(bool(review.get("changed")) for review in reviews),
            "confidence": confidence,
            "audit_note": " | ".join(
                str(review.get("audit_note", "")) for review in reviews if review.get("audit_note")
            ),
            "source_lines": audit.get("source_lines", []),
            "review_chunks": len(reviews),
            "chunk_size": chunk_size,
        },
    }
    validate_review(
        combined,
        str(sequence["sequence_id"]),
        str(sequence.get("target_character", "")),
        expected_sequence=sequence,
    )
    return str(sequence["sequence_id"]), combined, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/annotations/kurisu_two_volumes_v0.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/annotations/kurisu_two_volumes_v0_audit.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("data/annotations/luna_review_batches/SYSTEM_PROMPT.txt"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/sources"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/luna_review_results/reviewed.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/annotations/luna_review_results/errors.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sequence-ids", nargs="*", default=None)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("NEUROBRAIN_LUNA_API_BASE", "http://127.0.0.1:8891/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="NEUROBRAIN_LUNA_API_KEY",
        help="environment variable containing the API key",
    )
    parser.add_argument(
        "--preserve-events",
        action="store_true",
        help="require Luna to preserve event count, order, text, and delta_seconds",
    )
    args = parser.parse_args()

    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"API key environment variable {args.api_key_env} is not set")
    endpoint = args.endpoint

    prompt = args.prompt.read_text(encoding="utf-8-sig")
    sequences = read_jsonl(args.input)
    audits = build_audit_map(read_jsonl(args.audit))
    if args.sequence_ids is not None:
        selected = set(args.sequence_ids)
        sequences = [sequence for sequence in sequences if sequence.get("sequence_id") in selected]
        missing_ids = selected - {str(sequence.get("sequence_id")) for sequence in sequences}
        if missing_ids:
            raise RuntimeError(f"unknown sequence IDs: {', '.join(sorted(missing_ids))}")
    if args.limit > 0:
        sequences = sequences[: args.limit]
    missing = [str(s.get("sequence_id")) for s in sequences if s.get("sequence_id") not in audits]
    if missing:
        raise RuntimeError(f"missing audit records for: {', '.join(missing)}")

    reviewed: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                call_luna,
                sequence,
                audits[str(sequence["sequence_id"])],
                args.source_dir,
                prompt,
                endpoint,
                key,
                args.timeout,
                args.attempts,
                args.preserve_events,
                args.chunk_size,
            ): str(sequence["sequence_id"])
            for sequence in sequences
        }
        for future in concurrent.futures.as_completed(futures):
            sequence_id = futures[future]
            try:
                sid, value, error = future.result()
            except Exception as exc:  # noqa: BLE001
                sid, value, error = sequence_id, None, f"worker: {type(exc).__name__}: {exc}"
            if value is not None:
                reviewed[sid] = value
                print(f"OK {sid} events={len(value['events'])}", flush=True)
            else:
                errors[sid] = {"sequence_id": sid, "error": error or "unknown error"}
                print(f"ERROR {sid} {error}", flush=True)

    ordered_reviewed = [reviewed[sid] for sid in (str(s["sequence_id"]) for s in sequences) if sid in reviewed]
    ordered_errors = [errors[sid] for sid in (str(s["sequence_id"]) for s in sequences) if sid in errors]
    write_jsonl(args.out, ordered_reviewed)
    write_jsonl(args.errors, ordered_errors)
    print(f"completed={len(ordered_reviewed)} errors={len(ordered_errors)} out={args.out}")
    return 0 if not ordered_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
