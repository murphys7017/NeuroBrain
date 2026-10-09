"""Validate NeuroBrain annotation JSONL files against protocol v0.2."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurobrain import load_annotation_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    total_sequences = 0
    total_events = 0
    for path in args.paths:
        sequences = load_annotation_sequences(path)
        events = sum(sequence.length for sequence in sequences)
        total_sequences += len(sequences)
        total_events += events
        characters = sorted({sequence.target_character for sequence in sequences if sequence.target_character})
        print(f"{path}: valid; sequences={len(sequences)} events={events}")
        if characters:
            print(f"  characters: {', '.join(characters)}")
    print(f"total: sequences={total_sequences} events={total_events}")


if __name__ == "__main__":
    main()
