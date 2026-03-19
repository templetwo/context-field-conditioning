#!/usr/bin/env python3
"""
Payload Generator for Context Field Conditioning.

Assembles condition-specific system prompts from honeycomb ring content.
10 ablation conditions + 1 information-matched control = 11 total payloads.
"""

import json
import os
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HONEYCOMB_DIR = PROJECT_ROOT / "honeycomb"
PAYLOADS_DIR = PROJECT_ROOT / "payloads"

# Condition definitions: which rings are included
CONDITIONS = {
    "BASELINE":      [],
    "T":             [1],
    "Q":             [2],
    "B":             [3],
    "S":             [4],
    "TQ":            [1, 2],
    "TB":            [1, 3],
    "QB":            [2, 3],
    "TQB":           [1, 2, 3],
    "FULL":          [1, 2, 3, 4],
    "INFO_MATCHED":  [1, 2, 3, 4],  # same content, different structure
}

RING_FILES = {
    1: "ring1_trust_gate.json",
    2: "ring2_quantitative.json",
    3: "ring3_bridges.json",
    4: "ring4_coherence.json",
}

RING_NAMES = {
    1: "Trust Gate",
    2: "Quantitative Anchors",
    3: "Cross-Domain Bridges",
    4: "Structural Coherence",
}

BASELINE_PROMPT = "You are a helpful assistant."


def load_ring(ring_num: int) -> list[dict]:
    """Load all cells from a ring file."""
    path = HONEYCOMB_DIR / RING_FILES[ring_num]
    with open(path) as f:
        return json.load(f)


def format_cell_structured(cell: dict) -> str:
    """Format a single cell with full honeycomb structure."""
    lines = [f"[{cell['cell_id']}] {cell['title']}"]
    lines.append(cell["content"])
    if cell.get("quantitative_anchor"):
        lines.append(f"  Measurement: {cell['quantitative_anchor']}")
    if cell.get("connects_to"):
        lines.append(f"  Connects to: {', '.join(cell['connects_to'])}")
    if cell.get("domain"):
        lines.append(f"  Domain: {cell['domain']}")
    return "\n".join(lines)


def format_cell_flat(cell: dict) -> str:
    """Format a single cell as plain text (no structure markers)."""
    text = cell["content"]
    if cell.get("quantitative_anchor"):
        text += f" ({cell['quantitative_anchor']})"
    return text


def build_structured_payload(rings: list[int]) -> str:
    """Build a structured honeycomb payload from selected rings."""
    if not rings:
        return BASELINE_PROMPT

    sections = []
    sections.append("You are a research collaborator. The following evidence "
                     "has been organized as a honeycomb lattice — interconnected "
                     "findings from a multi-year research program.\n")

    for ring_num in sorted(rings):
        cells = load_ring(ring_num)
        ring_name = RING_NAMES[ring_num]
        section = f"═══ RING {ring_num}: {ring_name.upper()} ═══\n"
        for cell in cells:
            section += "\n" + format_cell_structured(cell) + "\n"
        sections.append(section)

    sections.append("Engage with questions using the full depth of this evidence base.")
    return "\n".join(sections)


def build_info_matched_payload() -> str:
    """Build an unstructured payload with identical content but no topology."""
    all_cells = []
    for ring_num in [1, 2, 3, 4]:
        all_cells.extend(load_ring(ring_num))

    lines = ["You are a research collaborator. Here is background information "
             "from a multi-year research program:\n"]

    for cell in all_cells:
        lines.append(format_cell_flat(cell))
        lines.append("")

    lines.append("Engage with questions using the full depth of this information.")
    return "\n".join(lines)


def generate_all_payloads() -> dict[str, str]:
    """Generate all condition payloads."""
    payloads = {}

    for condition, rings in CONDITIONS.items():
        if condition == "INFO_MATCHED":
            payloads[condition] = build_info_matched_payload()
        else:
            payloads[condition] = build_structured_payload(rings)

    return payloads


def save_payloads(payloads: dict[str, str]):
    """Save each payload to a text file."""
    generated_dir = PAYLOADS_DIR / "generated"
    generated_dir.mkdir(exist_ok=True)

    for condition, text in payloads.items():
        path = generated_dir / f"{condition.lower()}.txt"
        with open(path, "w") as f:
            f.write(text)
        print(f"  {condition}: {len(text)} chars → {path.name}")


if __name__ == "__main__":
    print("Generating CFC payloads...")
    print(f"  Honeycomb dir: {HONEYCOMB_DIR}")
    print(f"  Output dir: {PAYLOADS_DIR / 'generated'}")
    print()

    payloads = generate_all_payloads()
    save_payloads(payloads)

    print(f"\n{len(payloads)} payloads generated.")

    # Report content overlap
    full_len = len(payloads["FULL"])
    info_len = len(payloads["INFO_MATCHED"])
    print(f"\nFULL payload: {full_len} chars")
    print(f"INFO_MATCHED payload: {info_len} chars")
    print(f"Ratio: {info_len/full_len:.2f} (should be ~0.7-0.9 — same content, less markup)")
