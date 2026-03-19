#!/usr/bin/env python3
"""
CFC Experiment Runner.

Executes the full factorial ablation experiment:
  - 11 conditions × 20 probes × 5 repetitions = 1,100 runs (full)
  - 2 conditions × 5 probes × 2 repetitions = 20 runs (pilot)

Each trial: inject condition payload → present probe → collect response → reset.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runner.ollama_client import get_client, InferenceResult

CONFIG_PATH = PROJECT_ROOT / "runner" / "config.yaml"
PROBES_PATH = PROJECT_ROOT / "probes" / "probe_questions.json"
PAYLOADS_DIR = PROJECT_ROOT / "payloads" / "generated"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_probes() -> list[dict]:
    with open(PROBES_PATH) as f:
        data = json.load(f)
    return data["probes"]


def load_payload(condition: str) -> str:
    path = PAYLOADS_DIR / f"{condition.lower()}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Payload not found: {path}. Run payloads/generate_payloads.py first.")
    return path.read_text()


PRIORITY_CONDITIONS = {"BASELINE", "FULL", "INFO_MATCHED"}


def build_trial_list(config: dict, probes: list[dict], mode: str) -> list[dict]:
    """Build the list of all trials to execute."""
    conditions = config["experiment"]["conditions"]

    if mode == "pilot":
        # Pilot: 2 conditions × 5 probes × 2 reps = 20 runs
        conditions = ["BASELINE", "FULL"]
        probes = probes[:5]
        reps_priority = 2
        reps_standard = 2
    elif mode == "full":
        # Full: priority conditions get 5 reps, others get 3
        # 3 × 20 × 5 = 300  +  8 × 20 × 3 = 480  =  780 total
        reps_priority = 5
        reps_standard = 3
    else:
        reps_priority = config["experiment"]["repetitions"]
        reps_standard = reps_priority

    trials = []
    for condition in conditions:
        reps = reps_priority if condition in PRIORITY_CONDITIONS else reps_standard
        for probe in probes:
            for rep in range(reps):
                trials.append({
                    "condition": condition,
                    "probe_id": probe["id"],
                    "probe_category": probe["category"],
                    "probe_question": probe["question"],
                    "repetition": rep + 1,
                })

    return trials


def run_trial(client, trial: dict, payloads: dict) -> dict:
    """Execute a single trial and return the result record."""
    condition = trial["condition"]
    system_prompt = payloads[condition]

    result: InferenceResult = client.generate(
        prompt=trial["probe_question"],
        system=system_prompt,
    )

    return {
        "trial_id": f"{condition}_{trial['probe_id']}_r{trial['repetition']}",
        "condition": condition,
        "probe_id": trial["probe_id"],
        "probe_category": trial["probe_category"],
        "probe_question": trial["probe_question"],
        "repetition": trial["repetition"],
        "response_text": result.response_text,
        "tokens": result.tokens,
        "logprobs": result.logprobs,
        "has_logprobs": result.has_logprobs,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "model": result.model,
        "backend": result.backend,
        "timestamp": datetime.now().isoformat(),
    }


def save_result(record: dict, output_dir: Path):
    """Save a single trial result to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{record['trial_id']}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def load_completed(output_dir: Path) -> set:
    """Load set of already-completed trial IDs for resume support."""
    completed = set()
    if output_dir.exists():
        for f in output_dir.glob("*.json"):
            completed.add(f.stem)
    return completed


def main():
    parser = argparse.ArgumentParser(description="CFC Experiment Runner")
    parser.add_argument("--pilot", action="store_true", help="Run pilot (20 trials)")
    parser.add_argument("--full", action="store_true", help="Run full experiment (1,100 trials)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "llama-server"],
                        help="Inference backend")
    parser.add_argument("--resume", action="store_true", help="Skip completed trials")
    parser.add_argument("--dry-run", action="store_true", help="List trials without executing")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        print("Specify --pilot or --full")
        sys.exit(1)

    mode = "pilot" if args.pilot else "full"

    # Load config, probes
    config = load_config()
    probes = load_probes()

    # Generate payloads if not already done
    if not PAYLOADS_DIR.exists() or not list(PAYLOADS_DIR.glob("*.txt")):
        logger.info("Generating payloads...")
        from payloads.generate_payloads import generate_all_payloads, save_payloads
        payloads_text = generate_all_payloads()
        save_payloads(payloads_text)

    # Load all payloads into memory
    payloads = {}
    for condition in config["experiment"]["conditions"]:
        payloads[condition] = load_payload(condition)

    # Build trial list
    trials = build_trial_list(config, probes, mode)

    # Setup output
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_RAW_DIR / f"run_{run_id}"

    # Resume support
    completed = set()
    if args.resume:
        # Find most recent run directory
        if DATA_RAW_DIR.exists():
            existing = sorted(DATA_RAW_DIR.glob("run_*"))
            if existing:
                output_dir = existing[-1]
                completed = load_completed(output_dir)
                logger.info("Resuming from %s (%d completed)", output_dir.name, len(completed))

    remaining = [t for t in trials
                 if f"{t['condition']}_{t['probe_id']}_r{t['repetition']}" not in completed]

    logger.info("Mode: %s | Backend: %s", mode, args.backend)
    logger.info("Total trials: %d | Already done: %d | Remaining: %d",
                len(trials), len(completed), len(remaining))

    if args.dry_run:
        for t in remaining[:20]:
            print(f"  {t['condition']:15s} | {t['probe_id']} | rep {t['repetition']}")
        if len(remaining) > 20:
            print(f"  ... and {len(remaining) - 20} more")
        return

    # Initialize client
    client = get_client(backend=args.backend)
    if not client.health_check():
        logger.error("Backend '%s' is not available. Check that the server is running.", args.backend)
        sys.exit(1)

    logger.info("Backend healthy. Starting experiment...")
    logger.info("Output: %s", output_dir)

    # Add file handler for this run
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Save run metadata
    meta = {
        "run_id": run_id,
        "mode": mode,
        "backend": args.backend,
        "model": client.model if hasattr(client, "model") else "unknown",
        "total_trials": len(trials),
        "resumed_from": len(completed),
        "start_time": datetime.now().isoformat(),
        "config": config,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Execute
    errors = 0
    results_summary = []

    for trial in tqdm(remaining, desc=f"CFC {mode}", unit="trial"):
        try:
            record = run_trial(client, trial, payloads)
            save_result(record, output_dir)
            results_summary.append({
                "trial_id": record["trial_id"],
                "condition": record["condition"],
                "probe_id": record["probe_id"],
                "latency_ms": record["latency_ms"],
                "total_tokens": record["total_tokens"],
                "has_logprobs": record["has_logprobs"],
            })
        except Exception as e:
            errors += 1
            logger.error("Trial %s_%s_r%d failed: %s",
                         trial["condition"], trial["probe_id"], trial["repetition"], e)
            if errors > 10:
                logger.error("Too many errors (%d). Aborting.", errors)
                break

    # Save summary
    summary = {
        "run_id": run_id,
        "mode": mode,
        "completed": len(results_summary),
        "errors": errors,
        "end_time": datetime.now().isoformat(),
        "trials": results_summary,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. %d completed, %d errors. Output: %s",
                len(results_summary), errors, output_dir)


if __name__ == "__main__":
    main()
