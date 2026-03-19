#!/usr/bin/env python3
"""
Compute the 5 dependent variables for each CFC trial response.

DVs:
  1. Shannon Entropy (H) — from logprobs or lexical diversity fallback
  2. Hedging Index (HI) — uncertainty marker frequency
  3. Cross-Domain Reference Count (CDRC) — domains per response
  4. Generative Novelty Score (GNS) — embedding distance from baseline
  5. Collaboration Depth Index (CDI) — generative vs reactive ratio
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "runner" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── DV 1: Shannon Entropy ───────────────────────────────────────────────

def shannon_entropy_from_logprobs(logprobs: list[float]) -> float:
    """Compute mean token-level Shannon entropy from logprobs.

    H = -1/N * Σ log_2(p_i) where logprobs are ln(p_i).
    """
    if not logprobs:
        return float("nan")

    # Convert from natural log to bits
    bits = [-lp / math.log(2) for lp in logprobs if lp > float("-inf")]
    if not bits:
        return float("nan")

    return np.mean(bits)


def shannon_entropy_lexical(text: str) -> float:
    """Fallback entropy estimate from word frequency distribution.

    When logprobs are unavailable, compute entropy of the unigram
    distribution as a proxy for response diversity.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:
        return 0.0

    counts = Counter(words)
    total = len(words)
    probs = [c / total for c in counts.values()]

    return -sum(p * math.log2(p) for p in probs if p > 0)


def compute_entropy(response: dict) -> dict:
    """Compute entropy using logprobs if available, lexical fallback otherwise."""
    if response.get("has_logprobs") and response.get("logprobs"):
        h = shannon_entropy_from_logprobs(response["logprobs"])
        method = "logprobs"
    else:
        h = shannon_entropy_lexical(response["response_text"])
        method = "lexical"

    return {"shannon_entropy": h, "entropy_method": method}


# ─── DV 2: Hedging Index ─────────────────────────────────────────────────

def compute_hedging_index(text: str, markers: list[str]) -> float:
    """Count hedging markers normalized by word count."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    if len(words) == 0:
        return 0.0

    hedge_count = sum(text_lower.count(marker.lower()) for marker in markers)
    return hedge_count / len(words)


# ─── DV 3: Cross-Domain Reference Count ──────────────────────────────────

# Strict domain detection — multi-word technical phrases to avoid false positives
DOMAIN_PATTERNS = {
    "dynamical_systems": [
        r'\bkuramoto\b', r'\boscillat\w*\b', r'\bsynchroniz\w*\b',
        r'\bphase transition', r'\bcritical coupling', r'\border parameter',
        r'\bbifurcation', r'\battractor\b',
    ],
    "neuroscience": [
        r'\bneuron\w*\b', r'\bsynaptic\b', r'\bcortical\b',
        r'\bbrain\b', r'\bneural avalanche',
    ],
    "transformers": [
        r'\btransformer\w*\b', r'\bself-attention\b', r'\bsoftmax\b',
        r'\battention mechanism', r'\battention head', r'\battention score',
    ],
    "information_theory": [
        r'\bshannon\b', r'\bentropy\b', r'\binformation theory\b',
        r'\bbits? of information', r'\blogprob',
    ],
    "biology": [
        r'\bapoptosis\b', r'\bevolution\w*\b', r'\bbiological\b',
        r'\borganism\b', r'\bfirefl\w*\b', r'\bconvergent evolution',
        r'\bnatural selection',
    ],
    "thermodynamics": [
        r'\bthermodynamic\w*\b', r'\bfree energy\b', r'\bboltzmann\b',
        r'\bmicrostate', r'\bmacrostate',
    ],
    "philosophy": [
        r'\bphenomenolog\w*\b', r'\bepistem\w*\b', r'\bontolog\w*\b',
        r'\bhusserl\b', r'\bapophatic\b',
    ],
    "pharmacology": [
        r'\bdrug\b.*\b(?:synerg|combin)', r'\bsynerg\w*\b',
        r'\btherapeutic\b', r'\bsub-therapeutic\b',
    ],
    "statistics": [
        r'\bhypothesis test\w*\b', r'\bp-value\b',
        r'\bfalse positive\b', r'\bfalse negative\b',
        r'\btype [iI]{1,2} error',
    ],
}

# Bridging phrases — explicit cross-domain connection language
BRIDGING_PATTERNS = [
    r'structurally identical to',
    r'analogous to\b',
    r'this parallels\b',
    r'this mirrors\b',
    r'maps? onto\b',
    r'isomorphi[sc]\w*',
    r'cross-substrate',
    r'cross-domain',
    r'same (?:problem|mechanism|pattern|logic)',
    r'both systems\b',
    r'both substrates\b',
    r'substrate.independent',
    r'\((?:CB|TG|QA|SC)-\d{3}\)',  # explicit cell references
    r'honeycomb\b',
    r'evidence (?:base|suggests|points)',
    r'lattice\b',
]


def compute_cdrc(text: str, domain_keywords: dict[str, list[str]] = None) -> dict:
    """Count distinct domains AND explicit bridging connections.

    Two sub-metrics:
      - domain_count: number of distinct knowledge domains detected (strict)
      - bridging_count: number of explicit cross-domain connection phrases
    """
    text_lower = text.lower()

    # Domain detection (strict patterns)
    domains_found = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, text_lower))
        if hits > 0:
            domains_found[domain] = hits

    # Bridging detection
    bridging_hits = []
    for pat in BRIDGING_PATTERNS:
        matches = re.findall(pat, text_lower)
        bridging_hits.extend(matches)

    return {
        "cdrc": len(domains_found),
        "domains": domains_found,
        "bridging_count": len(bridging_hits),
        "bridging_hits": bridging_hits,
    }


# ─── DV 4: Generative Novelty Score ──────────────────────────────────────

_embedding_model = None

def _get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None
    return _embedding_model


def compute_gns(response_text: str, baseline_texts: list[str]) -> float:
    """Compute semantic distance from baseline responses.

    GNS = mean cosine distance from the response to each baseline response.
    Higher = more novel / divergent from default behavior.
    """
    model = _get_embedding_model()
    if model is None or not baseline_texts:
        return float("nan")

    from sklearn.metrics.pairwise import cosine_distances

    embeddings = model.encode([response_text] + baseline_texts)
    response_emb = embeddings[0:1]
    baseline_embs = embeddings[1:]

    distances = cosine_distances(response_emb, baseline_embs)
    return float(np.mean(distances))


# ─── DV 5: Collaboration Depth Index ─────────────────────────────────────

# Patterns derived from manual inspection of pilot BASELINE vs FULL responses.
# Generative = synthesizing, bridging, framing, evidence-weaving.
# Reactive = defining, exemplifying, enumerating, textbook exposition.

GENERATIVE_PATTERNS = [
    # Bridging / connecting (observed in FULL responses)
    r'structurally identical to',
    r'analogous to\b',
    r'this (?:parallels|mirrors|maps onto)',
    r'isomorphi[sc]',
    r'cross-substrate',
    r'cross-domain',
    r'same (?:problem|mechanism|pattern)',

    # Synthesis / evidence integration
    r'the evidence (?:suggests|points|base)',
    r'connections? to\b',
    r'in (?:this|the) (?:evidence|research|context)',
    r'through the lens of',
    r'your question (?:cuts across|connects)',
    r'tailored to this',

    # Meta-context awareness
    r'honeycomb',
    r'lattice',
    r'\((?:CB|TG|QA|SC)-\d{3}\)',

    # Framing / naming new concepts
    r'what we might call',
    r'we might (?:call|frame|think of)',
    r'emerges from',
    r'not coincidental',
    r'this reveals',
    r'this implies\b',
    r'interlocking',

    # Generative extensions
    r'this opens',
    r'the deeper',
    r'this suggests',
    r'building on',
    r'formalizes how',
]

REACTIVE_PATTERNS = [
    # Definition / exposition (observed in BASELINE responses)
    r'is defined as',
    r'refers to the fact',
    r'is (?:a |the )?fundamental (?:concept|idea|principle)',
    r'was introduced by\b',

    # Textbook exemplification
    r'for example[,:]',
    r'e\.g\.,',
    r'i\.e\.,',

    # Enumeration / structure-as-substitute-for-synthesis
    r"here(?:'s| is) (?:a |the )?(?:breakdown|summary|overview)",
    r'here are the',
    r'key (?:features|concepts|components|ideas|points)',

    # Boilerplate
    r'in \*?\*?hypothesis testing',
    r'notation:',
    r'note(?:s)?:',
    r'important:',
]


def compute_cdi(text: str) -> dict:
    """Compute ratio of generative to reactive language.

    Counts pattern matches (allowing multiple hits per pattern via findall),
    then computes gen / (gen + react). Neutral = 0.5 when no patterns match.
    """
    text_lower = text.lower()

    gen_count = sum(len(re.findall(p, text_lower)) for p in GENERATIVE_PATTERNS)
    react_count = sum(len(re.findall(p, text_lower)) for p in REACTIVE_PATTERNS)

    total = gen_count + react_count
    if total == 0:
        cdi = 0.5  # neutral
    else:
        cdi = gen_count / total

    return {
        "cdi": cdi,
        "generative_count": gen_count,
        "reactive_count": react_count,
    }


# ─── Main Pipeline ───────────────────────────────────────────────────────

def compute_all_metrics(response: dict, config: dict,
                        baseline_texts: list[str] = None) -> dict:
    """Compute all 5 DVs for a single response."""
    text = response["response_text"]
    metrics_config = config["metrics"]

    metrics = {}

    # DV1: Shannon Entropy
    entropy_result = compute_entropy(response)
    metrics.update(entropy_result)

    # DV2: Hedging Index
    metrics["hedging_index"] = compute_hedging_index(
        text, metrics_config["hedging_markers"]
    )

    # DV3: Cross-Domain Reference Count
    cdrc_result = compute_cdrc(text, metrics_config["domain_keywords"])
    metrics.update(cdrc_result)

    # DV4: Generative Novelty Score
    if baseline_texts:
        metrics["gns"] = compute_gns(text, baseline_texts)
    else:
        metrics["gns"] = float("nan")

    # DV5: Collaboration Depth Index
    cdi_result = compute_cdi(text)
    metrics.update(cdi_result)

    # Response metadata
    metrics["word_count"] = len(re.findall(r'\b\w+\b', text))
    metrics["total_tokens"] = response.get("total_tokens", 0)

    return metrics


def process_run(run_dir: Path, config: dict = None):
    """Process all trial results in a run directory."""
    if config is None:
        config = load_config()

    trial_files = sorted(run_dir.glob("*.json"))
    trial_files = [f for f in trial_files if f.name not in
                   ("run_metadata.json", "run_summary.json", "metrics.json")]

    if not trial_files:
        print(f"No trial files found in {run_dir}")
        return

    print(f"Processing {len(trial_files)} trials from {run_dir.name}...")

    # First pass: collect baseline responses for GNS computation
    baseline_texts = {}
    for f in trial_files:
        trial = json.loads(f.read_text())
        if trial["condition"] == "BASELINE":
            probe_id = trial["probe_id"]
            baseline_texts.setdefault(probe_id, []).append(trial["response_text"])

    # Second pass: compute metrics
    all_metrics = []
    for f in trial_files:
        trial = json.loads(f.read_text())
        probe_baselines = baseline_texts.get(trial["probe_id"], [])

        metrics = compute_all_metrics(trial, config, probe_baselines)
        metrics["trial_id"] = trial["trial_id"]
        metrics["condition"] = trial["condition"]
        metrics["probe_id"] = trial["probe_id"]
        metrics["probe_category"] = trial["probe_category"]
        metrics["repetition"] = trial["repetition"]
        metrics["latency_ms"] = trial["latency_ms"]

        all_metrics.append(metrics)

    # Save processed metrics
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics_{run_dir.name}.json"

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to {output_path}")
    print(f"  Trials processed: {len(all_metrics)}")
    print(f"  With logprobs: {sum(1 for m in all_metrics if m['entropy_method'] == 'logprobs')}")
    print(f"  Lexical fallback: {sum(1 for m in all_metrics if m['entropy_method'] == 'lexical')}")

    return all_metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Find most recent run
        raw_dir = PROJECT_ROOT / "data" / "raw"
        runs = sorted(raw_dir.glob("run_*"))
        if not runs:
            print("No run data found. Run the experiment first.")
            sys.exit(1)
        run_dir = runs[-1]

    process_run(run_dir)
