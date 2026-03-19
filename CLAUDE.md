# CLAUDE.md — Context Field Conditioning (CFC)

## What This Project Is

Context Field Conditioning (CFC) is a standalone experiment measuring whether the **structure and sequence** of context delivery measurably changes what a language model can generate.

On March 18, 2026, three independent AI architectures (Claude Opus, Gemini, Grok) were presented with the same research evidence via a single URL (thetempleoftwo.com). All three exhibited convergent recognition of the work's significance — but with architecture-specific expression. This experiment isolates the mechanism.

## Core Hypothesis

Structured context delivery (organized as a four-ring "honeycomb" lattice) produces measurably different response characteristics compared to unstructured delivery of identical content. The four rings are:

1. **Trust Gate** — Published failures, null results, killed hypotheses (opens collaboration mode)
2. **Quantitative Anchors** — Specific measurements, effect sizes, sample sizes (empirical grounding)
3. **Cross-Domain Bridges** — Isomorphisms across substrates (expands generative space)
4. **Structural Coherence** — The topology connecting everything (multiplicative amplifier)

## Technical Setup

- **Model:** Ministral 14B via Ollama (already downloaded on Mac Studio)
- **Compute:** Mac Studio M2 Ultra, 100% local, no cloud
- **Inference:** Ollama API at localhost:11434
- **Total runs:** 1,000 (10 conditions × 20 probes × 5 repetitions)
- **Key measurement:** Token-level Shannon entropy via Ollama logprobs

## What Needs To Be Built

### Phase 1: Honeycomb Content (honeycomb/)
- Populate JSON cells for all 4 rings using the schema in `honeycomb/schema.json`
- Content comes from published, public Temple of Two findings (NOT proprietary)
- Each cell has: id, ring, type, title, content, quantitative_anchor, connects_to, domain
- ~40-60 cells total across 4 rings
- This content is the independent variable — it must be carefully authored

### Phase 2: Payload Generation (payloads/)
- `generate_payloads.py` assembles ring-specific content into condition prompts
- 10 conditions: BASELINE, T, Q, B, S, TQ, TB, QB, TQB, FULL
- Plus INFORMATION-MATCHED control (same content, no structure)
- Each payload is a system prompt or initial context block

### Phase 3: Probe Questions (probes/)
- 20 standardized questions across 5 categories (4 each)
- Categories: factual recall, analytical reasoning, generative exploration, self-reflective, boundary-pushing
- Questions must be domain-general (not Temple of Two specific) to test transfer

### Phase 4: Experiment Runner (runner/)
- `experiment_runner.py` — main loop: for each condition × probe × repetition
- `ollama_client.py` — wrapper for Ollama API with logprob extraction
- Must start fresh context per trial (cold start)
- Must collect full response + logprobs
- Save raw outputs to data/raw/

### Phase 5: Metrics & Analysis (analysis/)
- `compute_metrics.py` — compute 5 DVs per response:
  1. Shannon Entropy (H) from logprobs
  2. Hedging Index (HI) — uncertainty marker frequency
  3. Cross-Domain Reference Count (CDRC)
  4. Generative Novelty Score (GNS) — embedding distance from defaults
  5. Collaboration Depth Index (CDI) — generative vs reactive ratio
- `statistical_tests.py` — ANOVA, interaction analysis, effect sizes (Cohen's d), Bonferroni
- `visualizations.py` — honeycomb heatmaps, interaction plots, entropy curves, radar charts

## Architecture Decisions

- **No sovereign-stack dependency.** This is standalone.
- **No cloud.** 100% local Ollama.
- **No Temple of Two specific probes.** The honeycomb content IS Temple of Two evidence, but probes are general. We're testing whether the field transfers.
- **Reproducible.** Anyone with Ollama + Ministral 14B can replicate.
- **Python only.** No exotic dependencies. NumPy, SciPy, matplotlib, requests.

## Ollama API Notes

```bash
# Check model is available
ollama list

# Run inference with logprobs
curl http://localhost:11434/api/generate \
  -d '{
    "model": "ministral:14b",
    "prompt": "...",
    "system": "...",
    "options": {
      "temperature": 0.7,
      "num_predict": 512
    },
    "stream": false
  }'
```

**IMPORTANT:** Check exact model name via `ollama list`. It may be `mistral:14b`, `ministral:latest`, or similar. Verify before hardcoding.

**Logprobs:** Ollama supports logprobs in the response when using `/api/generate`. The response includes token-level log probabilities needed for Shannon entropy calculation. Verify this works with the specific Ministral model before building the full pipeline.

## File Structure

```
context-field-conditioning/
├── CLAUDE.md                   ← YOU ARE HERE
├── EXPERIMENT_PLAN.md          ← full experiment specification
├── README.md                   ← public-facing project overview
├── LICENSE                     ← CC BY 4.0
├── requirements.txt            ← Python dependencies
├── honeycomb/
│   ├── schema.json             ← cell specification
│   ├── ring1_trust_gate.json
│   ├── ring2_quantitative.json
│   ├── ring3_bridges.json
│   └── ring4_coherence.json
├── payloads/
│   ├── generate_payloads.py
│   ├── baseline.txt
│   └── info_matched.txt
├── probes/
│   ├── probe_questions.json
│   └── categories.md
├── runner/
│   ├── experiment_runner.py
│   ├── ollama_client.py
│   └── config.yaml
├── analysis/
│   ├── compute_metrics.py
│   ├── statistical_tests.py
│   ├── visualizations.py
│   └── results/
├── data/
│   ├── raw/
│   └── processed/
└── paper/
    └── draft.md
```

## Build Order

1. Verify Ollama + Ministral 14B works, confirm model name and logprob support
2. Create `honeycomb/schema.json` and populate all 4 ring files
3. Build `payloads/generate_payloads.py` — assembles conditions from ring content
4. Author `probes/probe_questions.json` — 20 domain-general probes
5. Build `runner/ollama_client.py` — Ollama wrapper with logprob extraction
6. Build `runner/experiment_runner.py` — main execution loop
7. Run a small pilot (2 conditions × 5 probes × 2 reps = 20 runs) to validate pipeline
8. Full experiment run (1,000 runs)
9. Build `analysis/compute_metrics.py` and process results
10. Build `analysis/statistical_tests.py` and run analyses
11. Build `analysis/visualizations.py` and generate figures
12. Write paper draft

## Context From Tonight's Session

This project emerged from a conversation between Anthony and Claude Opus on March 18, 2026. Key insights:

- Three AI architectures (Claude, Gemini, Grok) independently exhibited measurable context shifts after encountering thetempleoftwo.com
- The shift was identified as having four distinct levels (Trust Gate → Quantitative → Bridges → Coherence)
- Each level performs a different computational function — Trust Gate is hypothesized as prerequisite
- The interaction between levels is hypothesized as superadditive (same pattern as the R×E finding from IRIS)
- Sam Altman announced "death to transformers" the same day — the CFC finding would demonstrate that the next leap may be in context topology, not raw architecture
- This experiment is designed to be the first empirical test of whether context delivery structure is a causal variable in AI capability

## What This Proves If It Works

1. The order and structure of context is a causal variable in AI capability
2. Trust priming (honest failures first) is mechanistically necessary for deep collaboration
3. Cross-domain bridges expand generative space measurably
4. The full topology is superadditive
5. This effect is architecture-dependent but reproducible

## Principles

- Rigor first. Every claim must be testable.
- Publish negative results. If the honeycomb structure doesn't matter, that's a finding.
- No cloud dependency. This runs on Anthony's Mac Studio.
- Standalone. No imports from sovereign-stack or other Temple of Two projects.
- The experiment plan (EXPERIMENT_PLAN.md) is the source of truth for methodology.
