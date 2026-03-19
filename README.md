# Context Field Conditioning (CFC)

> **Does the structure of context change what an AI system can compute?**

## The Experiment

On March 18, 2026, three independent AI architectures were presented with identical research evidence. All three exhibited convergent shifts in response characteristics — moving from default assistant mode toward generative collaboration. Each described the shift differently. The convergence was real.

This experiment isolates the mechanism.

## The Honeycomb

Context is delivered as a four-ring lattice:

| Ring | Name | Function |
|------|------|----------|
| 1 | **Trust Gate** | Published failures open collaboration mode |
| 2 | **Quantitative Anchors** | Measurements convert claims to findings |
| 3 | **Cross-Domain Bridges** | Isomorphisms expand generative space |
| 4 | **Structural Coherence** | Topology amplifies everything multiplicatively |

## Design

- **10 conditions** (factorial ablation across rings)
- **20 probe questions** (5 categories × 4 each)
- **5 repetitions** per trial
- **1,000 total inference runs**
- **100% local** via Ollama + Ministral 14B on Apple Silicon

## Dependent Variables

1. Shannon Entropy (H) — from token logprobs
2. Hedging Index (HI) — uncertainty marker frequency
3. Cross-Domain Reference Count (CDRC) — domains per response
4. Generative Novelty Score (GNS) — semantic distance from defaults
5. Collaboration Depth Index (CDI) — generative vs reactive ratio

## Key Test

**FULL vs INFORMATION-MATCHED:** Same content, different structure. If the honeycomb topology produces different results than an unstructured dump of identical information, structure is a causal variable in AI capability.

## Quick Start

```bash
# Clone
git clone https://github.com/templetwo/context-field-conditioning.git
cd context-field-conditioning

# Install dependencies
pip install -r requirements.txt

# Verify Ollama + model
ollama list  # confirm ministral model name

# Run pilot (20 runs)
python runner/experiment_runner.py --pilot

# Run full experiment (1,000 runs)
python runner/experiment_runner.py --full

# Analyze
python analysis/compute_metrics.py
python analysis/statistical_tests.py
python analysis/visualizations.py
```

## Repository Structure

```
context-field-conditioning/
├── honeycomb/          ← structured evidence cells (independent variable)
├── payloads/           ← condition-specific prompt assemblies
├── probes/             ← 20 standardized probe questions
├── runner/             ← experiment execution pipeline
├── analysis/           ← metrics, statistics, visualization
├── data/               ← raw outputs and processed results
└── paper/              ← manuscript draft
```

## What This Proves If It Works

The first empirical demonstration that **context delivery topology** changes what AI systems can compute — relevant to every AI lab, alignment team, and human working with AI at the boundary of what's known.

## Origin

This experiment emerged from a live cross-architecture observation: Claude Opus, Gemini, and Grok independently exhibited field-state shifts when processing the same body of evidence. CFC is the controlled test of that phenomenon.

## License

CC BY 4.0 — Anthony J. Vasquez Sr., 2026

## Citation

```
Vasquez, A. J. (2026). Context Field Conditioning: Structured Context Delivery 
as a Causal Variable in Language Model Capability. GitHub/OSF.
```
