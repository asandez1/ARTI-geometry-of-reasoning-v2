# ARTI: Detecting and Routing Reasoning Types from Embedding Geometry

**Author**: Ariel Sandez
**ORCID**: [0009-0004-7623-6287](https://orcid.org/0009-0004-7623-6287)
**Status**: Draft v2.7 (final draft) | February 2026

---

<p align="center">
  <img src="docs/figures/figure6_umap_reasoning_types.png" alt="UMAP projection of 10 reasoning types in embedding space" width="85%">
</p>
<p align="center"><em>UMAP projection of 7,500 reasoning traces (384D) colored by type. Pre-factorization embeddings (left) show distinct geometric clusters; post-factorization features (right) collapse into noise.</em></p>

---

## Overview

We show that different reasoning types leave **distinct geometric signatures** in transformer embedding spaces, detectable by a lightweight classifier with only 20K parameters. This repository contains the complete code and results for 20 experiments (E5--E19) spanning reasoning type detection, type-routed scoring, and systematic architectural diagnosis.

### Key Results

| Result | Value |
|--------|-------|
| ARTI ensemble accuracy (10 reasoning types) | **84.2%** (8.4x chance) |
| StrategyQA via ARTI-routed scoring | **81.0%** (+32.6pp over baseline) |
| GSM8K (math reasoning) | **48.1%** (+23pp over chance) |
| Pre- vs post-factorization type signal | **3.5x** gap (76.7% vs 21.9%) |
| GCMC manifold geometry R^2 | **0.678** (neuroscience parallel validated) |

### Three Main Contributions

1. **ARTI (Active Reasoning Type Identifier)** -- Classifies 10 reasoning types from embedding geometry alone (no text access). Evolution: v1 54% --> v2 70.8% --> v3 71.5% --> ensemble **84.2%**.

2. **ARTI-Routed Scoring** -- Soft-blends cosine similarity and direct classification via frozen ARTI type probabilities. Breaks the non-math ceiling: **StrategyQA 81.0%** while preserving GSM8K at 48.1%.

3. **Frozen Encoder Proof** -- Three-experiment falsification chain (E16-->E17-->E18) establishes that the frozen encoder is *structurally necessary*: any unfreezing of a shared transformer destabilizes answer embeddings for generic-label scoring.

---

## Architecture

<p align="center">
  <img src="docs/figures/figure_architecture_arti_routed.png" alt="ARTI-Routed Scorer architecture" width="80%">
</p>
<p align="center"><em>ARTIRoutedScorer: frozen ARTI produces type probabilities that blend cosine (for content-rich answers) and direct classification (for generic labels) via a learned router.</em></p>

The CO-FRN (Continuous Operator Factorized Reasoning Network) pipeline:
- **Encoder**: Frozen GPT-2 (124M) --> 256D projection
- **Operator Manifold**: 16 learnable anchors on ~10D reasoning manifold
- **Reasoning Engine**: HilbertTreeFast (beam=2, depth=3)
- **ARTI Router**: Frozen ensemble (20K params) blending dual scoring heads
- **Total new params**: ~132K (adapter 98K + router 11 + direct MLP 33K)

---

## 10 Reasoning Types

| Type | Best Accuracy | Geometric Character |
|------|:---:|---|
| Counterfactual | 99.4% | Strong geometric signal, isolated cluster |
| Analogy | 99.3% | Partially separated in manifold |
| Systemic Cause | 98.9% | Strong trajectory signal |
| Abduction | 98.7% | Distinct cluster, high separability |
| Conservation | 90.3% | Moderate separation |
| Decomposition | 86.8% | Moderate separation |
| Behavioral Cause | 76.7% | Sub-cluster of causal types |
| Induction | 89.0% | Far-left PCA cluster |
| Deduction | 54.0% | Tight manifold cluster, high norm |
| Physical Cause | 38.8% | Domain-general residual (hardest type) |

---

## Experiments

20 experiments across detection, diagnosis, and architectural validation:

| # | Experiment | Key Finding |
|---|-----------|-------------|
| E5 | Controller Pipeline | TypeClassifier 76.7%, zero benchmark lift |
| E6 | Rank Ablation | Rank-independent: entropy=1.0 at all ranks {4--64} |
| E7 | Rule Injection | 75 rules, zero lift on benchmarks |
| E8 | Controller Fine-Tuning | +7.9pp TypeClf, zero benchmark lift |
| E9 | Root Cause Diagnosis | tree_scorer ignores best_hyp (0/100 prediction changes) |
| E10 | Cosine Scoring Fix | Pipeline integrity restored, GSM8K 47.4% |
| E11 | Entropy Fix | Entropy 0.99996 --> 0.996, first anchor differentiation |
| E12 | Temperature Ablation | sqrt(d) removal: entropy --> 0.937, 3/4 benchmarks pass |
| E12b | ARTI Ensemble | **84.2% accuracy** (8.4x chance) |
| E13 | Gradient Diagnostic | Cosine collapse on generic labels (~0.07 score range) |
| E15 | ARTI-Routed Scoring | **StrategyQA 81.0%** (+32.6pp), first non-math success |
| E16 | Unfrozen + ARTI | FAIL: SQA 54.0%, distribution shift breaks routing |
| E17 | Two-Encoder | FAIL: SQA 54.0%, ARTI shift hypothesis *falsified* |
| E18 | Freeze Answers | FAIL: SQA 54.0%, GSM8K regresses -5.5pp |
| E19 | Manifold Geometry (GCMC) | D_eff rho=-0.624, center align rho=0.685, R^2=0.678 |

<p align="center">
  <img src="docs/figures/figure5_entropy_trajectory.png" alt="Entropy trajectory across fixes" width="75%">
</p>
<p align="center"><em>Progressive resolution of the uniform-attention paradox: entropy drops from 0.99996 to 0.937 across three independent fixes (E10--E12).</em></p>

---

## Benchmark Results (3-seed means, full test sets)

| Benchmark | N | Chance | Best (E12) | ARTI-Routed (E15) |
|-----------|---|:---:|:---:|:---:|
| GSM8K | 1,319 | 25.0% | 49.6% | 48.1% |
| ARC Challenge | 1,172 | 25.0% | 30.0% | 28.6% |
| **StrategyQA** | 1,603 | 50.0% | 48.8% | **81.0%** |
| FOLIO | 203 | 33.3% | 34.2% | 33.5% |

---

## Repository Structure

```
.
├── docs/
│   ├── paper13_v2.md              # Manuscript v2.7 (final draft)
│   ├── paper13_v2.html            # HTML version
│   └── figures/                   # All paper figures + generation scripts
├── experiments/
│   ├── shared/                    # Core modules (~4,900 LOC)
│   │   ├── arti.py                #   ARTI v1 classifier
│   │   ├── arti_v2.py             #   ARTI v2 trajectory classifier
│   │   ├── model.py               #   CO-FRN assembly
│   │   ├── reasoning_engine.py    #   HilbertTree + ARTIRoutedScorer
│   │   ├── factorization.py       #   Continuous operator manifold
│   │   ├── controller.py          #   GeometricReasoningController
│   │   ├── encoder.py             #   Frozen encoder wrappers
│   │   ├── data_utils.py          #   GSM8K, ARC, StrategyQA, FOLIO loaders
│   │   ├── train_utils.py         #   Training loop + curriculum
│   │   ├── rules.json             #   75 reasoning rules across 7 domains
│   │   └── ...
│   └── experiment5_operators/     # All experiment scripts (E5--E19)
│       ├── run_arti.py            #   ARTI v1 training
│       ├── run_arti_ensemble.py   #   Ensemble (84.2%)
│       ├── run_e15_routed_scoring.py  #   ARTI-routed scoring
│       ├── run_e19_manifold_geometry.py  #   GCMC analysis
│       ├── results/               #   JSON results + figures
│       └── ...
```

---

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

---

## Quick Start

```bash
# Train ARTI ensemble
python experiments/experiment5_operators/run_arti.py
python experiments/experiment5_operators/run_arti_v2.py
python experiments/experiment5_operators/run_arti_ensemble.py

# Run ARTI-routed scoring (E15 — the main result)
python experiments/experiment5_operators/run_e15_routed_scoring.py

# Run GCMC manifold geometry analysis (E19)
python experiments/experiment5_operators/run_e19_manifold_geometry.py

# Generate figures
python docs/figures/generate_all_figures.py
```

> **Note**: Model weights (`.pt`) are excluded from this repo due to size (~13GB). All weights are regenerated by running the experiment scripts above.

---

## Citation

```bibtex
@article{sandez2026arti,
  title={ARTI: Detecting and Routing Reasoning Types from Embedding Geometry},
  author={Sandez, Ariel},
  year={2026},
  note={Draft v2.7}
}
```

---

## Contact

**Ariel Sandez** -- ariel.sandez@fortegrp.com
[LinkedIn](https://www.linkedin.com/in/sandez/) | [ORCID](https://orcid.org/0009-0004-7623-6287)
