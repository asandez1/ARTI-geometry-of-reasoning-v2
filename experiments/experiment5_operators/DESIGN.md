# Experiment 5: Cross-Domain Reasoning Operator Extraction

## Motivation

Papers 11 and 12 (E1-E4) showed CO-FRN extracts mathematical reasoning operators
(+24pp on GSM8K, 100% SVAMP transfer) but fails on all other domains. Two root causes:

1. **Encoder ceiling**: Frozen GPT-2 embeddings carry no signal for ARC/StrategyQA/FOLIO.
   All methods collapse to chance.
2. **Uniform attention**: Entropy regularizer (lambda_entropy=0.01) forces all anchors to
   1/16. No operator specialization. Rho = 2.0 (maximal, opposite of target).

Paper 11 E4 proved domain-invariant reasoning structure EXISTS (consistency = 0.95) when
measured with a sentence-transformer encoder (all-MiniLM-L6-v2, 384D) on reasoning
transitions. The gap: Paper 12 used GPT-2 and the entropy regularizer prevented
specialization.

## Design: Three Sequential Phases with Gates

### Phase 1: Break the Encoder Ceiling

**Question**: Do sentence-transformer embeddings carry cross-domain reasoning signal?

**Conditions**: 4 encoders x 4 benchmarks x {CO-FRN, MLP baseline} = 32 runs, 1 seed

| Encoder | Dim | Rationale |
|---------|-----|-----------|
| gpt2 (control) | 768 | Paper 12 baseline, known to fail |
| all-MiniLM-L6-v2 | 384 | Paper 11 E4's encoder |
| all-mpnet-base-v2 | 768 | Stronger ST, same dim as GPT-2 |
| intfloat/e5-small-v2 | 384 | Instruction-tuned |

**Gate**: Best non-GPT-2 encoder > chance + 5pp on >= 2/3 non-math benchmarks.

### Phase 2: Fix Operator Specialization

**Question**: Does removing entropy regularizer allow input-dependent operator selection?

**Conditions**: 3 entropy strategies x 2 lambda_factorize values x 4 benchmarks = 24 runs

| Entropy Strategy | lambda_entropy | Description |
|-----------------|----------------|-------------|
| uniform (control) | 0.01 | Paper 12 default |
| disabled | 0.0 | Free specialization |
| annealed | 0.01 -> 0.0 over 10 epochs | Start uniform, relax |

Cross with lambda_factorize in {0.0, 0.1}.

**Gate**: (a) participation ratio in [3, 12], (b) accuracy >= Phase 1 on >= 3/4 benchmarks,
(c) rho < 1.5.

### Phase 3: Cross-Domain Operator Transfer

**Question**: Do reasoning operators transfer across domains?

Using best config from Phases 1-2:
- **3a**: Source training on GSM8K + ARC (3 seeds, 30 epochs)
- **3b**: Zero-shot transfer (SVAMP, OpenBookQA, StrategyQA, FOLIO)
- **3c**: Few-shot adaptation (16, 64, 256, 1024 examples, 3 seeds)
- **3d**: Reverse direction (train StrategyQA+FOLIO, test GSM8K+SVAMP)
- **3e**: Operator analysis (anchor heatmap, JS divergence, rho, causal intervention)

## Success Criteria

| # | Criterion | Threshold | Phase |
|---|-----------|-----------|-------|
| C1 | Encoder ceiling broken | > chance + 5pp on >= 2/3 non-math | 1 |
| C2 | Anchor specialization | Participation ratio in [3, 12] | 2 |
| C3 | Rho improvement | Rho < 1.5 (down from 2.0) | 2 |
| C4 | Far-domain zero-shot | Transfer ratio > 0.3 on both far-domain | 3 |
| C5 | Few-shot beats baseline | CO-FRN > LoRA at n=64, p<0.05 | 3 |
| C6 | Domain-invariant operators | JS divergence < 0.5 for >= 4/6 pairs | 3 |

## CLI

```bash
python run_experiment.py --quick    # Phase 1 only, 1 seed, 5 epochs
python run_experiment.py --full     # All phases, 3 seeds (Phase 3)
python run_experiment.py --phase 1  # Run only Phase 1
python run_experiment.py --phase 2  # Phase 2 (needs Phase 1 results)
python run_experiment.py --phase 3  # Phase 3 (needs Phase 2 results)
```
