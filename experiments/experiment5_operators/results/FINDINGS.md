# Experiment 5: Cross-Domain Reasoning Operator Extraction — Findings

**Generated**: 2026-02-22 23:11:38

## Phase 1: Encoder Ceiling

**Gate**: PASS
**Best encoder**: intfloat/e5-small-v2
**Non-math benchmarks above threshold**: 2/3

### Accuracy Matrix

| Encoder | GSM8K | ARC | StrategyQA | FOLIO |
|---------|-------|-----|------------|-------|
| gpt2 | 50.7% ** | 30.1% ** | 51.5% | 35.5% |
| all-MiniLM-L6-v2 | 48.3% ** | 28.3% | 55.3% ** | 35.5% |
| all-mpnet-base-v2 | 50.3% ** | 25.9% | 56.9% ** | 36.5% |
| intfloat/e5-small-v2 | 48.7% ** | 32.2% ** | 56.0% ** | 35.5% |

Chance levels: GSM8K=25%, ARC=25%, StrategyQA=50%, FOLIO=33.3%

## Phase 2: Operator Specialization

**Gate**: FAIL
**Best config**: anneal_lf0.1

### Diagnostics

| Config | Mean PR | Rho | Gate Score |
|--------|---------|-----|-----------|
| constant_lf0.0 | 16.00 | 1.2737318882718682 | 1/3 |
| constant_lf0.1 | 16.00 | 1.28116645788153 | 2/3 |
| disabled_lf0.0 | 15.99 | 1.2798995058983564 | 2/3 |
| disabled_lf0.1 | 15.99 | 1.2870673810442288 | 2/3 |
| anneal_lf0.0 | 16.00 | 1.2599989573160808 | 2/3 |
| anneal_lf0.1 | 16.00 | 1.2888999053587515 | 2/3 |

## Success Criteria Evaluation

| # | Criterion | Threshold | Result | Status |
|---|-----------|-----------|--------|--------|
| C1 | Encoder ceiling broken | >chance+5pp on >=2/3 non-math | 2/3 benchmarks | PASS |
| C2 | Anchor specialization | PR in [3, 12] | PR=16.00 | FAIL |
| C3 | Rho improvement | rho < 1.5 | rho=1.289 | PASS |
| C4 | Far-domain zero-shot | TR > 0.3 on both far-domain | TRs: ['0.000', '0.000'] | FAIL |
| C5 | Few-shot beats baseline | CO-FRN > LoRA at n=64, p<0.05, >=1/2 far-domain | 0/2 wins | FAIL |
| C6 | Domain-invariant operators | JS < 0.5 for >= 4/6 pairs | 0/0 pairs below 0.5 | FAIL |

**Overall: 2/6 criteria met**
