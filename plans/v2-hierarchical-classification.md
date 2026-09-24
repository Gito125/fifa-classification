# V2 Plan: Hierarchical Two-Stage Classification

> **Status**: Planned
> **Created**: 2026-09-24
> **Depends on**: v1 baseline results for comparison

## Overview

Replace the flat 10-class classifier from v1 with a two-stage hierarchical
pipeline that first classifies players into one of 4 general position groups,
then narrows down to the specific position within that group. Uses **soft
routing** so that Stage 1 probabilities flow into Stage 2, reducing error
propagation.

---

## Architecture

```text
                     ┌──────────────┐
  51 features ──────►│   STAGE 1    │──── probabilities ──┐
                     │  4 classes   │                     │
                     └──────────────┘                     │
                                                          ▼
                     ┌────────────────────────────────────────────┐
                     │              SOFT ROUTING                  │
                     │                                            │
                     │  P(GK)  × Sub_GK(pos | X)   → P(GK)      │
                     │  P(DEF) × Sub_DEF(pos | X)  → P(CB/LB/RB)│
                     │  P(MID) × Sub_MID(pos | X)  → P(CDM/CM/CAM)│
                     │  P(ATT) × Sub_ATT(pos | X)  → P(ST/LW/RW)│
                     │                                            │
                     │  Final prediction = argmax over all 10     │
                     └────────────────────────────────────────────┘
```

## Stage 1: General Position Group (4 Classes)

| Group | Label | Source Positions |
|-------|-------|------------------|
| Goalkeeper | `GK` | GK |
| Defender | `DEF` | CB, LB, RB |
| Midfielder | `MID` | CDM, CM, CAM |
| Attacker | `ATT` | ST, LW, RW |

### Expected behavior
- GK vs everyone else: trivially separable (GK base stats, goalkeeping stats).
- DEF vs MID vs ATT: should achieve high accuracy since defending/attacking
  skill profiles diverge significantly (v1 already separates these well).

## Stage 2: Specific Position Sub-Classifiers

### Sub-classifier: DEF → CB vs LB vs RB
- **Training data**: Only rows where `group == DEF`
- **Key challenge**: LB ↔ RB mirror confusion (near-identical skill profiles)
- **Classes**: 3

### Sub-classifier: MID → CDM vs CM vs CAM
- **Training data**: Only rows where `group == MID`
- **Key challenge**: CDM ↔ CM boundary is soft; CAM leans attacking
- **Key distinguishing features**: `mentality_interceptions`, `defending_*`,
  `attacking_finishing`, `mentality_positioning`
- **Classes**: 3

### Sub-classifier: ATT → ST vs LW vs RW
- **Training data**: Only rows where `group == ATT`
- **Key challenge**: LW ↔ RW mirror confusion; LW/RW are minority classes
- **Classes**: 3

### Sub-classifier: GK
- No sub-classifier needed (single position in group).
- Stage 2 output for GK is always `P(GK | GK group) = 1.0`.

## Soft Routing (Probability Combination)

For a given input X, the final probability of each specific position is:

```
P(position_j) = P(group_i | X) × P(position_j | X, group_i)
```

Where:
- `P(group_i | X)` comes from Stage 1's `predict_proba()`
- `P(position_j | X, group_i)` comes from the Stage 2 sub-classifier for group_i

All 3 (or 4) Stage 2 sub-classifiers run on every input. The final prediction
is `argmax` over all 10 position probabilities.

**Why soft routing over hard routing:**
- If Stage 1 gives P(DEF)=0.45, P(MID)=0.44, hard routing picks DEF and
  permanently discards the MID hypothesis. Soft routing preserves it — if
  Sub_MID strongly predicts CDM, it can still win.
- Self-correcting: a borderline CDM/CB player gets fair consideration from both
  sub-classifiers.

## Models to Train & Evaluate

All 5 model types from v1, at **every stage**:

| # | Model | Notes |
|---|-------|-------|
| 1 | KNN | With StandardScaler pipeline |
| 2 | Logistic Regression | With StandardScaler pipeline |
| 3 | Random Forest | No scaler needed |
| 4 | SVM | With StandardScaler pipeline, `probability=True` for soft routing |
| 5 | Soft-voting Ensemble | Combination of the above 4 |

This means training:
- 5 models for Stage 1 (4-class)
- 5 models × 3 sub-classifiers for Stage 2 (DEF, MID, ATT)
- **Total: 20 model artifacts** (excluding the trivial GK pass-through)

### Evaluation Strategy
Each of the 5 model types gets its own full hierarchical pipeline (same model
type at both stages). The ensemble pipeline uses ensemble at both stages.

Compare all 5 hierarchical pipelines against their flat v1 counterparts.

## Features

Use the same 51 features from v1 at both stages. The sub-classifiers may
implicitly learn to ignore irrelevant features (e.g., DEF sub-classifier
learning to downweight `attacking_finishing`).

> **Future consideration**: If v2 results plateau, try feature selection per
> sub-classifier to reduce noise. But start with all 51 for a clean comparison
> against v1.

## Open Decision: Mirror Position Merging

The user is **considering** merging left/right mirror positions:

| Merge | Before | After | Effect |
|-------|--------|-------|--------|
| Fullback | LB + RB | FB | DEF sub-classifier: 2 classes instead of 3 |
| Winger | LW + RW | W | ATT sub-classifier: 2 classes instead of 3 |

**If merged**: Final output is 7 classes instead of 10.

**Arguments for merging:**
- Mirror positions have genuinely identical skill profiles in the data.
- Players frequently switch sides in real football.
- LW/RW are minority classes (17/19 test samples) — merging improves balance.
- Metrics will jump significantly since LB↔RB and LW↔RW confusion disappears.

**Arguments against:**
- Loses granularity — a scout might want to know left vs right.
- If the model can learn even a weak left/right signal, it has value.

**Decision**: Deferred. Build v2 with all 10 classes first, then run an
experiment with merged mirrors as v2.1 to quantify the metric improvement.

## Data Pipeline

```text
data/raw/players_20.csv
    │
    ▼
scripts/02-pre-process-data.py  (same as v1, reuse processed data)
    │
    ▼
data/processed/train_data.csv   (51 features + team_position)
data/processed/test_data.csv
    │
    ▼
scripts/v2/01-train-stage1.py
    │   Creates group labels: GK/DEF/MID/ATT
    │   Trains 5 Stage 1 models
    │   Saves to models/v2/stage1/
    │
    ▼
scripts/v2/02-train-stage2.py
    │   Splits training data by group
    │   Trains 5 models × 3 sub-classifiers
    │   Saves to models/v2/stage2/{def,mid,att}/
    │
    ▼
scripts/v2/03-evaluate-hierarchical.py
    │   Loads Stage 1 + Stage 2 models
    │   Implements soft routing
    │   Evaluates on test_data.csv
    │   Compares against v1 baselines
    │   Saves figures to figures/v2/
    │
    ▼
models/v2/README.md   (auto-generated evaluation summary)
```

## Success Criteria

| Metric | v1 Best (SVM) | v2 Target | Notes |
|--------|---------------|-----------|-------|
| Accuracy | 0.86 | ≥ 0.88 | Modest improvement expected |
| Macro F1 | 0.64 | ≥ 0.68 | Main target; hierarchical should help minority classes |
| Stage 1 Accuracy | N/A | ≥ 0.95 | 4-class problem should be much easier |
| LW/RW Recall | 0.00–0.24 | ≥ 0.30 | Hardest to improve; accept small gains |
| CDM F1 | 0.72 | ≥ 0.75 | Should benefit from MID-only sub-classifier |

## File & Directory Structure

```text
fifa-20-classification/
├── scripts/
│   └── v2/
│       ├── 01-train-stage1.py
│       ├── 02-train-stage2.py
│       └── 03-evaluate-hierarchical.py
├── models/
│   └── v2/
│       ├── stage1/
│       │   ├── knn_stage1.pkl
│       │   ├── logistic_regression_stage1.pkl
│       │   ├── random_forest_stage1.pkl
│       │   ├── svm_stage1.pkl
│       │   └── ensemble_stage1.pkl
│       ├── stage2/
│       │   ├── def/
│       │   │   ├── knn_def.pkl
│       │   │   ├── logistic_regression_def.pkl
│       │   │   ├── random_forest_def.pkl
│       │   │   ├── svm_def.pkl
│       │   │   └── ensemble_def.pkl
│       │   ├── mid/
│       │   │   └── ... (same 5 models)
│       │   └── att/
│       │       └── ... (same 5 models)
│       └── README.md
├── figures/
│   └── v2/
│       ├── stage1_confusion_matrices/
│       ├── stage2_confusion_matrices/
│       └── hierarchical_vs_flat_comparison.png
└── plans/
    └── v2-hierarchical-classification.md   ← this file
```
