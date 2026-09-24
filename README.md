# FIFA Player Position Classification

A machine-learning project predicting football player positions from in-game skill attributes, physical traits, and player characteristics using FIFA player datasets.

---

## Quick Navigation

- [Project Structure](#project-structure)
- [Setup with uv](#setup-with-uv)
- [How to Select a Version (v1 vs v1.1)](#how-to-select-a-version-v1-vs-v11)
- [How to Run the Pipeline (Step-by-Step)](#how-to-run-the-pipeline-step-by-step)
  - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
  - [Step 2: Model Training (`--tune` vs `--train`)](#step-2-model-training---tune-vs---train)
  - [Step 3: Multi-Model Evaluation & Comparison](#step-3-multi-model-evaluation--comparison)
- [Version Overview](#version-overview)
- [Available Models](#available-models)
- [Future Roadmap (v1.2 & v2)](#future-roadmap)

---

## Project Structure

```text
fifa-20-classification/
├── configs/                          # Config-driven execution files
│   ├── v1.json                       # Version 1 baseline config (51 features)
│   └── v1.1.json                     # Version 1.1 feature-engineered config (77 features)
├── data/
│   ├── raw/                          # Raw datasets from FIFA 15 to FIFA 21
│   │   ├── players_15.csv ... players_21.csv
│   └── processed/                    # Version-specific processed datasets
│       ├── v1/                       # Output for v1 (train_data.csv, test_data.csv)
│       └── v1.1/                     # Output for v1.1 (train_data.csv, test_data.csv)
├── models/                           # Serialized model artifacts (.pkl)
│   ├── v1/                           # Saved models for v1 + v1 README
│   └── v1.1/                         # Saved models for v1.1
├── figures/                          # Generated evaluation plots & figures
│   ├── v1/                           # Confusion matrices & plots for v1
│   └── v1.1/                         # Charts, heatmaps & feature importances for v1.1
├── plans/                            # Future architectural roadmaps
│   └── v2-hierarchical-classification.md
├── scripts/                          # Reusable, config-driven pipeline scripts
│   ├── common.py                     # Config loader, data loader & evaluation utilities
│   ├── 01-understand-data.py         # Exploratory data analysis
│   ├── 02-pre-process-data.py        # Config-driven data cleaning & feature engineering
│   ├── 03-knn.py                     # K-Nearest Neighbors (--tune / --train)
│   ├── 04-svm.py                     # Support Vector Machine (--tune / --train)
│   ├── 05-logistic-regression.py     # Logistic Regression (--tune / --train)
│   ├── 06-random-forest.py           # Random Forest (--tune / --train)
│   ├── 07-ensemble.py                # Soft-Voting Ensemble (--tune / --train)
│   └── 08-evaluate-all.py            # Comprehensive evaluation & multi-model comparison
├── pyproject.toml                    # Dependencies & project metadata
└── README.md                         # This guide
```

---

## Setup with `uv`

The project uses [`uv`](https://docs.astral.sh/uv/) for reproducible Python environment management.

Install dependencies into `.venv`:

```bash
uv sync
```

Dependencies include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib`.

---

## How to Select a Version (v1 vs v1.1)

All scripts are **config-driven**. You select which version to run by passing the `--config` parameter:

- For **v1** (baseline 51 features):
  ```bash
  --config configs/v1.json
  ```
- For **v1.1** (engineered physical, demographic & trait features):
  ```bash
  --config configs/v1.1.json
  ```

When you pass a config file:
1. The script reads `"model_version"` and displays it at the top of the terminal.
2. Processed datasets automatically go to `data/processed/<version>/`.
3. Model artifacts automatically save to `models/<version>/`.
4. Figures and charts automatically save to `figures/<version>/`.
5. Hyperparameters automatically save and load from that specific JSON config.

*(If you omit `--config`, scripts default to `configs/v1.json`).*

---

## How to Run the Pipeline (Step-by-Step)

Here is the exact step-by-step workflow for running **v1.1**:

### Step 1: Data Preprocessing

Cleans the raw FIFA 20 data, removes target leakage columns, maps positions into 10 classes, and engineers the new physical and trait features:

```bash
uv run python scripts/02-pre-process-data.py --config configs/v1.1.json
```

**Output**:
- Creates `data/processed/v1.1/train_data.csv` (7,735 players, 78 columns)
- Creates `data/processed/v1.1/test_data.csv` (1,934 players, 78 columns)

---

### Step 2: Model Training (`--tune` vs `--train`)

Every model script provides two modes:

| Mode | What It Does | When to Use |
|---|---|---|
| **`--tune`** | Runs 5-fold cross-validation `GridSearchCV`, finds the best parameters, **automatically writes them to the config file**, and saves the model. | Use the **first time** you run a version, or when searching for new parameters. |
| **`--train`** | **Directly loads** the best parameters from the config file and trains the model in seconds without running grid search. | Use for **fast retraining**, testing, or reproducing results. |

#### Tuning and Training the Models for v1.1

Run the tuning commands one by one:

```bash
# 1. K-Nearest Neighbors
uv run python scripts/03-knn.py --config configs/v1.1.json --tune

# 2. Support Vector Machine
uv run python scripts/04-svm.py --config configs/v1.1.json --tune

# 3. Logistic Regression
uv run python scripts/05-logistic-regression.py --config configs/v1.1.json --tune

# 4. Random Forest
uv run python scripts/06-random-forest.py --config configs/v1.1.json --tune

# 5. Soft-Voting Ensemble
# (Automatically reads the best parameters of the 4 models above from configs/v1.1.json and tunes the voting weights)
uv run python scripts/07-ensemble.py --config configs/v1.1.json --tune
```

> **Note on Fast Retraining**:  
> Once tuned, the parameters are saved in `configs/v1.1.json`. You can retrain any model instantly without searching:
> ```bash
> uv run python scripts/03-knn.py --config configs/v1.1.json --train
> uv run python scripts/07-ensemble.py --config configs/v1.1.json --train
> ```

---

### Step 3: Multi-Model Evaluation & Comparison

After training the models, run the evaluation script:

```bash
uv run python scripts/08-evaluate-all.py --config configs/v1.1.json
```

This script:
1. Loads all 5 trained models from `models/v1.1/`.
2. Evaluates them against `data/processed/v1.1/test_data.csv`.
3. Prints a side-by-side metrics table (Accuracy, Macro F1, Precision, Recall).
4. Generates a comprehensive visualization suite saved to `figures/v1.1/`:
   - `model_comparison_v1.1.png`: Grouped bar chart comparing all 5 models.
   - `per_class_f1_heatmap_v1.1.png`: Heatmap showing F1 score across all 10 positions for each model.
   - `feature_importance_rf_v1.1.png`: Top 25 feature importances from Random Forest.
   - `height_weight_by_position_v1.1.png`: Box plots showing height and weight distributions by position.
   - `v1_vs_v1_1_comparison.png`: Direct before/after comparison chart (v1 baseline vs v1.1).

---

## Version Overview

### Version 1 (Baseline)
- **Features (51)**: Overall, potential, value, wage, weak foot, skill moves, 6 aggregate stats, 6 GK stats, and 28 detailed technical/movement/mentality sub-skills.
- **Config**: [`configs/v1.json`](configs/v1.json)
- **Detailed Documentation**: See [`models/v1/README.md`](models/v1/README.md) for full v1 evaluation metrics and feature documentation.

### Version 1.1 (Engineered Features)
- **Features (77)**: All 51 baseline features + 26 newly engineered features:
  - `height_cm`, `weight_kg`: Raw physical attributes.
  - `preferred_foot_right`: Binary footedness (Right=1, Left=0).
  - `work_rate_attack`, `work_rate_defense`: Ordinal work rate (Low=0, Med=1, High=2).
  - `body_type_lean`, `body_type_stocky`: One-hot body build model (star player custom bodies mapped to standard builds).
  - `international_reputation`: Player global profile rating (1–5).
  - 18 high-signal `player_traits` (e.g. `Comes For Crosses`, `Playmaker`, `Dives Into Tackles`, `Power Header`, etc.) encoded as multi-hot indicators.
- **Config**: [`configs/v1.1.json`](configs/v1.1.json)

---

## Available Models

All 5 algorithms classify players into 10 position classes (`CAM`, `CB`, `CDM`, `CM`, `GK`, `LB`, `LW`, `RB`, `RW`, `ST`):

1. **KNN** (`03-knn.py`): K-Nearest Neighbors with `StandardScaler`.
2. **SVM** (`04-svm.py`): Support Vector Classifier with `StandardScaler` and probability estimates enabled.
3. **Logistic Regression** (`05-logistic-regression.py`): L2-regularized multinomial logistic regression with `StandardScaler`.
4. **Random Forest** (`06-random-forest.py`): Multi-tree ensemble providing non-linear feature interactions and feature importance rankings.
5. **Soft-Voting Ensemble** (`07-ensemble.py`): Weighted combination averaging the predicted probability distributions of all 4 base models.

---

## Future Roadmap

### v1.2: Multi-Season Out-of-Time Validation
- Train models on historical seasons (**FIFA 15 through FIFA 20**, ~50,000+ player-seasons).
- Test on an unseen future season (**FIFA 21**, ~10,000 players).
- Supported directly in `scripts/02-pre-process-data.py` using `"split_strategy": "by_files"`.

### v2: Two-Stage Hierarchical Classification
- **Stage 1**: Predict broad position group (`GK`, `DEF`, `MID`, `ATT`).
- **Stage 2**: Sub-classifiers for specific positions within the predicted group (`CB/LB/RB`, `CDM/CM/CAM`, `ST/LW/RW`).
- **Soft Routing**: Blends Stage 1 group probabilities into Stage 2 to prevent cascading routing errors.
- Full architectural plan: [`plans/v2-hierarchical-classification.md`](plans/v2-hierarchical-classification.md).