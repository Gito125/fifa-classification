# FIFA Player Position Classifier

Predicts a football player's position from their attribute ratings. Started
on a synthetic 2,000-row dataset with 6 aggregate stats (accuracy capped at
~48%), diagnosed *why* it capped there, then rebuilt on the real FIFA 20
player dataset (18,483 players, 30 granular sub-skills) — accuracy: **60%**,
with much cleaner per-position behavior. Full history below.

## Project Structure

```text
fifa-20-classification/
├── data/
│   ├── fifa-20-data.csv        # Cleaned dataset: 7,348 rows × 52 columns
├── models/                     # Serialized trained model artifacts (.joblib)
├── results/                    # Generated evaluation charts, confusion matrices & metrics
├── scripts/                    # Pipeline scripts (data preparation, model training)
│   ├── prepare_data.py         # Cleans raw data, filters roles & engineers features
│   └── train.py                # Trains, evaluates & compares all benchmark models
├── pyproject.toml              # Project dependencies & metadata
├── uv.lock                     # Reproducible dependency lockfile
├── .gitignore                  # Environment, cache & artifact ignore patterns
└── README.md                   # Project documentation & benchmark findings
```

## Data

- **Raw Data (`data/raw/players_20.csv`)**: The FIFA 20 "Complete Player Dataset"
  (originally [stefanoleone992 on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset)),
  mirror URL:
  `https://raw.githubusercontent.com/asumapng/FIFA_EDA/main/players_20.csv`
  - 18,483 real players, 106 columns: identity/club/nation info, 6 aggregate
    ratings (`pace/shooting/passing/dribbling/defending/physic`), 30 granular
    sub-skills (`attacking_finishing`, `defending_standing_tackle`,
    `mentality_interceptions`, etc.), and goalkeeper-specific ratings.

- **Cleaned Data (`data/fifa-20-data.csv`)**: The output of `prepare_data.py`:
  7,348 rows × 52 columns, ready to load and model on directly (see **Data cleaning** below).

## Setup (`uv`)

Install dependencies using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

This reads [`pyproject.toml`](file:///home/gideon/Documents/CODE/LEARNING/DataCamp/Practice_Projects/fifa-20-classification/pyproject.toml) and sets up `.venv` with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib`.

> [!NOTE]
> If training XGBoost as part of the tree baseline and voting ensemble, add it via:
> ```bash
> uv add xgboost
> ```

## Running the Pipeline

```bash
# 1. Prepare and clean raw data, engineering composite features
uv run python scripts/prepare_data.py

# 2. Train and evaluate all models, outputting comparison metrics and confusion matrices
uv run python scripts/train.py
```

## Models & Benchmark Results

Evaluated classifiers: **KNN, Random Forest, Logistic Regression, SVM**, and a **Voting Ensemble**. `train.py` also includes XGBoost as a second tree-based baseline alongside Random Forest, then builds the ensemble as a soft-voting average of those two (Random Forest + XGBoost), since tree models had the most reliable `predict_proba` output for voting on this feature set.

| Model | Accuracy | Macro F1 |
|---|---|---|
| KNN (k=15) | 0.554 | 0.331 |
| SVM (rbf) | **0.602** | 0.371 |
| Logistic Regression | 0.597 | 0.367 |
| Random Forest | 0.592 | 0.353 |
| XGBoost | 0.590 | 0.369 |
| Ensemble (RF + XGBoost, soft voting) | 0.599 | **0.377** |

*(Single 80/20 stratified split, `random_state=42` — see **Limitations** on why these numbers are a bit noisy.)*

## Data Cleaning Pipeline

Applied in `scripts/prepare_data.py`, in order:

1. **Dropped SUB/RES rows**: `team_position` of "SUB" (bench, 7,914 rows) or
   "RES" (reserve, 2,981 rows) is a squad-status label, not an active tactical position.
2. **Collapsed position-slot variants** into base roles:
   `LCB`/`RCB` → `CB`, `LDM`/`RDM` → `CDM`, `LCM`/`RCM` → `CM`, `LAM`/`RAM` → `CAM`, `LS`/`RS` → `ST`, `LF`/`RF` → `CF`.
3. **Dropped `defending_marking`**: 100% null across all 18,483 rows (EA
   retired that stat as of FIFA 20; the scraper kept the empty column).
4. **Filled position-conditional NaNs with 0, not dropped**: The 6 aggregate
   ratings are only computed for outfield players (NaN for all GKs); the 6
   `gk_*` ratings are only computed for GKs (NaN for everyone else). Both are
   "not applicable," not missing data — 0 keeps the row and doubles as a free
   signal for the GK/outfield split.
5. **Engineered 3 features** (see below) — `attack_minus_defend` came out as
   the single most important feature by Random Forest importance.
6. **Safety-net dropna** on sub-skills + physical columns — confirmed 0 rows
   had anything missing after step 4.

**Net**: 18,483 raw rows → 7,348 cleaned rows, 15 single-label position classes.

## Engineered Features

- `attack_minus_defend`:
  $$\text{attacking\_finishing} + \text{skill\_dribbling} + \text{power\_shot\_power} - \text{defending\_standing\_tackle} - \text{defending\_sliding\_tackle} - \text{mentality\_interceptions}$$
- `defensive_composite`:
  Mean of `defending_standing_tackle`, `defending_sliding_tackle`, and `mentality_interceptions`.
- `playmaking_composite`:
  Mean of `skill_long_passing`, `attacking_short_passing`, and `mentality_vision`.

## What Actually Improved, and What Didn't

Switching from the synthetic 6-column dataset to this one fixed:
- **CB**: 0% recall → 92% recall
- **CDM**: 0% recall (every CDM predicted as CM) → 42% recall, with a real
  defensive gradient now visible (mean `mentality_interceptions`: CDM 66 →
  CM 61 → CAM 44)
- **GK, ST**: already strong, stayed strong (100%, 88% recall)

What it did *not* fix, on purpose:
- **Left vs. right mirror positions** (LB↔RB, LM↔RM, LW↔RW, LWB↔RWB) are
  still heavily confused even with all 30 real sub-skills on 18k real
  players. Checked `preferred_foot` as a possible proxy for side — it's
  uncorrelated with which side a player plays (~75% right-footed on *both*
  left- and right-side players). Conclusion: this isn't a data-quality or
  feature-engineering problem, it's that mirrored positions genuinely have
  near-identical skill profiles in real football. No amount of feature
  engineering will manufacture a side signal that isn't there.

## Limitations

- **Class imbalance**: CF (56 total), LWB (58), RWB (58) are small enough
  that their test-set metrics (0% recall on all three) are more about
  sample size (~11 test rows each) than true unlearnability. A single 80/20
  split leaves them noisy — stratified k-fold CV would give a more honest
  number.
- **`team_position` is a snapshot**, not a true label of "what position is
  this player" — it reflects one squad-role assignment, while the raw data's
  `player_positions` field lists every position a player can play (e.g.
  Messi: "RW, CF, ST"). Multi-label modeling against that field is a
  possible follow-up if single-label accuracy isn't enough.

## Possible Next Steps

- Merge L/R pairs into unified roles (fullback, wide-mid, winger) — should
  push accuracy well past 75-80%, since that's the confusion driving most of
  the current error.
- Merge CF into ST and LWB/RWB into a wingback class to fix the small-sample
  classes.
- Stratified k-fold CV instead of a single split, given the small-class
  noise above.
- Hyperparameter tuning (only default params used so far for all 5 models).