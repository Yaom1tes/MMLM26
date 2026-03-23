# March Madness ML Bracket Predictor

An ensemble machine learning model for predicting NCAA tournament outcomes, built for [Kaggle's March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition. (And used for my research lab's friendly competition). This model predicts win probabilities for every possible tournament matchup and simulates full brackets via Monte Carlo.

## Model Architecture

The predictor uses a weighted ensemble of three models with isotonic calibration:

| Model | Weight | Role |
|-------|--------|------|
| XGBoost | 0.45 | Primary predictor; by far the strongest on structured tabular data |
| LightGBM | 0.35 | Adds diversity with different tree-building strategy |
| Logistic Regression | 0.20 | Stabilizes calibration on tail probabilities |

Each model is wrapped in `CalibratedClassifierCV` (isotonic) in order to  improve Brier score which is the competition's implicit evaluation metric. 

Hyperparameters are tuned via **randomized search over 150 parameter combinations**, evaluated with walk-forward cross-validation on 2018-2025 tournament data.

## Features

The model engineers **~90 matchup-differential features** from multiple data sources:

**Box Score Stats** - Computed from Kaggle's `RegularSeasonDetailedResults`: offensive/defensive efficiency, shooting percentages, rebounding, turnover rates, pace (possessions), plus late-season and last-5/last-10 game form.

**Barttorvik Metrics** *(men's only)* - Adjusted offensive/defensive efficiency (`adjoe`, `adjde`), Barthag (win probability vs. average team), WAB (wins above bubble), strength of schedule (`sos`, `elite SOS`, `ncsos`), conference-adjusted metrics, and quality record stats. Integrated via fuzzy name-matching crosswalk (~360 teams matched).

**Elo Ratings** - Custom Elo system (K=20) computed from regular season game-by-game results.

**Massey Ordinals** - Average ranking from multiple computer ranking systems (men's only).

**Coach Experience** - I decided to also factor in historical tournament games and win percentage for each team's coach.

**Seed Interactions** - Seed differential, sum, ratio, plus interaction terms with Elo, net efficiency, Barttorvik metrics, and coach experience. (I realize that this is lowkey turning my model very biased towards seeds but will work around it next year) 

## Backtest Results

I decided to only use up to 2018 (I would have prefered to only use the years after 
Walk-forward validation: train on all data before year *Y*, evaluate on year *Y*'s tournament.

| Year | Brier Score | Notes |
|------|-------------|-------|
| 2018 | ~0.19 | |
| 2019 | ~0.17 | |
| 2020 | N/A | COVID (cancelled) |
| 2021 | ~0.21 | Bubble tournament, high variance |
| 2022 | ~0.22 | Historically chaotic (that St. Peter's run) |
| 2023 | ~0.19 | |
| 2024 | ~0.15 | Best backtest year |
| 2025 | ~0.19 | |

**Mean Brier: ~0.19** (coin-flip baseline is 0.25; decent Kaggle entries are typically 0.10–0.14)

## Outputs

| File | Description |
|------|-------------|
| `submission_2026.csv` | Kaggle submission, win probabilities for all possible matchups |
| `bracket_2026_rounds_{M/W}.txt` | Predicted bracket (use for actual bracket submissions) |
| `bracket_2026_simulated_{M/W}.csv` | Championship probabilities from 10,000 Monte Carlo simulations |
| `model_feature_importance_{M/W}.png` | Visual for Top 30 features by XGBoost importance |
| `hyperparam_search_results.csv` | All 150 hyperparameter combos with Brier scores |

### Requirements

```
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib
```

### Data Setup

1. Download the dataset from [Kaggle's March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and place the CSVs in a `data/` directory.
2. *(Optional, men's only)* Download team results from [Barttorvik](https://barttorvik.com) and place `mens_all_team_results.csv` in the project root.

### Run

```bash
python march_madness.py
```


Full pipeline takes around 2-3 hours (using a 150-combo hyperparameter search on a Mac M2 Air base model), so will vary by system and parameters.
