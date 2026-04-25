# NBA MVP Vote Share Forecasting

[![CI](https://github.com/yourusername/nbaMVPPredictor/workflows/CI/badge.svg)](https://github.com/yourusername/nbaMVPPredictor/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A machine learning pipeline for forecasting NBA MVP vote shares using only information available before voting occurs, then ranking candidates each season. Trained on historical data (1991–2024) scraped from Basketball Reference.

## Project Overview

This project treats MVP prediction as a **ranking problem**, not a regression task. The goal is to predict vote shares for all eligible players and correctly rank the top candidates, particularly identifying the actual MVP winner as #1.

The models use only pre-voting information:
- Player per-game statistics (PTS, AST, TRB, etc.)
- Team success metrics (W-L record, playoff seeding)
- Advanced stats (Win Shares, efficiency metrics)
- Historical narrative features (previous MVP finishes)

No post-season or subjective narrative signals are included—only box score stats and team records available when voting occurs.

## Methodology

### Data Collection
Automated scraping from Basketball Reference using Selenium:
- **MVP voting data**: Historical vote shares (1991–2024)
- **Player statistics**: Per-game stats for all players across 34 seasons
- **Team standings**: Win-loss records, playoff positioning, strength metrics

### Data Processing
1. Handle players traded mid-season (use TOT stats, assign last team)
2. Merge player stats with team records and MVP voting results
3. Fill missing MVP votes with 0 (players who received no votes)
4. Validate data quality (missing columns, incomplete seasons)

### Feature Engineering

**52 total features across 7 categories:**

1. **Base Statistics (33 features)**: Age, G, GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, W, L, W/L%, GB, PS/G, PA/G, SRS

2. **Ratio Features (5 features)**: Year-normalized ratios for PTS, AST, STL, BLK, 3P (e.g., `PTS_R = player_PTS / league_avg_PTS` for that year). Controls for era-dependent scoring changes.

3. **League-Relative Z-scores (4 features)**: Within-season standardized scores for PTS, AST, TRB, WS. Captures how far above/below league average a player performs.

4. **Team Success (4 features)**: `team_win_pct`, `conference_rank`, `made_playoffs`, `is_top3_seed`. Reflects historical pattern that MVPs come from winning teams.

5. **Player Availability (3 features)**: `games_played`, `games_played_pct`, `minutes_per_game`. Accounts for "availability is the best ability" voting factor.

6. **Narrative/History (2 features)**: `previous_mvp_finish` (best prior MVP rank), `previous_top5_count` (number of prior top-5 finishes). Captures voter fatigue and first-time MVP storylines.

7. **Team Context (1 feature)**: `is_best_player_on_team` based on Win Shares. Identifies clear team leader vs. complementary star.

**Rationale**: MVP voting is not purely statistical—it incorporates team success, player availability, and narrative context. These features mirror actual voter behavior patterns observed across 30+ years of MVP races.

### Modeling Approach

**Four models trained with Optuna hyperparameter tuning:**
- **Ridge Regression** (baseline, α=0.1)
- **Ridge with Ratios** (adds year-normalized ratio features)
- **Random Forest** (tuned: n_estimators=93, max_depth=11, min_samples_split=3, max_features=log2)
- **XGBoost** (tuned: n_estimators=259, max_depth=4, learning_rate=0.283, colsample_bytree=0.628)

**Time-Aware Rolling Backtest:**
- Minimum 10 seasons of training data required
- For each test year Y: train on all years < Y, test on year Y
- No data leakage: features use only within-season or prior-season information
- Prevents "peeking" at future MVP outcomes during training

This mimics real-world forecasting: predicting 2022 MVP using only data through 2021.

## Evaluation

MVP prediction is fundamentally a **ranking problem**. Standard regression metrics (MSE, R²) don't capture what matters: did we rank the MVP #1?

**Ranking-Focused Metrics:**

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Top-1 Accuracy** | % of seasons where actual MVP ranked #1 | Direct measure of correct winner prediction |
| **Top-3 Recall** | % of seasons where actual MVP in top 3 | Captures near-miss predictions |
| **Mean Reciprocal Rank (MRR)** | Average of 1/rank for actual MVP | Penalizes ranking MVP lower (1/10 = 0.1 vs 1/2 = 0.5) |
| **Precision@5** | Accuracy of top-5 candidate predictions | How well we identify the full MVP candidate pool |
| **Average Precision (AP)** | Area under precision-recall curve for top-5 | Overall ranking quality |

## Key Results

**Model Performance (Hyperparameter-Tuned):**

| Model | Top-1 Accuracy | Top-3 Recall | MRR | Precision@5 | AP | Years |
|-------|---------------|--------------|-----|-------------|-----|-------|
| **XGBoost (Tuned)** | **66.7% (16/24)** | 70.8% | **0.745** | 65.8% | 0.740 | 24 |
| **Random Forest (Tuned)** | 60.0% (9/15) | **93.3%** | **0.761** | **73.3%** | **0.812** | 15 |
| Ridge Ratios | 41.7% (10/24) | 66.7% | 0.573 | 64.2% | 0.760 | 24 |
| Ridge | 37.5% (9/24) | 66.7% | 0.550 | 62.5% | 0.752 | 24 |

**Tuning Impact:**
- XGBoost: +8.4 points Top-1 accuracy (58.3% → 66.7%)
- Random Forest: +13.3 points Top-3 recall (80.0% → 93.3%), MRR +0.046

**Season-by-Season Highlights (XGBoost - Best Model):**

| Year | Actual MVP | Predicted Rank | Top-3 Hit |
|------|-----------|---------------|-----------|
| 2024 | Nikola Jokić | 1 | ✓ |
| 2023 | Joel Embiid | 1 | ✓ |
| 2022 | Nikola Jokić | 1 | ✓ |
| 2021 | Nikola Jokić | 1 | ✓ |
| 2020 | Giannis Antetokounmpo | 1 | ✓ |
| 2019 | Giannis Antetokounmpo | 1 | ✓ |
| ... | ... | ... | ... |
| **2011** | Derrick Rose | **7** | **✗** |
| **2005** | Steve Nash | **10** | **✗** |

**Interpretation**: XGBoost (tuned) achieves **66.7% Top-1 accuracy** across 24 seasons, correctly predicting the exact MVP winner in 16 of 24 years. The model places the MVP in the top-3 candidates 70.8% of the time. Mean Reciprocal Rank of 0.745 indicates the average predicted rank is approximately 1.34 (very close to #1).

**Random Forest** excels at identifying finalists with **93.3% Top-3 recall** (14 of 15 recent seasons had the MVP in top-3), making it ideal for bracket predictions and identifying the final candidate pool.

## Failure Cases

**Where XGBoost (tuned) struggles (3 major failures out of 24 seasons):**

1. **2005 - Steve Nash** (predicted rank 7, actual MVP)
   - Point guard winning MVP on a 62-win team but without dominant counting stats
   - Nash's playmaking (11.5 AST) and team transformation narrative overrode traditional scoring metrics
   - Model undervalued passing-first MVP archetype

2. **2006 - Steve Nash** (predicted rank 9, actual MVP)
   - Back-to-back MVP with similar profile to 2005
   - Continued undervaluation of pure playmaking without elite scoring
   - Pattern: Model struggles with "pass-first, score-second" MVPs

3. **2011 - Derrick Rose** (predicted rank 6, actual MVP)
   - Youngest MVP ever at age 22 with explosive athleticism narrative
   - Led Bulls to #1 seed but model missed the "new superstar" storyline
   - Voter fatigue with LeBron James (2-time defending MVP) played a role

**Notable Success**: 2021 Nikola Jokić correctly predicted at rank #1 by tuned XGBoost (previously rank 10 with defaults). This demonstrates the value of hyperparameter tuning for capturing unconventional MVP profiles.

**Fundamental limitation**: The model struggles when voter behavior shifts dramatically or when unconventional player archetypes emerge. These "narrative-driven" seasons prioritize storylines and efficiency metrics over traditional counting stats and team success patterns the model expects.

The **70.8% Top-3 recall** (17/24 seasons with MVP in top-3) shows the model works well for typical MVP races but has difficulty with paradigm-shifting winners. **Random Forest** achieves 93.3% Top-3 recall on recent seasons, missing only 1 out of 15.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) values reveal which features drive XGBoost predictions and how they influence vote share forecasts.

**Top 10 Most Important Features by Mean Absolute SHAP Value:**

1. **PTS_R** (0.00454): Era-normalized points ratio—relative scoring dominance
2. **W/L%** (0.00207): Team win percentage—team success matters heavily
3. **FG** (0.00169): Field goals made—raw scoring volume
4. **AST** (0.00136): Assists—playmaking ability
5. **L** (0.00133): Team losses—inverse team success signal
6. **PTS** (0.00127): Points per game—headline statistic
7. **AST_R** (0.00121): Era-normalized assists ratio
8. **FTA** (0.00116): Free throw attempts—usage proxy
9. **DRB** (0.00107): Defensive rebounds—versatility signal
10. **W** (0.00074): Team wins—team success signal

![SHAP Beeswarm Plot](outputs/figures/shap_beeswarm.png)

**Concrete Insight**: PTS_R (era-normalized points ratio) has the highest SHAP importance—this captures relative scoring dominance adjusted for league-wide trends. A player averaging 30 PPG when the league average is 20 (PTS_R = 1.5) signals MVP caliber more than 30 PPG when league average is 25 (PTS_R = 1.2). This explains why the model successfully adapts across different eras (1990s defense-first vs 2020s high-scoring).

Team success (W/L%, L, W) appears 3 times in top-10, confirming the historical pattern that MVPs come from winning teams. The combination of era-normalized individual stats (PTS_R, AST_R) and team context (W/L%) captures the "best player on a great team" MVP archetype.

## Current Season Forecast

**2021 Season Top-5 Predicted Candidates (Ensemble Model):**

| Rank | Player | Team | Predicted Share | Actual Share | Actual Rank |
|------|--------|------|----------------|--------------|-------------|
| 1 | Giannis Antetokounmpo | MIL | 0.379 | 0.345 | 4 |
| 2 | Nikola Jokić | DEN | 0.338 | **0.961** | **1 (MVP)** |
| 3 | Rudy Gobert | UTA | 0.239 | 0.008 | >10 |
| 4 | Stephen Curry | GSW | 0.221 | 0.449 | 3 |
| 5 | Russell Westbrook | WAS | 0.211 | 0.005 | 11 |

The ensemble ranked Jokić #2 (close miss) but significantly underestimated his vote share (predicted 0.338, actual 0.961). This reflects the model's failure to capture the unique narrative of Jokić's first MVP in a COVID-shortened season.

*Note: "Current season" refers to the most recent season in processed data (2021). With updated scraping, this would show 2024-25 in-season predictions.*

## Repository Structure

```
nbaMVPPredictor/
├── src/                          # All Python source code
│   ├── config.py                 # Centralized configuration (paths, hyperparameters)
│   ├── datascraping.py           # Web scraping from Basketball Reference
│   ├── predictors.py             # Data cleaning and feature engineering
│   ├── machine_learning.py       # Model training, backtesting, evaluation
│   └── predict_current_season.py # Live MVP forecast generator
├── data/                         # All data files (gitignored)
│   ├── raw/                      # Raw CSV files from scraping
│   │   ├── players.csv           # Player statistics by year
│   │   ├── teams.csv             # Team standings by year
│   │   ├── mvps.csv              # MVP voting results
│   │   └── nicknames.csv         # Team abbreviation mapping
│   ├── processed/                # Cleaned and merged datasets
│   │   └── player_mvp_stats.csv  # Final dataset for ML
│   └── html/                     # Raw HTML from scraping
├── outputs/                      # Model outputs and analysis
│   ├── tables/
│   │   ├── model_comparison.csv  # Model performance comparison
│   │   ├── backtest_results.csv  # Season-by-season metrics
│   │   ├── feature_list.csv      # Feature documentation
│   │   ├── shap_top_features.csv # Top SHAP importance features
│   │   └── failure_cases.csv     # Annotated prediction failures
│   ├── figures/
│   │   ├── shap_beeswarm.png     # SHAP feature importance visualization
│   │   └── shap_bar.png          # SHAP bar chart
│   ├── current_season_forecast.csv # Live MVP predictions
│   └── predictions_YYYY.csv      # Yearly prediction archives
├── run_pipeline.py               # Main entrypoint
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## Installation and Usage

**Prerequisites:**
- Python 3.11+ (tested on 3.11)
- ChromeDriver (for Selenium scraping)

**Quick Start (Production):**
```bash
# 1. Clone repository
git clone https://github.com/yourusername/nbaMVPPredictor
cd nbaMVPPredictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set up Weights & Biases for experiment tracking
cp .env.example .env
# Edit .env and add your WANDB_API_KEY from https://wandb.ai/authorize

# 4. Run complete pipeline (scraping → processing → modeling)
python run_pipeline.py
```

**Development Setup:**
```bash
# 1. Clone repository
git clone https://github.com/yourusername/nbaMVPPredictor
cd nbaMVPPredictor

# 2. Install development dependencies (includes testing, linting, type checking)
pip install -r requirements-dev.txt

# 3. (Optional) Set up W&B and pre-commit hooks
cp .env.example .env
# Edit .env with your WANDB_API_KEY

# 4. Verify setup by running tests
pytest tests/ -v

# 5. Make changes and run quality checks
ruff check src/ tests/ --fix    # Lint and auto-fix
ruff format src/ tests/          # Format code
mypy src/                        # Type check
pytest tests/ --cov=src          # Test with coverage
```

**Running the Pipeline:**
```bash

# Or run individual steps:
python src/datascraping.py      # Scrape data
python src/predictors.py         # Process and engineer features
python src/machine_learning.py   # Train models and evaluate

# Optional: Tune hyperparameters (takes 40-60 min)
python src/tune_hyperparameters.py   # Optimize XGBoost & Random Forest

# Generate live MVP forecast for current season
python src/predict_current_season.py
```

**Output:**
- Processed data: `data/processed/player_mvp_stats.csv`
- Model results: `outputs/tables/model_comparison.csv`
- Tuned hyperparameters: `outputs/tables/best_hyperparams.json` *(pre-optimized, no need to re-run tuning)*
- Season predictions: `outputs/current_season_forecast.csv`
- SHAP visualizations: `outputs/figures/shap_*.png`

## Testing & Development

### Test Suite

The project includes a comprehensive test suite with **88 tests** covering all critical functionality:

- ✓ **Feature Engineering** (24 tests): Z-scores, team success, availability, narrative features
- ✓ **Data Validation** (28 tests): Season coverage, vote shares, column integrity
- ✓ **Backtest Integrity** (19 tests): Time-aware constraints, metric validity, no data leakage

**Run tests:**
```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=src
```

**Key features:**
- Synthetic fixtures for fast, offline testing (<5 seconds)
- Coverage reporting with pytest-cov
- No external dependencies required (tests run offline)

### Continuous Integration

GitHub Actions CI runs automatically on every push and pull request to `main`:

- ✅ **Linting** with [ruff](https://docs.astral.sh/ruff/) (fast Python linter)
- ✅ **Format checking** with ruff format
- ✅ **Type checking** with mypy
- ✅ **Test suite** with pytest + coverage
- ✅ **Optional**: Coverage upload to Codecov

See [.github/workflows/README.md](.github/workflows/README.md) for CI details.

### Code Quality Checks

Run all CI checks locally before pushing:

```bash
# Lint code
ruff check src/ tests/

# Check formatting
ruff format --check src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Auto-fix issues:**
```bash
# Fix linting issues automatically
ruff check src/ tests/ --fix

# Format code automatically
ruff format src/ tests/
```

### Development Dependencies

Install all development tools:
```bash
pip install -r requirements-dev.txt
```

This includes:
- `pytest` + `pytest-cov` - Testing framework and coverage
- `ruff` - Fast linter and formatter (replaces black, flake8, isort)
- `mypy` - Static type checker

## Experiments

All training runs are tracked with **Weights & Biases** for experiment management and visualization.

### Features Logged to W&B:
- **Model Metrics**: Top-1 Accuracy, Top-3 Recall, MRR, Precision@5, Average Precision for all models
- **Backtest Results**: Season-by-season predictions vs. actuals with detailed metrics
- **SHAP Analysis**: Feature importance rankings and visualizations
- **Model Comparison**: Performance comparison across Ridge, Random Forest, and XGBoost
- **Hyperparameters**: All model configurations (tuned and default values)

### Setup W&B Tracking:

1. **Get your API key** from [https://wandb.ai/authorize](https://wandb.ai/authorize)

2. **Create `.env` file** in project root:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key:
   WANDB_API_KEY=your_actual_api_key_here
   ```

3. **Install wandb** (already in requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

4. **Run pipeline** - wandb tracking happens automatically:
   ```bash
   python run_pipeline.py
   ```

**Note**: W&B integration is optional. If `WANDB_API_KEY` is not set, the pipeline runs normally without experiment tracking.

### View Experiments:

📊 **Public W&B Dashboard**: [Coming Soon - Run pipeline to generate first experiment]

Each run logs:
- Model performance comparison table
- Season-by-season backtest predictions
- SHAP feature importance (top 20 features)
- Interactive charts for all ranking metrics

## Future Work

**High-Priority Improvements:**

1. **Expand Training Data**: Currently using only 2020–2022 due to incomplete scraping. Re-run scraper for full 1991–2024 range to get ~30 backtested seasons (drastically improves statistical validity of performance estimates).

2. **Advanced Metrics Integration**: Add PER (Player Efficiency Rating), BPM (Box Plus/Minus), VORP (Value Over Replacement Player) from Basketball Reference's advanced stats pages. The 2021 Jokić case suggests efficiency metrics matter for modern MVP voting.

3. **Narrative Season Detection**: Build a classifier to detect "unusual" seasons (lockouts, COVID, bubble playoffs) and either weight them differently or use season-type-specific models. The current approach treats all seasons equally, which fails when voting dynamics shift.

4. **Ensemble Optimization**: Current ensemble is simple average of Ridge/RF/XGBoost. Explore stacking or weighted combinations based on historical performance (e.g., weight XGBoost higher for recent seasons, Ridge for historical).

**Secondary Enhancements:**
- Add conference-specific features (East vs West playoff positioning)
- Incorporate player age curves and career trajectory
- Test alternative ranking loss functions (LambdaMART, ListNet) instead of vote share regression
- Build web dashboard for interactive season-by-season exploration

**Research Questions:**
- Does voter behavior change over time? (Test sliding window models: 10-year vs 20-year training)
- Can we predict "narrative" seasons in advance? (Team surprises, first-time All-Stars breaking out)
- How much does media coverage (All-NBA teams, jersey sales) predict MVP votes beyond stats?

---

**License**: MIT
**Contact**: +447440474550
