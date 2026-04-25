"""
Configuration file for NBA MVP Predictor
Contains all paths, years, and configurable parameters
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HTML_DATA_DIR = DATA_DIR / "html"

# HTML subdirectories
MVP_HTML_DIR = HTML_DATA_DIR / "mvp"
PLAYER_HTML_DIR = HTML_DATA_DIR / "player"
TEAM_HTML_DIR = HTML_DATA_DIR / "team"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Raw data files
NICKNAMES_FILE = RAW_DATA_DIR / "nicknames.csv"
PLAYERS_RAW_FILE = RAW_DATA_DIR / "players.csv"
TEAMS_RAW_FILE = RAW_DATA_DIR / "teams.csv"
MVPS_RAW_FILE = RAW_DATA_DIR / "mvps.csv"

# Processed data files
PLAYER_MVP_STATS_FILE = PROCESSED_DATA_DIR / "player_mvp_stats.csv"

# Output files
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"
CORRELATIONS_FILE = OUTPUT_DIR / "correlations.csv"

# Table files
BACKTEST_RESULTS_FILE = TABLES_DIR / "backtest_results.csv"
FAILURE_CASES_FILE = TABLES_DIR / "failure_cases.csv"
FEATURE_LIST_FILE = TABLES_DIR / "feature_list.csv"
MODEL_COMPARISON_FILE = TABLES_DIR / "model_comparison.csv"
SHAP_TOP_FEATURES_FILE = TABLES_DIR / "shap_top_features.csv"
SEASON_EXPLANATIONS_FILE = TABLES_DIR / "season_explanations.csv"
CURRENT_SEASON_FORECAST_FILE = OUTPUT_DIR / "current_season_forecast.csv"

# Figure files
SHAP_BEESWARM_FILE = FIGURES_DIR / "shap_beeswarm.png"
SHAP_BAR_FILE = FIGURES_DIR / "shap_bar.png"

# Scraping configuration
START_YEAR = 1991
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Backtesting configuration
MIN_TRAINING_SEASONS = 10  # Minimum seasons required for training (time-aware backtest)
BACKTEST_START_YEAR = START_YEAR + MIN_TRAINING_SEASONS  # First year for backtesting
RANDOM_FOREST_START_YEAR = 2010  # Use recent years for Random Forest

# Prediction years (most recent years to predict)
PREDICTION_YEARS = [2021, 2022, 2023, 2024]

# Model configuration
RIDGE_ALPHA = 0.1
RANDOM_FOREST_N_ESTIMATORS = 50
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5
RANDOM_FOREST_RANDOM_STATE = 1

# XGBoost configuration
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 4
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_MIN_CHILD_WEIGHT = 3
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8
XGBOOST_RANDOM_STATE = 1

# Feature definitions
BASE_PREDICTORS = [
    "Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA', '3P%',
    '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',
    'GB', 'PS/G', 'PA/G', 'SRS'
]

RATIO_STATS = ["PTS", "AST", "STL", "BLK", "3P"]

# Domain-informed MVP features
LEAGUE_RELATIVE_FEATURES = ["PTS_zscore", "AST_zscore", "TRB_zscore", "WS_zscore"]
TEAM_SUCCESS_FEATURES = ["team_win_pct", "conference_rank", "made_playoffs", "is_top3_seed"]
AVAILABILITY_FEATURES = ["games_played", "games_played_pct", "minutes_per_game"]
NARRATIVE_FEATURES = ["previous_mvp_finish", "previous_top5_count"]
TEAM_CONTEXT_FEATURES = ["is_best_player_on_team"]

# All MVP-specific features combined
MVP_FEATURES = (
    LEAGUE_RELATIVE_FEATURES +
    TEAM_SUCCESS_FEATURES +
    AVAILABILITY_FEATURES +
    NARRATIVE_FEATURES +
    TEAM_CONTEXT_FEATURES
)

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        HTML_DATA_DIR,
        MVP_HTML_DIR,
        PLAYER_HTML_DIR,
        TEAM_HTML_DIR,
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("✓ All directories ensured")

if __name__ == "__main__":
    # Test configuration
    ensure_directories()
    print("\nConfiguration paths:")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"\nYears: {START_YEAR} to {END_YEAR} ({len(YEARS)} years)")
