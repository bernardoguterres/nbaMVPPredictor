"""
Live MVP Forecast Script
Generates ranked MVP predictions for the current or specified season
"""
import argparse
import sys
import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from config import (
    PLAYER_MVP_STATS_FILE,
    CURRENT_SEASON_FORECAST_FILE,
    RIDGE_ALPHA,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_MIN_SAMPLES_SPLIT,
    RANDOM_FOREST_RANDOM_STATE,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_MAX_DEPTH,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_SUBSAMPLE,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_RANDOM_STATE,
    BASE_PREDICTORS,
    RATIO_STATS,
    MVP_FEATURES,
    OUTPUT_DIR
)
from predictors import (
    add_league_relative_features,
    add_team_success_features,
    add_narrative_features,
    add_team_context_features
)
from machine_learning import create_ratio_features

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load processed data and apply full feature engineering pipeline"""
    logger.info("Loading historical data...")
    stats = pd.read_csv(PLAYER_MVP_STATS_FILE)

    # Validation: Check if data was loaded
    if len(stats) == 0:
        logger.error("No data loaded from processed file")
        sys.exit(1)

    # Apply all feature engineering steps
    logger.info("Applying feature engineering pipeline...")
    create_ratio_features(stats)
    add_league_relative_features(stats)
    add_team_success_features(stats, None)  # Team data already merged in processing
    add_narrative_features(stats)
    add_team_context_features(stats)

    return stats


def get_all_predictors(stats):
    """Get all available predictor columns from the dataset"""
    # Start with base predictors
    predictors = [col for col in BASE_PREDICTORS if col in stats.columns]

    # Add ratio features
    ratio_features = [f"{stat}_R" for stat in RATIO_STATS if f"{stat}_R" in stats.columns]
    predictors.extend(ratio_features)

    # Add MVP-specific features
    mvp_features = [col for col in MVP_FEATURES if col in stats.columns]
    predictors.extend(mvp_features)

    return predictors


def train_models(stats, predictors, target_season):
    """Train all models on historical data (excluding target season)"""
    print(f"Training models on historical data (excluding {target_season})...")

    # Use all seasons before target for training
    train_data = stats[stats['Year'] < target_season].copy()

    if len(train_data) < 10:
        print(f"Warning: Only {len(train_data)} seasons available for training. Results may be unreliable.")

    # Prepare training data
    X_train = train_data[predictors].fillna(0)
    y_train = train_data['Share'].fillna(0)

    # Train Ridge Regression
    print("  Training Ridge Regression...")
    ridge_model = Ridge(alpha=RIDGE_ALPHA)
    ridge_model.fit(X_train, y_train)

    # Train Random Forest
    print("  Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_FOREST_RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        random_state=XGBOOST_RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)

    return {
        'Ridge': ridge_model,
        'RandomForest': rf_model,
        'XGBoost': xgb_model
    }


def generate_predictions(stats, models, predictors, target_season):
    """Generate predictions for target season using all models"""
    print(f"Generating predictions for {target_season} season...")

    # Get target season data
    season_data = stats[stats['Year'] == target_season].copy()

    if len(season_data) == 0:
        print(f"Error: No data found for season {target_season}")
        sys.exit(1)

    # Check for missing data
    total_teams = 30
    avg_players_per_team = len(season_data) / total_teams if len(season_data) > 0 else 0
    if avg_players_per_team < 10:
        print(f"Warning: Limited data for {target_season} (only {len(season_data)} players). "
              f"Season may be incomplete.")

    # Prepare features
    X_season = season_data[predictors].fillna(0)

    # Generate predictions from each model
    results = season_data[['Player', 'Tm', 'Year', 'Share']].copy()

    for model_name, model in models.items():
        predictions = model.predict(X_season)
        results[f'{model_name}_pred'] = predictions
        results[f'{model_name}_rank'] = results[f'{model_name}_pred'].rank(ascending=False, method='min')

    # Calculate ensemble average
    pred_cols = [f'{name}_pred' for name in models.keys()]
    results['Ensemble_pred'] = results[pred_cols].mean(axis=1)
    results['Ensemble_rank'] = results['Ensemble_pred'].rank(ascending=False, method='min')

    # Calculate model consensus (average rank across models)
    rank_cols = [f'{name}_rank' for name in models.keys()]
    results['Consensus_rank'] = results[rank_cols].mean(axis=1)

    # Sort by ensemble prediction
    results = results.sort_values('Ensemble_pred', ascending=False)

    return results


def format_output(results, top_n=10):
    """Format results for display"""
    output = results.head(top_n).copy()

    # Round predictions to 3 decimals
    pred_cols = [col for col in output.columns if '_pred' in col]
    for col in pred_cols:
        output[col] = output[col].round(3)

    # Round ranks to integers
    rank_cols = [col for col in output.columns if '_rank' in col]
    for col in rank_cols:
        output[col] = output[col].astype(int)

    return output


def print_forecast_table(results, season, top_n=10):
    """Print formatted forecast table to console"""
    print(f"\n{'='*100}")
    print(f"NBA MVP FORECAST - {season} Season")
    print(f"{'='*100}\n")

    display = results.head(top_n).copy()

    # Create display table
    print(f"{'Rank':<6} {'Player':<25} {'Team':<6} {'Ensemble':<10} {'Ridge':<8} {'RF':<8} {'XGB':<8} {'Consensus':<10}")
    print("-" * 100)

    for idx, row in enumerate(display.itertuples(), 1):
        print(f"{idx:<6} {row.Player:<25} {row.Tm:<6} "
              f"{row.Ensemble_pred:>8.3f}  "
              f"{int(row.Ridge_rank):>6}  "
              f"{int(row.RandomForest_rank):>6}  "
              f"{int(row.XGBoost_rank):>6}  "
              f"{row.Consensus_rank:>8.1f}")

    print("-" * 100)

    # Show actual MVP if available
    if 'Share' in results.columns:
        actual_mvp = results[results['Share'] > 0].sort_values('Share', ascending=False)
        if len(actual_mvp) > 0:
            mvp = actual_mvp.iloc[0]
            mvp_predicted_rank = list(results['Player'].values).index(mvp['Player']) + 1
            print(f"\nActual MVP: {mvp['Player']} (Model predicted rank: {mvp_predicted_rank})")

    print()


def save_forecast(results, season):
    """Save forecast to CSV file"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = CURRENT_SEASON_FORECAST_FILE
    results.to_csv(output_file, index=False)
    print(f"✓ Forecast saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate live NBA MVP forecast')
    parser.add_argument(
        '--season',
        type=int,
        default=None,
        help='Season year to predict (default: most recent available)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top candidates to display (default: 10)'
    )

    args = parser.parse_args()

    # Load and prepare data
    stats = load_and_prepare_data()

    # Determine target season
    if args.season is None:
        target_season = stats['Year'].max()
        print(f"Using most recent season: {target_season}")
    else:
        target_season = args.season
        if target_season not in stats['Year'].values:
            print(f"Error: Season {target_season} not found in dataset")
            print(f"Available seasons: {sorted(stats['Year'].unique())}")
            sys.exit(1)

    # Get all predictors
    predictors = get_all_predictors(stats)
    print(f"Using {len(predictors)} features for prediction")

    # Train models on historical data
    models = train_models(stats, predictors, target_season)

    # Generate predictions
    results = generate_predictions(stats, models, predictors, target_season)

    # Display forecast
    print_forecast_table(results, target_season, args.top)

    # Save forecast
    save_forecast(results, target_season)

    print(f"✓ Live MVP forecast complete for {target_season} season")


if __name__ == "__main__":
    main()
