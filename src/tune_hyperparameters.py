"""
Hyperparameter Tuning with Optuna

Uses time-series cross-validation to find optimal hyperparameters for:
- XGBoost
- Random Forest

Optimization metric: Mean Reciprocal Rank (MRR) across CV folds
"""

import pandas as pd
import numpy as np
import optuna
import json
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings

from config import (
    PLAYER_MVP_STATS_FILE, BASE_PREDICTORS, RATIO_STATS,
    TABLES_DIR, START_YEAR, MIN_TRAINING_SEASONS
)

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load processed data and create features"""
    print("Loading data...")
    stats = pd.read_csv(PLAYER_MVP_STATS_FILE)
    stats = stats.fillna(0)

    # Create ratio features
    for stat in RATIO_STATS:
        if stat in stats.columns:
            yearly_means = stats.groupby("Year")[stat].mean()
            stats[f"{stat}_R"] = stats.apply(
                lambda row: row[stat] / yearly_means[row["Year"]] if yearly_means[row["Year"]] != 0 else 0,
                axis=1
            )

    # Create categorical encodings
    if 'Pos' in stats.columns:
        stats['NPos'] = stats['Pos'].astype('category').cat.codes
    if 'Tm' in stats.columns:
        stats['NTm'] = stats['Tm'].astype('category').cat.codes

    return stats


def calculate_mrr(y_true_share, y_pred, player_names):
    """
    Calculate Mean Reciprocal Rank for one season

    Args:
        y_true_share: Actual MVP vote shares
        y_pred: Predicted vote shares
        player_names: Player names for identification

    Returns:
        MRR value (1.0 = perfect, 0.1 = ranked 10th, etc.)
    """
    # Find actual MVP (max share)
    actual_mvp_idx = y_true_share.argmax()

    # Rank predictions (higher is better)
    predicted_ranks = np.argsort(y_pred)[::-1]

    # Find rank of actual MVP in predictions (1-indexed)
    mvp_rank_position = np.where(predicted_ranks == actual_mvp_idx)[0][0] + 1

    return 1.0 / mvp_rank_position


def time_series_cv_score(model, stats, predictors, n_folds=10):
    """
    Perform time-series cross-validation

    Uses last n_folds backtest seasons. For each fold:
    - Train on all years before fold year
    - Evaluate MRR on fold year

    Args:
        model: Scikit-learn compatible model
        stats: DataFrame with all data
        predictors: List of predictor column names
        n_folds: Number of CV folds (default 10)

    Returns:
        Mean MRR across all folds
    """
    # Get available years that meet minimum training requirement
    available_years = sorted(stats['Year'].unique())
    valid_years = [y for y in available_years if y >= START_YEAR + MIN_TRAINING_SEASONS]

    # Use last n_folds years for CV
    cv_years = valid_years[-n_folds:] if len(valid_years) >= n_folds else valid_years

    if len(cv_years) < 3:
        print(f"Warning: Only {len(cv_years)} CV folds available")
        return 0.0

    mrr_scores = []

    for fold_year in cv_years:
        # Train on all years before fold_year
        train = stats[stats["Year"] < fold_year].copy()
        test = stats[stats["Year"] == fold_year].copy()

        if train.empty or test.empty:
            continue

        # Filter predictors to available columns
        available_preds = [p for p in predictors if p in train.columns]

        if len(available_preds) == 0:
            continue

        try:
            # Train model
            model.fit(train[available_preds], train["Share"])

            # Predict
            predictions = model.predict(test[available_preds])

            # Calculate MRR for this fold
            mrr = calculate_mrr(test["Share"].values, predictions, test["Player"].values)
            mrr_scores.append(mrr)

        except Exception as e:
            print(f"  Error in fold {fold_year}: {e}")
            continue

    if len(mrr_scores) == 0:
        return 0.0

    return np.mean(mrr_scores)


def objective_xgboost(trial, stats, predictors):
    """
    Optuna objective function for XGBoost

    Searches hyperparameters and returns mean MRR from time-series CV
    """
    # Suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Create model
    model = xgb.XGBRegressor(**params)

    # Evaluate with time-series CV
    mean_mrr = time_series_cv_score(model, stats, predictors, n_folds=10)

    return mean_mrr


def objective_random_forest(trial, stats, predictors):
    """
    Optuna objective function for Random Forest

    Searches hyperparameters and returns mean MRR from time-series CV
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    max_features_choice = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5])

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'max_features': max_features_choice,
        'random_state': 1
    }

    # Create model
    model = RandomForestRegressor(**params)

    # Evaluate with time-series CV
    mean_mrr = time_series_cv_score(model, stats, predictors, n_folds=10)

    return mean_mrr


def tune_xgboost(stats, predictors, n_trials=100):
    """
    Tune XGBoost hyperparameters

    Args:
        stats: DataFrame with processed data
        predictors: List of predictor names
        n_trials: Number of Optuna trials (default 100)

    Returns:
        Dictionary with best parameters and score
    """
    print("\n" + "="*70)
    print("TUNING XGBOOST HYPERPARAMETERS")
    print("="*70)
    print(f"Optimization metric: Mean Reciprocal Rank (MRR)")
    print(f"CV strategy: Time-series (last 10 backtest seasons)")
    print(f"Number of trials: {n_trials}")
    print()

    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective_xgboost(trial, stats, predictors),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n" + "="*70)
    print("XGBOOST TUNING RESULTS")
    print("="*70)
    print(f"Best MRR: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    return {
        'best_params': study.best_params,
        'best_mrr': study.best_value,
        'n_trials': n_trials
    }


def tune_random_forest(stats, predictors, n_trials=100):
    """
    Tune Random Forest hyperparameters

    Args:
        stats: DataFrame with processed data
        predictors: List of predictor names
        n_trials: Number of Optuna trials (default 100)

    Returns:
        Dictionary with best parameters and score
    """
    print("\n" + "="*70)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*70)
    print(f"Optimization metric: Mean Reciprocal Rank (MRR)")
    print(f"CV strategy: Time-series (last 10 backtest seasons)")
    print(f"Number of trials: {n_trials}")
    print()

    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='random_forest_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective_random_forest(trial, stats, predictors),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n" + "="*70)
    print("RANDOM FOREST TUNING RESULTS")
    print("="*70)
    print(f"Best MRR: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    return {
        'best_params': study.best_params,
        'best_mrr': study.best_value,
        'n_trials': n_trials
    }


def main():
    """Main hyperparameter tuning pipeline"""
    print("="*70)
    print("NBA MVP PREDICTOR - HYPERPARAMETER TUNING")
    print("="*70)

    # Load data
    stats = load_and_prepare_data()
    print(f"Loaded {len(stats)} player-seasons across {stats['Year'].nunique()} years")

    # Get predictors
    predictors_base = [p for p in BASE_PREDICTORS if p in stats.columns]
    ratio_features = [f"{s}_R" for s in RATIO_STATS if f"{s}_R" in stats.columns]
    predictors = predictors_base + ratio_features

    # Add categorical features if available
    if 'NPos' in stats.columns:
        predictors.append('NPos')
    if 'NTm' in stats.columns:
        predictors.append('NTm')

    print(f"Using {len(predictors)} predictors")

    # Load existing XGBoost results if available
    best_hyperparams_file = TABLES_DIR / 'best_hyperparams.json'
    if best_hyperparams_file.exists():
        print("\n✓ Loading existing XGBoost results from previous tuning")
        with open(best_hyperparams_file, 'r') as f:
            existing_results = json.load(f)
            xgb_results = existing_results.get('XGBoost', {})
            print(f"  Loaded XGBoost best MRR: {xgb_results.get('best_mrr', 'N/A')}")
    else:
        # Tune XGBoost
        xgb_results = tune_xgboost(stats, predictors, n_trials=100)

    # Tune Random Forest
    rf_results = tune_random_forest(stats, predictors, n_trials=100)

    # Save results
    results = {
        'XGBoost': xgb_results,
        'RandomForest': rf_results
    }

    # Ensure tables directory exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = TABLES_DIR / 'best_hyperparams.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_file}")
    print("\nSummary:")
    print(f"  XGBoost best MRR:       {xgb_results['best_mrr']:.4f}")
    print(f"  Random Forest best MRR: {rf_results['best_mrr']:.4f}")


if __name__ == "__main__":
    main()
