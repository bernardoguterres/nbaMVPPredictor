import pandas as pd
import numpy as np
import logging
import json
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings
from config import (
    PLAYER_MVP_STATS_FILE, FEATURE_IMPORTANCE_FILE, CORRELATIONS_FILE,
    OUTPUT_DIR, BASE_PREDICTORS, RATIO_STATS, RIDGE_ALPHA,
    RANDOM_FOREST_N_ESTIMATORS, RANDOM_FOREST_MIN_SAMPLES_SPLIT,
    RANDOM_FOREST_RANDOM_STATE, BACKTEST_START_YEAR,
    RANDOM_FOREST_START_YEAR, PREDICTION_YEARS,
    BACKTEST_RESULTS_FILE, FAILURE_CASES_FILE, TABLES_DIR,
    FEATURE_LIST_FILE, MVP_FEATURES, MODEL_COMPARISON_FILE,
    LEAGUE_RELATIVE_FEATURES, TEAM_SUCCESS_FEATURES,
    AVAILABILITY_FEATURES, NARRATIVE_FEATURES, TEAM_CONTEXT_FEATURES,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE,
    XGBOOST_MIN_CHILD_WEIGHT, XGBOOST_SUBSAMPLE, XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_RANDOM_STATE, MIN_TRAINING_SEASONS, START_YEAR,
    SHAP_BEESWARM_FILE, SHAP_BAR_FILE, SHAP_TOP_FEATURES_FILE,
    SEASON_EXPLANATIONS_FILE, FIGURES_DIR
)
warnings.filterwarnings('ignore')

# Optional Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Optional python-dotenv for loading .env files
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def init_wandb(tuned_params=None):
    """
    Initialize Weights & Biases experiment tracking

    Args:
        tuned_params: Dictionary of tuned hyperparameters (optional)

    Returns:
        bool: True if wandb initialized successfully, False otherwise
    """
    if not WANDB_AVAILABLE:
        logger.info("Weights & Biases not available (wandb not installed)")
        return False

    # Load .env file if python-dotenv is available
    if DOTENV_AVAILABLE:
        load_dotenv()

    # Check if API key is set
    if not os.getenv('WANDB_API_KEY'):
        logger.info("WANDB_API_KEY not set - skipping experiment tracking")
        logger.info("To enable W&B tracking, set WANDB_API_KEY in .env file")
        return False

    try:
        # Load tuned hyperparameters if available
        best_hyperparams_file = TABLES_DIR / 'best_hyperparams.json'
        if tuned_params is None and best_hyperparams_file.exists():
            with open(best_hyperparams_file, 'r') as f:
                tuned_params = json.load(f)

        # Build config dict with all hyperparameters
        config = {
            # Data configuration
            'start_year': START_YEAR,
            'backtest_start_year': BACKTEST_START_YEAR,
            'min_training_seasons': MIN_TRAINING_SEASONS,
            'random_forest_start_year': RANDOM_FOREST_START_YEAR,

            # Ridge hyperparameters
            'ridge_alpha': RIDGE_ALPHA,

            # Random Forest hyperparameters (default or tuned)
            'rf_n_estimators': RANDOM_FOREST_N_ESTIMATORS,
            'rf_min_samples_split': RANDOM_FOREST_MIN_SAMPLES_SPLIT,
            'rf_random_state': RANDOM_FOREST_RANDOM_STATE,

            # XGBoost hyperparameters (default or tuned)
            'xgb_n_estimators': XGBOOST_N_ESTIMATORS,
            'xgb_max_depth': XGBOOST_MAX_DEPTH,
            'xgb_learning_rate': XGBOOST_LEARNING_RATE,
            'xgb_min_child_weight': XGBOOST_MIN_CHILD_WEIGHT,
            'xgb_subsample': XGBOOST_SUBSAMPLE,
            'xgb_colsample_bytree': XGBOOST_COLSAMPLE_BYTREE,
            'xgb_random_state': XGBOOST_RANDOM_STATE,

            # Feature counts
            'n_base_predictors': len(BASE_PREDICTORS),
            'n_ratio_stats': len(RATIO_STATS),
            'n_mvp_features': len(MVP_FEATURES),
        }

        # Override with tuned parameters if available
        if tuned_params:
            config['hyperparameters_tuned'] = True
            if 'XGBoost' in tuned_params and 'best_params' in tuned_params['XGBoost']:
                for key, value in tuned_params['XGBoost']['best_params'].items():
                    config[f'xgb_{key}'] = value
            if 'RandomForest' in tuned_params and 'best_params' in tuned_params['RandomForest']:
                for key, value in tuned_params['RandomForest']['best_params'].items():
                    config[f'rf_{key}'] = value
        else:
            config['hyperparameters_tuned'] = False

        # Initialize wandb
        wandb.init(
            project="nba-mvp-predictor",
            name=f"mvp-backtest-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
            config=config,
            tags=['backtest', 'ranking', 'mvp-prediction']
        )

        logger.info("✓ Weights & Biases initialized successfully")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        return False

def load_and_prepare_data():
    """Load and prepare the data for machine learning"""
    print("Loading player MVP stats data...")
    stats = pd.read_csv(PLAYER_MVP_STATS_FILE)

    print(f"Dataset shape: {stats.shape}")
    print(f"Years available: {stats['Year'].min()} to {stats['Year'].max()}")

    # Check for missing values
    missing_summary = pd.isnull(stats).sum()
    print(f"Columns with missing values: {missing_summary[missing_summary > 0].shape[0]}")

    # Fill missing values with 0
    stats = stats.fillna(0)

    return stats

def create_ratio_features(stats):
    """Create ratio features normalized by year"""
    logger.info("Creating ratio features...")

    # Validation: Check that required stats exist
    missing_stats = [stat for stat in RATIO_STATS if stat not in stats.columns]
    if missing_stats:
        logger.warning(f"Missing stats for ratio features: {missing_stats}")

    # Calculate ratios for key stats relative to yearly averages
    for stat in RATIO_STATS:
        if stat in stats.columns:
            yearly_means = stats.groupby("Year")[stat].mean()
            stats[f"{stat}_R"] = stats.apply(lambda row: row[stat] / yearly_means[row["Year"]] if yearly_means[row["Year"]] != 0 else 0, axis=1)

            # Validation: Check for nulls in created feature
            null_count = stats[f"{stat}_R"].isnull().sum()
            if null_count > 0:
                logger.warning(f"{stat}_R has {null_count} null values")
        else:
            logger.warning(f"Skipping {stat}_R - column {stat} not found")

    logger.info("Ratio features created successfully")
    return stats

def find_ap(combination):
    """Calculate Average Precision for top 5 MVP candidates"""
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)

    ps = []
    found = 0
    seen = 1

    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1

    return sum(ps) / len(ps) if ps else 0

def add_ranks(predictions):
    """Add ranking columns to predictions"""
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1, predictions.shape[0] + 1))
    predictions["Diff"] = (predictions["Rk"] - predictions["Predicted_Rk"])
    return predictions

def calculate_top1_accuracy(combination):
    """
    Calculate Top-1 Accuracy: Did the model rank the actual MVP first?
    Returns 1 if yes, 0 if no
    """
    actual_mvp = combination.sort_values("Share", ascending=False).iloc[0]["Player"]
    predicted_top1 = combination.sort_values("predictions", ascending=False).iloc[0]["Player"]
    return 1 if actual_mvp == predicted_top1 else 0

def calculate_top3_recall(combination):
    """
    Calculate Top-3 Recall: Was the actual MVP in the model's top 3?
    Returns 1 if yes, 0 if no
    """
    actual_mvp = combination.sort_values("Share", ascending=False).iloc[0]["Player"]
    predicted_top3 = combination.sort_values("predictions", ascending=False).head(3)["Player"].values
    return 1 if actual_mvp in predicted_top3 else 0

def calculate_mrr(combination):
    """
    Calculate Mean Reciprocal Rank (MRR)
    MRR = 1 / rank of actual MVP in predicted rankings
    Returns value between 0 and 1 (1 = perfect, 0.5 = ranked 2nd, 0.33 = ranked 3rd, etc.)
    """
    actual_mvp = combination.sort_values("Share", ascending=False).iloc[0]["Player"]
    predicted_ranked = combination.sort_values("predictions", ascending=False)

    # Find rank of actual MVP in predictions (1-indexed)
    mvp_predicted_rank = predicted_ranked[predicted_ranked["Player"] == actual_mvp].index[0]
    rank_position = list(predicted_ranked.index).index(mvp_predicted_rank) + 1

    return 1.0 / rank_position

def calculate_precision_at_k(combination, k=5):
    """
    Calculate Precision@K: Of model's top K, how many were actually in top K?

    Args:
        combination: DataFrame with predictions and actual shares
        k: Number of top predictions to consider (default 5)

    Returns:
        Precision@K value (between 0 and 1)
    """
    actual_topk = set(combination.sort_values("Share", ascending=False).head(k)["Player"].values)
    predicted_topk = set(combination.sort_values("predictions", ascending=False).head(k)["Player"].values)

    # How many overlap?
    overlap = len(actual_topk.intersection(predicted_topk))
    return overlap / k

def calculate_all_metrics(combination, year):
    """
    Calculate all ranking metrics for a single season

    Returns:
        Dictionary with all metrics
    """
    # Get actual MVP info
    actual_mvp_row = combination.sort_values("Share", ascending=False).iloc[0]
    actual_mvp = actual_mvp_row["Player"]
    actual_share = actual_mvp_row["Share"]

    # Get predicted rank of actual MVP
    predicted_ranked = combination.sort_values("predictions", ascending=False)
    mvp_predicted_rank = list(predicted_ranked["Player"].values).index(actual_mvp) + 1

    return {
        'Year': year,
        'Actual_MVP': actual_mvp,
        'MVP_Share': actual_share,
        'Predicted_Rank': mvp_predicted_rank,
        'Top1_Accuracy': calculate_top1_accuracy(combination),
        'Top3_Recall': calculate_top3_recall(combination),
        'MRR': calculate_mrr(combination),
        'Precision@5': calculate_precision_at_k(combination, k=5),
        'AP': find_ap(combination)
    }

def backtest(stats, model, years, predictors, use_scaler=False):
    """Backtest the model across multiple years and collect detailed metrics"""
    print(f"Backtesting model across {len(years)} years...")

    all_predictions = []
    season_metrics = []
    sc = StandardScaler() if use_scaler else None

    for year in years:
        train = stats[stats["Year"] < year].copy()
        test = stats[stats["Year"] == year].copy()

        # Skip if no test data or no training data
        if test.empty or train.empty:
            continue

        if use_scaler:
            sc.fit(train[predictors])
            train_scaled = train.copy()
            test_scaled = test.copy()
            train_scaled[predictors] = sc.transform(train[predictors])
            test_scaled[predictors] = sc.transform(test[predictors])
        else:
            train_scaled = train
            test_scaled = test

        model.fit(train_scaled[predictors], train_scaled["Share"])
        predictions = model.predict(test_scaled[predictors])
        predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=test.index)

        combination = pd.concat([test[["Player", "Share"]], predictions_df], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)

        # Calculate all metrics for this season
        metrics = calculate_all_metrics(combination, year)
        season_metrics.append(metrics)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(season_metrics) if season_metrics else pd.DataFrame()

    # Calculate average metrics
    avg_metrics = {}
    if not metrics_df.empty:
        avg_metrics = {
            'mean_top1_accuracy': metrics_df['Top1_Accuracy'].mean(),
            'mean_top3_recall': metrics_df['Top3_Recall'].mean(),
            'mean_mrr': metrics_df['MRR'].mean(),
            'mean_precision_at_5': metrics_df['Precision@5'].mean(),
            'mean_ap': metrics_df['AP'].mean()
        }

    return avg_metrics, metrics_df, pd.concat(all_predictions) if all_predictions else pd.DataFrame()

def generate_backtest_table(metrics_df, model_name):
    """
    Generate and save season-by-season backtest results table

    Args:
        metrics_df: DataFrame with per-season metrics
        model_name: Name of the model for the output file
    """
    if metrics_df.empty:
        print("No metrics to save")
        return

    # Ensure tables directory exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Round numeric columns for readability
    output_df = metrics_df.copy()
    output_df['MVP_Share'] = output_df['MVP_Share'].round(3)
    output_df['MRR'] = output_df['MRR'].round(4)
    output_df['Precision@5'] = output_df['Precision@5'].round(3)
    output_df['AP'] = output_df['AP'].round(4)

    # Save to CSV
    filename = TABLES_DIR / f"backtest_results_{model_name}.csv"
    output_df.to_csv(filename, index=False)
    print(f"✓ Saved backtest results to {filename}")

    return output_df

def identify_failure_cases(metrics_df, model_name, threshold_rank=5):
    """
    Identify and save seasons where the model performed poorly

    A failure case is defined as:
    - Actual MVP ranked worse than threshold_rank by the model

    Args:
        metrics_df: DataFrame with per-season metrics
        model_name: Name of the model
        threshold_rank: Consider it a failure if MVP ranked worse than this (default 5)
    """
    if metrics_df.empty:
        print("No metrics to analyze for failure cases")
        return

    # Identify failures
    failures = metrics_df[metrics_df['Predicted_Rank'] > threshold_rank].copy()

    if failures.empty:
        print(f"✓ No major failures found (all MVPs ranked in top {threshold_rank})")
        return

    # Sort by worst predictions first
    failures = failures.sort_values('Predicted_Rank', ascending=False)

    # Add a note column for manual annotation
    failures['Note'] = ''

    # Add severity rating
    def severity(rank):
        if rank > 20:
            return 'Critical'
        elif rank > 10:
            return 'High'
        else:
            return 'Medium'

    failures['Severity'] = failures['Predicted_Rank'].apply(severity)

    # Reorder columns
    cols = ['Year', 'Actual_MVP', 'Predicted_Rank', 'Severity', 'MVP_Share', 'MRR', 'Top1_Accuracy', 'Top3_Recall', 'Precision@5', 'Note']
    failures = failures[cols]

    # Ensure tables directory exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    filename = TABLES_DIR / f"failure_cases_{model_name}.csv"
    failures.to_csv(filename, index=False)
    print(f"✓ Saved {len(failures)} failure cases to {filename}")

    return failures

def generate_feature_list(stats):
    """
    Generate comprehensive feature list with categories and descriptions

    Args:
        stats: DataFrame with all features

    Returns:
        DataFrame with feature metadata
    """
    print("\n" + "="*60)
    print("GENERATING FEATURE LIST")
    print("="*60)

    feature_data = []

    # Base features
    for feat in BASE_PREDICTORS:
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Base Statistics',
                'description': 'Raw per-game or team statistic from Basketball Reference',
                'source': 'Basketball Reference'
            })

    # Ratio features
    for stat in RATIO_STATS:
        feat = f"{stat}_R"
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Era-Normalized Ratios',
                'description': f'{stat} divided by league average for that season',
                'source': 'Engineered'
            })

    # League-relative features
    for feat in LEAGUE_RELATIVE_FEATURES:
        if feat in stats.columns:
            base_stat = feat.replace('_zscore', '')
            feature_data.append({
                'feature': feat,
                'category': 'League-Relative (Z-scores)',
                'description': f'Z-score of {base_stat} within season (compares to league average)',
                'source': 'Engineered'
            })

    # Team success features
    descriptions = {
        'team_win_pct': 'Team win percentage (W/L%)',
        'conference_rank': 'Rank within conference (1-15)',
        'made_playoffs': 'Binary: 1 if team made playoffs',
        'is_top3_seed': 'Binary: 1 if team was top 3 seed in conference'
    }
    for feat in TEAM_SUCCESS_FEATURES:
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Team Success',
                'description': descriptions.get(feat, 'Team success metric'),
                'source': 'Engineered from team standings'
            })

    # Availability features
    descriptions = {
        'games_played': 'Number of games played in season',
        'games_played_pct': 'Games played as percentage of 82-game season',
        'minutes_per_game': 'Average minutes per game'
    }
    for feat in AVAILABILITY_FEATURES:
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Player Availability',
                'description': descriptions.get(feat, 'Availability metric'),
                'source': 'Engineered'
            })

    # Narrative features
    descriptions = {
        'previous_mvp_finish': 'Best MVP finish in prior seasons (0 if never voted)',
        'previous_top5_count': 'Number of prior top-5 MVP finishes'
    }
    for feat in NARRATIVE_FEATURES:
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Narrative/History',
                'description': descriptions.get(feat, 'Historical metric'),
                'source': 'Engineered from prior seasons'
            })

    # Team context features
    for feat in TEAM_CONTEXT_FEATURES:
        if feat in stats.columns:
            feature_data.append({
                'feature': feat,
                'category': 'Team Context',
                'description': 'Binary: 1 if highest Win Shares on team that season',
                'source': 'Engineered'
            })

    # Create DataFrame
    feature_df = pd.DataFrame(feature_data)

    # Ensure tables directory exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    feature_df.to_csv(FEATURE_LIST_FILE, index=False)
    print(f"✓ Saved feature list to {FEATURE_LIST_FILE}")

    # Print summary
    print("\nFeature Summary:")
    category_counts = feature_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} features")
    print(f"\nTotal features: {len(feature_df)}")

    return feature_df

def print_summary_table(avg_metrics, model_name):
    """
    Print a formatted summary table of evaluation metrics

    Args:
        avg_metrics: Dictionary of average metrics
        model_name: Name of the model
    """
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("="*60)

    if not avg_metrics:
        print("No metrics available")
        return

    # Format the metrics table
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 60)
    print(f"{'Top-1 Accuracy':<30} {avg_metrics.get('mean_top1_accuracy', 0):>9.1%}")
    print(f"{'Top-3 Recall':<30} {avg_metrics.get('mean_top3_recall', 0):>9.1%}")
    print(f"{'Mean Reciprocal Rank (MRR)':<30} {avg_metrics.get('mean_mrr', 0):>10.4f}")
    print(f"{'Precision@5':<30} {avg_metrics.get('mean_precision_at_5', 0):>9.1%}")
    print(f"{'Average Precision (AP)':<30} {avg_metrics.get('mean_ap', 0):>10.4f}")
    print("="*60)

    # Interpretation guide
    print("\n📊 Interpretation:")
    print(f"  • Top-1: {avg_metrics.get('mean_top1_accuracy', 0):.1%} of seasons predicted exact MVP")
    print(f"  • Top-3: {avg_metrics.get('mean_top3_recall', 0):.1%} of seasons had MVP in top 3")
    print(f"  • MRR: Average reciprocal rank of {avg_metrics.get('mean_mrr', 0):.4f} (1.0 = always #1)")

def get_available_predictors(stats, base_predictors):
    """
    Get list of available predictors from the dataset

    Args:
        stats: DataFrame with all features
        base_predictors: List of base predictor names

    Returns:
        List of predictor names that exist in the dataset
    """
    available = []
    for pred in base_predictors:
        if pred in stats.columns:
            available.append(pred)
        else:
            print(f"  ⚠ Predictor '{pred}' not found in dataset, skipping")

    return available

def generate_model_comparison_table(results):
    """
    Generate comprehensive model comparison table

    Args:
        results: Dictionary with model results

    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "="*70)
    print("GENERATING MODEL COMPARISON TABLE")
    print("="*70)

    comparison_data = []

    for model_name, result in results.items():
        avg_metrics = result.get('avg_metrics', {})
        metrics_df = result.get('metrics_df')

        # Get backtest years count
        n_years = len(metrics_df) if metrics_df is not None and not metrics_df.empty else 0

        comparison_data.append({
            'Model': model_name,
            'Top1_Accuracy': avg_metrics.get('mean_top1_accuracy', 0),
            'Top3_Recall': avg_metrics.get('mean_top3_recall', 0),
            'MRR': avg_metrics.get('mean_mrr', 0),
            'Precision@5': avg_metrics.get('mean_precision_at_5', 0),
            'AP': avg_metrics.get('mean_ap', 0),
            'Backtest_Years': n_years
        })

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    if comparison_df.empty:
        print("No model results to compare")
        return comparison_df

    # Sort by MRR (best overall ranking metric)
    comparison_df = comparison_df.sort_values('MRR', ascending=False)

    # Ensure tables directory exists
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    comparison_df.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"✓ Saved model comparison to {MODEL_COMPARISON_FILE}")

    # Print formatted table
    print("\n" + "="*90)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*90)
    print(f"{'Model':<20} {'Top-1':>8} {'Top-3':>8} {'MRR':>8} {'Prec@5':>8} {'AP':>8} {'Years':>6}")
    print("-"*90)

    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<20} "
              f"{row['Top1_Accuracy']:>7.1%} "
              f"{row['Top3_Recall']:>7.1%} "
              f"{row['MRR']:>8.4f} "
              f"{row['Precision@5']:>7.1%} "
              f"{row['AP']:>8.4f} "
              f"{int(row['Backtest_Years']):>6}")

    print("="*90)

    # Identify best model per metric
    print("\n📊 Best Model by Metric:")
    print(f"  • Top-1 Accuracy: {comparison_df.iloc[comparison_df['Top1_Accuracy'].argmax()]['Model']}")
    print(f"  • Top-3 Recall:   {comparison_df.iloc[comparison_df['Top3_Recall'].argmax()]['Model']}")
    print(f"  • MRR (Overall):  {comparison_df.iloc[comparison_df['MRR'].argmax()]['Model']}")
    print(f"  • Precision@5:    {comparison_df.iloc[comparison_df['Precision@5'].argmax()]['Model']}")

    return comparison_df

def train_models(stats, predictors):
    """Train and evaluate different models"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)

    # Get available years
    available_years = sorted(stats['Year'].unique())
    print(f"Available years: {available_years}")

    # Define years for backtesting
    backtest_years = [year for year in available_years if year >= BACKTEST_START_YEAR]
    print(f"Backtesting years: {backtest_years}")

    results = {}

    # Load tuned hyperparameters if available
    best_hyperparams_file = TABLES_DIR / 'best_hyperparams.json'
    tuned_params = {}
    if best_hyperparams_file.exists():
        print(f"\n✓ Loading tuned hyperparameters from {best_hyperparams_file}")
        with open(best_hyperparams_file, 'r') as f:
            tuned_params = json.load(f)
    else:
        print(f"\n⚠ No tuned hyperparameters found at {best_hyperparams_file}, using defaults")

    # Ridge Regression
    print("\n1. Training Ridge Regression...")
    reg = Ridge(alpha=RIDGE_ALPHA)
    avg_metrics, metrics_df, all_predictions = backtest(stats, reg, backtest_years, predictors)
    results['Ridge'] = {
        'avg_metrics': avg_metrics,
        'metrics_df': metrics_df,
        'predictions': all_predictions
    }
    print_summary_table(avg_metrics, "Ridge Regression")

    # Log to wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            'ridge/top1_accuracy': avg_metrics.get('mean_top1_accuracy', 0),
            'ridge/top3_recall': avg_metrics.get('mean_top3_recall', 0),
            'ridge/mrr': avg_metrics.get('mean_mrr', 0),
            'ridge/precision_at_5': avg_metrics.get('mean_precision_at_5', 0),
            'ridge/average_precision': avg_metrics.get('mean_ap', 0),
        })

    # Ridge with ratio features
    print("\n2. Training Ridge Regression with ratio features...")
    predictors_with_ratios = predictors + [f"{stat}_R" for stat in RATIO_STATS]
    avg_metrics_ratios, metrics_df_ratios, all_predictions_ratios = backtest(
        stats, reg, backtest_years, predictors_with_ratios
    )
    results['Ridge_Ratios'] = {
        'avg_metrics': avg_metrics_ratios,
        'metrics_df': metrics_df_ratios,
        'predictions': all_predictions_ratios
    }
    print_summary_table(avg_metrics_ratios, "Ridge with Ratios")

    # Log to wandb
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            'ridge_ratios/top1_accuracy': avg_metrics_ratios.get('mean_top1_accuracy', 0),
            'ridge_ratios/top3_recall': avg_metrics_ratios.get('mean_top3_recall', 0),
            'ridge_ratios/mrr': avg_metrics_ratios.get('mean_mrr', 0),
            'ridge_ratios/precision_at_5': avg_metrics_ratios.get('mean_precision_at_5', 0),
            'ridge_ratios/average_precision': avg_metrics_ratios.get('mean_ap', 0),
        })

    # Random Forest (using recent years only)
    print("\n3. Training Random Forest...")

    # Create categorical encodings
    stats['NPos'] = stats['Pos'].astype('category').cat.codes
    stats['NTm'] = stats['Tm'].astype('category').cat.codes

    # Use tuned hyperparameters if available, otherwise defaults
    if 'RandomForest' in tuned_params:
        rf_params = tuned_params['RandomForest']['best_params'].copy()
        rf_params['random_state'] = RANDOM_FOREST_RANDOM_STATE
        print(f"  Using tuned hyperparameters: {rf_params}")
        rf = RandomForestRegressor(**rf_params)
    else:
        rf = RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            random_state=RANDOM_FOREST_RANDOM_STATE,
            min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT
        )
    rf_years = [year for year in available_years if year >= RANDOM_FOREST_START_YEAR]

    if len(rf_years) > 0:
        print(f"Random Forest years: {rf_years}")
        avg_metrics_rf, metrics_df_rf, all_predictions_rf = backtest(
            stats, rf, rf_years, predictors_with_ratios + ["NPos", "NTm"]
        )
        results['RandomForest'] = {
            'avg_metrics': avg_metrics_rf,
            'metrics_df': metrics_df_rf,
            'predictions': all_predictions_rf
        }
        print_summary_table(avg_metrics_rf, "Random Forest")

        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'random_forest/top1_accuracy': avg_metrics_rf.get('mean_top1_accuracy', 0),
                'random_forest/top3_recall': avg_metrics_rf.get('mean_top3_recall', 0),
                'random_forest/mrr': avg_metrics_rf.get('mean_mrr', 0),
                'random_forest/precision_at_5': avg_metrics_rf.get('mean_precision_at_5', 0),
                'random_forest/average_precision': avg_metrics_rf.get('mean_ap', 0),
            })
    else:
        print("Not enough recent years for Random Forest")

    # XGBoost with time-aware rolling backtest
    print("\n4. Training XGBoost...")

    # Use tuned hyperparameters if available, otherwise defaults
    if 'XGBoost' in tuned_params:
        xgb_params = tuned_params['XGBoost']['best_params'].copy()
        xgb_params['random_state'] = XGBOOST_RANDOM_STATE
        xgb_params['objective'] = 'reg:squarederror'
        xgb_params['eval_metric'] = 'rmse'
        print(f"  Using tuned hyperparameters: {xgb_params}")
        xgb_model = xgb.XGBRegressor(**xgb_params)
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=XGBOOST_N_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
            random_state=XGBOOST_RANDOM_STATE,
            objective='reg:squarederror',
            eval_metric='rmse'
        )

    # Use all available years that meet minimum training requirement
    # Ensure at least MIN_TRAINING_SEASONS of training data
    xgb_years = [year for year in available_years if year >= START_YEAR + MIN_TRAINING_SEASONS]

    if len(xgb_years) > 0:
        print(f"XGBoost years: {xgb_years}")
        print(f"Minimum training window: {MIN_TRAINING_SEASONS} seasons")
        avg_metrics_xgb, metrics_df_xgb, all_predictions_xgb = backtest(
            stats, xgb_model, xgb_years, predictors_with_ratios + ["NPos", "NTm"]
        )
        results['XGBoost'] = {
            'avg_metrics': avg_metrics_xgb,
            'metrics_df': metrics_df_xgb,
            'predictions': all_predictions_xgb
        }
        print_summary_table(avg_metrics_xgb, "XGBoost")

        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'xgboost/top1_accuracy': avg_metrics_xgb.get('mean_top1_accuracy', 0),
                'xgboost/top3_recall': avg_metrics_xgb.get('mean_top3_recall', 0),
                'xgboost/mrr': avg_metrics_xgb.get('mean_mrr', 0),
                'xgboost/precision_at_5': avg_metrics_xgb.get('mean_precision_at_5', 0),
                'xgboost/average_precision': avg_metrics_xgb.get('mean_ap', 0),
            })
    else:
        print("Not enough years for XGBoost with minimum training window")

    return results

def log_predictions_to_wandb(results):
    """
    Log backtest predictions vs actuals to Weights & Biases as tables

    Args:
        results: Dictionary with model results containing metrics_df and predictions
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        for model_name, result in results.items():
            metrics_df = result.get('metrics_df')

            if metrics_df is None or metrics_df.empty:
                continue

            # Create table with predictions vs actuals for each season
            table_data = []
            for _, row in metrics_df.iterrows():
                table_data.append([
                    int(row['Year']),
                    row['Actual_MVP'],
                    float(row['MVP_Share']),
                    int(row['Predicted_Rank']),
                    float(row['MRR']),
                    int(row['Top1_Accuracy']),
                    int(row['Top3_Recall']),
                    float(row['Precision@5'])
                ])

            # Create wandb table
            table = wandb.Table(
                columns=['Year', 'Actual_MVP', 'MVP_Share', 'Predicted_Rank',
                         'MRR', 'Top1_Accuracy', 'Top3_Recall', 'Precision@5'],
                data=table_data
            )

            # Log table
            wandb.log({f'{model_name.lower()}/predictions_table': table})

        logger.info("✓ Logged predictions tables to W&B")

    except Exception as e:
        logger.warning(f"Failed to log predictions to W&B: {e}")


def log_shap_to_wandb(shap_values, X_sample):
    """
    Log SHAP feature importance to Weights & Biases

    Args:
        shap_values: SHAP values array
        X_sample: DataFrame with feature data
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        # Create table for top 20 features
        top_features = feature_importance_df.head(20)
        table_data = [[feat, float(shap_val)] for feat, shap_val in
                      zip(top_features['feature'], top_features['mean_abs_shap'])]

        table = wandb.Table(
            columns=['Feature', 'Mean_Abs_SHAP'],
            data=table_data
        )

        # Log table
        wandb.log({'shap/feature_importance_table': table})

        # Log as bar chart
        wandb.log({
            'shap/feature_importance': wandb.plot.bar(
                table, 'Feature', 'Mean_Abs_SHAP',
                title='SHAP Feature Importance (Top 20)'
            )
        })

        # Log SHAP images if they exist
        if SHAP_BEESWARM_FILE.exists():
            wandb.log({'shap/beeswarm_plot': wandb.Image(str(SHAP_BEESWARM_FILE))})
        if SHAP_BAR_FILE.exists():
            wandb.log({'shap/bar_plot': wandb.Image(str(SHAP_BAR_FILE))})

        logger.info("✓ Logged SHAP analysis to W&B")

    except Exception as e:
        logger.warning(f"Failed to log SHAP to W&B: {e}")


def analyze_results(results, stats):
    """Analyze and visualize results, generate tables"""
    # Generate comprehensive model comparison table
    comparison_df = generate_model_comparison_table(results)

    # Log predictions to wandb
    log_predictions_to_wandb(results)

    # Log model comparison to wandb
    if WANDB_AVAILABLE and wandb.run is not None and not comparison_df.empty:
        try:
            # Create comparison table
            table_data = []
            for _, row in comparison_df.iterrows():
                table_data.append([
                    row['Model'],
                    float(row['Top1_Accuracy']),
                    float(row['Top3_Recall']),
                    float(row['MRR']),
                    float(row['Precision@5']),
                    float(row['AP']),
                    int(row['Backtest_Years'])
                ])

            table = wandb.Table(
                columns=['Model', 'Top1_Accuracy', 'Top3_Recall', 'MRR',
                         'Precision@5', 'AP', 'Backtest_Years'],
                data=table_data
            )
            wandb.log({'model_comparison': table})

            # Set best model metrics to summary
            best_model_row = comparison_df.iloc[0]
            wandb.summary['best_model'] = best_model_row['Model']
            wandb.summary['best_top1_accuracy'] = best_model_row['Top1_Accuracy']
            wandb.summary['best_top3_recall'] = best_model_row['Top3_Recall']
            wandb.summary['best_mrr'] = best_model_row['MRR']
            wandb.summary['best_precision_at_5'] = best_model_row['Precision@5']
            wandb.summary['best_ap'] = best_model_row['AP']

            logger.info("✓ Logged model comparison to W&B")

        except Exception as e:
            logger.warning(f"Failed to log model comparison to W&B: {e}")

    # Generate detailed backtest tables for best model
    print("\n" + "="*70)
    print("GENERATING DETAILED EVALUATION TABLES")
    print("="*70)

    # Find best model by MRR
    if not comparison_df.empty:
        best_model = comparison_df.iloc[0]['Model']
        print(f"Best model (by MRR): {best_model}")

        if best_model in results:
            metrics_df = results[best_model].get('metrics_df')
            if metrics_df is not None and not metrics_df.empty:
                model_name_safe = best_model.lower().replace(' ', '_')
                generate_backtest_table(metrics_df, model_name_safe)
                identify_failure_cases(metrics_df, model_name_safe, threshold_rank=5)
    elif 'Ridge_Ratios' in results:
        # Fallback to Ridge_Ratios if comparison failed
        metrics_df = results['Ridge_Ratios'].get('metrics_df')
        if metrics_df is not None and not metrics_df.empty:
            generate_backtest_table(metrics_df, 'ridge_ratios')
            identify_failure_cases(metrics_df, 'ridge_ratios', threshold_rank=5)

    # Show feature importance for Ridge model
    print("\n" + "="*50)
    print("RIDGE REGRESSION ANALYSIS")
    print("="*50)
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(stats[BASE_PREDICTORS], stats["Share"])

    # Feature coefficients
    feature_importance = pd.DataFrame({
        'Feature': BASE_PREDICTORS,
        'Coefficient': reg.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    # Correlation analysis (numeric columns only)
    print("\nTop 10 Correlations with MVP Share:")
    numeric_stats = stats.select_dtypes(include=[np.number])
    correlations = numeric_stats.corr()["Share"].sort_values(ascending=False)
    print(correlations.head(10).to_string())

    return feature_importance, correlations

def predict_multiple_years(stats, predictors, years_to_predict):
    """Make predictions for multiple years"""
    print("\n" + "="*60)
    print(f"MAKING PREDICTIONS FOR YEARS: {years_to_predict}")
    print("="*60)

    # Add ratio features
    predictors_with_ratios = predictors + [f"{stat}_R" for stat in RATIO_STATS]

    reg = Ridge(alpha=RIDGE_ALPHA)
    all_year_predictions = {}

    for year in years_to_predict:
        if year not in stats['Year'].values:
            print(f"\n❌ No data available for {year}")
            continue

        print(f"\n📊 Predictions for {year}")
        print("-" * 40)

        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]

        if train.empty:
            print(f"❌ No training data available for {year}")
            continue

        reg.fit(train[predictors_with_ratios], train["Share"])

        predictions = reg.predict(test[predictors_with_ratios])
        predictions_df = pd.DataFrame(predictions, columns=["predictions"], index=test.index)

        combination = pd.concat([test[["Player", "Share"]], predictions_df], axis=1)
        combination = add_ranks(combination)

        # Show top predictions vs actual
        print("Top 10 Predicted vs Actual:")
        top_predictions = combination.sort_values("predictions", ascending=False).head(10)
        display_cols = ["Player", "Share", "predictions", "Predicted_Rk"]
        if "Rk" in top_predictions.columns:
            display_cols.append("Rk")

        # Format for better display
        display_df = top_predictions[display_cols].copy()
        display_df["Share"] = display_df["Share"].round(3)
        display_df["predictions"] = display_df["predictions"].round(3)
        print(display_df.to_string(index=False))

        # Calculate metrics
        mse = mean_squared_error(combination["Share"], combination["predictions"])
        print(f"\n📈 Mean Squared Error: {mse:.6f}")

        # Calculate AP if there are actual MVP votes
        if combination["Share"].sum() > 0:
            ap = find_ap(combination)
            print(f"🎯 Average Precision: {ap:.4f}")

        # Show actual MVP winner if available
        actual_winner = combination[combination["Share"] == combination["Share"].max()]
        if not actual_winner.empty and actual_winner["Share"].iloc[0] > 0:
            winner = actual_winner.iloc[0]
            print(f"🏆 Actual MVP: {winner['Player']} (Share: {winner['Share']:.3f})")
            print(f"🤖 Our prediction rank for MVP: #{int(winner['Predicted_Rk'])}")

        all_year_predictions[year] = combination

    return all_year_predictions

def generate_shap_analysis(stats, predictors):
    """
    Generate SHAP values and visualizations for XGBoost model

    Args:
        stats: DataFrame with all data
        predictors: List of predictor features

    Returns:
        tuple: (shap_values, explainer, X_data)
    """
    print("\n" + "="*70)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*70)

    # Train XGBoost on all available data
    print("Training XGBoost model on all historical data...")

    # Create categorical encodings if not present
    if 'NPos' not in stats.columns:
        stats['NPos'] = stats['Pos'].astype('category').cat.codes
    if 'NTm' not in stats.columns:
        stats['NTm'] = stats['Tm'].astype('category').cat.codes

    # Add ratio features
    predictors_with_ratios = predictors + [f"{stat}_R" for stat in RATIO_STATS]
    all_predictors = predictors_with_ratios + ["NPos", "NTm"]

    # Filter to available features
    available_predictors = [p for p in all_predictors if p in stats.columns]

    # Remove rows with missing values
    stats_clean = stats.dropna(subset=available_predictors + ['Share'])

    X = stats_clean[available_predictors]
    y = stats_clean['Share']

    print(f"Training data shape: {X.shape}")
    print(f"Features used: {len(available_predictors)}")

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
        subsample=XGBOOST_SUBSAMPLE,
        colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
        random_state=XGBOOST_RANDOM_STATE,
        objective='reg:squarederror'
    )

    xgb_model.fit(X, y)
    print("✓ XGBoost model trained")

    # Compute SHAP values
    print("\nComputing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(xgb_model)

    # Use a sample for visualization if dataset is large
    if len(X) > 500:
        X_sample = X.sample(n=500, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        print(f"✓ SHAP values computed for {len(X_sample)} samples (sampled for visualization)")
    else:
        X_sample = X
        shap_values = explainer.shap_values(X)
        print(f"✓ SHAP values computed for {len(X)} samples")

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating SHAP visualizations...")

    # 1. Beeswarm plot (feature importance with value distribution)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved beeswarm plot to {SHAP_BEESWARM_FILE}")

    # 2. Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved bar plot to {SHAP_BAR_FILE}")

    # 3. Top features table
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    # Save top 15 features
    top_features = feature_importance_df.head(15)
    top_features.to_csv(SHAP_TOP_FEATURES_FILE, index=False)
    print(f"✓ Saved top features to {SHAP_TOP_FEATURES_FILE}")

    print("\nTop 10 Features by Mean Absolute SHAP Value:")
    print(top_features.head(10).to_string(index=False))

    return shap_values, explainer, X, xgb_model, available_predictors

def explain_season(stats, year, model, predictors, explainer, top_n=5):
    """
    Explain predictions for a specific season using SHAP values

    Args:
        stats: DataFrame with all data
        year: Season year to explain
        model: Trained XGBoost model
        predictors: List of predictor features
        explainer: SHAP explainer object
        top_n: Number of top candidates to show

    Returns:
        DataFrame with explanations
    """
    # Filter to season
    season_data = stats[stats['Year'] == year].copy()

    if season_data.empty:
        print(f"No data found for {year}")
        return None

    # Get predictions
    X_season = season_data[predictors]
    predictions = model.predict(X_season)
    season_data['prediction'] = predictions

    # Sort by prediction
    season_data = season_data.sort_values('prediction', ascending=False)

    # Get top N candidates
    top_candidates = season_data.head(top_n)

    # Compute SHAP values for this season
    shap_values_season = explainer.shap_values(X_season)

    print(f"\n{'='*70}")
    print(f"SEASON {year} — TOP {top_n} MVP CANDIDATES")
    print(f"{'='*70}")

    explanations = []

    for idx, (i, row) in enumerate(top_candidates.iterrows()):
        player = row['Player']
        pred_rank = idx + 1
        actual_share = row['Share']

        # Get SHAP values for this player
        player_idx = season_data.index.get_loc(i)
        player_shap = shap_values_season[player_idx]

        # Get top 3 SHAP drivers
        shap_df = pd.DataFrame({
            'feature': predictors,
            'shap_value': player_shap,
            'feature_value': X_season.iloc[player_idx].values
        })

        # Sort by absolute SHAP value
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        top_drivers = shap_df.sort_values('abs_shap', ascending=False).head(3)

        print(f"\n{pred_rank}. {player} (predicted rank: {pred_rank}, actual share: {actual_share:.3f})")

        for _, driver in top_drivers.iterrows():
            feat = driver['feature']
            shap_val = driver['shap_value']
            feat_val = driver['feature_value']
            sign = "+" if shap_val > 0 else ""
            print(f"   {sign}{feat}: {sign}{shap_val:.3f} (value: {feat_val:.2f})")

            # Store for CSV
            explanations.append({
                'Year': year,
                'Player': player,
                'Predicted_Rank': pred_rank,
                'Actual_Share': actual_share,
                'Feature': feat,
                'SHAP_Value': shap_val,
                'Feature_Value': feat_val
            })

    return pd.DataFrame(explanations)

def generate_season_explanations(stats, model, predictors, explainer, last_n_seasons=5):
    """
    Generate explanations for the last N backtested seasons

    Args:
        stats: DataFrame with all data
        model: Trained XGBoost model
        predictors: List of predictor features
        explainer: SHAP explainer object
        last_n_seasons: Number of recent seasons to explain

    Returns:
        DataFrame with all explanations
    """
    print("\n" + "="*70)
    print(f"GENERATING SEASON EXPLANATIONS (Last {last_n_seasons} seasons)")
    print("="*70)

    # Get last N seasons
    available_years = sorted(stats['Year'].unique())
    recent_years = available_years[-last_n_seasons:] if len(available_years) >= last_n_seasons else available_years

    all_explanations = []

    for year in recent_years:
        exp_df = explain_season(stats, year, model, predictors, explainer, top_n=5)
        if exp_df is not None:
            all_explanations.append(exp_df)

    # Combine all explanations
    if all_explanations:
        combined_df = pd.concat(all_explanations, ignore_index=True)
        combined_df.to_csv(SEASON_EXPLANATIONS_FILE, index=False)
        print(f"\n✓ Saved season explanations to {SEASON_EXPLANATIONS_FILE}")
        return combined_df
    else:
        print("No explanations generated")
        return None

def print_pipeline_summary():
    """Print comprehensive pipeline summary with accuracy and failure analysis"""
    logger.info("\n" + "="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)

    try:
        # Load model comparison results
        if MODEL_COMPARISON_FILE.exists():
            comparison = pd.read_csv(MODEL_COMPARISON_FILE)
            logger.info("\n📊 MODEL PERFORMANCE:")

            for _, row in comparison.iterrows():
                model_name = row['Model']
                top1 = row['Top1_Accuracy']
                top3 = row['Top3_Recall']
                n_years = int(row['Backtest_Years'])

                top1_correct = int(top1 * n_years)
                logger.info(f"\n  {model_name}:")
                logger.info(f"    ✓ Top-1 Accuracy: {top1:.1%} ({top1_correct}/{n_years} seasons correct)")
                logger.info(f"    ✓ Top-3 Recall: {top3:.1%}")
                logger.info(f"    ✓ MRR: {row['MRR']:.3f}")
        else:
            logger.warning("Model comparison file not found")

    except Exception as e:
        logger.error(f"Error loading model comparison: {e}")

    # Analyze failure cases
    try:
        # Look for annotated failure cases
        failure_files = list(TABLES_DIR.glob("failure_cases_*.csv"))

        if failure_files:
            logger.info("\n🔍 FAILURE ANALYSIS:")

            all_failures = []
            for file in failure_files:
                df = pd.read_csv(file)
                if not df.empty:
                    all_failures.append(df)

            if all_failures:
                failures_df = pd.concat(all_failures, ignore_index=True)

                # Count failures
                total_failures = len(failures_df)
                logger.info(f"\n  Total major failures (MVP ranked >5): {total_failures}")

                # Check if failures have been annotated with reasons
                if 'Failure_Reason' in failures_df.columns:
                    reason_counts = failures_df['Failure_Reason'].value_counts()
                    logger.info("\n  Common failure types:")
                    for reason, count in reason_counts.items():
                        if pd.notna(reason) and reason != '':
                            logger.info(f"    • {reason}: {count}")

                    if len(reason_counts) > 0:
                        most_common = reason_counts.index[0]
                        logger.info(f"\n  ⚠️  Most common failure type: {most_common}")
                else:
                    logger.info("  (Failure reasons not yet annotated)")

                # Show specific failures
                logger.info("\n  Failure details:")
                for _, row in failures_df.head(3).iterrows():
                    year = int(row['Year'])
                    mvp = row['Actual_MVP']
                    rank = int(row['Predicted_Rank'])
                    severity = row['Severity']
                    logger.info(f"    • {year}: {mvp} predicted at rank {rank} ({severity})")
        else:
            logger.info("\n  No major failures recorded")

    except Exception as e:
        logger.error(f"Error analyzing failures: {e}")

    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*70)

def main():
    """Main function"""
    print("="*70)
    print("NBA MVP PREDICTION MODEL - MACHINE LEARNING PIPELINE")
    print("="*70)

    # Initialize Weights & Biases (optional)
    init_wandb()

    # Load and prepare data
    stats = load_and_prepare_data()

    print(f"Using {len(BASE_PREDICTORS)} base predictor variables")

    # Create ratio features
    stats = create_ratio_features(stats)

    # Generate comprehensive feature list
    generate_feature_list(stats)

    # Get available predictors (filter out any that don't exist)
    available_base = get_available_predictors(stats, BASE_PREDICTORS)
    print(f"\nAvailable base predictors: {len(available_base)}")

    # Add MVP features if available
    available_mvp = get_available_predictors(stats, MVP_FEATURES)
    print(f"Available MVP features: {len(available_mvp)}")

    # Combine all available features
    all_predictors = available_base + available_mvp
    print(f"Total available predictors: {len(all_predictors)}")

    # Train models with base features
    results = train_models(stats, available_base)

    # Analyze results
    feature_importance, correlations = analyze_results(results, stats)

    # Generate SHAP explainability analysis
    try:
        shap_values, explainer, X_shap, xgb_model_shap, shap_predictors = generate_shap_analysis(stats, available_base)

        # Log SHAP to wandb
        log_shap_to_wandb(shap_values, X_shap)

        # Generate season-level explanations
        season_explanations = generate_season_explanations(stats, xgb_model_shap, shap_predictors, explainer, last_n_seasons=5)
    except Exception as e:
        print(f"\n⚠ SHAP analysis failed: {e}")
        print("Continuing with pipeline...")

    # Make predictions for recent years
    available_years = sorted(stats['Year'].unique())
    recent_years = [year for year in PREDICTION_YEARS if year in available_years]

    if recent_years:
        predictions = predict_multiple_years(stats, BASE_PREDICTORS, recent_years)

        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)

        feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
        correlations.to_csv(CORRELATIONS_FILE)

        # Save predictions for each year
        for year, pred_df in predictions.items():
            filename = OUTPUT_DIR / f"predictions_{year}.csv"
            pred_df.to_csv(filename, index=False)
            print(f"Saved predictions for {year} to {filename}")

        print("\n✅ Analysis complete! Files saved:")
        print(f"   - {FEATURE_IMPORTANCE_FILE.name}: Feature importance from Ridge regression")
        print(f"   - {CORRELATIONS_FILE.name}: Correlations with MVP share")
        print("   - predictions_[year].csv: Predictions for each year")
    else:
        print(f"\n❌ No recent years {PREDICTION_YEARS} found in dataset")
        print("Available years:", available_years)

    # Print comprehensive pipeline summary
    print_pipeline_summary()

    # Finish wandb run
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        logger.info("✓ Weights & Biases run completed")

if __name__ == "__main__":
    main()
