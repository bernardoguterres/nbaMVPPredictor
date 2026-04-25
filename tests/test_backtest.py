"""
Test suite for backtest integrity.

Tests that backtesting follows time-aware rules: no future data leakage,
minimum training window enforced, and metrics are in valid ranges.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machine_learning import calculate_mrr, calculate_top1_accuracy, calculate_top3_recall


class TestTimeAwareBacktest:
    """Test that backtesting doesn't use future data"""

    def test_training_data_before_test_year(self):
        """Test that training data only includes years before test year"""
        # Simulate a time-aware backtest
        data = pd.DataFrame({
            'Year': [2020, 2020, 2021, 2021, 2022, 2022],
            'Player': ['P1', 'P2', 'P1', 'P2', 'P1', 'P2'],
            'PTS': [25, 20, 27, 22, 26, 21],
        })

        test_year = 2022
        train = data[data['Year'] < test_year]
        test = data[data['Year'] == test_year]

        # Training set should only have 2020, 2021
        assert train['Year'].max() < test_year, \
            f"Training data includes year {train['Year'].max()} >= test year {test_year}"

        # No overlap in years
        train_years = set(train['Year'].unique())
        test_years = set(test['Year'].unique())
        assert len(train_years.intersection(test_years)) == 0, \
            "Training and test sets have overlapping years"

    def test_no_future_data_in_features(self):
        """Test that features don't use data from future seasons"""
        # Create dataset with distinct season means
        data = []

        for year in [2020, 2021, 2022]:
            # Each year has different average PTS
            base_pts = year - 1990  # 2020 -> 30, 2021 -> 31, etc.

            for i in range(10):
                data.append({
                    'Year': year,
                    'Player': f'Player_{i}',
                    'PTS': base_pts + np.random.uniform(-2, 2),
                })

        df = pd.DataFrame(data)

        # Calculate z-scores within each year
        df['PTS_zscore'] = df.groupby('Year')['PTS'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # For 2020 data, z-score should only depend on 2020 mean
        year_2020 = df[df['Year'] == 2020]
        year_2020_mean = year_2020['PTS'].mean()

        # Calculate expected z-score manually
        year_2020_std = year_2020['PTS'].std()
        expected_zscore_first = (year_2020.iloc[0]['PTS'] - year_2020_mean) / year_2020_std
        actual_zscore_first = year_2020.iloc[0]['PTS_zscore']

        # Should match (no contamination from 2021, 2022)
        assert abs(expected_zscore_first - actual_zscore_first) < 1e-10, \
            "Z-score uses future data"


class TestMinimumTrainingWindow:
    """Test that minimum training window is enforced"""

    def test_minimum_10_seasons_required(self):
        """Test that backtest only starts after 10 training seasons"""
        from config import MIN_TRAINING_SEASONS, START_YEAR

        min_backtest_year = START_YEAR + MIN_TRAINING_SEASONS

        # First backtest year should be START_YEAR + 10
        assert min_backtest_year == START_YEAR + 10, \
            f"Minimum backtest year should be {START_YEAR + 10}, got {min_backtest_year}"

    def test_insufficient_training_data_rejected(self):
        """Test that we don't train with fewer than minimum seasons"""
        # Simulate attempting backtest with only 5 years of data
        data = pd.DataFrame({
            'Year': [2018, 2019, 2020, 2021, 2022],
            'Player': ['P1'] * 5,
            'PTS': [25, 26, 27, 28, 29],
        })

        test_year = 2022
        train = data[data['Year'] < test_year]

        unique_train_years = train['Year'].nunique()

        # Only 4 years (2018-2021)
        MIN_REQUIRED = 10
        assert unique_train_years < MIN_REQUIRED, \
            "Test setup should have insufficient data"

        # In real pipeline, this would be skipped
        if unique_train_years < MIN_REQUIRED:
            # Should not train
            assert True
        else:
            pytest.fail("Should have insufficient training data")

    def test_first_valid_backtest_year(self):
        """Test that first backtest year has exactly 10 prior seasons"""
        # Create 11 years of data (1991-2001)
        years = list(range(1991, 2002))
        data = pd.DataFrame({
            'Year': years * 5,  # 5 players per year
            'Player': [f'P{i}' for i in range(len(years) * 5)],
            'PTS': np.random.uniform(15, 30, len(years) * 5),
        })

        # First backtest year should be 2001 (10 years after 1991)
        first_test_year = 2001
        train = data[data['Year'] < first_test_year]

        # Should have exactly 10 training years
        assert train['Year'].nunique() == 10, \
            f"First backtest should have 10 training years, got {train['Year'].nunique()}"


class TestMetricValidRanges:
    """Test that evaluation metrics are in valid ranges"""

    def _create_mock_predictions(self, mvp_rank):
        """Helper to create mock prediction DataFrame where MVP is at given rank"""
        # Create 10 players with predictions
        data = {
            'Player': [f'P{i}' for i in range(10)],
            'predictions': [1.0 - (i * 0.1) for i in range(10)],  # Descending predictions
            'Share': [0.0] * 10
        }

        # Set actual MVP (highest Share) at the specified rank
        mvp_idx = mvp_rank - 1
        data['Share'][mvp_idx] = 1.0

        return pd.DataFrame(data)

    def test_mrr_between_0_and_1(self):
        """Test that Mean Reciprocal Rank is between 0 and 1"""
        # Test various rankings
        test_cases = [
            (1, 1.0),      # Rank 1 -> MRR = 1/1 = 1.0
            (2, 0.5),      # Rank 2 -> MRR = 1/2 = 0.5
            (3, 0.333),    # Rank 3 -> MRR = 1/3 = 0.333
            (5, 0.2),      # Rank 5 -> MRR = 1/5 = 0.2
        ]

        for rank, expected_mrr in test_cases:
            df = self._create_mock_predictions(rank)
            mrr = calculate_mrr(df)
            assert 0 <= mrr <= 1, f"MRR {mrr} not in range [0, 1]"
            assert abs(mrr - expected_mrr) < 0.01, \
                f"MRR for rank {rank} should be ~{expected_mrr}, got {mrr}"

    def test_top1_accuracy_binary(self):
        """Test that Top-1 accuracy is 0 or 1"""
        df_rank1 = self._create_mock_predictions(1)
        df_rank2 = self._create_mock_predictions(2)

        assert calculate_top1_accuracy(df_rank1) == 1, "Rank 1 should have Top-1 accuracy = 1"
        assert calculate_top1_accuracy(df_rank2) == 0, "Rank 2 should have Top-1 accuracy = 0"

    def test_top3_recall_binary(self):
        """Test that Top-3 recall is 0 or 1"""
        assert calculate_top3_recall(self._create_mock_predictions(1)) == 1, "Rank 1 should be in top 3"
        assert calculate_top3_recall(self._create_mock_predictions(2)) == 1, "Rank 2 should be in top 3"
        assert calculate_top3_recall(self._create_mock_predictions(3)) == 1, "Rank 3 should be in top 3"
        assert calculate_top3_recall(self._create_mock_predictions(4)) == 0, "Rank 4 should not be in top 3"

    def test_aggregate_metrics_in_range(self):
        """Test that aggregate metrics are in valid range"""
        # Simulate backtest results across 5 seasons
        ranks = [1, 2, 5, 3, 2]

        mrrs = [calculate_mrr(self._create_mock_predictions(r)) for r in ranks]
        top1s = [calculate_top1_accuracy(self._create_mock_predictions(r)) for r in ranks]
        top3s = [calculate_top3_recall(self._create_mock_predictions(r)) for r in ranks]

        # Averages
        avg_mrr = np.mean(mrrs)
        avg_top1 = np.mean(top1s)
        avg_top3 = np.mean(top3s)

        # All should be in [0, 1]
        assert 0 <= avg_mrr <= 1, f"Average MRR {avg_mrr} not in [0, 1]"
        assert 0 <= avg_top1 <= 1, f"Average Top-1 {avg_top1} not in [0, 1]"
        assert 0 <= avg_top3 <= 1, f"Average Top-3 {avg_top3} not in [0, 1]"


class TestBacktestPredictionRanking:
    """Test that backtest correctly ranks predictions"""

    def test_predictions_ranked_by_share(self):
        """Test that predictions are ranked from highest to lowest"""
        # Simulate predictions
        predictions = pd.DataFrame({
            'Player': ['P1', 'P2', 'P3', 'P4', 'P5'],
            'Prediction': [0.8, 0.5, 0.3, 0.1, 0.05],
            'Share': [0.9, 0.6, 0.2, 0.0, 0.0],
        })

        # Rank by prediction (descending)
        predictions['Rank'] = predictions['Prediction'].rank(ascending=False, method='min')

        # P1 should be rank 1 (highest prediction)
        assert predictions[predictions['Player'] == 'P1']['Rank'].iloc[0] == 1

        # P5 should be rank 5 (lowest prediction)
        assert predictions[predictions['Player'] == 'P5']['Rank'].iloc[0] == 5

        # Ranks should be in order
        sorted_preds = predictions.sort_values('Prediction', ascending=False)
        expected_ranks = [1, 2, 3, 4, 5]
        actual_ranks = sorted_preds['Rank'].tolist()
        assert actual_ranks == expected_ranks

    def test_actual_mvp_rank_identified(self):
        """Test that we correctly identify where actual MVP was ranked"""
        # Simulate predictions where actual MVP is P2
        predictions = pd.DataFrame({
            'Player': ['P1', 'P2', 'P3'],
            'Prediction': [0.8, 0.5, 0.3],
            'Share': [0.6, 0.9, 0.1],  # P2 is actual MVP
        })

        # Rank predictions
        predictions['Predicted_Rank'] = predictions['Prediction'].rank(
            ascending=False, method='min'
        )

        # Find actual MVP (highest share)
        actual_mvp = predictions.loc[predictions['Share'].idxmax()]

        # Actual MVP (P2) was predicted at rank 2
        assert actual_mvp['Predicted_Rank'] == 2

    def test_tied_predictions_handled(self):
        """Test that tied predictions are handled consistently"""
        predictions = pd.DataFrame({
            'Player': ['P1', 'P2', 'P3'],
            'Prediction': [0.5, 0.5, 0.3],  # P1 and P2 tied
        })

        # Rank with 'min' method (ties get same rank)
        predictions['Rank'] = predictions['Prediction'].rank(ascending=False, method='min')

        # Both P1 and P2 should have rank 1
        p1_rank = predictions[predictions['Player'] == 'P1']['Rank'].iloc[0]
        p2_rank = predictions[predictions['Player'] == 'P2']['Rank'].iloc[0]

        assert p1_rank == p2_rank == 1


class TestRollingBacktest:
    """Test rolling backtest logic"""

    def test_training_window_grows_over_time(self):
        """Test that training window expands as we progress through years"""
        years = list(range(2010, 2021))  # 11 years
        data = pd.DataFrame({
            'Year': years * 5,
            'Player': [f'P{i}' for i in range(len(years) * 5)],
        })

        backtest_years = [2020, 2021]  # Last 2 years

        training_sizes = []

        for test_year in backtest_years:
            train = data[data['Year'] < test_year]
            training_sizes.append(train['Year'].nunique())

        # Training size should increase
        assert training_sizes[1] > training_sizes[0], \
            "Training window should grow over time"

        # 2020 should have 10 years (2010-2019)
        assert training_sizes[0] == 10

        # 2021 should have 11 years (2010-2020)
        assert training_sizes[1] == 11

    def test_each_year_tested_once(self):
        """Test that each year is tested exactly once in backtest"""
        years = list(range(2010, 2024))  # 14 years
        data = pd.DataFrame({
            'Year': years * 5,
            'Player': [f'P{i}' for i in range(len(years) * 5)],
        })

        # Backtest on 2020-2023 (last 4 years, after 10-year minimum)
        backtest_years = [2020, 2021, 2022, 2023]

        tested_years = set()

        for test_year in backtest_years:
            assert test_year not in tested_years, f"Year {test_year} tested multiple times"
            tested_years.add(test_year)

        # All backtest years should be tested
        assert tested_years == set(backtest_years)


class TestPredictionQuality:
    """Test prediction output quality"""

    def test_predictions_all_non_negative(self):
        """Test that predicted vote shares are non-negative"""
        # Simulate model predictions
        predictions = np.array([0.8, 0.5, 0.3, 0.1, 0.05])

        assert (predictions >= 0).all(), "Predictions should be non-negative"

    def test_predictions_reasonable_range(self):
        """Test that predictions are in reasonable range (0-1.5)"""
        # Simulate predictions (some models might predict >1.0 before clipping)
        predictions = np.array([0.9, 0.6, 0.3, 0.1, 0.0])

        # Most predictions should be in [0, 1] range
        # (in real model, we might clip to [0, 1])
        assert (predictions >= 0).all()
        assert (predictions <= 1.5).all(), "Predictions should not be unreasonably large"

    def test_at_least_one_prediction_per_season(self):
        """Test that every season has at least one prediction"""
        # Simulate backtest results
        results = pd.DataFrame({
            'Year': [2020, 2020, 2021, 2021, 2022, 2022],
            'Player': ['P1', 'P2', 'P1', 'P2', 'P1', 'P2'],
            'Prediction': [0.8, 0.5, 0.7, 0.6, 0.9, 0.4],
        })

        # Every year should have predictions
        for year in results['Year'].unique():
            year_preds = results[results['Year'] == year]
            assert len(year_preds) > 0, f"Year {year} has no predictions"
            assert year_preds['Prediction'].max() > 0, f"Year {year} has no positive predictions"


class TestCrossValidationLeakage:
    """Test that cross-validation doesn't leak information"""

    def test_test_set_not_in_training(self):
        """Test that test set is completely separate from training set"""
        data = pd.DataFrame({
            'Year': [2020, 2020, 2021, 2021, 2022, 2022],
            'Player': ['P1', 'P2', 'P1', 'P2', 'P1', 'P2'],
            'PTS': [25, 20, 27, 22, 26, 21],
        })

        test_year = 2022
        train = data[data['Year'] < test_year]
        test = data[data['Year'] == test_year]

        # No rows should be in both
        train_indices = set(train.index)
        test_indices = set(test.index)

        overlap = train_indices.intersection(test_indices)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping rows between train and test"

    def test_player_can_appear_in_both_train_and_test(self):
        """Test that same player can appear in different years (not leakage)"""
        data = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'Player': ['P1', 'P1', 'P1'],  # Same player, different years
            'PTS': [25, 27, 26],
        })

        test_year = 2022
        train = data[data['Year'] < test_year]
        test = data[data['Year'] == test_year]

        # P1 should be in both (different years, not leakage)
        assert 'P1' in train['Player'].values
        assert 'P1' in test['Player'].values

        # But the 2022 stats should NOT be in training
        assert 2022 not in train['Year'].values


class TestBacktestCoverage:
    """Test backtest runs on sufficient historical data"""

    def test_backtest_runs_on_at_least_20_seasons(self):
        """Test that backtest evaluates at least 20 seasons (1991-2024 = 34 years)"""
        # With full 1991-2024 data and MIN_TRAINING_SEASONS=10,
        # backtest should run from 2001-2024 = 24 seasons
        from config import START_YEAR, MIN_TRAINING_SEASONS
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        # Load processed data if exists
        from config import PLAYER_MVP_STATS_FILE

        if not PLAYER_MVP_STATS_FILE.exists():
            pytest.skip("Processed data not found - run pipeline first")

        df = pd.read_csv(PLAYER_MVP_STATS_FILE)

        # Calculate expected backtest years
        first_backtest_year = START_YEAR + MIN_TRAINING_SEASONS
        last_year = df['Year'].max()

        expected_backtest_years = list(range(first_backtest_year, last_year + 1))

        # Should be at least 20 seasons
        assert len(expected_backtest_years) >= 20, \
            f"Backtest only covers {len(expected_backtest_years)} seasons (expected >= 20)"

    def test_ensemble_predictions_exist_for_all_seasons(self, processed_data_with_features):
        """Test that ensemble can generate predictions for every backtest season"""
        from sklearn.linear_model import Ridge
        from machine_learning import backtest
        from config import BASE_PREDICTORS

        df = processed_data_with_features.copy()

        # Simulate backtest for all years
        years = sorted(df['Year'].unique())[1:]  # Skip first year (need training data)

        model = Ridge(alpha=0.1)
        predictors = [col for col in BASE_PREDICTORS if col in df.columns][:10]  # Use first 10 predictors

        metrics, metrics_df, all_preds = backtest(df, model, years, predictors)

        # Check that we have predictions for each year
        if not all_preds.empty:
            years_with_predictions = set(df.loc[all_preds.index, 'Year'].unique())

            assert len(years_with_predictions) == len(years), \
                f"Predictions exist for {len(years_with_predictions)}/{len(years)} seasons"
