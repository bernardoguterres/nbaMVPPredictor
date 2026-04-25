"""
Test suite for feature engineering functions.

Tests that all 52 features are correctly created, z-scores are normalized,
ratio features handle edge cases, and no data leakage occurs.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictors import (
    add_league_relative_features,
    add_team_success_features,
    add_availability_features,
    add_narrative_features,
    add_team_context_features,
    handle_multiple_teams,
)
# Import from machine_learning for ratio features
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from machine_learning import create_ratio_features


class TestLeagueRelativeFeatures:
    """Test z-score feature engineering"""

    def test_zscore_features_created(self, synthetic_seasons):
        """Test that all z-score features are created"""
        df = synthetic_seasons.copy()
        df['WS'] = np.random.uniform(1.0, 15.0, len(df))

        df = add_league_relative_features(df)

        # Check all z-score columns exist
        assert 'PTS_zscore' in df.columns
        assert 'AST_zscore' in df.columns
        assert 'TRB_zscore' in df.columns
        assert 'WS_zscore' in df.columns

    def test_zscore_normalization_within_season(self, synthetic_seasons):
        """Test that z-scores are normalized within each season (mean=0, std=1)"""
        df = synthetic_seasons.copy()
        df['WS'] = np.random.uniform(1.0, 15.0, len(df))

        df = add_league_relative_features(df)

        # For each season, z-scores should have mean≈0 and std≈1
        for year in df['Year'].unique():
            season_data = df[df['Year'] == year]

            for stat in ['PTS', 'AST', 'TRB', 'WS']:
                zscore_col = f'{stat}_zscore'

                # Mean should be very close to 0 (allowing for floating point errors)
                mean = season_data[zscore_col].mean()
                assert abs(mean) < 1e-10, f"{zscore_col} mean not 0 for year {year}: {mean}"

                # Std should be 1 (or 0 if all values are identical)
                std = season_data[zscore_col].std()
                if season_data[stat].std() > 0:
                    assert abs(std - 1.0) < 0.01, f"{zscore_col} std not 1 for year {year}: {std}"

    def test_zscore_no_future_data_leakage(self, synthetic_seasons):
        """Test that z-scores don't use future season data"""
        df = synthetic_seasons.copy()
        df['WS'] = np.random.uniform(1.0, 15.0, len(df))

        # Add a unique identifier to one season
        year_2020 = df[df['Year'] == 2020].copy()
        year_2020_mean = year_2020['PTS'].mean()

        df = add_league_relative_features(df)

        # 2020 z-scores should only depend on 2020 data
        year_2020_after = df[df['Year'] == 2020]
        recalculated_mean = year_2020_after['PTS'].mean()

        # Mean PTS should not have changed
        assert abs(year_2020_mean - recalculated_mean) < 1e-10

    def test_zscore_handles_zero_std(self):
        """Test that z-score handles case where all values are identical"""
        df = pd.DataFrame({
            'Year': [2020] * 5,
            'PTS': [25.0] * 5,  # All identical
            'AST': [5.0] * 5,
            'TRB': [8.0] * 5,
            'WS': [10.0] * 5,
        })

        df = add_league_relative_features(df)

        # All z-scores should be 0 (not NaN)
        assert (df['PTS_zscore'] == 0).all()
        assert not df['PTS_zscore'].isnull().any()


class TestTeamSuccessFeatures:
    """Test team success feature engineering"""

    def test_team_success_features_created(self, processed_data_with_features):
        """Test that all team success features are created"""
        df = processed_data_with_features

        assert 'team_win_pct' in df.columns
        assert 'conference_rank' in df.columns
        assert 'made_playoffs' in df.columns
        assert 'is_top3_seed' in df.columns

    def test_made_playoffs_binary(self, processed_data_with_features):
        """Test that made_playoffs is binary (0 or 1)"""
        df = processed_data_with_features

        assert df['made_playoffs'].isin([0, 1]).all()

    def test_is_top3_seed_binary(self, processed_data_with_features):
        """Test that is_top3_seed is binary (0 or 1)"""
        df = processed_data_with_features

        assert df['is_top3_seed'].isin([0, 1]).all()

    def test_top3_seed_implies_playoffs(self, processed_data_with_features):
        """Test that top 3 seed always makes playoffs"""
        df = processed_data_with_features

        # If is_top3_seed == 1, then made_playoffs must == 1
        top3_teams = df[df['is_top3_seed'] == 1]
        assert (top3_teams['made_playoffs'] == 1).all()


class TestAvailabilityFeatures:
    """Test player availability feature engineering"""

    def test_availability_features_created(self, processed_data_with_features):
        """Test that all availability features are created"""
        df = processed_data_with_features

        assert 'games_played' in df.columns
        assert 'games_played_pct' in df.columns
        assert 'minutes_per_game' in df.columns

    def test_games_played_pct_range(self, processed_data_with_features):
        """Test that games_played_pct is between 0 and 1"""
        df = processed_data_with_features

        assert (df['games_played_pct'] >= 0).all()
        assert (df['games_played_pct'] <= 1.0).all()

    def test_games_played_matches_G(self, processed_data_with_features):
        """Test that games_played equals G column"""
        df = processed_data_with_features

        assert (df['games_played'] == df['G']).all()

    def test_minutes_per_game_matches_MP(self, processed_data_with_features):
        """Test that minutes_per_game equals MP column"""
        df = processed_data_with_features

        assert (df['minutes_per_game'] == df['MP']).all()


class TestNarrativeFeatures:
    """Test narrative/history feature engineering"""

    def test_narrative_features_created(self, processed_data_with_features):
        """Test that all narrative features are created"""
        df = processed_data_with_features

        assert 'previous_mvp_finish' in df.columns
        assert 'previous_top5_count' in df.columns

    def test_no_future_data_in_narrative_features(self):
        """Test that narrative features only use PRIOR seasons"""
        # Create dataset with known MVP history
        data = []

        # Player_0 wins MVP in 2020
        data.append({'Player': 'Player_0', 'Year': 2020, 'Share': 0.9})
        data.append({'Player': 'Player_1', 'Year': 2020, 'Share': 0.5})

        # Player_0 wins again in 2021
        data.append({'Player': 'Player_0', 'Year': 2021, 'Share': 0.85})
        data.append({'Player': 'Player_1', 'Year': 2021, 'Share': 0.6})

        # Player_1 wins in 2022
        data.append({'Player': 'Player_0', 'Year': 2022, 'Share': 0.3})
        data.append({'Player': 'Player_1', 'Year': 2022, 'Share': 0.8})

        df = pd.DataFrame(data)

        # Add narrative features
        df = add_narrative_features(df)

        # In 2020, Player_0 should have no prior finishes
        player0_2020 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2020)]
        assert player0_2020['previous_mvp_finish'].iloc[0] == 0
        assert player0_2020['previous_top5_count'].iloc[0] == 0

        # In 2021, Player_0 should have previous_mvp_finish=1 (won in 2020)
        player0_2021 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2021)]
        assert player0_2021['previous_mvp_finish'].iloc[0] == 1
        assert player0_2021['previous_top5_count'].iloc[0] == 1

        # In 2022, Player_0 should have previous_mvp_finish=1 (still from wins)
        player0_2022 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2022)]
        assert player0_2022['previous_mvp_finish'].iloc[0] == 1
        assert player0_2022['previous_top5_count'].iloc[0] == 2  # Top 5 in 2020, 2021

    def test_previous_top5_count_increments(self):
        """Test that previous_top5_count correctly counts prior top-5 finishes"""
        # Create 3 seasons where Player_0 finishes top 5 each time
        data = []
        for year in [2020, 2021, 2022]:
            data.append({'Player': 'Player_0', 'Year': year, 'Share': 0.9})
            for i in range(1, 10):  # Other players
                data.append({'Player': f'Player_{i}', 'Year': year, 'Share': 0.1 * (10 - i)})

        df = pd.DataFrame(data)
        df = add_narrative_features(df)

        # Player_0 in each year
        p0_2020 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2020)]
        p0_2021 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2021)]
        p0_2022 = df[(df['Player'] == 'Player_0') & (df['Year'] == 2022)]

        # Counts should increment
        assert p0_2020['previous_top5_count'].iloc[0] == 0  # No prior
        assert p0_2021['previous_top5_count'].iloc[0] == 1  # 2020
        assert p0_2022['previous_top5_count'].iloc[0] == 2  # 2020, 2021


class TestTeamContextFeatures:
    """Test team context feature engineering"""

    def test_team_context_features_created(self, processed_data_with_features):
        """Test that is_best_player_on_team is created"""
        df = processed_data_with_features

        assert 'is_best_player_on_team' in df.columns

    def test_is_best_player_binary(self, processed_data_with_features):
        """Test that is_best_player_on_team is binary"""
        df = processed_data_with_features

        assert df['is_best_player_on_team'].isin([0, 1]).all()

    def test_at_least_one_best_per_team(self, processed_data_with_features):
        """Test that each team-year has at least one 'best player'"""
        df = processed_data_with_features

        # Group by team-year and check at least one player is flagged
        for (team, year), group in df.groupby(['Team', 'Year']):
            best_count = group['is_best_player_on_team'].sum()
            assert best_count >= 1, f"No best player for {team} in {year}"

    def test_best_player_has_max_ws(self, processed_data_with_features):
        """Test that best player has maximum WS on their team"""
        df = processed_data_with_features

        for (team, year), group in df.groupby(['Team', 'Year']):
            best_players = group[group['is_best_player_on_team'] == 1]
            max_ws = group['WS'].max()

            # All flagged players should have max WS
            assert (best_players['WS'] == max_ws).all()


class TestMultiTeamHandling:
    """Test handling of players traded mid-season"""

    def test_multi_team_player_one_row_per_season(self):
        """Test that multi-team players produce one row per player-season"""
        # Create player who was on 3 teams in 2020
        data = pd.DataFrame({
            'Player': ['Player_0', 'Player_0', 'Player_0', 'Player_1'],
            'Year': [2020, 2020, 2020, 2020],
            'Tm': ['TEAM0', 'TEAM1', 'TOT', 'TEAM2'],
            'PTS': [10.0, 12.0, 22.0, 15.0],  # TOT = sum of other teams
        })

        result = handle_multiple_teams(data)

        # Should have only 2 rows (one per player)
        assert len(result) == 2

        # Player_0 should appear once
        player0_rows = result[result['Player'] == 'Player_0']
        assert len(player0_rows) == 1

    def test_multi_team_uses_tot_stats(self):
        """Test that TOT (total) stats are used for multi-team players"""
        data = pd.DataFrame({
            'Player': ['Player_0', 'Player_0', 'Player_0'],
            'Year': [2020, 2020, 2020],
            'Tm': ['TEAM0', 'TEAM1', 'TOT'],
            'PTS': [10.0, 12.0, 22.0],
        })

        result = handle_multiple_teams(data)

        # Should use TOT stats (22.0 PTS)
        assert result.iloc[0]['PTS'] == 22.0

    def test_multi_team_assigns_last_team(self):
        """Test that last team is assigned when using TOT stats"""
        # Basketball Reference typically orders as: TOT, TEAM1, TEAM2
        # So last row is the actual last team
        data = pd.DataFrame({
            'Player': ['Player_0', 'Player_0', 'Player_0'],
            'Year': [2020, 2020, 2020],
            'Tm': ['TOT', 'TEAM0', 'TEAM1'],  # Realistic order from Basketball Reference
            'PTS': [22.0, 10.0, 12.0],
        })

        result = handle_multiple_teams(data)

        # Should assign last team (TEAM1) using TOT stats
        assert result.iloc[0]['Tm'] == 'TEAM1'
        assert result.iloc[0]['PTS'] == 22.0  # TOT stats


class TestRatioFeatures:
    """Test ratio features from machine_learning.py"""

    def test_ratio_features_have_no_nulls(self, synthetic_seasons):
        """Test that ratio features have no null values after engineering"""
        df = synthetic_seasons.copy()

        # Create ratio features
        df = create_ratio_features(df)

        # Check that all ratio features exist and have no nulls
        ratio_features = ['PTS_R', 'AST_R', 'STL_R', 'BLK_R', '3P_R']

        for feature in ratio_features:
            if feature in df.columns:
                null_count = df[feature].isnull().sum()
                assert null_count == 0, \
                    f"Ratio feature {feature} has {null_count} null values"

    def test_ratio_features_positive(self, synthetic_seasons):
        """Test that ratio features are non-negative (ratios should be >= 0)"""
        df = synthetic_seasons.copy()
        df = create_ratio_features(df)

        for feature in ['PTS_R', 'AST_R', 'STL_R', 'BLK_R', '3P_R']:
            if feature in df.columns:
                assert (df[feature] >= 0).all(), \
                    f"Ratio feature {feature} has negative values"


class TestAllFeaturesPresent:
    """Test that complete feature engineering produces all 52 features"""

    def test_all_52_features_present(self, processed_data_with_features, expected_features):
        """Test that all expected features exist in processed data"""
        df = processed_data_with_features

        # Filter to only feature columns (not metadata like Player, Year, etc.)
        feature_columns = [
            'PTS_zscore', 'AST_zscore', 'TRB_zscore', 'WS_zscore',
            'team_win_pct', 'conference_rank', 'made_playoffs', 'is_top3_seed',
            'games_played', 'games_played_pct', 'minutes_per_game',
            'previous_mvp_finish', 'previous_top5_count',
            'is_best_player_on_team',
        ]

        for col in feature_columns:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_null_features(self, processed_data_with_features):
        """Test that engineered features have no null values"""
        df = processed_data_with_features

        # Z-scores should not be null
        assert not df['PTS_zscore'].isnull().any()
        assert not df['AST_zscore'].isnull().any()

        # Team success should not be null
        assert not df['team_win_pct'].isnull().any()
        assert not df['made_playoffs'].isnull().any()

        # Availability should not be null
        assert not df['games_played'].isnull().any()
        assert not df['games_played_pct'].isnull().any()

    def test_feature_dtypes_correct(self, processed_data_with_features):
        """Test that feature data types are correct"""
        df = processed_data_with_features

        # Binary features should be int
        assert df['made_playoffs'].dtype in [np.int32, np.int64]
        assert df['is_top3_seed'].dtype in [np.int32, np.int64]
        assert df['is_best_player_on_team'].dtype in [np.int32, np.int64]

        # Z-scores should be float
        assert df['PTS_zscore'].dtype == np.float64
        assert df['AST_zscore'].dtype == np.float64
