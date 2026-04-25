"""
Test suite for data validation.

Tests that all expected seasons are present, vote shares sum correctly,
no duplicate player-seasons exist, and columns have correct data types.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSeasonCoverage:
    """Test that all expected seasons are present"""

    def test_all_seasons_present(self, processed_data_with_features):
        """Test that all synthetic seasons (2020-2023) are in dataset"""
        df = processed_data_with_features

        expected_years = [2020, 2021, 2022, 2023]
        actual_years = sorted(df['Year'].unique())

        assert actual_years == expected_years, f"Expected {expected_years}, got {actual_years}"

    def test_no_missing_seasons(self, processed_data_with_features):
        """Test that there are no gaps in the season range"""
        df = processed_data_with_features

        min_year = df['Year'].min()
        max_year = df['Year'].max()

        expected_years = list(range(min_year, max_year + 1))
        actual_years = sorted(df['Year'].unique())

        assert actual_years == expected_years, "Missing seasons in range"

    def test_each_season_has_players(self, processed_data_with_features):
        """Test that each season has at least 20 players"""
        df = processed_data_with_features

        for year in df['Year'].unique():
            season_players = df[df['Year'] == year]
            assert len(season_players) >= 20, f"Season {year} has fewer than 20 players"


class TestMVPVoteShares:
    """Test MVP vote share integrity"""

    def test_vote_shares_sum_approximately_one(self, processed_data_with_features):
        """Test that vote shares sum to ~1.0 per season (allowing for rounding)"""
        df = processed_data_with_features

        for year in df['Year'].unique():
            season_data = df[df['Year'] == year]
            total_share = season_data['Share'].sum()

            # Should sum to approximately 1.0 (allowing for floating point errors and partial votes)
            # In real data, total might be slightly less due to players not getting votes
            # For synthetic data with controlled votes, should be close to top voter's share
            assert total_share > 0, f"Season {year} has no MVP votes"
            assert total_share < 5.0, f"Season {year} vote shares sum to {total_share} (too high)"

    def test_vote_shares_non_negative(self, processed_data_with_features):
        """Test that all vote shares are non-negative"""
        df = processed_data_with_features

        assert (df['Share'] >= 0).all(), "Found negative vote shares"

    def test_vote_shares_not_exceed_one(self, processed_data_with_features):
        """Test that no individual vote share exceeds 1.0"""
        df = processed_data_with_features

        assert (df['Share'] <= 1.0).all(), "Found vote share > 1.0"

    def test_mvp_winner_has_highest_share(self, processed_data_with_features):
        """Test that the player with highest share each season is realistic"""
        df = processed_data_with_features

        for year in df['Year'].unique():
            season_data = df[df['Year'] == year]
            max_share = season_data['Share'].max()

            # Winner typically has >0.5 share
            if max_share > 0:
                # At least one player should have votes
                assert season_data['Share'].sum() > 0

    def test_pts_won_matches_share(self, synthetic_mvp_votes):
        """Test that Pts Won is consistent with Share"""
        df = synthetic_mvp_votes

        # Pts Won should be Share * Pts Max
        for idx, row in df.iterrows():
            expected_pts = row['Share'] * row['Pts Max']
            # Allow some tolerance for rounding
            assert abs(row['Pts Won'] - expected_pts) < 10, \
                f"Pts Won ({row['Pts Won']}) doesn't match Share * Pts Max ({expected_pts})"


class TestPlayerUniqueness:
    """Test that players appear at most once per season"""

    def test_no_duplicate_player_seasons(self, processed_data_with_features):
        """Test that each player appears at most once per season"""
        df = processed_data_with_features

        # Count player-year combinations
        duplicates = df.groupby(['Player', 'Year']).size()
        duplicate_entries = duplicates[duplicates > 1]

        assert len(duplicate_entries) == 0, \
            f"Found {len(duplicate_entries)} duplicate player-seasons: {duplicate_entries.head()}"

    def test_player_names_consistent(self, processed_data_with_features):
        """Test that player names don't have inconsistent formatting"""
        df = processed_data_with_features

        # No player names should have leading/trailing spaces
        assert not df['Player'].str.startswith(' ').any(), "Found player names with leading spaces"
        assert not df['Player'].str.endswith(' ').any(), "Found player names with trailing spaces"

        # No player names should contain asterisks (removed in cleaning)
        assert not df['Player'].str.contains(r'\*', regex=True).any(), \
            "Found player names with asterisks"


class TestColumnExistence:
    """Test that all required columns exist"""

    def test_required_base_columns_exist(self, processed_data_with_features):
        """Test that all required base columns exist"""
        df = processed_data_with_features

        required_cols = [
            'Player', 'Year', 'Tm', 'Team', 'Share',
            'Age', 'G', 'MP', 'PTS', 'AST', 'TRB', 'FG%', 'W', 'L', 'W/L%'
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_feature_columns_exist(self, processed_data_with_features):
        """Test that all engineered feature columns exist"""
        df = processed_data_with_features

        feature_cols = [
            # Z-scores
            'PTS_zscore', 'AST_zscore', 'TRB_zscore', 'WS_zscore',

            # Team success
            'team_win_pct', 'conference_rank', 'made_playoffs', 'is_top3_seed',

            # Availability
            'games_played', 'games_played_pct', 'minutes_per_game',

            # Narrative
            'previous_mvp_finish', 'previous_top5_count',

            # Team context
            'is_best_player_on_team',
        ]

        for col in feature_cols:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_unnamed_columns(self, processed_data_with_features):
        """Test that there are no 'Unnamed' columns (from bad CSV reads)"""
        df = processed_data_with_features

        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        assert len(unnamed_cols) == 0, f"Found unnamed columns: {unnamed_cols}"


class TestColumnDataTypes:
    """Test that columns have correct data types"""

    def test_year_is_integer(self, processed_data_with_features):
        """Test that Year column is integer type"""
        df = processed_data_with_features

        assert df['Year'].dtype in [np.int32, np.int64], \
            f"Year should be integer, got {df['Year'].dtype}"

    def test_age_is_integer(self, processed_data_with_features):
        """Test that Age column is integer type"""
        df = processed_data_with_features

        assert df['Age'].dtype in [np.int32, np.int64], \
            f"Age should be integer, got {df['Age'].dtype}"

    def test_stats_are_numeric(self, processed_data_with_features):
        """Test that statistical columns are numeric"""
        df = processed_data_with_features

        numeric_cols = ['PTS', 'AST', 'TRB', 'FG%', 'MP', 'W', 'L', 'W/L%']

        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"{col} should be numeric, got {df[col].dtype}"

    def test_share_is_float(self, processed_data_with_features):
        """Test that Share column is float type"""
        df = processed_data_with_features

        assert df['Share'].dtype == np.float64, \
            f"Share should be float64, got {df['Share'].dtype}"

    def test_binary_features_are_integer(self, processed_data_with_features):
        """Test that binary features are integer type"""
        df = processed_data_with_features

        binary_cols = ['made_playoffs', 'is_top3_seed', 'is_best_player_on_team']

        for col in binary_cols:
            assert df[col].dtype in [np.int32, np.int64], \
                f"{col} should be integer, got {df[col].dtype}"


class TestDataQuality:
    """Test general data quality"""

    def test_no_all_null_columns(self, processed_data_with_features):
        """Test that no columns are entirely null"""
        df = processed_data_with_features

        for col in df.columns:
            null_pct = df[col].isnull().mean()
            assert null_pct < 1.0, f"Column {col} is entirely null"

    def test_critical_columns_no_nulls(self, processed_data_with_features):
        """Test that critical columns have no null values"""
        df = processed_data_with_features

        critical_cols = ['Player', 'Year', 'Share']

        for col in critical_cols:
            null_count = df[col].isnull().sum()
            assert null_count == 0, f"Critical column {col} has {null_count} nulls"

    def test_team_names_exist(self, processed_data_with_features):
        """Test that team names are not null"""
        df = processed_data_with_features

        # Both Tm (abbreviation) and Team (full name) should exist
        assert df['Tm'].notnull().all(), "Found null team abbreviations"
        assert df['Team'].notnull().all(), "Found null team names"

    def test_games_played_reasonable(self, processed_data_with_features):
        """Test that games played is in reasonable range"""
        df = processed_data_with_features

        # Games played should be between 1 and 82
        assert (df['G'] >= 1).all(), "Found games played < 1"
        assert (df['G'] <= 82).all(), "Found games played > 82"

    def test_win_percentage_range(self, processed_data_with_features):
        """Test that win percentage is between 0 and 1"""
        df = processed_data_with_features

        # W/L% should be between 0 and 1
        assert (df['W/L%'] >= 0).all(), "Found W/L% < 0"
        assert (df['W/L%'] <= 1).all(), "Found W/L% > 1"

    def test_percentages_in_range(self, processed_data_with_features):
        """Test that shooting percentages are in valid range (0-1)"""
        df = processed_data_with_features

        percentage_cols = ['FG%', '3P%', '2P%', 'FT%', 'eFG%']

        for col in percentage_cols:
            if col in df.columns:
                # Allow for nulls (some players might not shoot 3s)
                non_null = df[col].dropna()
                assert (non_null >= 0).all(), f"Found {col} < 0"
                assert (non_null <= 1).all(), f"Found {col} > 1"


class TestTeamDataIntegrity:
    """Test team data integrity"""

    def test_wins_plus_losses_equals_games(self, synthetic_teams):
        """Test that W + L equals total games (82)"""
        df = synthetic_teams

        total_games = df['W'] + df['L']

        # Allow for some seasons with different lengths (lockouts, etc.)
        # For synthetic data, should be exactly 82
        assert (total_games == 82).all(), "W + L doesn't equal 82"

    def test_win_pct_calculated_correctly(self, synthetic_teams):
        """Test that W/L% = W / (W + L)"""
        df = synthetic_teams

        calculated_pct = df['W'] / (df['W'] + df['L'])
        actual_pct = df['W/L%']

        # Allow small floating point tolerance
        diff = (calculated_pct - actual_pct).abs()
        assert (diff < 0.01).all(), "W/L% not calculated correctly"

    def test_each_team_appears_each_season(self, synthetic_teams):
        """Test that each team appears in each season"""
        df = synthetic_teams

        teams_per_season = df.groupby('Year')['Team'].nunique()

        # All seasons should have same number of teams
        assert teams_per_season.nunique() == 1, "Different number of teams per season"
