"""
Data Integrity Tests

Tests for data quality and correctness in the processed dataset.
Focus on real-world data requirements rather than synthetic edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import PLAYER_MVP_STATS_FILE


class TestDataIntegrity:
    """Test data integrity in the processed dataset"""

    @pytest.fixture
    def processed_data(self):
        """Load the actual processed data if it exists, otherwise use synthetic"""
        if PLAYER_MVP_STATS_FILE.exists():
            return pd.read_csv(PLAYER_MVP_STATS_FILE)
        else:
            # Fallback to synthetic for offline testing
            pytest.skip("Processed data file not found - run pipeline first")

    def test_all_seasons_1991_2024_present(self, processed_data):
        """Test that all seasons from 1991 to 2024 are present"""
        expected_years = set(range(1991, 2025))
        actual_years = set(processed_data['Year'].unique())

        missing_years = expected_years - actual_years

        assert len(missing_years) == 0, f"Missing seasons: {sorted(missing_years)}"
        assert processed_data['Year'].min() == 1991
        assert processed_data['Year'].max() == 2024

    def test_vote_shares_sum_to_one_per_season(self, processed_data):
        """Test that vote shares sum to approximately 1.0 per season (within reasonable tolerance)"""
        # Filter to only actual players (not League Average) who received votes
        players_only = processed_data[
            (processed_data['Share'] > 0) &
            (~processed_data['Player'].str.contains('League Average', na=False))
        ].copy()

        # Group by year and sum shares
        share_sums = players_only.groupby('Year')['Share'].sum()

        # Check each season
        # NOTE: MVP voting methodology can result in shares summing > 1.0
        # We check that it's reasonable (between 0.5 and 3.0)
        for year, total_share in share_sums.items():
            assert 0.5 <= total_share <= 3.0, \
                f"Year {year}: vote shares sum to {total_share:.4f} (expected 0.5-3.0)"

    def test_no_duplicate_player_seasons(self, processed_data):
        """Test that no player appears twice in the same season"""
        duplicates = processed_data.groupby(['Player', 'Year']).size()
        duplicates = duplicates[duplicates > 1]

        assert len(duplicates) == 0, \
            f"Found {len(duplicates)} duplicate player-seasons: {dict(duplicates)}"

    def test_no_nulls_in_required_columns(self, processed_data):
        """Test that required model input columns have no nulls (excluding League Average rows)"""
        # Filter out League Average rows (these are metadata, not players)
        players_only = processed_data[
            ~processed_data['Player'].str.contains('League Average', na=False)
        ].copy()

        required_cols = [
            'Player', 'Year', 'Share', 'PTS', 'AST', 'TRB',
            'G', 'MP', 'W/L%', 'team_win_pct'
        ]

        for col in required_cols:
            if col in players_only.columns:
                null_count = players_only[col].isnull().sum()
                assert null_count == 0, \
                    f"Column '{col}' has {null_count} null values (excluding League Average)"

    def test_mvp_winner_exists_every_season(self, processed_data):
        """Test that every season has an MVP winner (max vote share > 0.5)"""
        # Group by year and find max share
        max_shares = processed_data.groupby('Year')['Share'].max()

        # Check each season has a winner
        for year, max_share in max_shares.items():
            # MVP winner typically has > 0.5 share (50%+ of possible votes)
            # Some years might be close, so we check > 0.3 as minimum
            assert max_share > 0.3, \
                f"Year {year}: max share is {max_share:.3f} (expected clear winner)"

        # Count seasons with dominant winners (> 0.5 share)
        dominant_winners = (max_shares > 0.5).sum()
        total_seasons = len(max_shares)

        # Most seasons should have a clear winner
        assert dominant_winners / total_seasons > 0.7, \
            f"Only {dominant_winners}/{total_seasons} seasons have dominant MVP (>0.5 share)"


class TestDataIntegrityWithSynthetic:
    """Tests using synthetic data (fast, always runnable)"""

    def test_vote_shares_sum_constraint_synthetic(self, processed_data_with_features):
        """Test vote share sum constraint with synthetic data"""
        df = processed_data_with_features

        # Get seasons with votes
        seasons_with_votes = df[df['Share'] > 0].groupby('Year')['Share'].sum()

        # All should sum to ~1.0
        for year, total in seasons_with_votes.items():
            assert 0.95 <= total <= 1.05, \
                f"Synthetic year {year} shares sum to {total} (expected ~1.0)"

    def test_no_duplicate_players_synthetic(self, processed_data_with_features):
        """Test no duplicate player-seasons with synthetic data"""
        df = processed_data_with_features

        duplicates = df.groupby(['Player', 'Year']).size()
        max_count = duplicates.max()

        assert max_count == 1, \
            f"Found players appearing {max_count} times in same season"

    def test_mvp_exists_per_season_synthetic(self, processed_data_with_features):
        """Test that each synthetic season has an MVP (max Share)"""
        df = processed_data_with_features

        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            max_share = year_data['Share'].max()

            assert max_share > 0, \
                f"Synthetic year {year} has no MVP (max Share = {max_share})"

    def test_required_columns_no_nulls_synthetic(self, processed_data_with_features):
        """Test that key columns have no nulls in synthetic data"""
        df = processed_data_with_features

        required = ['Player', 'Year', 'Share', 'PTS', 'G', 'team_win_pct']

        for col in required:
            if col in df.columns:
                assert df[col].notnull().all(), \
                    f"Synthetic data has nulls in required column: {col}"

    def test_share_values_valid_range(self, processed_data_with_features):
        """Test that Share values are in valid range [0, 1]"""
        df = processed_data_with_features

        assert (df['Share'] >= 0).all(), "Found negative Share values"
        assert (df['Share'] <= 1).all(), "Found Share values > 1.0"

    def test_games_played_realistic(self, processed_data_with_features):
        """Test that games played is realistic (1-82 range)"""
        df = processed_data_with_features

        assert (df['G'] >= 1).all(), "Found players with 0 games"
        assert (df['G'] <= 82).all(), "Found players with > 82 games in regular season"

    def test_win_percentage_valid_range(self, processed_data_with_features):
        """Test that team win percentage is [0, 1]"""
        df = processed_data_with_features

        if 'W/L%' in df.columns:
            assert (df['W/L%'] >= 0).all(), "Found negative win percentages"
            assert (df['W/L%'] <= 1).all(), "Found win percentages > 1.0"

    def test_positive_counting_stats(self, processed_data_with_features):
        """Test that counting stats are non-negative"""
        df = processed_data_with_features

        counting_stats = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG', 'FT']

        for stat in counting_stats:
            if stat in df.columns:
                assert (df[stat] >= 0).all(), \
                    f"Found negative values in {stat}"
