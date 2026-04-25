"""
Shared test fixtures for NBA MVP Predictor tests

Creates small synthetic datasets to test pipeline without requiring real data.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def synthetic_seasons():
    """
    Generate 4 synthetic seasons (2020-2023) for testing.
    Small enough to run fast, large enough to test all features.
    """
    np.random.seed(42)

    seasons_data = []

    for year in [2020, 2021, 2022, 2023]:
        # 30 players per season (mimics actual data scale)
        n_players = 30

        # Base player stats
        players = pd.DataFrame({
            'Player': [f'Player_{i}' for i in range(n_players)],
            'Year': year,
            'Age': np.random.randint(20, 38, n_players),
            'G': np.random.randint(40, 82, n_players),
            'GS': np.random.randint(35, 82, n_players),
            'MP': np.random.uniform(20.0, 38.0, n_players),

            # Shooting stats
            'FG': np.random.uniform(3.0, 12.0, n_players),
            'FGA': np.random.uniform(8.0, 24.0, n_players),
            'FG%': np.random.uniform(0.40, 0.55, n_players),
            '3P': np.random.uniform(0.5, 4.0, n_players),
            '3PA': np.random.uniform(1.0, 10.0, n_players),
            '3P%': np.random.uniform(0.30, 0.45, n_players),
            '2P': np.random.uniform(2.5, 10.0, n_players),
            '2PA': np.random.uniform(5.0, 18.0, n_players),
            '2P%': np.random.uniform(0.45, 0.60, n_players),
            'eFG%': np.random.uniform(0.48, 0.62, n_players),

            # Free throws
            'FT': np.random.uniform(1.0, 10.0, n_players),
            'FTA': np.random.uniform(2.0, 12.0, n_players),
            'FT%': np.random.uniform(0.70, 0.90, n_players),

            # Rebounds
            'ORB': np.random.uniform(0.5, 3.0, n_players),
            'DRB': np.random.uniform(2.0, 10.0, n_players),
            'TRB': np.random.uniform(3.0, 12.0, n_players),

            # Other stats
            'AST': np.random.uniform(1.0, 10.0, n_players),
            'STL': np.random.uniform(0.5, 2.5, n_players),
            'BLK': np.random.uniform(0.2, 2.5, n_players),
            'TOV': np.random.uniform(1.0, 4.0, n_players),
            'PF': np.random.uniform(1.5, 3.5, n_players),
            'PTS': np.random.uniform(8.0, 30.0, n_players),

            # Win Shares (for team context features)
            'WS': np.random.uniform(1.0, 15.0, n_players),

            # Team (distribute across 10 teams)
            'Tm': [f'TEAM{i % 10}' for i in range(n_players)],
        })

        seasons_data.append(players)

    return pd.concat(seasons_data, ignore_index=True)


@pytest.fixture
def synthetic_mvp_votes():
    """
    Generate synthetic MVP voting data.
    Top 5 players per season get votes, with decreasing shares.
    """
    np.random.seed(42)

    mvp_data = []

    for year in [2020, 2021, 2022, 2023]:
        # Top 5 vote-getters per season
        shares = []
        for rank in range(1, 6):
            player_idx = rank - 1  # Player_0, Player_1, etc.

            # MVP share decreases with rank
            # Rank 1: ~0.8-1.0, Rank 2: ~0.5-0.7, etc.
            if rank == 1:
                share = np.random.uniform(0.8, 1.0)
            elif rank == 2:
                share = np.random.uniform(0.5, 0.7)
            elif rank == 3:
                share = np.random.uniform(0.3, 0.5)
            elif rank == 4:
                share = np.random.uniform(0.15, 0.3)
            else:
                share = np.random.uniform(0.05, 0.15)

            shares.append(share)

        # Normalize shares to sum to 1.0
        total_share = sum(shares)
        normalized_shares = [s / total_share for s in shares]

        for rank, share in enumerate(normalized_shares, start=1):
            player_idx = rank - 1

            mvp_data.append({
                'Player': f'Player_{player_idx}',
                'Year': year,
                'Pts Won': int(share * 1000),
                'Pts Max': 1000,
                'Share': share,
                'WS': np.random.uniform(10.0, 15.0),  # MVP candidates have high WS
            })

    return pd.DataFrame(mvp_data)


@pytest.fixture
def synthetic_teams():
    """
    Generate synthetic team standings data.
    10 teams with varying records.
    """
    np.random.seed(42)

    teams_data = []

    for year in [2020, 2021, 2022, 2023]:
        for team_idx in range(10):
            # Win percentage varies from 0.3 to 0.7
            win_pct = np.random.uniform(0.30, 0.70)
            wins = int(82 * win_pct)
            losses = 82 - wins

            # Recalculate W/L% from actual W and L for consistency
            actual_win_pct = wins / (wins + losses)

            teams_data.append({
                'Team': f'Team{team_idx}',  # Full name
                'Year': year,
                'W': wins,
                'L': losses,
                'W/L%': actual_win_pct,  # Use calculated value for consistency
                'GB': np.random.uniform(0, 20),  # Games behind
                'PS/G': np.random.uniform(100, 120),  # Points scored
                'PA/G': np.random.uniform(100, 115),  # Points allowed
                'SRS': np.random.uniform(-5, 8),  # Simple Rating System
            })

    return pd.DataFrame(teams_data)


@pytest.fixture
def synthetic_nicknames():
    """
    Team abbreviation to full name mapping.
    """
    return pd.DataFrame({
        'Abbreviation': [f'TEAM{i}' for i in range(10)],
        'Name': [f'Team{i}' for i in range(10)],
    })


@pytest.fixture
def temp_data_dir(synthetic_seasons, synthetic_mvp_votes, synthetic_teams, synthetic_nicknames):
    """
    Create a temporary directory with synthetic CSV files.
    Mimics the real data structure.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Create subdirectories
        raw_dir = Path(temp_dir) / "raw"
        processed_dir = Path(temp_dir) / "processed"
        raw_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)

        # Save CSV files
        synthetic_seasons.to_csv(raw_dir / "players.csv", index=False)
        synthetic_mvp_votes.to_csv(raw_dir / "mvps.csv", index=False)
        synthetic_teams.to_csv(raw_dir / "teams.csv", index=False)

        # Save nicknames as CSV
        synthetic_nicknames.to_csv(raw_dir / "nicknames.csv", index=False)

        yield temp_dir
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


@pytest.fixture
def processed_data_with_features(synthetic_seasons, synthetic_mvp_votes, synthetic_teams):
    """
    Fully processed dataset with all 52 features.
    This mimics the output of predictors.py
    """
    # Merge players and MVP votes
    df = synthetic_seasons.merge(
        synthetic_mvp_votes[['Player', 'Year', 'Share', 'Pts Won', 'Pts Max']],
        on=['Player', 'Year'],
        how='left'
    )
    df[['Share', 'Pts Won', 'Pts Max']] = df[['Share', 'Pts Won', 'Pts Max']].fillna(0)

    # Add team names
    df['Team'] = df['Tm'].str.replace('TEAM', 'Team')

    # Merge team stats
    df = df.merge(synthetic_teams, on=['Team', 'Year'], how='left')

    # Add league-relative features (z-scores)
    for stat in ['PTS', 'AST', 'TRB', 'WS']:
        df[f'{stat}_zscore'] = df.groupby('Year')[stat].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # Add team success features
    df['team_win_pct'] = df['W/L%']
    df['league_rank'] = df.groupby('Year')['W/L%'].rank(ascending=False, method='min')
    df['conference_rank'] = df['league_rank']
    df['made_playoffs'] = (df['league_rank'] <= 16).astype(int)
    df['is_top3_seed'] = (df['league_rank'] <= 6).astype(int)

    # Add availability features
    df['games_played'] = df['G']
    df['games_played_pct'] = df['G'] / 82.0
    df['minutes_per_game'] = df['MP']

    # Add narrative features (simplified - no historical lookback)
    df['previous_mvp_finish'] = 0  # No previous finishes in synthetic data
    df['previous_top5_count'] = 0

    # Add team context
    df['is_best_player_on_team'] = (
        df.groupby(['Team', 'Year'])['WS'].transform('max') == df['WS']
    ).astype(int)

    return df


@pytest.fixture
def expected_features():
    """
    List of all 52 expected features after processing.
    """
    return [
        # Base stats (33)
        'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
        '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
        'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',
        'GB', 'PS/G', 'PA/G', 'SRS',

        # League-relative (4 z-scores)
        'PTS_zscore', 'AST_zscore', 'TRB_zscore', 'WS_zscore',

        # Team success (4)
        'team_win_pct', 'conference_rank', 'made_playoffs', 'is_top3_seed',

        # Availability (3)
        'games_played', 'games_played_pct', 'minutes_per_game',

        # Narrative (2)
        'previous_mvp_finish', 'previous_top5_count',

        # Team context (1)
        'is_best_player_on_team',

        # Other required columns
        'Player', 'Year', 'Tm', 'Team', 'Share', 'WS',
    ]
