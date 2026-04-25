import pandas as pd
import numpy as np
import logging
from scipy import stats
from config import (
    MVPS_RAW_FILE, PLAYERS_RAW_FILE, TEAMS_RAW_FILE,
    NICKNAMES_FILE, PLAYER_MVP_STATS_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load MVP and player data"""
    logger.info("Loading raw data files...")
    mvps = pd.read_csv(MVPS_RAW_FILE)
    players = pd.read_csv(PLAYERS_RAW_FILE)
    teams = pd.read_csv(TEAMS_RAW_FILE)
    logger.info(f"MVP data loaded: {mvps.shape[0]} rows, {mvps.shape[1]} columns")
    logger.info(f"Player data loaded: {players.shape[0]} rows, {players.shape[1]} columns")
    logger.info(f"Team data loaded: {teams.shape[0]} rows, {teams.shape[1]} columns")

    # Validation: Check for minimum required data
    if len(mvps) == 0 or len(players) == 0 or len(teams) == 0:
        logger.error("One or more data files are empty!")
        raise ValueError("Cannot proceed with empty data files")

    return mvps, players, teams

def clean_mvps_data(mvps):
    """Clean MVP data"""
    logger.info("Cleaning MVP data...")
    # Keep relevant columns including WS for feature engineering
    cols_to_keep = ["Player", "Year", "Pts Won", "Pts Max", "Share"]

    # Add WS if available (for Win Shares feature engineering)
    if "WS" in mvps.columns:
        cols_to_keep.append("WS")
        logger.info("Win Shares (WS) column found and included")
    if "WS/48" in mvps.columns:
        cols_to_keep.append("WS/48")
        logger.info("WS/48 column found and included")

    # Validation: Check for required columns
    missing_cols = [col for col in ["Player", "Year", "Share"] if col not in mvps.columns]
    if missing_cols:
        logger.error(f"Missing required columns in MVP data: {missing_cols}")
        raise ValueError(f"MVP data missing required columns: {missing_cols}")

    mvps = mvps[cols_to_keep]

    # Validation: Check for seasons with fewer than 5 MVP vote-getters
    mvp_counts = mvps.groupby('Year').size()
    low_count_seasons = mvp_counts[mvp_counts < 5]
    if len(low_count_seasons) > 0:
        for year, count in low_count_seasons.items():
            logger.warning(f"Season {year} has only {count} MVP vote-getters (< 5) - may be unreliable")

    logger.info(f"MVP data after cleaning: {mvps.shape[0]} rows, {mvps.shape[1]} columns")
    return mvps

def clean_players_data(players):
    """Clean player data"""
    logger.info("Cleaning player data...")
    # Remove unnamed columns and rank
    if "Unnamed: 0" in players.columns:
        del players["Unnamed: 0"]
    if "Rk" in players.columns:
        del players["Rk"]

    # Rename "Team" to "Tm" for consistency with older code
    if "Team" in players.columns and "Tm" not in players.columns:
        players.rename(columns={"Team": "Tm"}, inplace=True)

    # Remove asterisks from player names
    players["Player"] = players["Player"].str.replace("*", "", regex=False)

    # Validation: Check for required columns
    required_cols = ["Player", "Year", "Tm"]
    missing_cols = [col for col in required_cols if col not in players.columns]
    if missing_cols:
        logger.error(f"Missing required columns in player data: {missing_cols}")
        raise ValueError(f"Player data missing required columns: {missing_cols}")

    logger.info(f"Player data after cleaning: {players.shape[0]} rows, {players.shape[1]} columns")
    return players

def handle_multiple_teams(df):
    """Handle players who played for multiple teams in one season"""
    logger.info("Handling players with multiple teams...")

    # Count multi-team players for logging
    multi_team_count = df.groupby(["Player", "Year"]).size()
    multi_team_players = (multi_team_count > 1).sum()
    if multi_team_players > 0:
        logger.info(f"Found {multi_team_players} player-season combinations with multiple teams")

    def single_team(df_group):
        if df_group.shape[0] == 1:
            return df_group
        else:
            # If player has TOT (total) row, use that but keep the last team
            tot_row = df_group[df_group["Tm"] == "TOT"]
            if not tot_row.empty:
                tot_row = tot_row.copy()
                tot_row["Tm"] = df_group.iloc[-1]["Tm"]  # Use last team
                return tot_row
            else:
                return df_group.iloc[-1:]  # Just take the last team

    # Group by player and year, then apply single_team function
    players_clean = df.groupby(["Player", "Year"]).apply(single_team)

    # Reset index
    players_clean.index = players_clean.index.droplevel()
    players_clean.index = players_clean.index.droplevel()

    logger.info(f"Player data after handling multiple teams: {players_clean.shape[0]} rows")
    return players_clean

def merge_data(players, mvps):
    """Merge player and MVP data"""
    logger.info("Merging player and MVP data...")

    # Validation: Check merge keys exist
    if "Player" not in players.columns or "Year" not in players.columns:
        logger.error("Player data missing merge keys (Player, Year)")
        raise ValueError("Cannot merge: missing Player or Year column in player data")
    if "Player" not in mvps.columns or "Year" not in mvps.columns:
        logger.error("MVP data missing merge keys (Player, Year)")
        raise ValueError("Cannot merge: missing Player or Year column in MVP data")

    players_before = len(players)
    combined = players.merge(mvps, how="outer", on=["Player", "Year"])

    # Fill NaN values in MVP columns with 0
    combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)

    # Validation: Check for unexpected data loss
    if len(combined) < players_before:
        logger.warning(f"Merge resulted in fewer rows ({len(combined)}) than input players ({players_before})")

    logger.info(f"Combined data shape: {combined.shape[0]} rows, {combined.shape[1]} columns")
    return combined

def add_team_names(combined):
    """Add full team names using nicknames mapping"""
    logger.info("Adding team names...")

    # Load nicknames mapping
    nicknames = {}
    try:
        with open(NICKNAMES_FILE) as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():  # Skip empty lines
                    parts = line.replace("\n", "").split(",")
                    if len(parts) >= 2:
                        abbrev = parts[0].strip()
                        name = parts[1].strip()
                        nicknames[abbrev] = name
        logger.info(f"Loaded {len(nicknames)} team abbreviation mappings")
    except FileNotFoundError:
        logger.warning(f"Nicknames file not found: {NICKNAMES_FILE}")
        return combined

    combined["Team"] = combined["Tm"].map(nicknames)

    # Check for unmapped teams
    unmapped = combined[combined["Team"].isnull()]["Tm"].unique()
    if len(unmapped) > 0:
        logger.warning(f"Found {len(unmapped)} unmapped team abbreviations: {list(unmapped)}")

    return combined

def merge_team_data(combined, teams):
    """Merge team statistics"""
    logger.info("Merging team data...")

    # Clean teams data
    if "Unnamed: 0" in teams.columns:
        del teams["Unnamed: 0"]

    # Remove asterisks and playoff indicators from team names
    teams["Team"] = teams["Team"].str.replace("*", "", regex=False)

    # Filter out division headers and other non-team rows
    teams = teams[~teams["W"].astype(str).str.contains("Division", na=False)]
    teams = teams[~teams["W"].astype(str).str.contains("Conference", na=False)]

    # Convert data types
    teams = teams.apply(pd.to_numeric, errors='ignore')

    # Handle GB column (Games Behind) - replace "—" with 0
    if "GB" in teams.columns:
        teams["GB"] = teams["GB"].astype(str).str.replace("—", "0")
        teams["GB"] = pd.to_numeric(teams["GB"], errors='coerce').fillna(0)

    # Validation: Check merge keys exist
    if "Team" not in combined.columns or "Year" not in combined.columns:
        logger.error("Combined data missing Team or Year column for team merge")
        raise ValueError("Cannot merge team data: missing Team or Year column")

    # Merge with combined data
    before_merge = combined.shape[0]
    train = combined.merge(teams, how="left", on=["Team", "Year"])
    after_merge = train.shape[0]

    # Validation: Check for unexpected row count changes
    if after_merge != before_merge:
        logger.warning(f"Team merge changed row count: {before_merge} → {after_merge}")

    # Validation: Check for expected team stat columns
    expected_team_cols = ["W", "L", "W/L%"]
    missing_team_cols = [col for col in expected_team_cols if col not in train.columns]
    if missing_team_cols:
        logger.warning(f"Missing expected team columns after merge: {missing_team_cols}")

    logger.info(f"Team data merged: {train.shape[0]} rows, {train.shape[1]} columns")

    return train

def add_league_relative_features(df):
    """
    Add z-score features for key stats within each season
    Z-score = (value - mean) / std for that season

    Why: MVP voting compares players to league averages, not absolute values
    No leakage: Calculated within season only, using available stats
    """
    logger.info("Adding league-relative features (z-scores)...")

    # Stats to z-score
    stats_to_zscore = ['PTS', 'AST', 'TRB']

    # Add WS if available
    if 'WS' in df.columns:
        stats_to_zscore.append('WS')

    for stat in stats_to_zscore:
        if stat in df.columns:
            # Calculate z-score within each year
            df[f'{stat}_zscore'] = df.groupby('Year')[stat].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            logger.info(f"  ✓ Created {stat}_zscore")
        else:
            logger.warning(f"Skipping {stat}_zscore - column {stat} not found")

    # Validation: Check for nulls in created features
    for stat in stats_to_zscore:
        zscore_col = f'{stat}_zscore'
        if zscore_col in df.columns:
            null_count = df[zscore_col].isnull().sum()
            if null_count > 0:
                logger.warning(f"{zscore_col} has {null_count} null values")

    logger.info("League-relative features added successfully")
    return df

def add_team_success_features(df, teams):
    """
    Add team success and playoff positioning features

    Features:
    - team_win_pct: Already exists as W/L%
    - conference_rank: Rank within conference (1-15)
    - made_playoffs: Binary, top 8 in conference
    - is_top3_seed: Binary, top 3 in conference

    Why: MVP almost always comes from top playoff teams
    No leakage: Final standings available before MVP voting
    """
    logger.info("Adding team success features...")

    # Validation: Check for required column
    if 'W/L%' not in df.columns:
        logger.error("W/L% column not found - cannot create team success features")
        raise ValueError("Missing W/L% column required for team success features")

    # Determine conference for each team-year
    # We need to figure out which conference each team belongs to
    # We'll infer from the original team standings which had East/West splits

    # Create conference rank by sorting within Year and using GB (Games Behind)
    # Teams with same year, sort by W/L% descending
    df = df.sort_values(['Year', 'Team', 'W/L%'], ascending=[True, True, False])

    # For conference_rank, we need to know which teams are in which conference
    # Since we don't have explicit conference info, we'll rank all teams
    # and assume top 15 per conference (30 teams total, 15 per conference)

    # Simple approach: rank by wins within year
    # Better approach: We need to preserve East/West from original scrape
    # For now, let's create a simplified version

    # Add team_win_pct (rename W/L% for clarity)
    df['team_win_pct'] = df['W/L%']

    # Rank teams by W/L% within each year
    df['league_rank'] = df.groupby('Year')['W/L%'].rank(ascending=False, method='min')

    # Simplified: assume top 16 make playoffs (8 per conference)
    # In reality, it's 8 per conference, but without conference info this is close
    df['made_playoffs'] = (df['league_rank'] <= 16).astype(int)

    # Top 6 seeds (3 per conference)
    df['is_top3_seed'] = (df['league_rank'] <= 6).astype(int)

    # For conference_rank, we'll use league_rank as approximation
    # This isn't perfect but captures the essence
    df['conference_rank'] = df['league_rank']  # Simplified

    logger.info("  ✓ Created team_win_pct")
    logger.info("  ✓ Created conference_rank (league-wide ranking)")
    logger.info("  ✓ Created made_playoffs")
    logger.info("  ✓ Created is_top3_seed")

    logger.info("Team success features added successfully")
    return df

def add_availability_features(df):
    """
    Add player availability features

    Features:
    - games_played: Already exists as G
    - games_played_pct: G / 82 (standard NBA season)
    - minutes_per_game: Already exists as MP

    Why: Voters penalize players who miss significant time
    No leakage: Season statistics only
    """
    logger.info("Adding availability features...")

    # Validation: Check for required columns
    if 'G' not in df.columns:
        logger.error("G (games) column not found - cannot create availability features")
        raise ValueError("Missing G column required for availability features")
    if 'MP' not in df.columns:
        logger.warning("MP (minutes per game) column not found")

    # Games played percentage (out of standard 82 game season)
    # Note: Some seasons have different lengths (lockouts, COVID)
    # Using 82 as baseline, actual season length would be better
    df['games_played'] = df['G']
    df['games_played_pct'] = df['G'] / 82.0

    # Clip at 1.0 in case of any data issues
    df['games_played_pct'] = df['games_played_pct'].clip(upper=1.0)

    if 'MP' in df.columns:
        df['minutes_per_game'] = df['MP']
    else:
        df['minutes_per_game'] = 0
        logger.warning("MP column missing - setting minutes_per_game to 0")

    logger.info("  ✓ Created games_played")
    logger.info("  ✓ Created games_played_pct")
    logger.info("  ✓ Created minutes_per_game")

    logger.info("Availability features added successfully")
    return df

def add_narrative_features(df):
    """
    Add narrative and historical features based on prior MVP voting

    Features:
    - previous_mvp_finish: Best prior MVP finish (1=won, 0=never voted)
    - previous_top5_count: Number of prior top-5 finishes

    Why: Narrative matters - voters consider past performance
    No leakage: Uses only PRIOR seasons (Year < current year)
    """
    print("Adding narrative/history features...")

    # Sort by player and year to ensure chronological order
    df = df.sort_values(['Player', 'Year'])

    # Calculate MVP rank from Share
    # Players with Share > 0 got votes, rank them by Share within each year
    df['mvp_rank'] = df.groupby('Year')['Share'].rank(ascending=False, method='min')
    df.loc[df['Share'] == 0, 'mvp_rank'] = 0  # No votes = no rank

    # For each player-year, look at all prior years
    def get_previous_mvp_finish(group):
        """Get best prior MVP finish for each row in player group"""
        result = []
        for idx, row in group.iterrows():
            current_year = row['Year']
            # Get all prior years for this player
            prior = group[group['Year'] < current_year]

            if len(prior) == 0 or prior['Share'].sum() == 0:
                # No prior MVP votes
                best_finish = 0
            else:
                # Best finish is minimum rank (1 = best)
                prior_ranks = prior[prior['Share'] > 0]['mvp_rank']
                if len(prior_ranks) > 0:
                    best_finish = prior_ranks.min()
                else:
                    best_finish = 0

            result.append(best_finish)
        return result

    def get_previous_top5_count(group):
        """Count prior top-5 finishes for each row in player group"""
        result = []
        for idx, row in group.iterrows():
            current_year = row['Year']
            prior = group[group['Year'] < current_year]

            if len(prior) == 0:
                count = 0
            else:
                # Count how many times they finished in top 5
                count = len(prior[(prior['mvp_rank'] > 0) & (prior['mvp_rank'] <= 5)])

            result.append(count)
        return result

    # Apply to each player group
    df['previous_mvp_finish'] = df.groupby('Player', group_keys=False).apply(
        lambda g: pd.Series(get_previous_mvp_finish(g), index=g.index)
    )

    df['previous_top5_count'] = df.groupby('Player', group_keys=False).apply(
        lambda g: pd.Series(get_previous_top5_count(g), index=g.index)
    )

    # Drop temporary mvp_rank column
    df = df.drop('mvp_rank', axis=1)

    logger.info("  ✓ Created previous_mvp_finish")
    logger.info("  ✓ Created previous_top5_count")

    logger.info("Narrative features added successfully")
    return df

def add_team_context_features(df):
    """
    Add team context features

    Features:
    - is_best_player_on_team: 1 if highest WS on their team that season

    Why: MVPs are typically the clear best player on their team
    No leakage: WS is calculated from box score stats available during season
    """
    logger.info("Adding team context features...")

    if 'WS' not in df.columns:
        logger.warning("WS column not found - setting is_best_player_on_team to 0")
        df['is_best_player_on_team'] = 0
    else:
        # For each team-year, find the player with highest WS
        df['is_best_player_on_team'] = (
            df.groupby(['Team', 'Year'])['WS'].transform('max') == df['WS']
        ).astype(int)

        # Handle ties and NaN
        df['is_best_player_on_team'] = df['is_best_player_on_team'].fillna(0).astype(int)

        logger.info("  ✓ Created is_best_player_on_team")

    logger.info("Team context features added successfully")
    return df

def main():
    """Main function to process all data"""
    logger.info("="*60)
    logger.info("STARTING DATA PROCESSING FOR NBA MVP PREDICTION")
    logger.info("="*60)

    # Load data
    mvps, players, teams = load_data()

    # Clean MVP data
    mvps = clean_mvps_data(mvps)

    # Clean player data
    players = clean_players_data(players)

    # Handle multiple teams per player per season
    players = handle_multiple_teams(players)

    # Merge player and MVP data
    combined = merge_data(players, mvps)

    # Add team names
    combined = add_team_names(combined)

    # Merge team data
    train = merge_team_data(combined, teams)

    # Convert data types
    train = train.apply(pd.to_numeric, errors='ignore')

    # FEATURE ENGINEERING
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)

    # Add domain-informed MVP features
    train = add_league_relative_features(train)
    train = add_team_success_features(train, teams)
    train = add_availability_features(train)
    train = add_narrative_features(train)
    train = add_team_context_features(train)

    logger.info("\n✓ Feature engineering completed")
    logger.info(f"Final dataset shape: {train.shape}")

    # Save final dataset
    train.to_csv(PLAYER_MVP_STATS_FILE, index=False)
    logger.info(f"Final dataset saved to {PLAYER_MVP_STATS_FILE}")

    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("DATA PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Final dataset shape: {train.shape}")
    logger.info(f"Years covered: {train['Year'].min()} to {train['Year'].max()}")
    logger.info(f"Number of unique players: {train['Player'].nunique()}")
    logger.info(f"MVP winners in dataset: {len(train[train['Share'] > 0.5])}")
    logger.info(f"Players with MVP votes: {len(train[train['Share'] > 0])}")

    # Show years with data
    years_available = sorted(train['Year'].unique())
    logger.info(f"Years with data: {years_available}")

    logger.info("\nData processing completed successfully!")

if __name__ == "__main__":
    main()
