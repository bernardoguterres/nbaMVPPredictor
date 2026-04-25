# NBA MVP Predictor - Test Suite

Comprehensive test suite for the NBA MVP prediction pipeline.

## Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared pytest fixtures (synthetic data)
├── test_features.py         # Feature engineering tests
├── test_validation.py       # Data validation tests
├── test_backtest.py         # Backtest integrity tests
└── README.md                # This file
```

## Test Coverage

### Feature Engineering Tests (`test_features.py`)

Tests that all 52 features are correctly created:

- **Z-score features** (4): PTS_zscore, AST_zscore, TRB_zscore, WS_zscore
  - ✓ Features created for all stats
  - ✓ Normalized within each season (mean=0, std=1)
  - ✓ No future data leakage
  - ✓ Handles zero standard deviation gracefully

- **Team success features** (4): team_win_pct, conference_rank, made_playoffs, is_top3_seed
  - ✓ All features created
  - ✓ Binary features are 0 or 1
  - ✓ Top-3 seed implies playoffs

- **Availability features** (3): games_played, games_played_pct, minutes_per_game
  - ✓ All features created
  - ✓ games_played_pct in [0, 1]
  - ✓ Values match source columns

- **Narrative features** (2): previous_mvp_finish, previous_top5_count
  - ✓ Features created
  - ✓ No future data leakage (only uses prior seasons)
  - ✓ Counts increment correctly over time

- **Team context features** (1): is_best_player_on_team
  - ✓ Feature created and binary
  - ✓ At least one best player per team
  - ✓ Best player has max Win Shares

- **Multi-team handling**:
  - ✓ One row per player per season
  - ✓ TOT (total) stats used for traded players
  - ✓ Last team assigned

### Data Validation Tests (`test_validation.py`)

Tests data integrity and quality:

- **Season coverage**:
  - ✓ All expected seasons present (2020-2023 for synthetic data)
  - ✓ No missing seasons in range
  - ✓ Each season has sufficient players

- **MVP vote shares**:
  - ✓ Vote shares sum to ~1.0 per season
  - ✓ Non-negative values
  - ✓ No individual share > 1.0
  - ✓ Pts Won consistent with Share

- **Player uniqueness**:
  - ✓ No duplicate player-seasons
  - ✓ Player names consistently formatted
  - ✓ No asterisks in names

- **Column existence**:
  - ✓ All required base columns exist
  - ✓ All engineered features exist
  - ✓ No 'Unnamed' columns

- **Data types**:
  - ✓ Year and Age are integers
  - ✓ Stats are numeric
  - ✓ Share is float
  - ✓ Binary features are integers

- **Data quality**:
  - ✓ No all-null columns
  - ✓ Critical columns have no nulls
  - ✓ Games played in [1, 82]
  - ✓ Win percentages in [0, 1]
  - ✓ Shooting percentages in [0, 1]

### Backtest Integrity Tests (`test_backtest.py`)

Tests time-aware backtesting:

- **Time-aware constraints**:
  - ✓ Training data only includes years before test year
  - ✓ No future data in features
  - ✓ No overlap between train and test sets

- **Minimum training window**:
  - ✓ 10 seasons required before backtesting
  - ✓ Insufficient data rejected
  - ✓ First backtest year has exactly 10 prior seasons

- **Metric validity**:
  - ✓ MRR (Mean Reciprocal Rank) in [0, 1]
  - ✓ Top-1 accuracy is binary (0 or 1)
  - ✓ Top-3 recall is binary (0 or 1)
  - ✓ Aggregate metrics in valid range

- **Prediction ranking**:
  - ✓ Predictions ranked highest to lowest
  - ✓ Actual MVP rank correctly identified
  - ✓ Tied predictions handled consistently

- **Rolling backtest**:
  - ✓ Training window grows over time
  - ✓ Each year tested exactly once

## Running Tests

### All tests
```bash
pytest
```

### Specific test file
```bash
pytest tests/test_features.py
pytest tests/test_validation.py
pytest tests/test_backtest.py
```

### Specific test class
```bash
pytest tests/test_features.py::TestLeagueRelativeFeatures
```

### Specific test
```bash
pytest tests/test_features.py::TestLeagueRelativeFeatures::test_zscore_normalization_within_season
```

### With coverage report
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Parallel execution (fast)
```bash
pytest -n auto
```

### Verbose output
```bash
pytest -v
```

### Stop on first failure
```bash
pytest -x
```

## Synthetic Test Data

Tests use **synthetic fixture data** instead of real NBA data:

- **4 seasons** (2020-2023)
- **30 players per season**
- **10 teams**
- **Top 5 MVP candidates per season** with realistic vote shares

### Why synthetic data?

1. **Fast**: Tests run in <1 second, no need to load 34 years of real data
2. **Offline**: No dependency on CSV files or database
3. **Controlled**: Exact values known, easy to test edge cases
4. **Reproducible**: Same data every run (seed=42)

### Fixtures

- `synthetic_seasons`: Player per-game stats
- `synthetic_mvp_votes`: MVP voting results
- `synthetic_teams`: Team standings
- `synthetic_nicknames`: Team abbreviation mapping
- `temp_data_dir`: Temporary directory with CSV files
- `processed_data_with_features`: Fully processed dataset with all 52 features

## Continuous Integration

GitHub Actions runs tests automatically on every push to `main`:

- **Python versions**: 3.9, 3.10, 3.11
- **Test execution**: pytest with coverage
- **Code quality**: black (formatting), flake8 (linting)
- **Coverage upload**: Codecov integration

See `.github/workflows/ci.yml` for full configuration.

## Adding New Tests

### 1. Create test function
```python
def test_new_feature():
    """Test description"""
    # Arrange
    data = ...

    # Act
    result = function_to_test(data)

    # Assert
    assert result == expected
```

### 2. Use fixtures for data
```python
def test_with_fixture(synthetic_seasons):
    """Use shared fixture data"""
    assert len(synthetic_seasons) > 0
```

### 3. Organize into classes
```python
class TestNewFeature:
    """Group related tests"""

    def test_feature_created(self):
        pass

    def test_feature_valid_range(self):
        pass
```

## Test Best Practices

✓ **Fast**: Use small synthetic datasets
✓ **Isolated**: Each test independent
✓ **Deterministic**: Same input → same output
✓ **Descriptive**: Clear test names and docstrings
✓ **Focused**: One assertion per test (when possible)
✓ **Maintainable**: Use fixtures to avoid duplication

## Coverage Goals

- **Overall**: >80% line coverage
- **Critical modules**:
  - `predictors.py`: >90% (feature engineering)
  - `machine_learning.py`: >70% (model training)

Current coverage: Run `pytest --cov=src` to see report.

## Troubleshooting

### Import errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest
```

### Missing dependencies
```bash
pip install -r requirements-dev.txt
```

### Failing tests after code changes
- Check if feature logic changed
- Update expected values in tests
- Ensure synthetic data still matches assumptions

## Future Test Additions

1. **Integration tests**: Full pipeline end-to-end
2. **Model tests**: Test Ridge, XGBoost predictions
3. **SHAP tests**: Test explainability outputs
4. **Edge cases**: Lockout seasons, COVID season
5. **Performance tests**: Ensure pipeline stays fast
