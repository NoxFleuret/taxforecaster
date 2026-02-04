"""
Unit tests for FeatureEngineer class.

Run with: pytest tests/test_feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_series(self):
        """Create a sample time series"""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        values = np.random.randn(50).cumsum() + 100
        return pd.Series(values, index=dates, name='revenue')
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with date column"""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        df = pd.DataFrame({
            'Tanggal': dates,
            'value': np.random.randn(50).cumsum() + 100
        })
        return df
    
    def test_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        
        assert engineer.lag_periods == 3
        assert engineer.rolling_window == 3
        assert engineer.include_holidays == True
    
    def test_add_lag_features(self, sample_series):
        """Test lag feature creation"""
        engineer = FeatureEngineer()
        df = engineer.add_lag_features(sample_series, lags=3)
        
        assert 'y' in df.columns
        assert 'lag_1' in df.columns
        assert 'lag_2' in df.columns
        assert 'lag_3' in df.columns
        assert len(df.columns) == 4
    
    def test_add_time_features(self, sample_df):
        """Test time feature creation"""
        engineer = FeatureEngineer()
        df = engineer.add_time_features(sample_df)
        
        assert 'Month' in df.columns
        assert 'Year' in df.columns
        assert 'Quarter' in df.columns
        assert 'Month_Sin' in df.columns
        assert 'Month_Cos' in df.columns
    
    def test_add_rolling_features(self, sample_series):
        """Test rolling statistics features"""
        engineer = FeatureEngineer()
        df = pd.DataFrame({'y': sample_series})
        df = engineer.add_rolling_features(df, target_col='y', window=3)
        
        assert 'rolling_mean_3' in df.columns
        assert 'rolling_std_3' in df.columns
        assert 'rolling_min_3' in df.columns
        assert 'rolling_max_3' in df.columns
    
    def test_add_holiday_features(self, sample_df):
        """Test holiday feature creation"""
        engineer = FeatureEngineer()
        df = engineer.add_holiday_features(sample_df)
        
        assert 'is_lebaran' in df.columns
        assert 'is_natal' in df.columns
        assert df['is_lebaran'].dtype == int
        assert df['is_natal'].dtype == int
    
    def test_create_ml_features(self, sample_series):
        """Test complete ML feature creation"""
        engineer = FeatureEngineer()
        features_df = engineer.create_ml_features(sample_series)
        
        # Check that features were created
        assert len(features_df) > 0
        assert 'y' in features_df.columns
        assert 'lag_1' in features_df.columns
        assert 'Month' in features_df.columns
        
        # Check that NaN rows were dropped
        assert not features_df.isnull().any().any()
    
    def test_get_feature_names(self):
        """Test feature name retrieval"""
        engineer = FeatureEngineer()
        feature_names = engineer.get_feature_names()
        
        assert 'y' in feature_names
        assert 'lag_1' in feature_names
        assert 'Month' in feature_names
        assert 'is_lebaran' in feature_names


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
