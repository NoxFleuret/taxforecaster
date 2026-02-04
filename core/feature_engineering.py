"""
Feature engineering module.

Handles creation of time-based features, lag features, rolling statistics,
and holiday features for tax forecasting.
"""

import pandas as pd
import numpy as np
import holidays
from typing import Optional
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from logger import get_logger
    from config_loader import get
except ImportError:
    def get_logger(*args):
        import logging
        return logging.getLogger(__name__)
    def get(key, default=None): return default


logger = get_logger(__name__)


class FeatureEngineer:
    """
    Handles all feature engineering operations for tax forecasting models.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer"""
        self.lag_periods = get('models.features.lag_periods', 3)
        self.rolling_window = get('models.features.rolling_window', 3)
        self.include_holidays = get('models.features.holiday_features', True)
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features (month, year, quarter)
        
        Args:
            df: DataFrame with DatetimeIndex or Tanggal column
            
        Returns:
            DataFrame with added time features
        """
        logger.debug("Adding time features")
        
        df = df.copy()
        
        # Get date column
        if hasattr(df.index, 'month'):
            date_idx = df.index
        elif 'Tanggal' in df.columns:
            date_idx = pd.to_datetime(df['Tanggal'])
        else:
            logger.warning("No date column found, skipping time features")
            return df
        
        df['Month'] = date_idx.month
        df['Year'] = date_idx.year
        df['Quarter'] = date_idx.quarter
        df['DayOfYear'] = date_idx.dayofyear
        
        # Cyclical encoding for month (to capture seasonality)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        logger.debug(f"Added time features: Month, Year, Quarter, Month_Sin, Month_Cos")
        return df
    
    def add_lag_features(self, series: pd.Series, lags: Optional[int] = None) -> pd.DataFrame:
        """
        Add lag features for a time series
        
        Args:
            series: Time series data
            lags: Number of lag periods (default from config)
            
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = self.lag_periods
        
        logger.debug(f"Adding {lags} lag features")
        
        df = pd.DataFrame(series)
        df.columns = ['y']
        
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str = 'y',
                            window: Optional[int] = None) -> pd.DataFrame:
        """
        Add rolling statistics features
        
        Args:
            df: DataFrame containing target column
            target_col: Name of target column
            window: Rolling window size (default from config)
            
        Returns:
            DataFrame with rolling features
        """
        if window is None:
            window = self.rolling_window
        
        logger.debug(f"Adding rolling features with window={window}")
        
        df = df.copy()
        
        # Rolling mean
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        
        # Rolling std
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        
        # Rolling min/max
        df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Indonesian holiday features (Lebaran, Natal)
        
        Args:
            df: DataFrame with DatetimeIndex or Tanggal column
            
        Returns:
            DataFrame with holiday features
        """
        if not self.include_holidays:
            logger.debug("Holiday features disabled in config")
            return df
        
        logger.debug("Adding Indonesian holiday features")
        
        df = df.copy()
        
        try:
            # Get date information
            if hasattr(df.index, 'year'):
                years = df.index.year.unique()
                date_col = df.index
            elif 'Tanggal' in df.columns:
                years = pd.to_datetime(df['Tanggal']).dt.year.unique()
                date_col = pd.to_datetime(df['Tanggal'])
            else:
                logger.warning("No date information found for holiday features")
                return df
            
            # Get Indonesian holidays
            try:
                id_holidays = holidays.Indonesia(years=years)
            except:
                id_holidays = {}
            
            # Extract specific holiday dates
            lebaran_dates = []
            natal_dates = []
            
            for date, name in id_holidays.items():
                if "Idul Fitri" in name:
                    lebaran_dates.append(date)
                elif "Hari Raya Natal" in name or "Christmas" in name:
                    natal_dates.append(date)
            
            # Initialize columns
            df['is_lebaran'] = 0
            df['is_natal'] = 0
            
            # Mark holiday months
            for idx in range(len(df)):
                current_date = date_col.iloc[idx] if hasattr(date_col, 'iloc') else date_col[idx]
                
                for d in lebaran_dates:
                    if d.year == current_date.year and d.month == current_date.month:
                        df.at[df.index[idx], 'is_lebaran'] = 1
                
                for d in natal_dates:
                    if d.year == current_date.year and d.month == current_date.month:
                        df.at[df.index[idx], 'is_natal'] = 1
            
            logger.debug(f"Added holiday features: {df['is_lebaran'].sum()} Lebaran, "
                        f"{df['is_natal'].sum()} Natal periods")
            
        except Exception as e:
            logger.warning(f"Holiday feature creation failed: {e}")
            df['is_lebaran'] = 0
            df['is_natal'] = 0
        
        return df
    
    def create_ml_features(self, series: pd.Series, include_exog: bool = False) -> pd.DataFrame:
        """
        Create complete feature set for ML models
        
        Args:
            series: Time series with DatetimeIndex
            include_exog: Whether to expect exogenous variables
            
        Returns:
            DataFrame with all features
        """
        logger.info(f"Creating ML features for {len(series)} data points")
        
        # Start with lag features
        df = self.add_lag_features(series)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Add rolling features
        df = self.add_rolling_features(df, target_col='y')
        
        # Remove rows with NaN (from lagging and rolling)
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows due to NaN from feature engineering")
        
        logger.info(f"Created {len(df.columns)} features from {len(series)} samples")
        
        return df
    
    def get_feature_names(self, include_exog: bool = False) -> list:
        """
        Get list of feature names that will be created
        
        Args:
            include_exog: Whether exogenous features are included
            
        Returns:
            List of feature column names
        """
        features = ['y']  # Target
        
        # Lag features
        for i in range(1, self.lag_periods + 1):
            features.append(f'lag_{i}')
        
        # Rolling features
        features.extend([
            f'rolling_mean_{self.rolling_window}',
            f'rolling_std_{self.rolling_window}',
            f'rolling_min_{self.rolling_window}',
            f'rolling_max_{self.rolling_window}'
        ])
        
        # Time features
        features.extend(['Month', 'Year', 'Quarter', 'DayOfYear', 'Month_Sin', 'Month_Cos'])
        
        # Holiday features
        if self.include_holidays:
            features.extend(['is_lebaran', 'is_natal'])
        
        return features


# Example usage
if __name__ == "__main__":
    # Test feature engineer
    engineer = FeatureEngineer()
    
    # Create sample time series
    dates = pd.date_range('2020-01-01', periods=50, freq='M')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates, name='revenue')
    
    print("Original series:")
    print(series.head())
    
    # Create features
    features_df = engineer.create_ml_features(series)
    
    print(f"\n✅ Created features: {features_df.shape}")
    print(f"📊 Columns: {list(features_df.columns)}")
    print("\nSample features:")
    print(features_df.head())
