"""
Intelligent Recommendation Engine for Tax Forecasting.

Analyzes data characteristics and provides actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from logger import get_logger
except ImportError:
    def get_logger(*args):
        import logging
        return logging.getLogger(__name__)


logger = get_logger(__name__)


class RecommendationEngine:
    """
    Generates intelligent recommendations for forecasting improvements.
    
    Analyzes:
    - Data characteristics (trend, seasonality, volatility)
    - Model suitability
    - Feature engineering opportunities
    - Data quality issues
    """
    
    def __init__(self):
        self.recommendations = []
        self.data_characteristics = {}
        
    def analyze_data(self, data: pd.DataFrame, value_col: str) -> Dict:
        """
        Analyze data characteristics.
        
        Args:
            data: Time series data
            value_col: Column name for values
            
        Returns:
            Dictionary of data characteristics
        """
        characteristics = {}
        
        # Basic stats
        characteristics['mean'] = data[value_col].mean()
        characteristics['std'] = data[value_col].std()
        characteristics['cv'] = characteristics['std'] / characteristics['mean'] if characteristics['mean'] != 0 else 0
        
        # Trend detection
        characteristics['trend'] = self._detect_trend(data[value_col])
        
        # Seasonality detection
        characteristics['seasonality'] = self._detect_seasonality(data[value_col])
        
        # Volatility
        characteristics['volatility'] = self._calculate_volatility(data[value_col])
        
        # Missing data
        characteristics['missing_pct'] = data[value_col].isna().sum() / len(data) * 100
        
        # Data sufficiency
        characteristics['data_points'] = len(data)
        characteristics['sufficient_data'] = len(data) >= 24
        
        self.data_characteristics = characteristics
        logger.info(f"Data analysis complete: {characteristics}")
        
        return characteristics
    
    def _detect_trend(self, series: pd.Series) -> str:
        """Detect trend direction"""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear regression slope
        x = np.arange(len(series))
        y = series.fillna(method='ffill').fillna(method='bfill').values
        
        slope = np.polyfit(x, y, 1)[0]
        slope_pct = slope / series.mean() * 100 if series.mean() != 0 else 0
        
        if abs(slope_pct) < 1:
            return 'stable'
        elif slope_pct > 1:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Detect if data has seasonal patterns"""
        if len(series) < 24:
            return False
        
        # Simple autocorrelation check at 12-month lag
        try:
            autocorr_12 = series.autocorr(lag=12)
            return autocorr_12 > 0.5 if not np.isnan(autocorr_12) else False
        except:
            return False
    
    def _calculate_volatility(self, series: pd.Series) -> str:
        """Calculate volatility level"""
        cv = series.std() / series.mean() if series.mean() != 0 else 0
        
        if cv < 0.1:
            return 'low'
        elif cv < 0.3:
            return 'medium'
        else:
            return 'high'
    
    def recommend_models(self) -> List[str]:
        """
        Recommend suitable models based on data characteristics.
        
        Returns:
            List of recommended model names
        """
        if not self.data_characteristics:
            return ['XGBoost', 'Prophet', 'ARIMAX']
        
        recommendations = []
        chars = self.data_characteristics
        
        # Prophet for seasonal data
        if chars.get('seasonality', False):
            recommendations.append('Prophet')
            recommendations.append('SARIMAX')
        
        # Tree models for complex patterns
        if chars.get('volatility') in ['medium', 'high']:
            recommendations.append('XGBoost')
            recommendations.append('LightGBM')
            recommendations.append('Random Forest')
        
        # Linear models for stable trends
        if chars.get('trend') == 'stable' and chars.get('volatility') == 'low':
            recommendations.append('Linear Regression')
            recommendations.append('Ridge')
        
        # LSTM for long sequences
        if chars.get('data_points', 0) > 60:
            recommendations.append('LSTM')
        
        # Ensemble for general robustness
        recommendations.append('Ensemble (Top 3)')
        
        return list(set(recommendations))[:5]  # Return top 5 unique
    
    def recommend_features(self) -> List[str]:
        """
        Recommend feature engineering strategies.
        
        Returns:
            List of feature recommendations
        """
        recommendations = []
        chars = self.data_characteristics
        
        # Always recommend basics
        recommendations.append("✅ Time features (month, quarter, year)")
        recommendations.append("✅ Lag features (1, 3, 6, 12 months)")
        
        # Seasonality
        if chars.get('seasonality', False):
            recommendations.append("📊 Seasonal decomposition features")
            recommendations.append("🔄 Cyclical encoding (sin/cos transforms)")
        
        # Trend
        if chars.get('trend') in ['increasing', 'decreasing']:
            recommendations.append("📈 Rolling statistics (mean, std)")
            recommendations.append("📉 Trend indicators")
        
        # Volatility
        if chars.get('volatility') == 'high':
            recommendations.append("⚡ Volatility measures (rolling std)")
            recommendations.append("🎯 Outlier indicators")
        
        # Macro features
        recommendations.append("🌍 Macro economic indicators (inflation, GDP)")
        recommendations.append("📅 Holiday indicators")
        
        return recommendations
    
    def recommend_data_quality_improvements(
        self, 
        missing_pct: float,
        outlier_count: int
    ) -> List[str]:
        """
        Recommend data quality improvements.
        
        Args:
            missing_pct: Percentage of missing data
            outlier_count: Number of detected outliers
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Missing data
        if missing_pct > 10:
            recommendations.append("🚨 HIGH: Significant missing data detected")
            recommendations.append("💡 Consider: Interpolation or forward-fill")
            recommendations.append("💡 Alternative: Remove incomplete periods")
        elif missing_pct > 0:
            recommendations.append("⚠️ MEDIUM: Some missing data points")
            recommendations.append("💡 Use interpolation for small gaps")
        
        # Outliers
        if outlier_count > 5:
            recommendations.append("🚨 HIGH: Multiple outliers detected")
            recommendations.append("💡 Investigate: Data entry errors or real events")
            recommendations.append("💡 Consider: Robust models (Huber, Quantile Regression)")
        elif outlier_count > 0:
            recommendations.append("⚠️ MEDIUM: Few outliers detected")
            recommendations.append("💡 Review flagged points manually")
        
        # General
        if not recommendations:
            recommendations.append("✅ Data quality looks good!")
        
        return recommendations
    
    def recommend_hyperparameters(self, model_name: str) -> Dict:
        """
        Recommend hyperparameter ranges for models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of recommended hyperparameters
        """
        chars = self.data_characteristics
        
        recommendations = {}
        
        if model_name == 'XGBoost':
            recommendations = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == 'Prophet':
            seasonality_mode = 'multiplicative' if chars.get('volatility') == 'high' else 'additive'
            recommendations = {
                'seasonality_mode': [seasonality_mode],
                'changepoint_prior_scale': [0.05, 0.1, 0.5],
                'seasonality_prior_scale': [1.0, 5.0, 10.0]
            }
        elif model_name == 'Random Forest':
            recommendations = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        
        return recommendations
    
    def generate_report(self) -> str:
        """
        Generate comprehensive recommendation report.
        
        Returns:
            Formatted report string
        """
        if not self.data_characteristics:
            return "❌ No data analyzed yet. Run analyze_data() first."
        
        chars = self.data_characteristics
        
        report = f"""
# 🎯 Forecasting Recommendations Report

## 📊 Data Characteristics

- **Data Points**: {chars['data_points']} months
- **Trend**: {chars['trend'].title()}
- **Seasonality**: {'✅ Detected' if chars['seasonality'] else '❌ Not detected'}
- **Volatility**: {chars['volatility'].title()}
- **Missing Data**: {chars['missing_pct']:.1f}%
- **Data Sufficiency**: {'✅ Sufficient' if chars['sufficient_data'] else '⚠️ Limited'}

## 🤖 Recommended Models

"""
        for model in self.recommend_models():
            report += f"- {model}\n"
        
        report += "\n## 🔧 Recommended Features\n\n"
        for feature in self.recommend_features():
            report += f"- {feature}\n"
        
        report += f"""
## 📈 Strategy Recommendations

**Best Approach**: 
- Use ensemble of top 3-5 models
- Include seasonal and lag features
- Cross-validate with time series split
- Monitor forecast accuracy monthly

**Training Tips**:
- Reserve last 20% for testing
- Use {'walk-forward validation' if chars['data_points'] > 60 else 'holdout validation'}
- Tune hyperparameters with {'Optuna' if chars['data_points'] > 40 else 'Grid Search'}
"""
        
        return report


# Convenience functions
def get_recommendations(data: pd.DataFrame, value_col: str) -> str:
    """Get recommendations report (convenience function)"""
    engine = RecommendationEngine()
    engine.analyze_data(data, value_col)
    return engine.generate_report()


if __name__ == "__main__":
    # Test recommendation engine
    print("Testing Recommendation Engine...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    trend = np.linspace(100, 150, 60)
    seasonal = 10 * np.sin(np.arange(60) * 2 * np.pi / 12)
    noise = np.random.normal(0, 5, 60)
    values = trend + seasonal + noise
    
    test_data = pd.DataFrame({
        'date': dates,
        'revenue': values
    })
    
    engine = RecommendationEngine()
    characteristics = engine.analyze_data(test_data, 'revenue')
    
    print(f"\n📊 Data Characteristics:\n{characteristics}")
    print(f"\n🤖 Recommended Models:\n{engine.recommend_models()}")
    print(f"\n📋 Full Report:\n{engine.generate_report()}")
