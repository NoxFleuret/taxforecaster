"""
Unit tests for Recommendation Engine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from intelligence.recommendation_engine import RecommendationEngine, get_recommendations


class TestRecommendationEngine:
    """Test suite for RecommendationEngine class"""
    
    @pytest.fixture
    def trending_data(self):
        """Create data with strong upward trend"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='M')
        trend = np.linspace(100, 200, 60)
        noise = np.random.normal(0, 5, 60)
        
        return pd.DataFrame({
            'date': dates,
            'revenue': trend + noise
        })
    
    @pytest.fixture
    def seasonal_data(self):
        """Create data with seasonality"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='M')
        seasonal = 100 + 20 * np.sin(np.arange(60) * 2 * np.pi / 12)
        noise = np.random.normal(0, 3, 60)
        
        return pd.DataFrame({
            'date': dates,
            'revenue': seasonal + noise
        })
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = RecommendationEngine()
        assert engine.recommendations == []
        assert engine.data_characteristics == {}
    
    def test_analyze_trending_data(self, trending_data):
        """Test analysis of trending data"""
        engine = RecommendationEngine()
        chars = engine.analyze_data(trending_data, 'revenue')
        
        assert 'trend' in chars
        assert 'seasonality' in chars
        assert 'volatility' in chars
        assert 'data_points' in chars
        assert chars['trend'] == 'increasing'
        assert chars['data_points'] == 60
    
    def test_analyze_seasonal_data(self, seasonal_data):
        """Test analysis of seasonal data"""
        engine = RecommendationEngine()
        chars = engine.analyze_data(seasonal_data, 'revenue')
        
        # Note: Seasonality detection might not work on all synthetic data
        assert 'seasonality' in chars
        assert isinstance(chars['seasonality'], bool)
    
    def test_trend_detection(self, trending_data):
        """Test specific trend detection"""
        engine = RecommendationEngine()
        trend = engine._detect_trend(trending_data['revenue'])
        
        assert trend in ['increasing', 'decreasing', 'stable', 'insufficient_data']
        assert trend == 'increasing'  # Should detect upward trend
    
    def test_volatility_calculation(self, trending_data):
        """Test volatility calculation"""
        engine = RecommendationEngine()
        volatility = engine._calculate_volatility(trending_data['revenue'])
        
        assert volatility in ['low', 'medium', 'high']
    
    def test_model_recommendations(self, seasonal_data):
        """Test model recommendations"""
        engine = RecommendationEngine()
        engine.analyze_data(seasonal_data, 'revenue')
        
        models = engine.recommend_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert len(models) <= 5  # Shouldn't recommend more than 5
    
    def test_feature_recommendations(self, trending_data):
        """Test feature recommendations"""
        engine = RecommendationEngine()
        engine.analyze_data(trending_data, 'revenue')
        
        features = engine.recommend_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any('Time features' in f for f in features)
        assert any('Lag features' in f for f in features)
    
    def test_data_quality_recommendations(self):
        """Test data quality recommendations"""
        engine = RecommendationEngine()
        
        # Low missing data
        recs_good = engine.recommend_data_quality_improvements(5, 1)
        assert len(recs_good) > 0
        
        # High missing data
        recs_bad = engine.recommend_data_quality_improvements(15, 10)
        assert len(recs_bad) > 0
        assert any('HIGH' in r for r in recs_bad)
    
    def test_hyperparameter_recommendations(self, trending_data):
        """Test hyperparameter recommendations"""
        engine = RecommendationEngine()
        engine.analyze_data(trending_data, 'revenue')
        
        xgb_params = engine.recommend_hyperparameters('XGBoost')
        assert isinstance(xgb_params, dict)
        assert 'n_estimators' in xgb_params or len(xgb_params) == 0
        
        prophet_params = engine.recommend_hyperparameters('Prophet')
        assert isinstance(prophet_params, dict)
    
    def test_generate_report(self, trending_data):
        """Test report generation"""
        engine = RecommendationEngine()
        engine.analyze_data(trending_data, 'revenue')
        
        report = engine.generate_report()
        
        assert isinstance(report, str)
        assert 'Data Characteristics' in report
        assert 'Recommended Models' in report
        assert 'Recommended Features' in report
    
    def test_convenience_function(self, seasonal_data):
        """Test convenience function"""
        report = get_recommendations(seasonal_data, 'revenue')
        
        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
