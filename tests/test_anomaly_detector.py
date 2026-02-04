"""
Unit tests for Anomaly Detector.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from intelligence.anomaly_detector import AnomalyDetector, detect_anomalies


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known anomalies"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='M')
        values = np.random.normal(100, 10, 100)
        
        # Insert anomalies
        values[10] = 200  # High outlier
        values[50] = 20   # Low outlier
        
        return pd.DataFrame({
            'date': dates,
            'revenue': values
        })
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = AnomalyDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert detector.anomalies == {}
    
    def test_zscore_detection(self, sample_data):
        """Test Z-score anomaly detection"""
        detector = AnomalyDetector()
        anomalies = detector.detect_zscore(sample_data['revenue'], threshold=3.0)
        
        assert isinstance(anomalies, pd.Series)
        assert len(anomalies) == len(sample_data)
        assert anomalies.sum() >= 2  # Should detect at least our 2 injected anomalies
    
    def test_iqr_detection(self, sample_data):
        """Test IQR anomaly detection"""
        detector = AnomalyDetector()
        anomalies = detector.detect_iqr(sample_data['revenue'])
        
        assert isinstance(anomalies, pd.Series)
        assert len(anomalies) == len(sample_data)
        assert anomalies.sum() >= 2
    
    def test_isolation_forest(self, sample_data):
        """Test Isolation Forest detection"""
        detector = AnomalyDetector()
        
        # Add some features
        sample_data['month'] = sample_data['date'].dt.month
        sample_data['value_lag1'] = sample_data['revenue'].shift(1).fillna(100)
        
        anomalies = detector.detect_isolation_forest(
            sample_data,
            features=['revenue', 'month', 'value_lag1']
        )
        
        assert isinstance(anomalies, pd.Series)
        assert len(anomalies) == len(sample_data)
    
    def test_detect_all_ensemble(self, sample_data):
        """Test ensemble detection method"""
        detector = AnomalyDetector()
        results = detector.detect_all(sample_data, 'revenue', method='ensemble')
        
        assert 'zscore' in results
        assert 'iqr' in results
        assert 'ensemble' in results
        assert all(isinstance(v, pd.Series) for v in results.values())
    
    def test_anomaly_summary(self, sample_data):
        """Test anomaly summary generation"""
        detector = AnomalyDetector()
        detector.detect_all(sample_data, 'revenue', method='ensemble')
        
        summary = detector.get_anomaly_summary(sample_data)
        
        assert isinstance(summary, pd.DataFrame)
        if len(summary) > 0:
            assert 'anomaly_score' in summary.columns
            assert 'severity' in summary.columns
    
    def test_explain_anomaly(self, sample_data):
        """Test anomaly explanation generation"""
        detector = AnomalyDetector()
        explanation = detector.explain_anomaly(sample_data, 10, 'revenue')
        
        assert isinstance(explanation, str)
        assert 'Anomaly Detected' in explanation
        assert 'Value' in explanation
    
    def test_recommendations(self):
        """Test recommendation generation"""
        detector = AnomalyDetector()
        
        recs_none = detector.get_recommendations(0)
        assert len(recs_none) > 0
        assert any('No significant anomalies' in r for r in recs_none)
        
        recs_many = detector.get_recommendations(10)
        assert len(recs_many) > 0
        assert any('Multiple anomalies' in r for r in recs_many)
    
    def test_convenience_function(self, sample_data):
        """Test convenience function"""
        results = detect_anomalies(sample_data, 'revenue')
        
        assert isinstance(results, dict)
        assert 'ensemble' in results


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
