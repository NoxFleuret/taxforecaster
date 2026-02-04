"""
Anomaly Detection System for Tax Revenue Data.

Detects unusual patterns using statistical and ML methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from datetime import datetime
import sys
import os

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


class AnomalyDetector:
    """
    Detects anomalies in time series data using multiple methods.
    
    Methods:
    - Z-Score: Statistical outlier detection
    - IQR: Interquartile range method
    - Isolation Forest: ML-based anomaly detection
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
        """
        self.contamination = contamination
        self.isolation_forest = None
        self.anomalies = {}
        
    def detect_zscore(
        self, 
        data: pd.Series, 
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect anomalies using Z-score method.
        
        Args:
            data: Time series data
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Boolean series indicating anomalies
        """
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return pd.Series([False] * len(data), index=data.index)
        
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > threshold
        
        logger.info(f"Z-Score detection found {anomalies.sum()} anomalies")
        return anomalies
    
    def detect_iqr(self, data: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Detect anomalies using IQR method.
        
        Args:
            data: Time series data
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Boolean series indicating anomalies
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        logger.info(f"IQR detection found {anomalies.sum()} anomalies")
        return anomalies
    
    def detect_isolation_forest(
        self, 
        data: pd.DataFrame,
        features: List[str]
    ) -> pd.Series:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: DataFrame with features
            features: List of feature column names
            
        Returns:
            Boolean series indicating anomalies
        """
        X = data[features].fillna(0)
        
        if len(X) < 10:
            logger.warning("Insufficient data for Isolation Forest")
            return pd.Series([False] * len(data), index=data.index)
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        
        predictions = self.isolation_forest.fit_predict(X)
        anomalies = pd.Series(predictions == -1, index=data.index)
        
        logger.info(f"Isolation Forest found {anomalies.sum()} anomalies")
        return anomalies
    
    def detect_all(
        self, 
        data: pd.DataFrame,
        value_col: str,
        method: str = 'ensemble'
    ) -> Dict[str, pd.Series]:
        """
        Run all detection methods and return results.
        
        Args:
            data: DataFrame with time series data
            value_col: Column name for values
            method: 'ensemble', 'zscore', 'iqr', or 'isolation_forest'
            
        Returns:
            Dictionary of anomaly boolean series
        """
        results = {}
        
        if method in ['zscore', 'ensemble']:
            results['zscore'] = self.detect_zscore(data[value_col])
        
        if method in ['iqr', 'ensemble']:
            results['iqr'] = self.detect_iqr(data[value_col])
        
        if method in ['isolation_forest', 'ensemble']:
            if len(data.columns) > 1:
                feature_cols = [col for col in data.columns if col != value_col]
                results['isolation_forest'] = self.detect_isolation_forest(
                    data, feature_cols[:5]  # Limit to 5 features
                )
        
        # Ensemble: flag as anomaly if detected by 2+ methods
        if method == 'ensemble' and len(results) > 1:
            results['ensemble'] = sum(results.values()) >= 2
        
        self.anomalies = results
        return results
    
    def get_anomaly_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of detected anomalies.
        
        Args:
            data: Original data with index
            
        Returns:
            DataFrame with anomaly details
        """
        if not self.anomalies:
            return pd.DataFrame()
        
        # Use ensemble if available, otherwise use first method
        anomaly_mask = self.anomalies.get('ensemble', list(self.anomalies.values())[0])
        
        anomaly_data = data[anomaly_mask].copy()
        anomaly_data['anomaly_score'] = self._calculate_anomaly_scores(data, anomaly_mask)
        anomaly_data['severity'] = anomaly_data['anomaly_score'].apply(
            lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.5 else 'Low'
        )
        
        return anomaly_data
    
    def _calculate_anomaly_scores(
        self, 
        data: pd.DataFrame, 
        anomaly_mask: pd.Series
    ) -> pd.Series:
        """Calculate anomaly severity scores (0-1)"""
        scores = pd.Series(0.0, index=data.index)
        
        # Count how many methods flagged each point
        method_count = sum(
            method_mask.astype(int) 
            for method_mask in self.anomalies.values()
        )
        
        # Normalize to 0-1
        max_methods = len(self.anomalies)
        scores = method_count / max_methods if max_methods > 0 else 0
        
        return scores
    
    def explain_anomaly(
        self, 
        data: pd.DataFrame,
        anomaly_index: int,
        value_col: str
    ) -> str:
        """
        Generate human-readable explanation for an anomaly.
        
        Args:
            data: Original data
            anomaly_index: Index of anomaly point
            value_col: Column name for values
            
        Returns:
            Explanation string
        """
        if anomaly_index not in data.index:
            return "Invalid anomaly index"
        
        value = data.loc[anomaly_index, value_col]
        mean = data[value_col].mean()
        std = data[value_col].std()
        
        deviation = abs(value - mean) / std if std > 0 else 0
        percent_deviation = abs((value - mean) / mean * 100) if mean != 0 else 0
        
        direction = "above" if value > mean else "below"
        
        explanation = f"""
**Anomaly Detected**: {deviation:.2f} standard deviations {direction} average

- **Value**: {value:,.2f}
- **Average**: {mean:,.2f}
- **Deviation**: {percent_deviation:.1f}%
- **Severity**: {'High' if deviation > 3 else 'Medium' if deviation > 2 else 'Low'}

**Possible Causes**:
- Data entry error
- Special economic event
- Seasonal anomaly
- Tax policy change
"""
        
        return explanation
    
    def get_recommendations(self, anomaly_count: int) -> List[str]:
        """
        Get recommendations based on anomaly detection results.
        
        Args:
            anomaly_count: Number of anomalies detected
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if anomaly_count == 0:
            recommendations.append("✅ No significant anomalies detected. Data quality looks good!")
        elif anomaly_count <= 3:
            recommendations.append("⚠️ Few anomalies detected. Review flagged data points manually.")
            recommendations.append("💡 Consider investigating economic events during anomaly periods.")
        else:
            recommendations.append("🚨 Multiple anomalies detected. Data quality issues possible.")
            recommendations.append("💡 Recommended actions:")
            recommendations.append("   - Verify data source accuracy")
            recommendations.append("   - Check for data entry errors")
            recommendations.append("   - Consider outlier removal or robust models")
            recommendations.append("   - Investigate external factors (policy changes, crises)")
        
        return recommendations


# Convenience functions
def detect_anomalies(
    data: pd.DataFrame,
    value_col: str,
    method: str = 'ensemble'
) -> Dict[str, pd.Series]:
    """Quick anomaly detection (convenience function)"""
    detector = AnomalyDetector()
    return detector.detect_all(data, value_col, method)


if __name__ == "__main__":
    # Test anomaly detector
    print("Testing Anomaly Detector...")
    
    # Create sample data with anomalies
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    values = np.random.normal(100, 10, 100)
    
    # Insert anomalies
    values[10] = 200  # High outlier
    values[50] = 20   # Low outlier
    
    test_data = pd.DataFrame({
        'date': dates,
        'revenue': values
    })
    
    detector = AnomalyDetector()
    results = detector.detect_all(test_data, 'revenue', method='ensemble')
    
    print(f"\n✅ Anomalies detected: {results['ensemble'].sum()}")
    
    summary = detector.get_anomaly_summary(test_data)
    print(f"\n📊 Anomaly Summary:\n{summary[['revenue', 'severity']]}")
