"""
Base model class for all forecasting models.

Defines the interface that all forecasting models must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
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


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    All forecasting models must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize base forecaster
        
        Args:
            name: Model name
            **kwargs: Additional model-specific parameters
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.scaler = None
        self.params = kwargs
        
        logger.debug(f"Initialized {self.name} model")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseForecaster':
        """
        Train the model on provided data
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted values
        """
        pass
    
    def forecast(self, periods: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Generate future forecast
        
        Args:
            periods: Number of periods to forecast
            exog: Exogenous variables for forecast period
            **kwargs: Additional forecasting parameters
            
        Returns:
            Forecasted values as Series
        """
        raise NotImplementedError(f"{self.name} does not support forecast method")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        self.params.update(params)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if supported by model
        
        Returns:
            Series of feature importances or None
        """
        return None
    
    def save(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseForecaster':
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model instance
        """
        import joblib
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self) -> str:
        """String representation"""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({status})"


class MLForecaster(BaseForecaster):
    """
    Base class for ML-based forecasters (XGBoost, RandomForest, etc.)
    
    Provides common functionality for sklearn-compatible models.
    """
    
    def __init__(self, name: str, model_class, **kwargs):
        """
        Initialize ML forecaster
        
        Args:
            name: Model name
            model_class: Sklearn-compatible model class
            **kwargs: Model parameters
        """
        super().__init__(name, **kwargs)
        self.model = model_class(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'MLForecaster':
        """
        Train the ML model
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional fit parameters
            
        Returns:
            Self
        """
        logger.info(f"Training {self.name} on {len(X)} samples with {len(X.columns)} features")
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        logger.info(f"{self.name} training complete")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        
        return self.model.predict(X, **kwargs)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from tree-based models"""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        
        return None


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    # Create sample data
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randn(100))
    
    # Test ML forecaster
    model = MLForecaster("Random Forest", RandomForestRegressor, n_estimators=10, random_state=42)
    print(f"Created: {model}")
    
    model.fit(X, y)
    print(f"After fit: {model}")
    
    predictions = model.predict(X[:10])
    print(f"\nPredictions shape: {predictions.shape}")
    
    importance = model.get_feature_importance()
    print(f"\nFeature importance:\n{importance}")
