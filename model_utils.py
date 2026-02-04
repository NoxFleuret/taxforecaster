import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def calculate_shap_values(model, X_train, X_test):
    """
    Calculates SHAP values for a given model.
    Returns the shap_values object and the explainer.
    Handles Tree-based and Linear models appropriately.
    """
    try:
        # Determine model type
        model_type = type(model).__name__
        
        # Tree-based
        if model_type in ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor']:
            explainer = shap.TreeExplainer(model)
            # Check for modern SHAP which returns Explanation object
            shap_obj = explainer(X_test)
            if hasattr(shap_obj, 'values'):
                shap_values = shap_obj.values
            else:
                shap_values = shap_obj
            
            # For random forest, it might return list (multi-output), take first
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        
        # Kernel-based (fallback for blackboxes like SVR, KNN, MLP)
        # Using a sample of X_train as background to speed up
        elif model_type in ['SVR', 'KNeighborsRegressor', 'MLPRegressor']:
            background = shap.sample(X_train, 50) # Use 50 samples as background
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test)
            
        # Linear
        else:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            
        return shap_values, explainer
    except Exception as e:
        print(f"SHAP Error: {e}")
        return None, None

def detect_drift(X_train, X_inference, threshold=0.05):
    """
    Detects data drift between training data and inference data using 
    Kolmogorov-Smirnov (KS) Test.
    
    Returns:
        drift_report (dict): {feature: {'drift': bool, 'p_value': float}}
        has_drift (bool): True if ANY feature has drift.
    """
    drift_report = {}
    has_drift = False
    
    # Align columns
    common_cols = [c for c in X_train.columns if c in X_inference.columns]
    
    for col in common_cols:
        # KS Test
        stat, p_value = ks_2samp(X_train[col], X_inference[col])
        
        is_drift = p_value < threshold
        if is_drift: has_drift = True
        
        drift_report[col] = {
            'drift': is_drift,
            'p_value': p_value,
            'stat': stat
        }
        
    return drift_report, has_drift

def calculate_confidence_intervals(predictions_list, alpha=0.95):
    """
    Calculates confidence intervals from a list of prediction arrays (Ensemble members).
    
    Args:
        predictions_list (list of arrays): e.g., [pred_model_A, pred_model_B, pred_model_C]
        alpha (float): Confidence level (default 0.95)
        
    Returns:
        lower (array), upper (array), mean (array)
    """
    if not predictions_list:
        return None, None, None
        
    # Stack predictions: shape (n_models, n_periods)
    stacked_preds = np.vstack(predictions_list)
    
    # Mean
    mean_pred = np.mean(stacked_preds, axis=0)
    
    # Standard Error / Variance based
    # Simple approach: percentile
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    lower = np.percentile(stacked_preds, lower_p, axis=0)
    upper = np.percentile(stacked_preds, upper_p, axis=0)
    
    return lower, upper, mean_pred
