"""
Centralized Error Handling for TaxForecaster Application

This module provides custom exception classes, user-friendly error messages,
and recovery suggestions for common issues.
"""

import streamlit as st
import traceback
import logging
from typing import Optional, Dict, Any
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taxforecaster_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TaxForecaster')


# ============================================================================
# Custom Exception Classes
# ============================================================================

class TaxForecasterError(Exception):
    """Base exception for all TaxForecaster errors"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()


class DataError(TaxForecasterError):
    """Raised when there are issues with data loading or validation"""
    pass


class ModelError(TaxForecasterError):
    """Raised when there are issues with model training or prediction"""
    pass


class ValidationError(TaxForecasterError):
    """Raised when data validation fails"""
    pass


class ConfigurationError(TaxForecasterError):
    """Raised when configuration is invalid or missing"""
    pass


class APIError(TaxForecasterError):
    """Raised when external API calls fail"""
    pass


# ============================================================================
# User-Friendly Error Messages
# ============================================================================

ERROR_MESSAGES = {
    # Data Errors
    'file_not_found': {
        'title': '📂 File Not Found',
        'message': 'The data file could not be located.',
        'suggestions': [
            'Verify the file path is correct',
            'Ensure the file exists in the expected directory',
            'Upload a new data file using the sidebar'
        ]
    },
    'invalid_csv_format': {
        'title': '⚠️ Invalid CSV Format',
        'message': 'The CSV file format is not recognized.',
        'suggestions': [
            'Check that the file has required columns: Tanggal, Jenis Pajak, Nominal (Milyar)',
            'Ensure dates are in DD/MM/YYYY format',
            'Verify the file is a valid CSV with comma separators'
        ]
    },
    'missing_columns': {
        'title': '🔍 Missing Required Columns',
        'message': 'The dataset is missing one or more required columns.',
        'suggestions': [
            'Required columns: Tanggal, Jenis Pajak, Nominal (Milyar)',
            'Check for typos in column names',
            'Ensure column headers are in the first row'
        ]
    },
    'insufficient_data': {
        'title': '📉 Insufficient Data',
        'message': 'Not enough historical data for reliable forecasting.',
        'suggestions': [
            'Minimum 24 months of data recommended',
            'Ensure data is available for all tax types',
            'Consider using a different time range'
        ]
    },
    
    # Model Errors
    'model_not_trained': {
        'title': '🤖 Model Not Trained',
        'message': 'No trained model is available for predictions.',
        'suggestions': [
            'Navigate to Dashboard and click "Train Models"',
            'Ensure data is loaded before training',
            'Check that at least one model trained successfully'
        ]
    },
    'training_failed': {
        'title': '❌ Training Failed',
        'message': 'Model training encountered an error.',
        'suggestions': [
            'Check data quality and remove outliers',
            'Reduce the number of trials if training times out',
            'Ensure you have sufficient system memory'
        ]
    },
    'prediction_failed': {
        'title': '⚠️ Prediction Failed',
        'message': 'Unable to generate forecast predictions.',
        'suggestions': [
            'Verify the model is properly trained',
            'Check that macro data is available for future periods',
            'Try reducing the forecast horizon'
        ]
    },
    
    # API Errors
    'api_timeout': {
        'title': '⏱️ API Timeout',
        'message': 'External data source is not responding.',
        'suggestions': [
            'Check your internet connection',
            'Try again in a few moments',
            'Use cached data if available'
        ]
    },
    'api_rate_limit': {
        'title': '🚦 Rate Limit Exceeded',
        'message': 'Too many requests to external API.',
        'suggestions': [
            'Wait a few minutes before retrying',
            'Use manual data upload instead',
            'Contact administrator to increase rate limits'
        ]
    },
    
    # Validation Errors
    'invalid_date_range': {
        'title': '📅 Invalid Date Range',
        'message': 'The selected date range is invalid.',
        'suggestions': [
            'Ensure end date is after start date',
            'Check that dates are within available data range',
            'Use YYYY-MM-DD format for date inputs'
        ]
    },
    'data_quality_issues': {
        'title': '⚠️ Data Quality Issues Detected',
        'message': 'The data contains anomalies or quality issues.',
        'suggestions': [
            'Review the Data Quality page',
            'Fix outliers using the preprocessing tool',
            'Check for missing values and duplicates'
        ]
    }
}


# ============================================================================
# Error Handling Functions
# ============================================================================

def handle_error(error: Exception, error_type: Optional[str] = None, context: Optional[Dict] = None):
    """
    Handle errors with user-friendly messages and logging
    
    Args:
        error: The exception that occurred
        error_type: Optional error type key for ERROR_MESSAGES
        context: Optional context information
    """
    # Log the full error with traceback
    logger.error(f"Error occurred: {str(error)}", exc_info=True)
    
    if context:
        logger.error(f"Context: {context}")
    
    # Get user-friendly message
    if error_type and error_type in ERROR_MESSAGES:
        error_info = ERROR_MESSAGES[error_type]
    else:
        # Generic error message
        error_info = {
            'title': '⚠️ An Error Occurred',
            'message': str(error),
            'suggestions': [
                'Check the error log for details',
                'Try refreshing the page',
                'Contact support if the issue persists'
            ]
        }
    
    # Display in Streamlit
    st.error(f"**{error_info['title']}**")
    st.write(error_info['message'])
    
    if error_info.get('suggestions'):
        st.write("**Suggestions:**")
        for suggestion in error_info['suggestions']:
            st.write(f"• {suggestion}")
    
    # Show technical details in expander
    with st.expander("🔧 Technical Details"):
        st.code(traceback.format_exc())
        if context:
            st.json(context)


def safe_execute(func, error_type: Optional[str] = None, default_return=None, context: Optional[Dict] = None):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        error_type: Error type key for ERROR_MESSAGES
        default_return: Value to return if error occurs
        context: Optional context information
        
    Returns:
        Result of func() or default_return if error occurs
    """
    try:
        return func()
    except Exception as e:
        handle_error(e, error_type, context)
        return default_return


def validate_data_file(df, required_columns=None):
    """
    Validate data file structure
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValidationError: If validation fails
    """
    if required_columns is None:
        required_columns = ['Tanggal', 'Jenis Pajak', 'Nominal (Milyar)']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValidationError(
            f"Missing required columns: {', '.join(missing_cols)}",
            context={'missing_columns': missing_cols, 'available_columns': list(df.columns)}
        )
    
    # Check for minimum data
    if len(df) < 12:
        raise ValidationError(
            "Insufficient data rows (minimum 12 required)",
            context={'row_count': len(df)}
        )
    
    return True


def show_error_toast(message: str, icon: str = "⚠️"):
    """Show a simple error toast message"""
    st.toast(f"{icon} {message}", icon="⚠️")


def log_operation(operation_name: str, success: bool, details: Optional[Dict] = None):
    """
    Log operation for audit trail
    
    Args:
        operation_name: Name of the operation
        success: Whether operation succeeded
        details: Additional details to log
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_name,
        'success': success,
        'details': details or {}
    }
    
    if success:
        logger.info(f"Operation succeeded: {operation_name}", extra=log_entry)
    else:
        logger.warning(f"Operation failed: {operation_name}", extra=log_entry)
    
    return log_entry
