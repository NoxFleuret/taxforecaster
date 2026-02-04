"""
Data loading and preprocessing module.

Handles loading CSV files, validating data structure, and merging macro data.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Optional, Tuple
import sys

# Add parent to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from error_handler import DataError, ValidationError, validate_data_file
    from logger import get_logger
    from config_loader import get
except ImportError:
    # Fallback for standalone usage
    class DataError(Exception): pass
    class ValidationError(Exception): pass
    def validate_data_file(*args, **kwargs): return True
    def get_logger(*args): 
        import logging
        return logging.getLogger(__name__)
    def get(key, default=None): return default


logger = get_logger(__name__)


class DataLoader:
    """
    Handles all data loading operations for tax history and macro data.
    """
    
    def __init__(self, history_file: str, macro_file: Optional[str] = None):
        """
        Initialize DataLoader
        
        Args:
            history_file: Path to tax history CSV or UploadedFile object
            macro_file: Optional path to macro data CSV or UploadedFile object
        """
        self.history_file = history_file
        self.macro_file = macro_file
        self.df = None
        self.macro_df = None
        self.metadata = {
            'loaded_at': None,
            'load_duration': None,
            'rows': 0,
            'tax_types': 0,
            'has_macro': False
        }
    
    def load(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load and validate both tax history and macro data
        
        Returns:
            Tuple of (tax_df, macro_df)
            
        Raises:
            DataError: If data loading fails
            ValidationError: If data validation fails
        """
        import time
        start_time = time.time()
        
        logger.info("Starting data loading process")
        
        # Load tax history
        self.df = self._load_tax_history()
        
        # Load macro data if provided
        if self.macro_file:
            self.macro_df = self._load_macro_data()
            self._merge_macro_data()
        
        # Update metadata
        load_duration = time.time() - start_time
        self.metadata.update({
            'loaded_at': datetime.now().isoformat(),
            'load_duration': round(load_duration, 2),
            'rows': len(self.df),
            'tax_types': len(self.df['Jenis Pajak'].unique()),
            'has_macro': self.macro_df is not None
        })
        
        logger.info(f"Data loaded successfully: {self.metadata['rows']} rows, "
                   f"{self.metadata['tax_types']} tax types, "
                   f"duration={load_duration:.2f}s")
        
        return self.df, self.macro_df
    
    def _load_tax_history(self) -> pd.DataFrame:
        """Load tax history CSV file"""
        try:
            # Handle UploadedFile vs file path
            if hasattr(self.history_file, 'name'):
                logger.info(f"Loading from UploadedFile: {self.history_file.name}")
                df = pd.read_csv(self.history_file)
            else:
                abs_path = os.path.abspath(self.history_file)
                logger.info(f"Loading from file: {abs_path}")
                
                if not os.path.exists(abs_path):
                    raise DataError(
                        f"Tax history file not found: {abs_path}",
                        context={'file_path': abs_path}
                    )
                
                df = pd.read_csv(self.history_file)
            
            # Validate structure
            validate_data_file(df)
            
            # Parse dates
            try:
                date_format = get('data.date_format', '%d/%m/%Y')
                df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
            except Exception as e:
                raise ValidationError(
                    "Invalid date format in Tanggal column",
                    context={'error': str(e)}
                )
            
            logger.debug(f"Successfully loaded {len(df)} rows from tax history")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load tax history: {e}")
            raise
    
    def _load_macro_data(self) -> Optional[pd.DataFrame]:
        """Load macro economic data CSV file"""
        try:
            # Handle UploadedFile vs file path
            if hasattr(self.macro_file, 'name'):
                logger.info(f"Loading macro from UploadedFile: {self.macro_file.name}")
                macro_df = pd.read_csv(self.macro_file)
            else:
                abs_path = os.path.abspath(self.macro_file)
                
                if not os.path.exists(abs_path):
                    logger.warning(f"Macro file not found: {abs_path}. Continuing without macro data.")
                    return None
                
                logger.info(f"Loading macro from file: {abs_path}")
                macro_df = pd.read_csv(self.macro_file)
            
            # Parse dates
            try:
                macro_df['Tanggal'] = pd.to_datetime(macro_df['Tanggal'], dayfirst=True)
            except Exception as e:
                logger.warning(f"Could not parse dates in macro data: {e}")
                return None
            
            logger.debug(f"Successfully loaded {len(macro_df)} rows from macro data")
            return macro_df
            
        except Exception as e:
            logger.warning(f"Error loading macro data: {e}. Continuing without macro data.")
            return None
    
    def _merge_macro_data(self):
        """Merge macro data with tax history"""
        if self.macro_df is None or self.df is None:
            return
        
        try:
            logger.debug("Merging macro data with tax history")
            
            # Merge on date
            original_rows = len(self.df)
            self.df = pd.merge(self.df, self.macro_df, on='Tanggal', how='left')
            
            # Fill missing values
            exclude_cols = {'Jenis Pajak', 'Nominal (Milyar)', 'Tanggal'}
            macro_cols = [col for col in self.df.columns if col not in exclude_cols]
            
            for col in macro_cols:
                if col in self.df.columns and self.df[col].dtype in ['float64', 'int64']:
                    self.df[col] = self.df[col].ffill().bfill()
            
            logger.info(f"Merged macro data: {len(macro_cols)} macro indicators added")
            
        except Exception as e:
            logger.error(f"Error merging macro data: {e}")
            raise
    
    def get_tax_types(self) -> list:
        """Get list of unique tax types"""
        if self.df is None:
            return []
        return self.df['Jenis Pajak'].unique().tolist()
    
    def get_data_for_tax_type(self, tax_type: str) -> pd.DataFrame:
        """
        Get data for a specific tax type
        
        Args:
            tax_type: Name of tax type
            
        Returns:
            Filtered and sorted DataFrame
        """
        if self.df is None:
            raise DataError("No data loaded")
        
        data = self.df[self.df['Jenis Pajak'] == tax_type].copy().sort_values('Tanggal')
        data.set_index('Tanggal', inplace=True)
        
        return data
    
    def get_macro_columns(self) -> list:
        """Get list of macro indicator column names"""
        if self.df is None:
            return []
        
        exclude_cols = {'Jenis Pajak', 'Nominal (Milyar)', 'Tanggal', 'is_lebaran', 'is_natal'}
        return [col for col in self.df.columns 
                if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]


# Example usage
if __name__ == "__main__":
    # Test data loader
    loader = DataLoader("tax_history.csv", "macro_data_auto.csv")
    
    try:
        tax_df, macro_df = loader.load()
        print(f"\n✅ Loaded {len(tax_df)} rows")
        print(f"📊 Tax types: {loader.get_tax_types()}")
        print(f"📈 Macro columns: {loader.get_macro_columns()}")
        print(f"\n⏱️ Metadata: {loader.metadata}")
    except Exception as e:
        print(f"❌ Error: {e}")
