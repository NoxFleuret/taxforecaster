"""
Unit tests for DataLoader class.

Run with: pytest tests/test_data_loader.py
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class"""
    
    @pytest.fixture
    def sample_tax_data(self, tmp_path):
        """Create a sample tax history CSV file"""
        data = {
            'Tanggal': ['01/01/2023', '01/02/2023', '01/03/2023'],
            'Jenis Pajak': ['PPh', 'PPh', 'PPh'],
            'Nominal (Milyar)': [100.0, 120.0, 110.0]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "tax_test.csv"
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    @pytest.fixture
    def sample_macro_data(self, tmp_path):
        """Create a sample macro data CSV file"""
        data = {
            'Tanggal': ['01/01/2023', '01/02/2023', '01/03/2023'],
            'Inflasi': [3.0, 3.1, 2.9],
            'Kurs_USD': [15000, 15100, 14900]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "macro_test.csv"
        df.to_csv(file_path, index=False)
        
        return str(file_path)
    
    def test_init(self, sample_tax_data):
        """Test DataLoader initialization"""
        loader = DataLoader(sample_tax_data)
        
        assert loader.history_file == sample_tax_data
        assert loader.macro_file is None
        assert loader.df is None
        assert loader.macro_df is None
    
    def test_load_tax_history_only(self, sample_tax_data):
        """Test loading tax history without macro data"""
        loader = DataLoader(sample_tax_data)
        tax_df, macro_df = loader.load()
        
        assert tax_df is not None
        assert len(tax_df) == 3
        assert 'Tanggal' in tax_df.columns
        assert 'Jenis Pajak' in tax_df.columns
        assert 'Nominal (Milyar)' in tax_df.columns
        assert macro_df is None
    
    def test_load_with_macro_data(self, sample_tax_data, sample_macro_data):
        """Test loading tax history with macro data"""
        loader = DataLoader(sample_tax_data, sample_macro_data)
        tax_df, macro_df = loader.load()
        
        assert tax_df is not None
        assert macro_df is not None
        
        # Check that macro columns are merged
        assert 'Inflasi' in tax_df.columns
        assert 'Kurs_USD' in tax_df.columns
    
    def test_get_tax_types(self, sample_tax_data):
        """Test retrieving tax types"""
        loader = DataLoader(sample_tax_data)
        loader.load()
        
        tax_types = loader.get_tax_types()
        assert len(tax_types) == 1
        assert 'PPh' in tax_types
    
    def test_get_macro_columns(self, sample_tax_data, sample_macro_data):
        """Test retrieving macro column names"""
        loader = DataLoader(sample_tax_data, sample_macro_data)
        loader.load()
        
        macro_cols = loader.get_macro_columns()
        assert 'Inflasi' in macro_cols
        assert 'Kurs_USD' in macro_cols
    
    def test_metadata_tracking(self, sample_tax_data):
        """Test that metadata is tracked correctly"""
        loader = DataLoader(sample_tax_data)
        loader.load()
        
        assert loader.metadata['loaded_at'] is not None
        assert loader.metadata['rows'] == 3
        assert loader.metadata['tax_types'] == 1
        assert loader.metadata['has_macro'] == False
        assert loader.metadata['load_duration'] >= 0
    
    def test_file_not_found(self):
        """Test handling of missing file"""
        loader = DataLoader("nonexistent_file.csv")
        
        with pytest.raises(Exception):
            loader.load()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
