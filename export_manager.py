"""
Export functionality for TaxForecaster application.

Supports exporting forecasts to CSV, Excel, and HTML formats.
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, Dict, Any
import os
import sys

sys.path.append(os.path.dirname(__file__))

try:
    from logger import get_logger
    from config_loader import get
except ImportError:
    def get_logger(*args):
        import logging
        return logging.getLogger(__name__)
    def get(key, default=None): return default


logger = get_logger(__name__)


class ExportManager:
    """Handles exporting forecast results to various formats"""
    
    @staticmethod
    def export_to_csv(forecast_df: pd.DataFrame, filename: str = None) -> str:
        """
        Export forecast data to CSV
        
        Args:
            forecast_df: DataFrame containing forecast results
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"forecast_export_{timestamp}.csv"
        
        filepath = os.path.join("exports", filename)
        os.makedirs("exports", exist_ok=True)
        
        forecast_df.to_csv(filepath, index=False)
        logger.info(f"Exported forecast to CSV: {filepath}")
        
        return filepath
    
    @staticmethod
    def export_to_excel(
        forecast_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        filename: str = None
    ) -> str:
        """
        Export forecast data to Excel with multiple sheets
        
        Args:
            forecast_df: DataFrame containing forecast results
            historical_df: Optional historical data
            metrics_df: Optional model performance metrics
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"forecast_export_{timestamp}.xlsx"
        
        filepath = os.path.join("exports", filename)
        os.makedirs("exports", exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write forecast data
            forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Write historical data if provided
            if historical_df is not None:
                historical_df.to_excel(writer, sheet_name='Historical Data', index=False)
            
            # Write metrics if provided
            if metrics_df is not None:
                metrics_df.to_excel(writer, sheet_name='Model Performance', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Export Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Application': ['TaxForecaster 2.0'],
                'Version': [get('app.version', '2.0.0')]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Exported forecast to Excel: {filepath}")
        return filepath
    
    @staticmethod
    def export_to_html(
        forecast_df: pd.DataFrame,
        charts: Optional[list] = None,
        title: str = "Tax Forecast Report",
        filename: str = None
    ) -> str:
        """
        Export forecast to interactive HTML report
        
        Args:
            forecast_df: DataFrame containing forecast results
            charts: Optional list of Plotly figure objects
            title: Report title
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"forecast_report_{timestamp}.html"
        
        filepath = os.path.join("exports", filename)
        os.makedirs("exports", exist_ok=True)
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3B82F6;
                    padding-bottom: 15px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #e0e0e0;
                }}
                th {{
                    background-color: #3B82F6;
                    color: white;
                    font-weight: 600;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                .metadata {{
                    margin-top: 40px;
                    padding: 20px;
                    background: #e8f4f8;
                    border-left: 4px solid #3B82F6;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏛️ {title}</h1>
                <div class="metadata">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>Application:</strong> TaxForecaster 2.0
                </div>
                
                <h2>📊 Forecast Data</h2>
                {forecast_df.to_html(index=False, classes='forecast-table')}
        """
        
        # Add charts if provided
        if charts:
            for i, fig in enumerate(charts):
                html_content += f"""
                <div class="chart-container">
                    <h3>Chart {i+1}</h3>
                    <div id="chart{i}"></div>
                    <script>
                        var data{i} = {fig.to_json()};
                        Plotly.newPlot('chart{i}', data{i}.data, data{i}.layout);
                    </script>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Exported forecast to HTML: {filepath}")
        return filepath
    
    @staticmethod
    def create_download_button_data(forecast_df: pd.DataFrame, format: str = 'csv') -> tuple:
        """
        Create data for Streamlit download button
        
        Args:
            forecast_df: DataFrame to export
            format: Export format ('csv' or 'excel')
            
        Returns:
            Tuple of (data, filename, mime_type)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            data = forecast_df.to_csv(index=False).encode('utf-8')
            filename = f"forecast_{timestamp}.csv"
            mime = 'text/csv'
        elif format == 'excel':
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            data = output.getvalue()
            filename = f"forecast_{timestamp}.xlsx"
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return data, filename, mime


# Convenience functions
def export_csv(df: pd.DataFrame, filename: str = None) -> str:
    """Export to CSV (convenience function)"""
    return ExportManager.export_to_csv(df, filename)


def export_excel(df: pd.DataFrame, filename: str = None) -> str:
    """Export to Excel (convenience function)"""
    return ExportManager.export_to_excel(df, filename=filename)


def export_html(df: pd.DataFrame, charts: list = None, filename: str = None) -> str:
    """Export to HTML (convenience function)"""
    return ExportManager.export_to_html(df, charts, filename=filename)


if __name__ == "__main__":
    # Test export manager
    test_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10, freq='M'),
        'Tax_Type': ['PPh'] * 10,
        'Forecast': [100 + i*5 for i in range(10)]
    })
    
    print("Testing export functions...")
    csv_path = export_csv(test_df, "test_export.csv")
    print(f"✅ CSV exported: {csv_path}")
    
    excel_path = export_excel(test_df, "test_export.xlsx")
    print(f"✅ Excel exported: {excel_path}")
    
    html_path = export_html(test_df, filename="test_report.html")
    print(f"✅ HTML exported: {html_path}")
