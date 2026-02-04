"""
Theme manager for TaxForecaster application.

Provides theme customization including dark/light mode and custom color schemes.
"""

import streamlit as st
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    from config_loader import get
except ImportError:
    def get(key, default=None): return default


class ThemeManager:
    """Manages application themes and styling"""
    
    # Predefined themes
    THEMES = {
        'dark': {
            'name': 'Dark Mode (Default)',
            'primary_color': '#3B82F6',
            'background_color': '#0E1117',
            'secondary_background': '#262730',
            'text_color': '#FAFAFA',
            'font': 'sans serif'
        },
        'light': {
            'name': 'Light Mode',
            'primary_color': '#3B82F6',
            'background_color': '#FFFFFF',
            'secondary_background': '#F0F2F6',
            'text_color': '#262730',
            'font': 'sans serif'
        },
        'ocean': {
            'name': 'Ocean Blue',
            'primary_color': '#0077BE',
            'background_color': '#F0F8FF',
            'secondary_background': '#E6F3FF',
            'text_color': '#1C3D5A',
            'font': 'sans serif'
        },
        'forest': {
            'name': 'Forest Green',
            'primary_color': '#2D5016',
            'background_color': '#F5F9F0',
            'secondary_background': '#E8F5E0',
            'text_color': '#1A3409',
            'font': 'sans serif'
        },
        'sunset': {
            'name': 'Sunset Orange',
            'primary_color': '#FF6B35',
            'background_color': '#FFF8F3',
            'secondary_background': '#FFE5D9',
            'text_color': '#4A1F14',
            'font': 'sans serif'
        }
    }
    
    @staticmethod
    def get_current_theme() -> str:
        """Get currently selected theme from session state"""
        if 'theme' not in st.session_state:
            st.session_state['theme'] = 'dark'
        return st.session_state['theme']
    
    @staticmethod
    def set_theme(theme_name: str):
        """Set the current theme"""
        if theme_name in ThemeManager.THEMES:
            st.session_state['theme'] = theme_name
    
    @staticmethod
    def apply_theme():
        """Apply current theme using custom CSS"""
        current_theme = ThemeManager.get_current_theme()
        theme_config = ThemeManager.THEMES.get(current_theme, ThemeManager.THEMES['dark'])
        
        custom_css = f"""
        <style>
        /* Main theme colors */
        :root {{
            --primary-color: {theme_config['primary_color']};
            --background-color: {theme_config['background_color']};
            --secondary-bg: {theme_config['secondary_background']};
            --text-color: {theme_config['text_color']};
        }}
        
        /* Custom metric cards with theme colors */
        .metric-card {{
            background: linear-gradient(135deg, {theme_config['secondary_background']} 0%, {theme_config['background_color']} 100%);
            border-left: 4px solid {theme_config['primary_color']};
            color: {theme_config['text_color']};
        }}
        
        /* Status indicators */
        .status-pulse {{
            box-shadow: 0 0 0 0 {theme_config['primary_color']};
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {theme_config['primary_color']};
            color: white;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {theme_config['secondary_background']};
            border-radius: 8px;
        }}
        
        /* Smooth animations */
        * {{
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        </style>
        """
        
        st.markdown(custom_css, unsafe_allow_html=True)
    
    @staticmethod
    def render_theme_selector():
        """Render theme selection widget in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Theme")
        
        current_theme = ThemeManager.get_current_theme()
        theme_options = {v['name']: k for k, v in ThemeManager.THEMES.items()}
        
        selected_theme_name = st.sidebar.selectbox(
            "Choose Theme",
            options=list(theme_options.keys()),
            index=list(theme_options.values()).index(current_theme) if current_theme in theme_options.values() else 0,
            key='theme_selector'
        )
        
        if theme_options[selected_theme_name] != current_theme:
            ThemeManager.set_theme(theme_options[selected_theme_name])
            st.rerun()
    
    @staticmethod
    def get_chart_colors() -> list:
        """Get color palette for charts based on current theme"""
        current_theme = ThemeManager.get_current_theme()
        
        # Default color palette
        default_colors = [
            '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6',
            '#EC4899', '#6366F1', '#14B8A6', '#F97316'
        ]
        
        # Theme-specific palettes
        theme_palettes = {
            'ocean': ['#0077BE', '#00A3E0', '#00C9FF', '#5DADE2', '#85C1E2'],
            'forest': ['#2D5016', '#52A447', '#7BC96F', '#A8E6A3', '#C8F2C3'],
            'sunset': ['#FF6B35', '#FF8F6B', '#FFB299', '#FFCDB3', '#FFE5D9']
        }
        
        return theme_palettes.get(current_theme, default_colors)


# Convenience function
def apply_theme():
    """Apply current theme (convenience function)"""
    ThemeManager.apply_theme()


def render_theme_selector():
    """Render theme selector (convenience function)"""
    ThemeManager.render_theme_selector()


if __name__ == "__main__":
    print("Theme Manager initialized")
    print(f"Available themes: {list(ThemeManager.THEMES.keys())}")
