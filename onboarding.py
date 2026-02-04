"""
 Onboarding wizard for first-time users.

Guides users through initial setup: data upload, validation, and first forecast.
"""

import streamlit as st
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    from logger import get_logger
except ImportError:
    def get_logger(*args):
        import logging
        return logging.getLogger(__name__)


logger = get_logger(__name__)


class OnboardingWizard:
    """Manages the first-time user onboarding experience"""
    
    STEPS = [
        {
            'title': '👋 Welcome to TaxForecaster',
            'description': 'Let\'s get you started with forecasting your tax revenue!',
            'icon': '🎯'
        },
        {
            'title': '📊 Upload Your Data',
            'description': 'Upload your historical tax data to begin',
            'icon': '📁'
        },
        {
            'title': '✅ Validate Data',
            'description': 'We\'ll check your data quality',
            'icon': '🔍'
        },
        {
            'title': '🚀 Train Models',
            'description': 'Build your first forecast model',
            'icon': '🤖'
        }
    ]
    
    @staticmethod
    def should_show_onboarding() -> bool:
        """Check if onboarding should be shown"""
        # Show if not completed OR if explicitly requested
        return not st.session_state.get('onboarding_completed', False) or \
               st.session_state.get('show_onboarding_wizard', False)
    
    @staticmethod
    def mark_completed():
        """Mark onboarding as completed"""
        st.session_state['onboarding_completed'] = True
        st.session_state['show_onboarding_wizard'] = False
        logger.info("Onboarding completed")
    
    @staticmethod
    def reset_onboarding():
        """Reset onboarding state"""
        st.session_state['onboarding_completed'] = False
        st.session_state['current_onboarding_step'] = 0
    
    @staticmethod
    def get_current_step() -> int:
        """Get current onboarding step"""
        if 'current_onboarding_step' not in st.session_state:
            st.session_state['current_onboarding_step'] = 0
        return st.session_state['current_onboarding_step']
    
    @staticmethod
    def advance_step():
        """Move to next step"""
        current = OnboardingWizard.get_current_step()
        if current < len(OnboardingWizard.STEPS) - 1:
            st.session_state['current_onboarding_step'] = current + 1
            st.rerun()
    
    @staticmethod
    def go_back():
        """Go to previous step"""
        current = OnboardingWizard.get_current_step()
        if current > 0:
            st.session_state['current_onboarding_step'] = current - 1
            st.rerun()
    
    @staticmethod
    def render(key_prefix: str = "onboarding"):
        """
        Render the onboarding wizard
        
        Args:
            key_prefix: Unique prefix for widget keys to prevent duplicate ID errors
        """
        current_step = OnboardingWizard.get_current_step()
        total_steps = len(OnboardingWizard.STEPS)
        step_info = OnboardingWizard.STEPS[current_step]
        
        # Custom CSS for wizard
        st.markdown("""
        <style>
        .wizard-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin: 10px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .wizard-step {
            font-size: 32px;
            text-align: center;
            margin: 5px 0;
        }
        .wizard-title {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin: 5px 0;
        }
        .wizard-description {
            font-size: 16px;
            text-align: center;
            opacity: 0.9;
            margin: 5px 0;
        }
        .progress-dots {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin: 15px 0;
        }
        .progress-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: rgba(255,255,255,0.3);
            transition: all 0.3s ease;
        }
        .progress-dot.active {
            background: white;
            transform: scale(1.3);
            box-shadow: 0 0 10px rgba(255,255,255,0.5);
        }
        .progress-dot.completed {
            background: #10B981;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Wizard container
        st.markdown(f"""
        <div class="wizard-container">
            <div class="wizard-step">{step_info['icon']}</div>
            <div class="wizard-title">{step_info['title']}</div>
            <div class="wizard-description">{step_info['description']}</div>
            <div class="progress-dots">
                {''.join([
                    f'<div class="progress-dot {" active" if i == current_step else ""}{" completed" if i < current_step else ""}"></div>'
                    for i in range(total_steps)
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Step-specific content
        if current_step == 0:
            OnboardingWizard._render_welcome()
        elif current_step == 1:
            OnboardingWizard._render_upload()
        elif current_step == 2:
            OnboardingWizard._render_validation()
        elif current_step == 3:
            OnboardingWizard._render_training()
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_step > 0:
                if st.button("← Back", key=f"{key_prefix}_back", use_container_width=True):
                    OnboardingWizard.go_back()
        
        with col3:
            if current_step < total_steps - 1:
                if st.button("Next →", key=f"{key_prefix}_next", use_container_width=True, type="primary"):
                    OnboardingWizard.advance_step()
            else:
                if st.button("🎉 Finish", key=f"{key_prefix}_finish", use_container_width=True, type="primary"):
                    OnboardingWizard.mark_completed()
                    st.success("✅ Onboarding complete! Welcome to TaxForecaster!")
                    st.balloons()
                    st.rerun()
        
        # Skip button
        with col2:
            if st.button("Skip Tutorial", key=f"{key_prefix}_skip", use_container_width=True):
                OnboardingWizard.mark_completed()
                st.rerun()
    
    @staticmethod
    def _render_welcome():
        """Render welcome step"""
        st.markdown("""
        ### What is TaxForecaster?
        
        TaxForecaster 2.0 is an AI-powered forecasting platform that helps you:
        
        - 📊 **Predict tax revenue** with machine learning models
        - 🎯 **Test scenarios** with different economic conditions
        - 📈 **Visualize trends** with interactive charts
        - 📄 **Export reports** in multiple formats
        
        ### Quick Start
        
        This wizard will guide you through:
        1. Uploading your tax history data
        2. Validating data quality
        3. Training your first forecast model
        4. Viewing predictions
        
        Let's get started! Click "Next" to continue.
        """)
    
    @staticmethod
    def _render_upload():
        """Render data upload step"""
        st.markdown("""
        ### Upload Your Data
        
        TaxForecaster needs historical tax data to build accurate forecasts.
        
        **Required Format:**
        - **CSV file** with columns: `Tanggal`, `Jenis Pajak`, `Nominal (Milyar)`
        - **Date format**: DD/MM/YYYY
        - **Minimum**: 24 months of data recommended
        
        **Optional:**
        - Macro economic data (Inflasi, Kurs_USD, etc.)
        """)
        
        # Show sample data
        with st.expander("📋 View Sample Data Format"):
            st.code("""
Tanggal,Jenis Pajak,Nominal (Milyar)
01/01/2023,PPh,150.5
01/02/2023,PPh,165.2
01/03/2023,PPh,158.7
01/01/2023,PPN,200.3
01/02/2023,PPN,215.8
            """, language="csv")
        
        st.info("💡 **Tip**: Go to the Dashboard page to upload your data files!")
    
    @staticmethod
    def _render_validation():
        """Render validation step"""
        st.markdown("""
        ### Data Validation
        
        Before training models, we'll check your data quality:
        
        ✅ **Required Columns** - Verify all necessary columns exist  
        ✅ **Date Format** - Check all dates are parseable  
        ✅ **Missing Values** - Identify and handle gaps  
        ✅ **Outliers** - Detect unusual values  
        ✅ **Data Range** - Ensure sufficient history  
        
        ### Auto-Corrections
        
        TaxForecaster can automatically:
        - Fill missing dates with interpolation
        - Smooth outliers using rolling averages
        - Merge macro economic indicators
        
        These corrections are optional and can be configured in the Dashboard.
        """)
        
        st.success("✅ Your data will be validated automatically when you upload it!")
    
    @staticmethod
    def _render_training():
        """Render training step"""
        st.markdown("""
        ### Train Your First Model
        
        TaxForecaster uses **15+ machine learning models**:
        
        🌲 **Tree-Based**: XGBoost, LightGBM, Random Forest  
        📊 **Statistical**: SARIMA, Prophet, Holt-Winters  
        🧠 **Neural Networks**: LSTM, MLP  
        🎯 **Ensemble**: Combines best models  
        
        ### Training Process
        
        1. **Feature Engineering** - Create time-based features
        2. **Hyperparameter Tuning** - Optimize each model
        3. **Cross-Validation** - Test accuracy
        4. **Ensemble Selection** - Pick top performers
        
        **Training time**: ~2-5 minutes depending on data size
        
        ### Next Steps
        
        After completing this tutorial:
        1. Upload your data on the **Dashboard** page
        2. Click "Train Models" to start
        3. View forecasts and download reports
        
        Ready to start? Click "Finish" to begin!
        """)


def show_onboarding_if_needed():
    """Show onboarding wizard if needed (convenience function)"""
    if OnboardingWizard.should_show_onboarding():
        OnboardingWizard.render()
        return True
    return False


# Example usage
if __name__ == "__main__":
    print("Onboarding wizard initialized")
    print(f"Total steps: {len(OnboardingWizard.STEPS)}")
