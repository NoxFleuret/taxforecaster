"""
Loading animations and skeleton screens for TaxForecaster.

Provides visual feedback during data loading and processing.
"""

import streamlit as st
import time


class LoadingAnimations:
    """Collection of loading animations and skeleton screens"""
    
    @staticmethod
    def skeleton_metric_card():
        """Render skeleton screen for metric card"""
        st.markdown("""
        <style>
        .skeleton {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 8px;
        }
        
        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        .skeleton-card {
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .skeleton-title {
            height: 20px;
            width: 60%;
            margin-bottom: 15px;
        }
        
        .skeleton-value {
            height: 40px;
            width: 40%;
            margin-bottom: 10px;
        }
        
        .skeleton-subtitle {
            height: 15px;
            width: 50%;
        }
        </style>
        
        <div class="skeleton-card">
            <div class="skeleton skeleton-title"></div>
            <div class="skeleton skeleton-value"></div>
            <div class="skeleton skeleton-subtitle"></div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def skeleton_chart():
        """Render skeleton screen for chart"""
        st.markdown("""
        <div class="skeleton-card" style="height: 400px;">
            <div class="skeleton" style="height: 100%; width: 100%;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def loading_spinner_custom(text: str = "Loading..."):
        """Custom loading spinner with animation"""
        st.markdown(f"""
        <style>
        .spinner-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }}
        
        .spinner {{
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3B82F6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .loading-text {{
            margin-top: 20px;
            font-size: 18px;
            color: #666;
            animation: pulse 1.5s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        </style>
        
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="loading-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar_animated(progress: float, label: str = ""):
        """Animated progress bar with gradient"""
        st.markdown(f"""
        <style>
        .progress-container {{
            width: 100%;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }}
        
        .progress-bar {{
            height: 30px;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6);
            width: {progress}%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }}
        
        .progress-label {{
            color: #666;
            margin-bottom: 10px;
            font-weight: 500;
        }}
        </style>
        
        <div>
            <div class="progress-label">{label}</div>
            <div class="progress-container">
                <div class="progress-bar">{int(progress)}%</div>
           </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def number_counter(value: float, label: str, duration: float = 1.0):
        """Animated number counter"""
        placeholder = st.empty()
        
        steps = 20
        step_value = value / steps
        step_duration = duration / steps
        
        for i in range(steps + 1):
            current_value = step_value * i
            placeholder.metric(label, f"{current_value:,.2f}")
            time.sleep(step_duration)
    
    @staticmethod
    def fade_in_content(content_html: str):
        """Fade in animation for content"""
        st.markdown(f"""
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.6s ease-out;
        }}
        </style>
        
        <div class="fade-in">
            {content_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def shimmer_effect():
        """Shimmer loading effect"""
        st.markdown("""
        <style>
        .shimmer-wrapper {
            width: 100%;
            height: 200px;
            background: #f6f7f8;
            background-image: linear-gradient(
                to right,
                #f6f7f8 0%,
                #edeef1 20%,
                #f6f7f8 40%,
                #f6f7f8 100%
            );
            background-repeat: no-repeat;
            background-size: 800px 200px;
            animation: shimmer 2s linear infinite;
            border-radius: 10px;
        }
        
        @keyframes shimmer {
            0% { background-position: -800px 0; }
            100% { background-position: 800px 0; }
        }
        </style>
        
        <div class="shimmer-wrapper"></div>
        """, unsafe_allow_html=True)


def show_loading_screen(text: str = "Loading..."):
    """Show loading screen (convenience function)"""
    LoadingAnimations.loading_spinner_custom(text)


def show_skeleton_cards(count: int = 3):
    """Show multiple skeleton cards"""
    for _ in range(count):
        LoadingAnimations.skeleton_metric_card()


# Example usage
if __name__ == "__main__":
    print("Loading animations module loaded")
