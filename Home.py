import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import style

# Import UX enhancements
try:
    from theme_manager import ThemeManager
    from onboarding import show_onboarding_if_needed
except ImportError:
    # Fallback if modules not available
    class ThemeManager:
        @staticmethod
        def apply_theme(): pass
        @staticmethod
        def render_theme_selector(): pass
    def show_onboarding_if_needed(): return False

# Set Page Config
st.set_page_config(
    page_title="TaxForecaster | Enterprise AI",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Apply Theme
style.apply_theme()
ThemeManager.apply_theme()  # Apply custom theme

# Show onboarding wizard if needed
if show_onboarding_if_needed():
    st.stop()  # Stop rendering rest of page if showing onboarding

# --- SYSTEM PULSE CHECKS ---
# Check Data Status
data_status = "MISSING"
data_color = "red"
data_msg = "Tax History Not Found"
data_icon = "❌"
tax_file = 'tax_history.csv'

if os.path.exists(tax_file):
    data_status = "ONLINE"
    data_color = "#10B981"
    data_icon = "✅"
    try:
        last_mod = os.path.getmtime(tax_file)
        dt_mod = datetime.fromtimestamp(last_mod).strftime("%d %b %H:%M")
        data_msg = f"Last Upd: {dt_mod}"
    except:
        data_msg = "Ready"

# Check Macro Status
macro_status = "OFFLINE"
macro_color = "gray"
macro_msg = "No Macro Data"
macro_icon = "⚪"
macro_file = 'macro_data_auto.csv'

if os.path.exists(macro_file):
    macro_status = "LIVE"
    macro_color = "#3B82F6" # Blue
    macro_icon = "🔵"
    try:
        df_m = pd.read_csv(macro_file)
        latest_date = pd.to_datetime(df_m['Tanggal'], dayfirst=True).max().strftime("%b %Y")
        macro_msg = f"Data until: {latest_date}"
    except:
        macro_msg = "Connected"

# Check Model Status (Session State)
model_status = "IDLE"
model_color = "orange"
model_acc = "N/A"
model_msg = "Not Trained"
model_icon = "🟡"

if 'forecaster_v6' in st.session_state and st.session_state['forecaster_v6']:
    fc = st.session_state['forecaster_v6']
    if fc.is_fitted:
        model_status = "ACTIVE"
        model_color = "#8B5CF6" # Purple
        model_icon = "🟣"
        
        # Get best accuracy from performance list
        try:
            acc_list = [float(x['Accuracy'].replace('%','')) for x in fc.model_performance]
            avg_acc = sum(acc_list) / len(acc_list)
            model_acc = f"{avg_acc:.1f}%"
            model_msg = f"Best Acc: {max(acc_list):.1f}%"
        except:
            model_msg = "Models Ready"
        
        # Memory usage tracking
        try:
            mem_info = fc.get_memory_usage()
            if mem_info['total_mb'] > 100:  # Alert if using more than 100MB
                st.toast(f"⚠️ High memory usage: {mem_info['total_mb']:.1f} MB", icon="⚠️")
        except:
            pass

# --- HERO SECTION ---
st.markdown("""
<div style="text-align: center; padding: 60px 0 40px 0; animation: fadeIn 0.8s ease-out;">
<div style="
display: inline-block;
padding: 5px 15px;
background: rgba(59, 130, 246, 0.2);
color: #60A5FA;
border-radius: 20px;
font-size: 0.85rem;
font-weight: 600;
letter-spacing: 1px;
border: 1px solid rgba(59, 130, 246, 0.4);
margin-bottom: 20px;
">FASYA_DEV</div>
<h1 style="
font-size: 4rem; 
font-weight: 800; 
background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin-bottom: 10px;
letter-spacing: -2px;
">TaxForecaster</h1>
<p style="
font-size: 1.25rem; 
color: #94A3B8; 
max-width: 600px; 
margin: 0 auto; 
line-height: 1.6;
">Advanced predictive intelligence system for fiscal revenue planning, risk simulation, and macroeconomic stress testing.</p>
</div>
""", unsafe_allow_html=True)


# --- SYSTEM PULSE DASHBOARD ---
st.markdown(f"""
<div style="
display: grid;
grid-template-columns: repeat(3, 1fr);
gap: 20px;
max-width: 900px;
margin: 0 auto 50px auto;
padding: 20px;
background: rgba(15, 23, 42, 0.6);
border-radius: 20px;
border: 1px solid rgba(255,255,255,0.05);
backdrop-filter: blur(10px);
animation: fadeIn 1s ease-out;
">
<!-- SYSTEM 1 -->
<div style="text-align: center; padding: 10px; border-right: 1px solid rgba(255,255,255,0.05);">
<div style="font-size: 0.75rem; color: #64748B; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;">Data Pipeline</div>
<div style="font-weight: 700; color: {data_color}; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
<span style="height: 8px; width: 8px; background-color: {data_color}; border-radius: 50%; box-shadow: 0 0 10px {data_color};"></span>
{data_status}
</div>
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">{data_msg}</div>
</div>

<!-- SYSTEM 2 -->
<div style="text-align: center; padding: 10px; border-right: 1px solid rgba(255,255,255,0.05);">
<div style="font-size: 0.75rem; color: #64748B; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;">AI Core</div>
<div style="font-weight: 700; color: {model_color}; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
<span style="height: 8px; width: 8px; background-color: {model_color}; border-radius: 50%; box-shadow: 0 0 10px {model_color};"></span>
{model_status}
</div>
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">{model_msg}</div>
</div>

<!-- SYSTEM 3 -->
<div style="text-align: center; padding: 10px;">
<div style="font-size: 0.75rem; color: #64748B; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;">Macro Feeds</div>
<div style="font-weight: 700; color: {macro_color}; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
<span style="height: 8px; width: 8px; background-color: {macro_color}; border-radius: 50%; box-shadow: 0 0 10px {macro_color};"></span>
{macro_status}
</div>
<div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">{macro_msg}</div>
</div>
</div>
""", unsafe_allow_html=True)

# st.markdown("---")

# --- NAVIGATION GRID ---
st.markdown("### 🚀 Module Workflow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
<div class="premium-card">
<div style="font-size: 2rem; margin-bottom: 15px;">🛡️</div>
<h3>Data Quality</h3>
<p>Pre-flight check for your datasets. Validate schema, detect anomalies, and auto-correct data issues.</p>
<a href="/Data_Quality" target="_self">
<button style="
background: rgba(59, 130, 246, 0.1); 
color: #60A5FA; 
border: 1px solid rgba(59, 130, 246, 0.3); 
padding: 8px 16px; 
border-radius: 8px; 
cursor: pointer; 
width: 100%;
font-weight: 600;
transition: all 0.2s;
">Launch Validator &rarr;</button>
</a>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="premium-card">
<div style="font-size: 2rem; margin-bottom: 15px;">⚡</div>
<h3>Forecasting Engine</h3>
<p>The command center. Train ensemble models (XGBoost, Prophet) and generate primary revenue projections.</p>
<a href="/Dashboard" target="_self">
<button style="
background: linear-gradient(135deg, #3B82F6, #2563EB); 
color: white; 
border: none; 
padding: 8px 16px; 
border-radius: 8px; 
cursor: pointer; 
width: 100%;
font-weight: 600;
box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
">Open Dashboard &rarr;</button>
</a>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="premium-card">
<div style="font-size: 2rem; margin-bottom: 15px;">🧠</div>
<h3>Model Lab</h3>
<p>Explainable AI. Understand feature importance (SHAP) and fine-tune model weights manually.</p>
<a href="/Model_Lab" target="_self">
<button style="
background: rgba(139, 92, 246, 0.1); 
color: #A78BFA; 
border: 1px solid rgba(139, 92, 246, 0.3); 
padding: 8px 16px; 
border-radius: 8px; 
cursor: pointer; 
width: 100%;
font-weight: 600;
">Enter Lab &rarr;</button>
</a>
</div>
""", unsafe_allow_html=True)

with col4:
    st.markdown("""
<div class="premium-card">
<div style="font-size: 2rem; margin-bottom: 15px;">🌪️</div>
<h3>Scenario Lab</h3>
<p>Stress Test. Simulate economic shocks (Crisis, Boom) and visualize probability distributions.</p>
<a href="/Scenario_Lab" target="_self">
<button style="
background: rgba(245, 158, 11, 0.1); 
color: #FBBF24; 
border: 1px solid rgba(245, 158, 11, 0.3); 
padding: 8px 16px; 
border-radius: 8px; 
cursor: pointer; 
width: 100%;
font-weight: 600;
">Run Simulation &rarr;</button>
</a>
</div>
""", unsafe_allow_html=True)


# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.8rem; margin-top: 20px;">
<p>&copy; 2025 Fasya_Dev | TaxForecaster</p>
<p>Powered by XGBoost, Prophet, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
