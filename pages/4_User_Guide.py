"""
User Guide page - Interactive tutorial + Documentation reference.
"""

import streamlit as st
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import importlib
    import onboarding
    importlib.reload(onboarding)
    from onboarding import OnboardingWizard
    
    from theme_manager import ThemeManager
    import style
    from model_info import MODEL_INFO
except ImportError:
    st.error("❌ Required modules not found.")
    st.stop()

# Page config
st.set_page_config(
    page_title="User Guide | TaxForecaster",
    page_icon="📚",
    layout="wide"
)

# Apply theme
style.apply_theme()
ThemeManager.apply_theme()

# Page header
st.title("📚 TaxForecaster User Guide & Documentation")
st.markdown("Interactive tutorial and comprehensive documentation for TaxForecaster 2.0")
st.markdown("---")

# Main tabs: Onboarding + Documentation + Quick Tips
main_tabs = st.tabs([
    "🎓 Interactive Tutorial",
    "🌍 Macro Data",
    "🤖 Algorithms",
    "📖 Glossary",
    "❓ FAQ",
    "💡 Quick Tips",
    "🎨 Theme"
])

# ==========================================
# TAB 1: INTERACTIVE TUTORIAL (Onboarding)
# ==========================================
with main_tabs[0]:
    st.markdown("### 🎓 Step-by-Step Interactive Tutorial")
    st.info("👋 New to TaxForecaster? Follow this guided tutorial to learn all the key features!")
    
    st.markdown("---")
    
    # Show onboarding wizard
    OnboardingWizard.render(key_prefix="user_guide")

# ==========================================
# TAB 2: MACRO DATA
# ==========================================
with main_tabs[1]:
    st.markdown("### 🌍 Macroeconomic Data Sources")
    st.markdown("Real-time values and specifications for the 15 core indicators used in regression models.")
    
    # Try to load latest values
    latest_vals = {}
    try:
        if os.path.exists("macro_data_auto.csv"):
            df_macro = pd.read_csv("macro_data_auto.csv")
            if not df_macro.empty:
                last_row = df_macro.iloc[-1]
                latest_vals = last_row.to_dict()
    except Exception:
        pass
        
    def get_val(key, fmt="{:,.2f}"):
        v = latest_vals.get(key, 0)
        return fmt.format(v)

    st.markdown(f"""
    #### 1. Macro Core (Foundation)
    | Indikator | Ticker | Last Value | Source | Impact |
    | :--- | :--- | :--- | :--- | :--- |
    | **GDP Growth** | `NY.GDP.MKTP.KD.ZG` | **{get_val('Pertumbuhan_Ekonomi', '{:.2f}%')}** | World Bank | Positive impact on all taxes. |
    | **Inflation (CPI)** | `FP.CPI.TOTL.ZG` | **{get_val('Inflasi', '{:.2f}%')}** | World Bank | Increases PPN nominals, may reduce volume. |
    | **Household Cons.** | `NE.CON.PRVT.ZS` | **{get_val('Konsumsi_RT_Growth', '{:.2f}%')}** | World Bank | Major driver of PPN Dalam Negeri. |
    | **Exports** | `NE.EXP.GNFS.ZS` | **{get_val('Ekspor_Growth', '{:.2f}%')}** | World Bank | Drives PPh Badan (Exporters) & Bea Keluar. |
    | **Imports** | `NE.IMP.GNFS.ZS` | **{get_val('Impor_Growth', '{:.2f}%')}** | World Bank | Drives PPN Impor & Bea Masuk. |
    | **PMI Manufaktur** | *Survey* | **{get_val('PMI_Manufaktur', '{:.1f}')}** | S&P Global | Business sentiment (>50 Expansion). Drives PPN & PPh. |

    #### 2. Financial Markets
    | Indikator | Ticker | Last Value | Source | Impact |
    | :--- | :--- | :--- | :--- | :--- |
    | **USD/IDR** | `IDR=X` | **{get_val('Kurs_USD', 'Rp {:,.0f}')}** | Yahoo | Weak IDR increases import tax & oil tax revenue. |
    | **IHSG (JCI)** | `^JKSE` | **{get_val('IHSG', '{:,.0f}')}** | Yahoo | Proxy for business sentiment & investment tax. |
    | **BI Rate** | *Sim* | **{get_val('BI_Rate', '{:.2f}%')}** | Bank Ind | High rates increase cost of funds, lowering corporate profit (PPh). |
    | **SBN 10Y** | `ID10YT=RR` | **{get_val('SBN_10Y', '{:.2f}%')}** | Yahoo/BI | Benchmark yield. Affects government cost of funds. |

    #### 3. Commodities & Energy
    | Indikator | Ticker | Last Value | Source | Impact |
    | :--- | :--- | :--- | :--- | :--- |
    | **Oil Price (ICP)** | `CL=F` | **{get_val('Harga_Minyak_ICP', '${:,.2f}')}** | Yahoo | Critical for PPh Migas & PNBP. |
    | **Coal Price** | `MTF=F` | **{get_val('Harga_Batubara', '${:,.2f}')}** | Yahoo | Major driver for Mining Corporate Tax. |
    | **CPO Price** | `PO=F` | **{get_val('Harga_CPO', '${:,.2f}')}** | Yahoo | Drives Palm Oil Export Levies. |
    | **Lifting Minyak** | *APBN* | **{get_val('Lifting_Minyak', '{:,.0f} bph')}** | ESDM | Volume base for PPh Migas (Oil). |
    | **Lifting Gas** | *APBN* | **{get_val('Lifting_Gas', '{:,.0f} boepd')}** | ESDM | Volume base for PPh Migas (Gas). |
    """)

# ==========================================
# TAB 3: ALGORITHMS
# ==========================================
with main_tabs[2]:
    st.markdown("### 🤖 Model Catalog")
    st.markdown("Technical details of the forecasting algorithms used in TaxForecaster.")
    
    models = sorted(MODEL_INFO.keys())
    
    for m_name in models:
        info = MODEL_INFO[m_name]
        with st.expander(f"📌 {m_name}", expanded=False):
            st.markdown(f"**{info['description']}**")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("##### ⚙️ Mechanism")
                st.write(info['mechanism'])
                st.markdown("##### 🎯 Best Use Case")
                st.info(info['suitable_for'])
            with c2:
                st.markdown("##### ✅ Pros")
                for p in info['pros']: st.write(f"• {p}")
                st.markdown("##### ⚠️ Cons")
                for c in info['cons']: st.write(f"• {c}")

# ==========================================
# TAB 4: GLOSSARY
# ==========================================
with main_tabs[3]:
    st.markdown("### 📖 Terminology Glossary")
    st.markdown("""
    | Term | Definition |
    | :--- | :--- |
    | **RMSE** | Root Mean Squared Error. The standard deviation of the prediction errors. Lower is better. |
    | **MAPE** | Mean Absolute Percentage Error. Accuracy metric representing average % deviation from actual. |
    | **Buoyancy** | A measure of tax system efficiency. `Tax Growth % / GDP Growth %`. Value > 1 indicates a highly responsive tax system. |
    | **Seasonality** | Recurring fluctuations in data that happen at regular intervals (e.g., tax spike in March/April). |
    | **Confidence Interval** | The range (95%) within which the actual value is expected to fall. |
    | **Drift** | A significant change in the statistical properties of the data (e.g., post-COVID macro patterns vs pre-COVID). |
    | **Ensemble** | Combining multiple models to create a more robust forecast. |
    | **SHAP** | SHapley Additive exPlanations - method to explain individual predictions. |
    | **Hyperparameter** | Model configuration settings (e.g., learning rate, number of trees). |
    | **Cross-Validation** | Testing technique that uses different subsets of data for training vs validation. |
    """)

# ==========================================
# TAB 5: FAQ
# ==========================================
with main_tabs[4]:
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("**Q: Why is my forecast showing a flat line?**"):
        st.markdown("""
        This usually happens if:
        - The model fails to find a signal in the data
        - Data is too limited (< 24 months)
        - Macro data is not loaded correctly
        
        **Solutions:**
        - Check if macro data file exists
        - Try a different model (Prophet, XGBoost)
        - Ensure sufficient historical data
        """)
    
    with st.expander("**Q: How do I update the macro data?**"):
        st.markdown("""
        The system automatically reads from `macro_data_auto.csv`. 
        
        To update:
        1. Run the data fetch script (if available)
        2. Or manually update the CSV file
        3. Restart the application to reload data
        """)
    
    with st.expander("**Q: Can I add a new Tax Type?**"):
        st.markdown("""
        Yes! Simply:
        1. Go to **Dashboard** page
        2. Upload a CSV with the new tax type in `Jenis Pajak` column
        3. System will detect it automatically
        4. Train models to include the new tax type
        """)
    
    with st.expander("**Q: How accurate are the forecasts?**"):
        st.markdown("""
        Accuracy varies by tax type and model:
        - **Typical MAPE**: 5-15% for stable tax types
        - **Best performers**: PPh, PPN (low volatility)
        - **Challenging**: Migas taxes (commodity-dependent)
        
        Check Model Lab page for detailed accuracy metrics.
        """)
    
    with st.expander("**Q: What does the Ensemble model do?**"):
        st.markdown("""
        The Ensemble model:
        - Combines predictions from top 3-5 models
        - Weights based on validation accuracy
        - Generally provides **most robust forecasts**
        - Reduces impact of individual model errors
        """)
    
    with st.expander("**Q: How do I export forecasts?**"):
        st.markdown("""
        Multiple export options available:
        1. **Dashboard**: Download CSV, Excel, or HTML reports
        2. **Executive Summary**: Generate comprehensive reports
        3. **Model Lab**: Export model performance metrics
        
        All exports include timestamps and metadata.
        """)

# ==========================================
# ==========================================
# TAB 6: QUICK TIPS
# ==========================================
with main_tabs[5]:
    st.markdown("### 💡 Quick Tips & Best Practices")
    st.markdown("Helpful shortcuts and recommendations for using TaxForecaster effectively.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Getting Started")
        st.info("""
        **Step-by-Step Workflow:**
        
        1. **Upload Data**
           - Go to Dashboard page
           - Upload tax history CSV file
           - Format: Tanggal, Jenis Pajak, Nominal
        
        2. **Validate Quality**
           - Check Data Quality page
           - Ensure quality score > 90%
           - Fix any schema errors
        
        3. **Train Models**
           - Click "Train Models" button
           - Wait for training completion
           - Review accuracy metrics
        
        4. **View Forecasts**
           - Explore forecast charts
           - Check confidence intervals
           - Review trend analysis
        
        5. **Export Reports**
           - Download in CSV, Excel, or HTML
           - Share with stakeholders
        """)
        
        st.markdown("#### 🎯 Model Selection Tips")
        st.success("""
        **When to use each model:**
        
        - **Ensemble**: Best for most cases (combines top models)
        - **Prophet**: Strong seasonal patterns
        - **XGBoost**: Complex non-linear relationships
        - **LSTM**: Long-term dependencies
        - **ARIMAX**: Clear trends with macro factors
        """)
    
    with col2:
        st.markdown("#### ✅ Best Practices")
        st.warning("""
        **For Accurate Forecasts:**
        
        - ✅ Use **minimum 24 months** of historical data
        - ✅ Check **Data Quality** page before training
        - ✅ Review **SHAP explanations** for insights
        - ✅ Test **multiple scenarios** for risk assessment
        - ✅ Use **Ensemble model** for robustness
        - ✅ Validate **macro data** is up-to-date
        - ✅ Export reports with **confidence intervals**
        - ✅ Monitor **model performance** regularly
        """)
        
        st.markdown("#### ⚡ Performance Tips")
        st.info("""
        **Speed up forecasting:**
        
        - Reduce hyperparameter search trials
        - Focus on specific tax types
        - Use cached models when available
        - Clear browser cache if slow
        - Train during off-peak hours
        """)
        
        st.markdown("#### 🔍 Troubleshooting")
        st.error("""
        **Common Issues:**
        
        - **Flat forecasts**: Check macro data loading
        - **Training fails**: Verify data format & completeness
        - **Slow performance**: Reduce trials or clear cache
        - **Theme not changing**: Hard refresh browser
        """)

# ==========================================
# TAB 7: THEME
# ==========================================
with main_tabs[6]:
    st.markdown("### 🎨 Theme Customization")
    st.markdown("Personalize your TaxForecaster experience.")
    
    st.info("Select a color theme below. The application will automatically reload with your new style.")
    
    # We need to manually render the selector here instead of using the sidebar one
    current_theme = ThemeManager.get_current_theme()
    theme_options = {v['name']: k for k, v in ThemeManager.THEMES.items()}
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_theme_name = st.selectbox(
            "Choose Theme",
            options=list(theme_options.keys()),
            index=list(theme_options.values()).index(current_theme) if current_theme in theme_options.values() else 0,
            key='main_theme_selector'
        )
    
        if theme_options[selected_theme_name] != current_theme:
            ThemeManager.set_theme(theme_options[selected_theme_name])
            st.rerun()
            
    # Preview color palettes
    st.markdown("#### Theme Preview")
    palettes = {
        'Dark': ['#0E1117', '#262730', '#FF4B4B'],
        'Light': ['#FFFFFF', '#F0F2F6', '#FF4B4B'],
        'Ocean': ['#0F1F2C', '#1E3A52', '#00B4D8'],
        'Forest': ['#0D1F12', '#1B3A24', '#2D8A4E'],
        'Sunset': ['#1F110D', '#3A2018', '#E05D3D']
    }
    
    cols = st.columns(len(palettes))
    for i, (name, colors) in enumerate(palettes.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            for c in colors:
                st.markdown(f'<div style="background-color:{c};height:20px;border-radius:4px;margin-bottom:4px;"></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("TaxForecaster | by fasya_dev")
st.caption("📚 Complete documentation and interactive tutorial")
