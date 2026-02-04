import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import style
from forecaster import TaxForecaster

# --- Page Config ---
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Model Lab | TaxForecaster", layout="wide", page_icon="🧪")
style.apply_theme()
sns.set_theme(style="whitegrid", context="notebook", palette="mako")

st.title("🧪 Model Laboratory")
st.markdown("Advanced diagnostics, explainability, and experimental A/B testing.")

# --- Session State for Shared Forecaster ---
if 'forecaster_v6' not in st.session_state or st.session_state['forecaster_v6'] is None:
    st.info("ℹ️ Please run a forecast in the **Dashboard** first to populate this page.")
    st.stop()
    
fc = st.session_state['forecaster_v6']
if not fc.is_fitted:
    st.warning("⚠️ Model is not fitted. Please train it in the Dashboard.")
    st.stop()

# --- Select Tax Type ---
tax_options = fc.df['Jenis Pajak'].unique()
selected_tax = st.selectbox("Select Tax Type to Analyze", tax_options)

# --- Check Advanced Capabilities ---
with st.expander("🔌 Backend Capabilities & Installed Engines", expanded=False):
    cap_cols = st.columns(4)
    
    # We check by trying imports or inspecting the forecaster module if possible
    # A safer way is to check the trained models dict to see if they ever appear
    
    has_xgb = 'XGBoost' in str(fc.trained_models)
    has_lstm = 'LSTM' in str(fc.trained_models)
    has_prophet = 'Prophet' in str(fc.trained_models)
    
    cap_cols[0].metric("XGBoost", "Active" if has_xgb else "Inactive", delta="Installed" if has_xgb else "Off", delta_color="normal")
    cap_cols[1].metric("Deep Learning (LSTM)", "Active" if has_lstm else "Inactive", delta="TensorFlow" if has_lstm else "No TF", delta_color="normal")
    cap_cols[2].metric("Prophet", "Active" if has_prophet else "Inactive", delta="Meta" if has_prophet else "Off", delta_color="normal")
    cap_cols[3].metric("Sklearn", "Active", delta="Core", delta_color="normal")

# --- Tabs ---
tab_explain, tab_ab, tab_sens, tab_drift = st.tabs(["🧠 Explainability (SHAP)", "⚖️ A/B Testing & Weights", "🌪️ Sensitivity Analysis", "📉 Drift Detection"])

with tab_explain:
    st.subheader(f"Why did the model predict this for {selected_tax}?")
    
    # Run Explain
    with st.spinner("Calculating SHAP values..."):
        shap_res = fc.explain_model(selected_tax)
        
    if shap_res and shap_res[0] is not None:
        shap_values, explainer, feature_names = shap_res
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Feature Importance (Summary Plot)")
            st.caption("Variables sorted by impact on the forecast.")
            
            # Styled SHAP Summary
            fig_summary, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, color="#3498DB")
            ax.set_xlabel("Average Absolute SHAP Value (Impact)")
            st.pyplot(fig_summary)
            
        with c2:
            st.markdown("#### Detailed Impact (Beeswarm)")
            st.caption("Color = Feature Value (Red: High, Blue: Low). X-Axis = Impact (+/-).")
            try:
                fig_swarm, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, feature_names=feature_names, show=False)
                st.pyplot(fig_swarm)
            except:
                st.info("Beeswarm plot unavailable for this model type.")
    else:
        st.info("Explainability is not available for the selected model (likely a Time-Series model like ARIMA/Prophet which doesn't support SHAP).")

with tab_ab:
    st.subheader("A/B Testing & Custom Weights")
    st.markdown("Compare the **Auto-Selected** model vs a **Custom Weighted** strategy.")
    
    # 1. Get Models
    model_group = fc.trained_models[selected_tax]
    avail_models = list(model_group['candidates'].keys())
    
    # 2. Controls
    c_list = st.columns(3)
    weights = {}
    
    with st.expander("⚙️ Configure Custom Weights", expanded=True):
        st.write("Assign weights to models (Sum doesn't need to be 100, we normalize).")
        cols = st.columns(3) # Fixed 3 Columns for clarity
        for i, m_name in enumerate(avail_models):
            with cols[i % 3]:
                weights[m_name] = st.slider(f"{m_name}", 0.0, 1.0, 0.0, key=f"w_{m_name}")
    
    # 3. Simulate
    if sum(weights.values()) > 0:
        # Run Predict A (Auto)
        res_a = fc.predict(forecast_periods=12, model_strategy='auto')
        # Extract just this tax
        df_a = [r['data'] for r in res_a if r['tax_type'] == selected_tax][0]
        model_name_a = [r['model'] for r in res_a if r['tax_type'] == selected_tax][0]
        
        # Run Predict B (Custom)
        res_b = fc.predict(forecast_periods=12, custom_weights=weights)
        df_b = [r['data'] for r in res_b if r['tax_type'] == selected_tax][0]
        
        # Plot Comparison
        st.markdown("#### Forecast Comparison")
        
        chart_data = pd.DataFrame({
            'Date': df_a['Tanggal'],
            f'Auto ({model_name_a})': df_a['Nominal (Milyar)'],
            'Custom Weighted': df_b['Nominal (Milyar)']
        }).set_index('Date')
        
        st.line_chart(chart_data)
        
        # Diff Metrics
        total_a = df_a['Nominal (Milyar)'].sum()
        total_b = df_b['Nominal (Milyar)'].sum()
        diff = total_b - total_a
        
        met1, met2, met3 = st.columns(3)
        met1.metric("Auto Forecast (Total)", f"Rp {total_a:,.0f} B")
        met2.metric("Custom Forecast (Total)", f"Rp {total_b:,.0f} B")
        met3.metric("Difference", f"Rp {diff:,.0f} B", delta=diff)
        
    else:
        st.info("Adjust sliders above to create a custom ensemble.")

with tab_sens:
    st.subheader("🌪️ Macro Sensitivity (Interactive Funnel)")
    st.info(f"Analyzing how each macroeconomic indicator affects the **Total {selected_tax} Forecast** (12 Months). We simulate a +/- 10% shock to each variable.")
    
    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Simulating shocks..."):
            # 1. Baseline
            if fc.macro_df is None:
                st.error("Macro data missing.")
                st.stop()
                
            base_macro = fc.macro_df.iloc[-1] # Last values extended
            
            # Helper to build simple constant future df
            def make_future(macro_row):
                dates = pd.date_range(start=fc.df['Tanggal'].max() + pd.DateOffset(months=1), periods=12, freq='ME')
                d = [macro_row] * 12
                df = pd.DataFrame(d)
                df['Tanggal'] = dates
                return df
            
            # Run Base
            res_base = fc.predict(12, custom_macro_future=make_future(base_macro), model_strategy='auto')
            base_total = [r['data'] for r in res_base if r['tax_type'] == selected_tax][0]['Nominal (Milyar)'].sum()
            
            # 2. Iterate Features
            macro_cols = [c for c in base_macro.index if fc.macro_df[c].dtype in ['float64', 'int64'] and c not in ['Year', 'Month']]
            
            impacts = []
            
            for col in macro_cols:
                # Shock Up (+10%)
                row_up = base_macro.copy()
                row_up[col] *= 1.10
                res_up = fc.predict(12, custom_macro_future=make_future(row_up), model_strategy='auto')
                total_up = [r['data'] for r in res_up if r['tax_type'] == selected_tax][0]['Nominal (Milyar)'].sum()
                
                # Shock Down (-10%)
                row_down = base_macro.copy()
                row_down[col] *= 0.90
                res_down = fc.predict(12, custom_macro_future=make_future(row_down), model_strategy='auto')
                total_down = [r['data'] for r in res_down if r['tax_type'] == selected_tax][0]['Nominal (Milyar)'].sum()
                
                # Calculate Delta
                delta_up = total_up - base_total
                delta_down = total_down - base_total
                
                impacts.append({
                    'Feature': col,
                    'Positive Shock (+10%)': delta_up,
                    'Negative Shock (-10%)': delta_down,
                    'Range': abs(delta_up - delta_down)
                })
            
            # 3. Visualize (Plotly Funnel)
            df_imp = pd.DataFrame(impacts).sort_values('Range', ascending=True)
            
            # Prepare data for Funnel/Bar Chart (Diverging)
            # Flatten for Plotly
            plot_data = []
            for _, row in df_imp.iterrows():
                plot_data.append({'Feature': row['Feature'], 'Impact': row['Positive Shock (+10%)'], 'Type': '+10% Shock'})
                plot_data.append({'Feature': row['Feature'], 'Impact': row['Negative Shock (-10%)'], 'Type': '-10% Shock'})
            
            df_plot = pd.DataFrame(plot_data)
            
            st.markdown("##### Revenue Impact (Milyar Rp)")
            
            fig_funnel = px.bar(
                df_plot, 
                y='Feature', 
                x='Impact', 
                color='Type',
                orientation='h',
                title="Sensitivity Analysis (Tornado Plot)",
                color_discrete_map={'+10% Shock': '#2ECC71', '-10% Shock': '#E74C3C'}
            )
            
            fig_funnel.update_layout(
                barmode='overlay',
                xaxis_title="Change in Revenue Forecast (Milyar Rp)",
                yaxis_title="Macro Indicator",
                height=500,
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Table
            with st.expander("View Detailed Sensitivity Data"):
                st.dataframe(df_imp.set_index('Feature').sort_values('Range', ascending=False).style.format("{:,.0f}"))


with tab_drift:
    st.subheader("📉 Data Drift Detection (KDE Analysis)")
    st.markdown("Check if the macroeconomic assumptions for the future are significantly different from history.")
    
    if st.button("Run Drift Check"):
        # We need future macro data. For now, let's extrapolate simple
        if fc.macro_df is not None:
             # Create dummy future macro
            future_dates = pd.date_range(start=fc.df['Tanggal'].max(), periods=12, freq='ME')
            last_row = fc.macro_df.iloc[-1]
            # Create a dataframe repeated
            future_macro = pd.DataFrame([last_row] * 12)
            # Add some noise to simulate "Different future"
            import numpy as np
            for c in future_macro.columns:
                if future_macro[c].dtype in ['float64', 'int64']:
                    future_macro[c] = future_macro[c] * np.random.uniform(0.9, 1.1, size=12)
            
            # Run Check
            drift_res, has_drift = fc.check_drift(selected_tax, future_macro)
            
            if drift_res:
                if has_drift:
                    st.error("🚨 Significant Data Drift Detected!", icon="⚠️")
                else:
                    st.success("✅ No significant drift detected.")
                
                # Visual Drift Inspection (Seaborn KDE)
                st.markdown("#### Distribution Overlap (History vs Future)")
                drift_cols = st.columns(2)
                
                # Pick top 2 most drifted features (lowest p-value)
                sorted_drift = sorted(drift_res.items(), key=lambda x: x[1]['p_value'])
                top_features = [x[0] for x in sorted_drift[:4]] # Take top 4
                
                # History vs Future Data
                hist_macro = fc.macro_df
                
                for i, feat in enumerate(top_features):
                    with drift_cols[i % 2]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.kdeplot(hist_macro[feat], label='History', fill=True, color="blue", alpha=0.3, ax=ax)
                        sns.kdeplot(future_macro[feat], label='Future (Simulated)', fill=True, color="orange", alpha=0.3, ax=ax)
                        ax.set_title(f"{feat} (p={drift_res[feat]['p_value']:.4f})")
                        ax.legend()
                        sns.despine()
                        st.pyplot(fig)
                
            else:
                st.warning("Could not calculate drift (missing macro data).")
