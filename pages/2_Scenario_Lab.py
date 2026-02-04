import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forecaster import TaxForecaster
from fetch_macro import fetch_macro_data
import style

# --- Page Config ---
st.set_page_config(page_title="Scenario Lab | TaxForecaster 2.0", layout="wide", page_icon="🧪")
style.apply_theme()
sns.set_theme(style="white", context="talk")

st.title("🧪 Scenario Lab")
st.markdown("### Advanced Simulation & Comparison Engine")

# --- Initialize Session State for Model ---
if 'forecaster_v6' not in st.session_state:
    st.session_state['forecaster_v6'] = None

# --- Helper: Visual Comparison ---
def plot_comparison(history_df, sc1_df, sc2_df, label1, label2):
    fig = go.Figure()
    
    # 1. History
    hist_agg = history_df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=hist_agg['Tanggal'], y=hist_agg['Nominal (Milyar)'],
        mode='lines', name='Realisasi (History)',
        line=dict(color='#BDC3C7', width=2),
        fill='tozeroy',
        fillcolor='rgba(189, 195, 199, 0.1)' # Light Grey Fill
    ))
    
    # 2. Scenario 1 (Neon Blue)
    sc1_agg = sc1_df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=sc1_agg['Tanggal'], y=sc1_agg['Nominal (Milyar)'],
        mode='lines+markers', name=label1,
        line=dict(color='#00d2ff', width=3, dash='solid'),
        marker=dict(size=6, color='#00d2ff', line=dict(width=1, color='white'))
    ))
    
    # 3. Scenario 2 (Neon Orange)
    sc2_agg = sc2_df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=sc2_agg['Tanggal'], y=sc2_agg['Nominal (Milyar)'],
        mode='lines+markers', name=label2,
        line=dict(color='#ff9f43', width=3, dash='solid'),
        marker=dict(size=6, color='#ff9f43', line=dict(width=1, color='white'))
    ))
    
    fig.update_layout(
        title="Comparison: Total Tax Revenue Forecast",
        yaxis_title="Total (Milyar Rupiah)",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    return fig

# --- SIDEBAR: Scenario Config ---
with st.sidebar:
    st.subheader("⚙️ Scenario Config")
    
    # Data Check
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    history_path = os.path.join(root_dir, 'tax_history.csv')
    macro_path = os.path.join(root_dir, 'macro_data_auto.csv')
    
    if not os.path.exists(history_path):
        st.error("⚠️ Data History not found. Please upload in Dashboard first.")
        st.stop()
        
    # Load Defaults (Full 15 Indicators)
    defs = {
        'Inflasi': 3.0, 'Pertumbuhan_Ekonomi': 5.0, 'Kurs_USD': 15000.0, 'Harga_Minyak_ICP': 80.0,
        'SBN_10Y': 6.6, 'Lifting_Minyak': 605.0, 'Lifting_Gas': 1005.0,
        'IHSG': 7300.0, 'BI_Rate': 6.25, 'Harga_Batubara': 145.0, 'Harga_CPO': 820.0,
        'Ekspor_Growth': 5.0, 'Impor_Growth': 4.0, 'Konsumsi_RT_Growth': 4.9, 'PMI_Manufaktur': 51.0
    }
    
    if os.path.exists(macro_path):
        try:
            m_df = pd.read_csv(macro_path)
            last_3 = m_df.tail(3).mean(numeric_only=True)
            for k in defs.keys():
                # Allow fallback alias for price commodity
                val = last_3.get(k)
                if pd.isna(val):
                    if k == 'Harga_Minyak_ICP': val = last_3.get('Harga_Komoditas', 80.0)
                if not pd.isna(val):
                    defs[k] = float(val)
        except: pass

    st.info("Define two scenarios to compare impacts.")
    
    with st.expander("📝 Dataset & Horizon", expanded=False):
        forecast_months = st.slider("Forecast Horizon", 6, 24, 12, format="%d Months")

    st.markdown("---")
    
    # --- PRESETS / STRESS TEST ---
    st.caption("🚀 One-Click Stress Test (Presets)")
    # Keys tracking
    keys = list(defs.keys())
    
    if st.button("Normal", use_container_width=True):
        for k in keys:
            st.session_state[f'a_{k}'] = defs[k]
            st.session_state[f'b_{k}'] = defs[k]
        
    if st.button("Optimis", use_container_width=True):
        # A = Normal
        for k in keys: st.session_state[f'a_{k}'] = defs[k]
        
        # B = Optimistic
        st.session_state['b_Inflasi'] = defs['Inflasi']
        st.session_state['b_Pertumbuhan_Ekonomi'] = defs['Pertumbuhan_Ekonomi'] + 1.2
        st.session_state['b_Kurs_USD'] = defs['Kurs_USD'] - 500 # Stronger
        st.session_state['b_Harga_Minyak_ICP'] = defs['Harga_Minyak_ICP'] + 5
        st.session_state['b_IHSG'] = defs['IHSG'] * 1.05
        st.session_state['b_SBN_10Y'] = defs['SBN_10Y'] - 0.5
        st.session_state['b_PMI_Manufaktur'] = 53.0
        st.session_state['b_Harga_CPO'] = defs['Harga_CPO'] * 1.1
        # Others same
        for k in keys:
            if f'b_{k}' not in st.session_state: st.session_state[f'b_{k}'] = defs[k]

    if st.button("Krisis", use_container_width=True):
        # A = Normal
        for k in keys: st.session_state[f'a_{k}'] = defs[k]
        
        # B = Crisis
        st.session_state['b_Inflasi'] = defs['Inflasi'] + 4.0
        st.session_state['b_Pertumbuhan_Ekonomi'] = defs['Pertumbuhan_Ekonomi'] - 2.5
        st.session_state['b_Kurs_USD'] = defs['Kurs_USD'] + 1500
        st.session_state['b_Harga_Minyak_ICP'] = defs['Harga_Minyak_ICP'] + 20
        st.session_state['b_IHSG'] = defs['IHSG'] * 0.8
        st.session_state['b_SBN_10Y'] = defs['SBN_10Y'] + 2.0
        st.session_state['b_PMI_Manufaktur'] = 45.0
        st.session_state['b_Harga_CPO'] = defs['Harga_CPO'] * 0.8
         # Others same
        for k in keys:
            if f'b_{k}' not in st.session_state: st.session_state[f'b_{k}'] = defs[k]
    
    # Initialize Session State if empty
    for k in keys:
        if f'a_{k}' not in st.session_state: st.session_state[f'a_{k}'] = defs[k]
        if f'b_{k}' not in st.session_state: st.session_state[f'b_{k}'] = defs[k]

    # --- INPUT WIDGETS ---
    def render_scenario_inputs(prefix, title):
        st.markdown(f"#### {title}")
        with st.expander(f"⚙️ Config {title}", expanded=False):
            st.caption("🏛️ Asumsi APBN")
            c1, c2 = st.columns(2)
            st.session_state[f'{prefix}_Pertumbuhan_Ekonomi'] = c1.number_input("GDP (%)", value=st.session_state[f'{prefix}_Pertumbuhan_Ekonomi'], key=f"in_{prefix}_gdp", step=0.1)
            st.session_state[f'{prefix}_Inflasi'] = c2.number_input("Inflasi (%)", value=st.session_state[f'{prefix}_Inflasi'], key=f"in_{prefix}_inf", step=0.1)
            
            c3, c4 = st.columns(2)
            st.session_state[f'{prefix}_Kurs_USD'] = c3.number_input("Kurs (IDR)", value=st.session_state[f'{prefix}_Kurs_USD'], key=f"in_{prefix}_kurs", step=100.0)
            st.session_state[f'{prefix}_SBN_10Y'] = c4.number_input("SBN 10Y (%)", value=st.session_state[f'{prefix}_SBN_10Y'], key=f"in_{prefix}_sbn", step=0.1)
            
            st.caption("🛢️ Energi & Komoditas")
            c5, c6 = st.columns(2)
            st.session_state[f'{prefix}_Harga_Minyak_ICP'] = c5.number_input("ICP ($)", value=st.session_state[f'{prefix}_Harga_Minyak_ICP'], key=f"in_{prefix}_oil", step=1.0)
            st.session_state[f'{prefix}_Harga_Batubara'] = c6.number_input("Coal ($)", value=st.session_state[f'{prefix}_Harga_Batubara'], key=f"in_{prefix}_coal", step=5.0)
            
            c7, c8 = st.columns(2)
            st.session_state[f'{prefix}_Harga_CPO'] = c7.number_input("CPO ($)", value=st.session_state[f'{prefix}_Harga_CPO'], key=f"in_{prefix}_cpo", step=10.0)
            st.session_state[f'{prefix}_Lifting_Minyak'] = c8.number_input("Lift Oil (bph)", value=st.session_state[f'{prefix}_Lifting_Minyak'], key=f"in_{prefix}_loil", step=10.0)
            
            st.caption("🏦 Finansial & Riil")
            c9, c10 = st.columns(2)
            st.session_state[f'{prefix}_IHSG'] = c9.number_input("IHSG", value=st.session_state[f'{prefix}_IHSG'], key=f"in_{prefix}_ihsg", step=50.0)
            st.session_state[f'{prefix}_BI_Rate'] = c10.number_input("BI Rate (%)", value=st.session_state[f'{prefix}_BI_Rate'], key=f"in_{prefix}_bi", step=0.25)
            
            st.session_state[f'{prefix}_PMI_Manufaktur'] = st.number_input("PMI Manuf", value=st.session_state[f'{prefix}_PMI_Manufaktur'], key=f"in_{prefix}_pmi", step=0.5)

    # Note: We act directly on session state via callback in number_input inputs doesn't work easily here, 
    # so we read the widget 'key' and update the main state manually or just let widget manage it.
    # To keep it simple, we just use the keys from session state initialization.
    
    # We need a different approach: The Presets update 'session_state', but widgets need to reflect that.
    # We use 'key' in widgets linked to session_state variable if possible, or manual.
    # Streamlit widgets sync with session_state if key matches. 
    
    # Correct Pattern:
    # 1. Preset button fixes session_state['a_Inflasi'].
    # 2. visual component: number_input(key='a_Inflasi') -> Auto syncs!
    
    # Let's rebuild the inputs properly mapped:
    
    def render_input_section(prefix, title):
        st.markdown(f"#### {title}")
        with st.expander(f"⚙️ Config {title}", expanded=False):
            # We map specific keys for widget binding
            c1, c2 = st.columns(2)
            st.number_input("GDP (%)", key=f"{prefix}_Pertumbuhan_Ekonomi", step=0.1)
            st.number_input("Inflasi (%)", key=f"{prefix}_Inflasi", step=0.1)
            
            c3, c4 = st.columns(2)
            st.number_input("Kurs (IDR)", key=f"{prefix}_Kurs_USD", step=100.0)
            st.number_input("SBN 10Y (%)", key=f"{prefix}_SBN_10Y", step=0.1)
            
            st.caption("🛢️ Energi & Komoditas")
            c5, c6 = st.columns(2)
            st.number_input("ICP ($)", key=f"{prefix}_Harga_Minyak_ICP", step=1.0)
            st.number_input("Coal ($)", key=f"{prefix}_Harga_Batubara", step=5.0)
            
            c7, c8 = st.columns(2)
            st.number_input("CPO ($)", key=f"{prefix}_Harga_CPO", step=10.0)
            st.number_input("Lift Oil (bph)", key=f"{prefix}_Lifting_Minyak", step=10.0)
            
            st.caption("🏦 Finansial & Riil")
            c9, c10 = st.columns(2)
            st.number_input("IHSG", key=f"{prefix}_IHSG", step=50.0)
            st.number_input("BI Rate (%)", key=f"{prefix}_BI_Rate", step=0.25)
            
            st.number_input("PMI Manuf", key=f"{prefix}_PMI_Manufaktur", step=0.5)

    render_input_section('a', "🔵 Scenario A")
    st.markdown("---")
    render_input_section('b', "🟠 Scenario B")

    st.markdown("###")
    run_btn = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

# --- MAIN CONTENT ---
tab_compare, tab_monte = st.tabs(["🆚 Scenario Comparison", "🎲 Monte Carlo Simulation"])

# Ensure Model Loaded
fc = st.session_state.get('forecaster_v6')
if not fc:
    if os.path.exists(history_path):
        with st.spinner("Initializing Model Engine..."):
            fc = TaxForecaster(history_path, macro_path if os.path.exists(macro_path) else None)
            fc.load_data()
            try:
                if not fc.is_fitted:
                    st.warning("⚠️ Model belum dilatih. Melakukan training cepat...")
                    fc.fit(n_trials=2, epochs=50) 
                st.session_state['forecaster_v6'] = fc
            except Exception as e:
                st.error(f"Model Init Error: {e}")
                st.stop()
    else:
        st.warning("Please go to Dashboard and upload data first.")
        st.stop()

# === TAB 1: SCENARIO COMPARISON ===
with tab_compare:
    if run_btn:
        with st.spinner("Simulating Scenarios (Advanced)..."):
            dates_future = pd.date_range(start=fc.df['Tanggal'].max() + pd.DateOffset(months=1), periods=forecast_months, freq='ME')
            
            # Build DataFrames
            cols_base = list(defs.keys())
            
            data_a = {k: st.session_state[f'a_{k}'] for k in cols_base}
            data_a['Tanggal'] = dates_future
            for k in cols_base:
                data_a[k] = [data_a[k]] * len(dates_future)
            df_a = pd.DataFrame(data_a)
            
            data_b = {k: st.session_state[f'b_{k}'] for k in cols_base}
            data_b['Tanggal'] = dates_future
            for k in cols_base:
                data_b[k] = [data_b[k]] * len(dates_future)
            df_b = pd.DataFrame(data_b)
            
            # Predict A
            fc.predict(forecast_periods=forecast_months, custom_macro_future=df_a, model_strategy='auto')
            res_a = pd.concat([r['data'] for r in fc.results], ignore_index=True)
            res_a['Scenario'] = 'Scenario A'
            
            # Predict B
            fc.predict(forecast_periods=forecast_months, custom_macro_future=df_b, model_strategy='auto')
            res_b = pd.concat([r['data'] for r in fc.results], ignore_index=True)
            res_b['Scenario'] = 'Scenario B'
            
            # --- RESULTS DISPLAY ---
            
            # 1. High Level Delta
            total_a = res_a.groupby('Tanggal')['Nominal (Milyar)'].sum().sum()
            total_b = res_b.groupby('Tanggal')['Nominal (Milyar)'].sum().sum()
            delta = total_b - total_a
            pct_delta = (delta / total_a * 100) if total_a != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                style.display_metric_card("Total Forecast (Scenario A)", f"{total_a:,.0f}", suffix="M")
            with col2:
                style.display_metric_card("Total Forecast (Scenario B)", f"{total_b:,.0f}", suffix="M")
            with col3:
                # Custom Delta Card
                color = "#27AE60" if delta > 0 else "#C0392B"
                arrow = "▲" if delta > 0 else "▼"
                st.markdown(f"""
                <div class="content-card" style="text-align:center; height:180px; display:flex; flex-direction:column; justify-content:center;">
                    <h4 style="color:#BDC3C7; margin:0;">DIFFERENCE (B - A)</h4>
                    <div style="font-size:2rem; font-weight:bold; color:{color};">
                        {arrow} Rp {abs(delta):,.0f} M
                    </div>
                    <div style="font-size:1.2rem; color:{color}; font-weight:bold;">
                        {pct_delta:+.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # 2. Main Comparison Chart
            st.subheader("📉 Projection Comparison (All Taxes)")
            hist_df = fc.df.copy()
            hist_df['Tanggal'] = pd.to_datetime(hist_df['Tanggal'])
            
            # --- NEW: Weighted Consensus ---
            prob_a = st.slider("Probability of Scenario A (%)", 0, 100, 50, step=5)
            prob_b = 100 - prob_a
            st.caption(f"Scenario A: {prob_a}% | Scenario B: {prob_b}%")
            
            sc1_agg = res_a.groupby('Tanggal')['Nominal (Milyar)'].sum().values
            sc2_agg = res_b.groupby('Tanggal')['Nominal (Milyar)'].sum().values
            weighted_agg = (sc1_agg * (prob_a/100)) + (sc2_agg * (prob_b/100))
            
            fig_comp = plot_comparison(hist_df, res_a, res_b, "Scenario A", "Scenario B")
            
            # Add Weighted Line
            # Get dates from agg
            plot_dates = res_a.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()['Tanggal']
            fig_comp.add_trace(go.Scatter(
                x=plot_dates, y=weighted_agg,
                mode='lines', name='Weighted Consensus',
                line=dict(color='#9B59B6', width=4, dash='dot')
            ))
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # 3. Breakdown by Tax Type
            st.subheader("📋 Tax Type Impact Analysis")
            
            sum_a = res_a.groupby('Jenis Pajak')['Nominal (Milyar)'].sum()
            sum_b = res_b.groupby('Jenis Pajak')['Nominal (Milyar)'].sum()
            
            delta_df = pd.DataFrame({'Scenario A': sum_a, 'Scenario B': sum_b})
            delta_df['Delta (Nominal)'] = delta_df['Scenario B'] - delta_df['Scenario A']
            delta_df['Delta (%)'] = (delta_df['Delta (Nominal)'] / delta_df['Scenario A']) * 100
            
            delta_df = delta_df.sort_values('Delta (Nominal)', ascending=False)
            st.dataframe(delta_df.style.format("{:,.1f}"), use_container_width=True)
            
            # 4. Input Summary
            with st.expander("🔎 View Scenario Inputs Used"):
                c1, c2 = st.columns(2)
                c1.markdown("**Scenario A Parameters:**")
                c1.json(data_a)
                c2.markdown("**Scenario B Parameters:**")
                c2.json(data_b)


    else:
        # Empty State Tab 1
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #444;'>🔬 Ready to Simulate</h2>
            <p>Konfigurasikan <b>Scenario A (Baseline)</b> dan <b>Scenario B (Alternative)</b> di sidebar sebelah kiri.<br>
            Sistem kini mendukung <b>15 Instrumen Makro Ekonomi</b> lengkap (SBN, CPO, Lifting, PMI, dll).<br>
            Klik tombol <b>🚀 Run Comparison</b> untuk memulai simulasi perbandingan.</p>
        </div>
        """, unsafe_allow_html=True)

# === TAB 2: MONTE CARLO ===
with tab_monte:
    st.markdown("### 🎲 Probabilistic Monte Carlo Simulation")
    st.info("Simulate thousands of possible economic futures by applying random shocks to macro indicators based on their historical volatility.")
    
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    n_sims = mc_col1.slider("Number of Simulations", 50, 500, 100, step=50)
    vol_scale = mc_col2.slider("Volatility Scale (Uncertainty)", 0.5, 2.0, 1.0, step=0.1, help="1.0 = Historical Volatility. >1.0 = More uncertain future.")
    mc_btn = mc_col3.button("🎲 Run Monte Carlo", type="primary", use_container_width=True)
    
    if mc_btn:
        with st.spinner(f"Running {n_sims} simulations across all models..."):
            # Call Backend
            mc_results = fc.predict_monte_carlo(forecast_periods=forecast_months, n_simulations=n_sims, volatility_scale=vol_scale)
            
            if mc_results:
                st.success("Simulation Complete!")
                
                # Plotting Fan Charts
                st.markdown("### 📊 Forecast Uncertainty Cones (Fan Charts)")
                
                # Checkbox to show specific taxes
                all_taxes = list(mc_results.keys())
                selected_mc_tax = st.selectbox("Select Tax Type to Visualize", all_taxes)
                
                res_df = mc_results[selected_mc_tax]
                
                # Create Fan Chart (Professional Blue)
                fig_fan = go.Figure()
                
                # P95 - P05 (90% Interval - Lightest)
                fig_fan.add_trace(go.Scatter(
                    x=res_df['Tanggal'], y=res_df['P95'],
                    mode='lines', line=dict(width=0),
                    name='95th Percentile', showlegend=False
                ))
                fig_fan.add_trace(go.Scatter(
                    x=res_df['Tanggal'], y=res_df['P05'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0, 210, 255, 0.1)', # Neon Blue Light
                    name='90% Confidence Interval'
                ))
                
                # P75 - P25 (50% Interval - Darker)
                fig_fan.add_trace(go.Scatter(
                    x=res_df['Tanggal'], y=res_df['P75'],
                    mode='lines', line=dict(width=0),
                    name='75th Percentile', showlegend=False
                ))
                fig_fan.add_trace(go.Scatter(
                    x=res_df['Tanggal'], y=res_df['P25'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0, 210, 255, 0.3)', # Neon Blue Medium
                    name='50% Confidence Interval'
                ))
                
                # Mean Line
                fig_fan.add_trace(go.Scatter(
                    x=res_df['Tanggal'], y=res_df['Mean'],
                    mode='lines', line=dict(color='#0984e3', width=3), # Strong Blue
                    name='Mean Forecast'
                ))
                
                fig_fan.update_layout(
                    title=f"Probabilistic Forecast: {selected_mc_tax}",
                    yaxis_title="Nominal (Milyar)",
                    hovermode="x unified",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig_fan, use_container_width=True)
                
                # Summary Stats
                st.markdown("#### Risk Metrics (End of Period)")
                last_row = res_df.iloc[-1]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Optimistic (P95)", f"{last_row['P95']:,.0f}")
                m2.metric("Base Case (Mean)", f"{last_row['Mean']:,.0f}")
                m3.metric("Pessimistic (P05)", f"{last_row['P05']:,.0f}")
                spread = last_row['P95'] - last_row['P05']
                m4.metric("Uncertainty Spread", f"{spread:,.0f}")
