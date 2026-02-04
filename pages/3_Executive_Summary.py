import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import style
import importlib
importlib.reload(style)
import report_generator # [NEW] Added for HTML Report

import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Executive Summary | TaxForecaster", layout="wide", page_icon="📈")
style.apply_theme()
sns.set_theme(style="white", context="talk") # Clean white theme for Executive Report

st.title("📈 Executive Summary")
st.markdown("High-level overview of revenue composition, seasonality, and strategic targets.")

# --- Check State ---
if 'forecaster_v6' not in st.session_state or st.session_state['forecaster_v6'] is None or not st.session_state['forecaster_v6'].results:
    st.info("ℹ️ Please run a forecast in the **Dashboard** first to populate this page.")
    st.stop()

# --- SIDEBAR DOWNLOAD ---
fc = st.session_state['forecaster_v6']
st.sidebar.subheader("📑 Executive Report")
if st.sidebar.button("📄 Generate Full Report", use_container_width=True):
    with st.spinner("Generating HTML Report..."):
        importlib.reload(report_generator)
        html_report = report_generator.generate_html_report(fc)
        st.sidebar.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name=f"Executive_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

# --- Data Preparation ---
fc = st.session_state['forecaster_v6']
# Extract all results
all_res = []
for r in fc.results:
    d = r['data'].copy()
    d['Model'] = r['model']
    all_res.append(d)

df = pd.concat(all_res, ignore_index=True)

# Aggregates
total_forecast = df['Nominal (Milyar)'].sum()
min_date = df['Tanggal'].min()
max_date = df['Tanggal'].max()
period_str = f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"

# --- NEW: AI SMART ANALYST & NEWS ---
import narrative_engine
import fetch_news
import importlib
importlib.reload(fetch_news)

# 1. Load Macro Context
macro_context = {}
if os.path.exists("macro_data_auto.csv"):
    try:
        m_df = pd.read_csv("macro_data_auto.csv")
        if not m_df.empty:
            macro_context = m_df.iloc[-1].to_dict()
    except: pass

# 2. Fetch News (Cached/Live) - DEFERRED LOADING
@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour, no spinner
def get_news_cached_v3():
    return fetch_news.get_latest_financial_news()

# Initialize news in session state
if 'executive_news_items' not in st.session_state:
    st.session_state['executive_news_items'] = None  # None means not loaded yet

# Defer news loading to end of page - load news AFTER page renders
def load_news_deferred():
    if st.session_state['executive_news_items'] is None:
        try:
            # Fetch news (will be instant if cached)
            st.session_state['executive_news_items'] = get_news_cached_v3()
        except:
            # On error, use empty list
            st.session_state['executive_news_items'] = []
    return st.session_state['executive_news_items']

# For now, use empty list for insights generation (news is optional)
# We'll load news after the page renders
news_items = st.session_state['executive_news_items'] if st.session_state['executive_news_items'] is not None else []

# Add refresh button for manual news update
if st.sidebar.button("🔄 Refresh News", use_container_width=True):
    get_news_cached_v3.clear()
    st.session_state['executive_news_items'] = None
    st.rerun()

# 3. Generate Insights
insights = narrative_engine.generate_insights(df, macro_context, news_items)

# --- DISPLAY INSIGHTS ---
st.markdown("### 🤖 AI Smart Analyst Insights")
col_i1, col_i2, col_i3 = st.columns([1, 1, 1])

with col_i1:
    st.info(f"**Executive Summary**\n\n" + "\n".join([f"- {s}" for s in insights['summary']]))

with col_i2:
    if insights['opportunities']:
        st.success(f"**Opportunities (Upside)**\n\n" + "\n".join([f"- {s}" for s in insights['opportunities']]))
    else:
        st.success("**Opportunities**\n\nNo specific upside signals detected.")

with col_i3:
    if insights['risks']:
        st.warning(f"**Risks (Downside)**\n\n" + "\n".join([f"- {s}" for s in insights['risks']]))
    else:
        st.warning("**Risks**\n\nNo immediate macro risks detected.")





# --- 1. KPI Cards (Glassmorphism) ---
st.subheader("🎯 Strategic Overview")
k1, k2, k3, k4 = st.columns(4)

with k1:
    style.display_metric_card("Total Projected Revenue", f"{total_forecast:,.0f} B")

with k2:
    style.display_metric_card("Forecast Horizon", f"{(max_date - min_date).days // 30 + 1}", suffix=" Months")

with k3:
    # Top Contributor
    by_tax = df.groupby('Jenis Pajak')['Nominal (Milyar)'].sum().sort_values(ascending=False)
    top_tax = by_tax.index[0]
    top_val = by_tax.iloc[0]
    style.display_metric_card("Top Contributor", top_tax, suffix=f"<br><span style='font-size:1.2rem; font-weight:600'>Rp {top_val:,.0f} B</span>")

with k4:
    # Avg Monthly
    avg_monthly = total_forecast / ((max_date - min_date).days // 30 + 1)
    style.display_metric_card("Avg. Monthly Run Rate", f"{avg_monthly:,.0f} B")

st.markdown("---")

# --- 2. Waterfall Chart (Revenue Composition) ---
col_waterfall, col_heatmap = st.columns([1, 1])

with col_waterfall:
    st.subheader("🌊 Revenue Composition (Waterfall)")
    st.markdown("Contribution of each tax type to the total.")
    
    # Prepare Waterfall Data
    wf_data = by_tax.reset_index()
    wf_data.columns = ['Measure', 'Value']
    
    # fig_wf colors (Professional Executive Style)
    fig_wf = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative"] * len(wf_data) + ["total"],
        x = list(wf_data['Measure']) + ["Total"],
        textposition = "outside",
        text = [f"{v/1000:.1f}T" for v in wf_data['Value']] + [f"{total_forecast/1000:.1f}T"],
        y = list(wf_data['Value']) + [0],
        connector = {"line":{"color":"#7f8c8d", "dash": "dot"}},
        # Custom Professional Colors
        increasing = {"marker":{"color":"#10B981"}}, # Emerald (Success)
        decreasing = {"marker":{"color":"#F43F5E"}}, # Rose (Deficit/Negative)
        totals = {"marker":{"color":"#0F172A"}},     # Dark Slate (Total)
    ))

    fig_wf.update_layout(
        title = "Revenue Bridge by Tax Type",
        showlegend = False, # Clean look
        waterfallgap = 0.2,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12, color="#2c3e50")
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# --- 3. Seasonality Heatmap (Seaborn Upgrade) ---
with col_heatmap:
    st.subheader("🔥 Seasonality Heatmap (Seaborn)")
    st.markdown("Revenue intensity by Month vs Year.")
    
    # ... (Data prep remains same) ...
    df['Year'] = df['Tanggal'].dt.year
    df['Month'] = df['Tanggal'].dt.strftime('%b')
    df['Month_Num'] = df['Tanggal'].dt.month
    
    # Pivot
    heatmap_data = df.groupby(['Year', 'Month', 'Month_Num'])['Nominal (Milyar)'].sum().reset_index()
    heatmap_data = heatmap_data.sort_values(['Year', 'Month_Num'])
    
    # Pivot for Heatmap format (Index=Month, Columns=Year)
    hm_pivot = heatmap_data.pivot(index='Month', columns='Year', values='Nominal (Milyar)')
    # Sort Index
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    hm_pivot = hm_pivot.reindex(month_order)
    
    # Seaborn Heatmap
    fig_hm, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        hm_pivot,
        annot=True,
        fmt=".0f", # Integer format
        cmap="vlag", # Diverging palette
        linewidths=.5,
        ax=ax,
        cbar_kws={'label': 'Revenue (Bn)'}
    )
    ax.set_title("Annual Revenue Intensity", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig_hm)

# --- 4. Monthly Trend Drilldown ---
st.markdown("---")
st.subheader("📅 Monthly Performance Breakdown")

# Multi-select for tax comparison
selected_taxes = st.multiselect("Compare Tax Types", df['Jenis Pajak'].unique(), default=[top_tax])

if selected_taxes:
    filtered_df = df[df['Jenis Pajak'].isin(selected_taxes)]
    
    fig_ln = px.line(
        filtered_df, 
        x='Tanggal', 
        y='Nominal (Milyar)', 
        color='Jenis Pajak',
        markers=True,
        title="Comparative Trend Analysis",
        color_discrete_sequence=style.get_plot_palette()
    )
    fig_ln.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_ln, use_container_width=True)
else:
    st.info("Select tax types to visualize trends.")

st.markdown("---")
st.subheader("📋 Monthly Forecast Detail (Aggregated)")

# Aggregate Monthly
agg_cols = {'Nominal (Milyar)': 'sum'}
if 'Nominal Lower' in df.columns: agg_cols['Nominal Lower'] = 'sum'
if 'Nominal Upper' in df.columns: agg_cols['Nominal Upper'] = 'sum'

monthly_agg = df.groupby('Tanggal').agg(agg_cols).sort_index()

# Add Bounds if missing
if 'Nominal Lower' not in monthly_agg.columns:
    monthly_agg['Nominal Lower'] = monthly_agg['Nominal (Milyar)'] * 0.95
if 'Nominal Upper' not in monthly_agg.columns:
    monthly_agg['Nominal Upper'] = monthly_agg['Nominal (Milyar)'] * 1.05

# Calculate Growth
monthly_agg['Growth (MoM)'] = monthly_agg['Nominal (Milyar)'].pct_change() * 100

# Formatting for Display
display_df = monthly_agg.copy()
display_df.index = display_df.index.strftime('%Y-%m')
display_df = display_df.rename(columns={
    'Nominal (Milyar)': 'Forecast (Bn)',
    'Nominal Lower': 'Lower Bound (95%)',
    'Nominal Upper': 'Upper Bound (95%)',
    'Growth (MoM)': 'Growth %'
})

st.dataframe(
    display_df.style.format({
        'Forecast (Bn)': '{:,.0f}',
        'Lower Bound (95%)': '{:,.0f}',
        'Upper Bound (95%)': '{:,.0f}',
        'Growth %': '{:+.2f}%'
    }),
    use_container_width=True
)

st.markdown("---")
st.subheader("🧩 Advanced Strategy Matrix (BCG Analysis)")

# --- 5. Tax Portfolio Matrix (Bubble Chart) ---
# X: Avg Growth Rate
# Y: Total Revenue Contribution
# Size: Volatility (Std Dev)

# Prepare Metrics per Tax Type
tax_metrics = []
for tax_type in df['Jenis Pajak'].unique():
    sub = df[df['Jenis Pajak'] == tax_type]
    
    total_rev = sub['Nominal (Milyar)'].sum()
    avg_growth = sub['Nominal (Milyar)'].pct_change().mean() * 100
    volatility = sub['Nominal (Milyar)'].std()
    cv = volatility / sub['Nominal (Milyar)'].mean() if sub['Nominal (Milyar)'].mean() != 0 else 0
    
    tax_metrics.append({
        'Jenis Pajak': tax_type,
        'Total Revenue': total_rev,
        'Avg Growth (%)': avg_growth,
        'Volatility (StdDev)': volatility,
        'Risk (CV)': cv
    })

df_matrix = pd.DataFrame(tax_metrics).fillna(0)

# Toggle View
view_mode = st.radio("View Mode:", ["Interactive Matrix (Plotly)", "Density Analysis (Seaborn)"], horizontal=True)

# --- Tax Portfolio Matrix (Full Width) ---
st.markdown("#### 🟢 Tax Portfolio Matrix")

if view_mode == "Interactive Matrix (Plotly)":
    st.markdown("Classifying tax types by **Growth** (X) vs **Revenue Size** (Y). Bubble Size = **Volatility**.")
    if not df_matrix.empty:
        fig_bubble = px.scatter(
            df_matrix,
            x="Avg Growth (%)",
            y="Total Revenue",
            size="Volatility (StdDev)",
            color="Jenis Pajak",
            hover_name="Jenis Pajak",
            text="Jenis Pajak",
            log_y=True,
            size_max=60,
            color_discrete_sequence=px.colors.qualitative.Bold # Professional Palette
        )
        
        # Enhanced Styling
        fig_bubble.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.8),
            textposition='top center'
        )
        
        # Add Quadrant Lines
        median_growth = df_matrix['Avg Growth (%)'].median()
        median_rev = df_matrix['Total Revenue'].median()
        
        fig_bubble.add_vline(x=median_growth, line_width=1, line_dash="dash", line_color="grey")
        fig_bubble.add_hline(y=median_rev, line_width=1, line_dash="dash", line_color="grey")
        
        fig_bubble.update_layout(
            height=550, 
            showlegend=False,
            plot_bgcolor='rgba(240, 242, 246, 0.5)',
            xaxis=dict(gridcolor='#bdc3c7'),
            yaxis=dict(gridcolor='#bdc3c7')
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        
else:
    st.markdown("Detailed density analysis of **Revenue vs Growth**.")
    if not df_matrix.empty:
        # Seaborn JointPlot
        st.caption("Distribution density with marginal histograms.")
        
        sns.set_style("whitegrid")
        g = sns.jointplot(
            data=df_matrix,
            x="Avg Growth (%)",
            y="Total Revenue",
            kind="reg", # Regression line + Scatter
            truncate=False,
            color="#0D9488", # Teal
            height=8,
            ratio=5,
            space=0.2,
            scatter_kws={"s": 100, "alpha": 0.6}
        )
        g.set_axis_labels("Average Growth (%)", "Total Revenue (Log Scale)", fontsize=12)
        
        # Apply Log Scale to Y manually since JointPlot is tricky with log scales
        g.ax_joint.set_yscale('log')
        
        # Add labels
        for i in range(df_matrix.shape[0]):
            g.ax_joint.text(
                df_matrix['Avg Growth (%)'].iloc[i]+0.2, 
                df_matrix['Total Revenue'].iloc[i], 
                df_matrix['Jenis Pajak'].iloc[i], 
                horizontalalignment='left', 
                size='small', 
                color='black', 
                weight='semibold'
            )
        
        # Clear titles to avoid clutter
        st.pyplot(g.figure)

# --- Risk & Buoyancy Section (Below Matrix) ---
st.markdown("---")
st.markdown("#### ⚖️ Risk & Buoyancy")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    # 1. Tax Buoyancy Calculation (Implied)
    # Tax Growth / GDP Growth
    # Get GDP Growth from Macro Context
    gdp_growth = macro_context.get('Pertumbuhan_Ekonomi', 5.0) # Default 5%
    total_tax_growth = monthly_agg['Growth (MoM)'].mean() # Simple proxy
    
    buoyancy = total_tax_growth / gdp_growth if gdp_growth != 0 else 0
    
    # Custom HTML for larger display
    delta_color = "#10B981" if buoyancy > 1 else "#F43F5E"  # Green if good, red if poor
    delta_arrow = "↑" if buoyancy > 1 else "↓"
    
    st.markdown(f"""
    <div style='padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 0.5rem; text-align: center; height: 300px; display: flex; flex-direction: column; justify-content: center;'>
        <p style='margin: 0; color: #94a3b8; font-size: 1.1rem; font-weight: 500;'>Implied Tax Buoyancy</p>
        <h1 style='margin: 0.5rem 0; font-size: 3.5rem; font-weight: 700;'>{buoyancy:.2f}x</h1>
        <p style='margin: 0; color: {delta_color}; font-size: 1.3rem; font-weight: 600;'>
            {delta_arrow} vs GDP Growth ({gdp_growth:.2f}%)
        </p>
        <p style='margin-top: 0.5rem; color: #64748b; font-size: 0.85rem;'>
            Ratio of Tax Revenue Growth to GDP Growth. >1 indicates elastic/efficient tax system.
        </p>
    </div>
    """, unsafe_allow_html=True)

with risk_col2:
    # 2. Volatility Ranking (Bar Chart)
    df_risk = df_matrix.sort_values('Risk (CV)', ascending=True) # Low risk top? or High risk top?
    # Let's show High Risk at top
    df_risk = df_matrix.sort_values('Risk (CV)', ascending=False).head(5)
    
    fig_risk = px.bar(
        df_risk,
        x="Risk (CV)",
        y="Jenis Pajak",
        orientation='h',
        color="Risk (CV)",
        color_continuous_scale="Reds"
    )
    fig_risk.update_layout(
        height=300,
        title=dict(
            text="Volatility Ranking (Risk)",
            font=dict(size=14, weight=600),
            x=0.5,
            xanchor='center'
        ),
        yaxis={'categoryorder':'total ascending'},
        margin=dict(l=20, r=20, t=40, b=20),  # Added top margin for title
        font=dict(size=12)  # Increased font size for better readability
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# --- DISPLAY NEWS TICKER (Moved to Bottom) ---
st.markdown("---")
st.subheader("📰 Live Market & Policy News")

# Load news here (deferred) - page has already rendered above content
news_items_display = load_news_deferred()

with st.expander("Show Latest Financial News", expanded=True):
    if news_items_display is None or len(news_items_display) == 0:
        st.info("⏳ Loading latest news... Please refresh if this persists.")
    else:
        col_news_1, col_news_2 = st.columns(2)
        
        local_news = [n for n in news_items_display if 'Local' in n.get('category', 'Local')]
        global_news = [n for n in news_items_display if 'Global' in n.get('category', '')]
        
        with col_news_1:
            st.markdown("##### 🇮🇩 Local Market")
            # Take top 30 (interleaved) then sort by Source
            display_local = sorted(local_news[:30], key=lambda x: x['source'])
            for n in display_local:
                st.markdown(f"**[{n['source']}]** [{n['title']}]({n['link']})")
                
        with col_news_2:
            st.markdown("##### 🌎 Global Market")
            if global_news:
                 # Take top 30 (interleaved) then sort by Source
                display_global = sorted(global_news[:30], key=lambda x: x['source'])
                for n in display_global:
                    st.markdown(f"**[{n['source']}]** [{n['title']}]({n['link']})")
            else:
                st.info("No global news feeds connected.")

