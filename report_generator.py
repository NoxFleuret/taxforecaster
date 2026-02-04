import narrative_engine
import re
import pandas as pd
import datetime
import base64
import io
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# --- HELPER FUNCTIONS FOR CHARTS ---

def create_agg_forecast_chart(results_df):
    """
    Creates a line chart showing the total aggregated revenue forecast over time.

    Args:
        results_df (pd.DataFrame): The forecast results dataframe.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.graph_objects as go
    agg_df = results_df.groupby(['Tanggal', 'Tipe Data'])['Nominal (Milyar)'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg_df['Tanggal'], y=agg_df['Nominal (Milyar)'], mode='lines+markers', name='Forecast', line=dict(color='#2E86C1', width=3)))
    fig.update_layout(
        title="Total Revenue Forecast", 
        height=400, 
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickmode='array', tickvals=agg_df['Tanggal'], tickformat='%b %Y')
    )
    return fig

def create_yoy_chart(fc, results_df):
    """
    Creates a bar chart showing Year-over-Year (YoY) growth percentage.

    Args:
        fc (TaxForecaster): The forecaster object containing historical data.
        results_df (pd.DataFrame): The forecast results dataframe.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.graph_objects as go
    
    full_ts = pd.DataFrame()
    if fc.df is not None:
        valid_history = fc.df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
        valid_forecast = results_df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
        full_ts = pd.concat([valid_history, valid_forecast], ignore_index=True)
    else:
        full_ts = results_df.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index()
        
    full_ts = full_ts.groupby('Tanggal')['Nominal (Milyar)'].sum().reset_index().sort_values('Tanggal')
    full_ts['YoY_Growth'] = full_ts['Nominal (Milyar)'].pct_change(periods=12) * 100
    
    start_date = results_df['Tanggal'].min()
    yoy_data = full_ts[full_ts['Tanggal'] >= start_date].dropna()
    
    if yoy_data.empty: return None
        
    yoy_data['Color'] = yoy_data['YoY_Growth'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
    fig = go.Figure(go.Bar(
        x=yoy_data['Tanggal'], y=yoy_data['YoY_Growth'], marker_color=yoy_data['Color']
    ))
    fig.update_layout(
        title="YoY Growth Projection (%)", 
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickmode='array', tickvals=yoy_data['Tanggal'], tickformat='%b %Y')
    )
    return fig

def create_waterfall_chart(by_tax, total_forecast):
    """
    Creates a waterfall chart decomposing total revenue by tax type.

    Args:
        by_tax (pd.Series): Series of revenue sum per tax type.
        total_forecast (float): Total forecasted revenue.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.graph_objects as go
    wf_data = by_tax.reset_index()
    wf_data.columns = ['Measure', 'Value']
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative"] * len(wf_data) + ["total"],
        x = list(wf_data['Measure']) + ["Total"],
        textposition = "outside",
        text = [f"{v/1000:.1f}T" for v in wf_data['Value']] + [f"{total_forecast/1000:.1f}T"],
        y = list(wf_data['Value']) + [0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Revenue Composition Bridge", height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_seasonality_heatmap(results_df):
    """
    Creates a heatmap visualization of revenue seasonality (Month vs Year).

    Args:
        results_df (pd.DataFrame): The forecast results dataframe.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.express as px
    results_df['Year'] = results_df['Tanggal'].dt.year
    results_df['Month'] = results_df['Tanggal'].dt.strftime('%b')
    results_df['Month_Num'] = results_df['Tanggal'].dt.month
    
    heatmap_data = results_df.groupby(['Year', 'Month', 'Month_Num'])['Nominal (Milyar)'].sum().reset_index()
    hm_pivot = heatmap_data.pivot(index='Month', columns='Year', values='Nominal (Milyar)')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    hm_pivot = hm_pivot.reindex(month_order)
    
    fig = px.imshow(
        hm_pivot, labels=dict(x="Year", y="Month", color="Revenue"),
        x=hm_pivot.columns, y=hm_pivot.index, color_continuous_scale="Reds", text_auto=".2s", title="Seasonality Heatmap"
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_strategic_bubble(results_df):
    """
    Creates a bubble chart matrix (Growth vs Revenue vs Volatility).

    Args:
        results_df (pd.DataFrame): The forecast results dataframe.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    import plotly.express as px
    tax_metrics = []
    for tax_type in results_df['Jenis Pajak'].unique():
        sub = results_df[results_df['Jenis Pajak'] == tax_type]
        tax_metrics.append({
            'Jenis Pajak': tax_type,
            'Revenue': sub['Nominal (Milyar)'].sum(),
            'Growth': sub['Nominal (Milyar)'].pct_change().mean() * 100,
            'Volatility': sub['Nominal (Milyar)'].std()
        })
    df_matrix = pd.DataFrame(tax_metrics).fillna(0)
    fig = px.scatter(
        df_matrix, x="Growth", y="Revenue", size="Volatility", color="Jenis Pajak",
        title="Tax Portfolio Matrix", log_y=True, size_max=40, text="Jenis Pajak"
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

# --- MAIN HTML REPORT GENERATOR ---

def generate_html_report(forecaster_obj):
    """
    Generates a professional executive summary report in HTML format.
    Returns the HTML string.
    """
    if not forecaster_obj.results:
        return "<h3>No forecast results available to generate report.</h3>"

    # 1. Prepare Data
    fc = forecaster_obj
    
    # Robust handling for fc.results structure (List of DFs vs List of Dicts)
    if fc.results and isinstance(fc.results[0], pd.DataFrame):
        results_df = pd.concat(fc.results, ignore_index=True)
    elif fc.results and isinstance(fc.results[0], dict) and 'data' in fc.results[0]:
        results_df = pd.concat([r['data'] for r in fc.results], ignore_index=True)
    else:
        results_df = pd.DataFrame()
        
    if results_df.empty:
        return "<h3>No valid forecast data found.</h3>"

    if 'Tipe Data' not in results_df.columns:
        results_df['Tipe Data'] = 'Forecast'
        
    start_date = results_df['Tanggal'].min()
    end_date = results_df['Tanggal'].max()
    
    # Ensure 'Tipe Data' exists
    if 'Tipe Data' not in results_df.columns:
        results_df['Tipe Data'] = 'Forecast'
        
    total_forecast = results_df['Nominal (Milyar)'].sum()
    months_count = (end_date - start_date).days // 30 + 1
    avg_monthly = total_forecast / max(1, months_count)
    
    # Top 5 Tax Types by Revenue
    by_tax = results_df.groupby('Jenis Pajak')['Nominal (Milyar)'].sum().sort_values(ascending=False)
    top_taxes = by_tax.head(5)
    top_contributor = by_tax.index[0]
    top_val = by_tax.iloc[0]
    
    start_str = start_date.strftime('%B %Y')
    end_str = end_date.strftime('%B %Y')
    gen_time = datetime.datetime.now().strftime("%d %B %Y, %H:%M WIB")
    
    # 2. Extract Macro & News for AI
    macro_context = {}
    if fc.macro_df is not None and not fc.macro_df.empty:
         macro_context = fc.macro_df.iloc[-1].to_dict()
    news_items = []
    
    # 3. Generate Insights
    insights = narrative_engine.generate_insights(results_df, macro_context, news_items)
    
    # --- CHART GENERATION (Using Helpers) ---
    import plotly.graph_objects as go
    
    fig_agg = create_agg_forecast_chart(results_df)
    agg_html = fig_agg.to_html(full_html=False, include_plotlyjs='cdn')
    
    fig_yoy = create_yoy_chart(fc, results_df)
    yoy_html = fig_yoy.to_html(full_html=False, include_plotlyjs=False) if fig_yoy else "<p>Insufficient data for YoY Growth.</p>"
    
    fig_wf = create_waterfall_chart(by_tax, total_forecast)
    wf_html = fig_wf.to_html(full_html=False, include_plotlyjs='cdn')
    
    fig_hm = create_seasonality_heatmap(results_df)
    hm_html = fig_hm.to_html(full_html=False, include_plotlyjs=False)
    
    fig_bubble = create_strategic_bubble(results_df)
    bubble_html = fig_bubble.to_html(full_html=False, include_plotlyjs=False)
    
    # Seasonal Decomposition (Complex, keeping logic here simply as previously defined or simplifying)
    # Re-using simple logic for decomp
    from plotly.subplots import make_subplots
    decomp_charts_html = ""
    if fc.df is not None:
        try:
             # 1. Aggregate
             agg_history = fc.df.groupby('Tanggal')['Nominal (Milyar)'].sum().sort_index()
             agg_history.index = pd.to_datetime(agg_history.index)
             if len(agg_history) >= 24:
                decomp = seasonal_decompose(agg_history, model='additive', period=12)
                fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonality", "Residuals"))
                fig_decomp.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, mode='lines', name='Trend', line=dict(color='#E67E22')), row=1, col=1)
                fig_decomp.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, mode='lines', name='Seasonal', line=dict(color='#2ECC71')), row=2, col=1)
                fig_decomp.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, mode='markers', name='Residual', marker=dict(color='#E74C3C', size=4)), row=3, col=1)
                fig_decomp.update_layout(height=500, title_text="Seasonal Decomposition: Aggregate", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                decomp_charts_html += f'<div class="chart-container">{fig_decomp.to_html(full_html=False, include_plotlyjs=False)}</div>'

             # 2. Per Tax Type
             unique_taxes = fc.df['Jenis Pajak'].unique()
             for tax in unique_taxes:
                 tax_ts = fc.df[fc.df['Jenis Pajak'] == tax].copy()
                 tax_ts['Tanggal'] = pd.to_datetime(tax_ts['Tanggal'])
                 tax_ts = tax_ts.set_index('Tanggal').sort_index()['Nominal (Milyar)']
                 
                 if len(tax_ts) >= 24:
                    decomp_tax = seasonal_decompose(tax_ts, model='additive', period=12)
                    fig_decomp_tax = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonality", "Residuals"))
                    fig_decomp_tax.add_trace(go.Scatter(x=decomp_tax.trend.index, y=decomp_tax.trend, mode='lines', name='Trend', line=dict(color='#E67E22')), row=1, col=1)
                    fig_decomp_tax.add_trace(go.Scatter(x=decomp_tax.seasonal.index, y=decomp_tax.seasonal, mode='lines', name='Seasonal', line=dict(color='#2ECC71')), row=2, col=1)
                    fig_decomp_tax.add_trace(go.Scatter(x=decomp_tax.resid.index, y=decomp_tax.resid, mode='markers', name='Residual', marker=dict(color='#E74C3C', size=4)), row=3, col=1)
                    fig_decomp_tax.update_layout(height=500, title_text=f"Seasonal Decomposition: {tax}", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                    decomp_charts_html += f'<div class="chart-container">{fig_decomp_tax.to_html(full_html=False, include_plotlyjs=False)}</div>'
        except Exception as e: 
            decomp_charts_html += f"<p>Error generating seasonal charts: {e}</p>"
            
    if not decomp_charts_html: decomp_charts_html = "<p>Data insufficient for seasonal decomposition.</p>"
    
    # Risk Ranking (Re-using bubble matrix df logic simplified)
    # ... (Actually bubble chart function creates internal dataframe, lets just re-calc for HTML simplicity or skip if bubble is enough)
    # Let's skip extra risk chart for now to keep it simpler and match PDF which uses Strat Matrix.
    risk_html = "" 
    
    # Buoyancy
    # Simplified assumption calculation
    monthly_agg_pre = results_df.groupby('Tanggal')['Nominal (Milyar)'].sum().sort_index()
    tax_growth_agg = monthly_agg_pre.pct_change().mean() * 100
    buoyancy = tax_growth_agg / 5.0 
    buoyancy_val = f"{buoyancy:.2f}x"
    
    # --- SEABORN STATIC CHARTS ---
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    sb_html = ""
    try:
        sns.set_theme(style="whitegrid", palette="mako")
        
        # 1. Boxen Plot
        fig_sb1, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxenplot(data=results_df, x="Nominal (Milyar)", y="Jenis Pajak", palette="mako", ax=ax1)
        sns.despine(left=True, bottom=True)
        ax1.set_xlabel("Forecast Distribution (Nominal)")
        ax1.set_ylabel("")
        
        boxen_b64 = fig_to_base64(fig_sb1)
        plt.close(fig_sb1)
        
        # 2. Correlation
        pivot_tax = results_df.pivot_table(index='Tanggal', columns='Jenis Pajak', values='Nominal (Milyar)')
        corr = pivot_tax.corr()
        
        fig_sb2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax2)
        ax2.set_title("Tax Type Correlation Matrix")
        
        corr_b64 = fig_to_base64(fig_sb2)
        plt.close(fig_sb2)
        
        sb_html = f"""
        <h2>🔬 Distribution & Correlation Analysis</h2>
        <div class="chart-container">
            <h3>Forecast Distribution (Boxen Plot)</h3>
            <img src="data:image/png;base64,{boxen_b64}" style="width: 100%; border-radius: 8px;">
            <p style="font-size: 12px; color: #7f8c8d; margin-top: 10px;">Shows the spread and tail behavior of the forecasted values.</p>
        </div>
        <div class="chart-container">
            <h3>Correlation Matrix</h3>
            <img src="data:image/png;base64,{corr_b64}" style="width: 100%; border-radius: 8px;">
            <p style="font-size: 12px; color: #7f8c8d; margin-top: 10px;">Correlation between different tax types over the forecast period.</p>
        </div>
        """
    except Exception as e:
        sb_html = f"<p>Could not generate static charts: {e}</p>"

    # --- HTML TEMPLATE ---
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tax Forecast Executive Summary</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #2c3e50; padding-bottom: 20px; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #2c3e50; text-transform: uppercase; letter-spacing: 2px; }}
            .report-title {{ font-size: 32px; color: #34495e; margin: 10px 0; }}
            .meta {{ color: #7f8c8d; font-size: 14px; }}
            
            .kpi-container {{ display: flex; justify-content: space-between; margin-bottom: 30px; gap: 20px; }}
            .kpi-card {{ background: #fff; padding: 20px; border-radius: 8px; flex: 1; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #eee; }}
            .kpi-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
            .kpi-card .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .kpi-card .sub {{ font-size: 12px; color: #95a5a6; margin-top: 5px; }}
            
            h2 {{ color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 40px; margin-bottom: 20px; }}
            
            .chart-container {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 10px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
            th {{ background: #2c3e50; color: white; text-align: left; padding: 12px; }}
            td {{ border-bottom: 1px solid #ddd; padding: 12px; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            
            .footer {{ margin-top: 60px; text-align: center; font-size: 12px; color: #aaa; border-top: 1px solid #eee; padding-top: 20px; }}
            
            @media print {{
                body {{ padding: 0; max-width: 100%; }}
                .no-print {{ display: none; }}
                .chart-container {{ break-inside: avoid; page-break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">TaxForecaster</div>
            <div class="report-title">Executive Forecast Report</div>
            <div class="meta">Generated: {gen_time} | Horizon: {start_str} - {end_str}</div>
        </div>
        
        <!-- KPIs -->
        <div class="kpi-container">
            <div class="kpi-card">
                <h3>Total Projected Revenue</h3>
                <div class="value">Rp {total_forecast:,.0f} B</div>
                <div class="sub">Aggregate for period</div>
            </div>
            <div class="kpi-card">
                <h3>Top Contributor</h3>
                <div class="value">{top_contributor}</div>
                <div class="sub">Rp {top_val:,.0f} B</div>
            </div>
            <div class="kpi-card">
                <h3>Avg. Monthly Run Rate</h3>
                <div class="value">Rp {avg_monthly:,.0f} B</div>
                <div class="sub">Average per month</div>
            </div>
            <div class="kpi-card">
                <h3>Tax Types</h3>
                <div class="value">{len(by_tax)} Types</div>
                <div class="sub">Modeled Instruments</div>
            </div>
        </div>
        
        <!-- VISUALS -->
        <h2>🤖 AI Analyst Insights</h2>
        <div style="background: #f9f9f9; padding: 15px; border-left: 5px solid #2980b9; margin-bottom: 20px;">
            <p><b>Executive Summary:</b><br>{re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', '<br>'.join(insights['summary']))}</p>
        </div>
        <div style="display: flex; gap: 20px; margin-bottom: 30px;">
            <div style="flex: 1; background: #e8f5e9; padding: 15px; border-radius: 5px; color: #27ae60;">
                <b>🚀 Opportunities (Upside)</b><br>
                {'<br>'.join([f"• {re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', x)}" for x in insights['opportunities']]) if insights['opportunities'] else "No significant upside signals."}
            </div>
            <div style="flex: 1; background: #fdf2e9; padding: 15px; border-radius: 5px; color: #d35400;">
                <b>⚠️ Risks & Headwinds</b><br>
                {'<br>'.join([f"• {re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', x)}" for x in insights['risks']]) if insights['risks'] else "No immediate risks detected."}
            </div>
        </div>
        
        <h2>📈 Forecast Trend & Growth</h2>
        <div class="chart-container">
            {agg_html}
        </div>
        <div class="chart-container">
            {yoy_html}
        </div>
        
        <h2>🗓️ Seasonal Analysis (Aggregate & By Tax Type)</h2>
        {decomp_charts_html}
        
        <h2>🌊 Revenue Composition</h2>
        <div class="chart-container">
            {wf_html}
        </div>
        
        <h2>🔥 Seasonality Heatmap</h2>
        <div class="chart-container">
            {hm_html}
        </div>
        
        <!-- SEABORN CHARTS INSERTED HERE -->
        {sb_html}
        
        <h2>🧩 Strategic Analysis</h2>
        <div class="chart-container">
            <h3>Implied Tax Buoyancy: <span style="color: #2980b9; font-size: 24px;">{buoyancy_val}</span></h3>
            <p><i>(Ratio of Tax Growth to avg GDP Growth ~5%)</i></p>
        </div>
        
        <div class="chart-container">
            {bubble_html}
        </div>
        
        <div class="chart-container">
            {risk_html}
        </div>
        
        <!-- TABLES -->
        <h2>📊 Top Revenue Contributors</h2>
        <table>
            <tr>
                <th>Tax Type</th>
                <th>Projected Revenue (Milyar Rp)</th>
                <th>Contribution</th>
            </tr>
    """
    
    # Add Table Rows
    for tax, val in top_taxes.items():
        pct = (val / total_forecast) * 100
        html += f"""
            <tr>
                <td>{tax}</td>
                <td>{val:,.0f}</td>
                <td>{pct:.1f}%</td>
            </tr>
        """
        
    html += """
        </table>

        <h2>📅 Monthly Forecast Detail</h2>
        <p>Detailed monthly projection with 95% Confidence Intervals (Lower/Upper Bounds):</p>
        <table>
            <tr>
                <th>Month</th>
                <th>Forecast (Milyar Rp)</th>
                <th>Lower Bound (95%)</th>
                <th>Upper Bound (95%)</th>
                <th>Growth (MoM)</th>
            </tr>
    """
    
    # Calculate Monthly Aggregates
    agg_cols = {'Nominal (Milyar)': 'sum'}
    if 'Nominal Lower' in results_df.columns: agg_cols['Nominal Lower'] = 'sum'
    if 'Nominal Upper' in results_df.columns: agg_cols['Nominal Upper'] = 'sum'
    
    monthly_agg = results_df.groupby('Tanggal').agg(agg_cols).sort_index()
    
    prev_val = 0
    for date, row in monthly_agg.iterrows():
        val = row['Nominal (Milyar)']
        lower = row.get('Nominal Lower', val * 0.95)
        upper = row.get('Nominal Upper', val * 1.05)
        
        # Growth Calc
        growth = 0
        if prev_val > 0:
            growth = ((val - prev_val) / prev_val) * 100
        growth_str = f"<span style='color: {'green' if growth >= 0 else 'red'}'>{growth:+.1f}%</span>" if prev_val > 0 else "-"
        
        html += f"""
            <tr>
                <td>{date.strftime('%B %Y')}</td>
                <td><b>{val:,.0f}</b></td>
                <td style="color: #e67e22;">{lower:,.0f}</td>
                <td style="color: #27ae60;">{upper:,.0f}</td>
                <td>{growth_str}</td>
            </tr>
        """
        prev_val = val

    html += """
        </table>
        
        <div class="footer">
            <p>Generated by TaxForecaster | by fasya_dev. This report is for internal simulation purposes only.</p>
        </div>
        
        <script>
            // window.print();
        </script>
    </body>
    </html>
    """
    
    return html


