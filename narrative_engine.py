import pandas as pd
import numpy as np

def generate_insights(forecast_df, macro_dict=None, news_list=None):
    """
    Generates structured natural language insights based on Forecast, Macro, and News.
    """
    insights = {
        "summary": [],
        "risks": [],
        "opportunities": []
    }
    
    # --- 1. FORECAST ANALYSIS ---
    total_revenue = forecast_df['Nominal (Milyar)'].sum()
    monthly_avg = forecast_df.groupby('Tanggal')['Nominal (Milyar)'].sum().mean()
    
    # Identify Peaks
    monthly_sum = forecast_df.groupby('Tanggal')['Nominal (Milyar)'].sum()
    peak_month = monthly_sum.idxmax()
    peak_val = monthly_sum.max()
    
    insights["summary"].append(
        f"Projected total revenue is **Rp {total_revenue:,.0f} B**, with an average run rate of **Rp {monthly_avg:,.0f} B/month**."
    )
    insights["summary"].append(
        f"Revenue is expected to peak in **{peak_month.strftime('%B %Y')}** (Rp {peak_val:,.0f} B)."
    )
    
    # --- 2. MACRO CONTEXT (External Data) ---
    if macro_dict:
        # Oil Price Analysis
        oil_price = macro_dict.get('Harga_Minyak_ICP', 0)
        if oil_price > 85:
            insights["opportunities"].append(
                f"**High Oil Price (ICP ${oil_price:,.2f})**: Windfall potential for PPh Migas and PNBP SDA."
            )
        elif oil_price < 60:
            insights["risks"].append(
                f"**Low Oil Price (ICP ${oil_price:,.2f})**: Risk of shortfall in Oil & Gas related revenue."
            )
            
        # Exchange Rate
        kurs = macro_dict.get('Kurs_USD', 0)
        if kurs > 16000:
            insights["opportunities"].append(
                f"**IDR Depreciation (Rp {kurs:,.0f})**: Boosts nominal value of Import Duties and export proceeds."
            )
            insights["risks"].append(
                "Warning: Import volume may contract due to high exchange rate costs."
            )
            
        # Inflation
        inflation = macro_dict.get('Inflasi', 0)
        if inflation > 4.5:
            insights["risks"].append(
                f"**High Inflation ({inflation:.2f}%)**: May suppress household purchasing power, affecting domestic PPN."
            )
            
        # Global Sentiment (S&P 500 / Gold)
        gold = macro_dict.get('Harga_Emas', 0)
        if gold > 2100:
            insights["opportunities"].append(
                f"**Gold Rally (${gold:,.0f})**: Positive signal for Mining Sector tax contributions."
            )

    # --- 3. NEWS SENTIMENT INTEGRATION ---
    if news_list:
        # Simple Keyword Matching for narrative coloring
        # In a real AI model, this would use LLM
        risk_keywords = ['deficit', 'slowdown', 'recession', 'turun', 'anjlok', 'defisit', 'inflation']
        opp_keywords = ['surplus', 'growth', 'naik', 'tumbuh', 'windfall', 'profit']
        
        found_risk = False
        found_opp = False
        
        recent_headlines = [n['title'] for n in news_list[:3]]
        
        # Check sentiment in top 3 headlines
        for title in recent_headlines:
            t_lower = title.lower()
            if any(k in t_lower for k in risk_keywords):
                found_risk = True
            if any(k in t_lower for k in opp_keywords):
                found_opp = True
                
        if found_risk:
            insights["risks"].append("Market headlines indicate mixed/cautious sentiment regarding economic details.")
        if found_opp:
            insights["opportunities"].append("Recent news highlights positive growth momentum in key sectors.")
            
    return insights
