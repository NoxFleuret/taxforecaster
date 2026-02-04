import streamlit as st

def apply_theme():
    # Custom CSS - Enterprise "Ministry of Finance" Theme (Premium V2)
    st.markdown("""
    <style>
        /* IMPORT FONT (Inter & JetBrains Mono) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Global Theme - SOPHISTICATED VISION OS DARK MODE */
        :root {
            --primary-bg: #0A0E17;       /* Void Navy */
            --secondary-bg: #151E2E;     /* Deep Slate */
            --primary-color: #3B82F6;    /* Electric Blue */
            --primary-glow: rgba(59, 130, 246, 0.5);
            --accent-color: #F59E0B;     /* Amber Gold */
            --success-color: #10B981;    /* Neon Emerald */
            --danger-color: #EF4444;     /* Alert Red */
            --text-primary: #F1F5F9;     /* Titanium White */
            --text-secondary: #94A3B8;   /* Muted Steel */
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-bg: rgba(30, 41, 59, 0.4);
        }
        
        .stApp {
            background: radial-gradient(circle at top right, #1e293b 0%, var(--primary-bg) 60%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif !important;
        }
        
        /* ANIMATIONS */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }

        /* HIDE STREAMLIT ANCHOR LINKS */
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, a.anchor-link {
            display: none !important;
        }

        /* HEADERS - Modern & Tight */
        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--text-primary) !important;
            font-weight: 700;
        }
        
        /* SIDEBAR styling */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 17, 32, 0.95) !important;
            border-right: 1px solid var(--glass-border);
            backdrop-filter: blur(20px);
        }
        
        /* METRIC CARDS - GLASSMORPHISM 2.0 */
        .metric-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.4));
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        }

        .metric-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary-color);
            box-shadow: 0 10px 30px -10px var(--primary-glow);
        }
        
        .metric-title {
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .metric-value {
            color: var(--text-primary);
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 4px;
            background: linear-gradient(180deg, #fff, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
        }
        
        /* PREMIUM CARDS (For Navigation) */
        .premium-card {
            background: var(--secondary-bg);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            height: 100%;
            animation: fadeIn 0.6s ease-out forwards;
        }
        .premium-card:hover {
            border-color: var(--primary-color);
            transform: translateY(-4px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            background: linear-gradient(145deg, var(--secondary-bg), #1e293b);
        }
        .premium-card h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }
        .premium-card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        /* Pill Badges for Delta */
        .metric-delta-pos {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .metric-delta-neg {
            background: rgba(239, 68, 68, 0.2);
            color: #F87171;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        /* CUSTOM BUTTONS */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), #2563EB);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px var(--primary-glow);
        }
        
    </style>
    """, unsafe_allow_html=True)

def display_sidebar_branding():
    """
    Consistent Sidebar Header.
    """
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #334155;">
            <h2 style="margin:0; font-size: 1.4rem; color: white;">🏛️ TaxForecaster</h2>
            <p style="margin:0; font-size: 0.8rem; color: #94A3B8;">Ministry of Finance AI Lab</p>
        </div>
        """, unsafe_allow_html=True)

def display_metric_card(title, value, delta=None, prefix="Rp", suffix=""):
    """
    Renders a Glassmorphism metric card.
    """
    delta_html = ""
    if delta is not None:
        color_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f"<div class='{color_class}'>{arrow} {abs(delta):.1f}% vs Last Year</div>"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{prefix} {value} {suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def get_plot_palette():
    """
    Returns a consistent Plotly color palette (Enterprise Blue/Gold/Teal).
    """
    return [
        "#3B82F6", # Blue
        "#10B981", # Emerald
        "#F59E0B", # Amber
        "#8B5CF6", # Violet
        "#EC4899", # Pink
        "#6366F1", # Indigo
        "#14B8A6", # Teal
    ]
