import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data_validator
import data_versioning
import style

# --- Page Config ---
st.set_page_config(page_title="Data Quality Center", layout="wide", page_icon="🛡️")

style.apply_theme()

st.title("🛡️ Data Quality Center")
st.markdown("Monitor health, validate data, and manage snapshots of your macroeconomic indicators.")

# Load Data
# Load Data - Selection
dataset_type = st.sidebar.radio("Select Dataset to Audit:", ["🌍 Macroeconomic Indicators", "💰 Tax Revenue History"])

if dataset_type == "🌍 Macroeconomic Indicators":
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'macro_data_auto.csv')
    validate_func = data_validator.validate_macro_data
    is_macro = True
else:
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tax_history.csv')
    validate_func = data_validator.validate_tax_data
    is_macro = False

if not os.path.exists(DATA_FILE):
    st.error(f"Data file not found at {DATA_FILE}. Please upload or fetch data first.")
    st.stop()

@st.cache_data(ttl=60) # Short cache for updates
def load_data(path):
    """
    Loads data from a CSV file with caching.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(path)

df = load_data(DATA_FILE)

# Run Validation
report, score, outliers = validate_func(df)

# --- Top Stats ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Data Quality Score", f"{score}/100", delta=score-100 if score < 100 else 0)
with c2:
    st.metric("Total Records", len(df))
with c3:
    st.metric("Missing Values", report['missing_count'], delta=-report['missing_count'], delta_color="inverse")
with c4:
    st.metric("Outliers Detected", report['outlier_count'], delta=-report['outlier_count'], delta_color="inverse")

import seaborn as sns
import matplotlib.pyplot as plt

# --- Tabs ---
tab_names = ["📊 Overview", "🧐 Interactive Review", "💾 Snapshots"]
tab_overview, tab_review, tab_snapshots = st.tabs(tab_names)

with tab_overview:
    st.subheader("Validation Report")
    
    if report['status'] == 'OK':
        st.success("✅ Data Quality is Good!")
    elif report['status'] == 'WARNING':
        st.warning("⚠️ Data Quality Issues Detected")
    else:
        st.error("🚨 Critical Data Quality Issues")
    
    # --- VISUAL: Missing Data Matrix (Seaborn Heatmap) ---
    st.markdown("#### 🧩 Data Completeness Matrix (Heatmap)")
    st.caption("Visual map of missing values (Yellow = Missing, Purple = Present).")
    
    try:
        fig_miss, ax = plt.subplots(figsize=(10, 4))
        # Create a boolean mask: True if missing
        missing_matrix = df.isnull()
        if missing_matrix.sum().sum() > 0 or True: # Plot anyway to show "clean" data
            sns.heatmap(missing_matrix, cbar=False, yticklabels=False, cmap="viridis", ax=ax)
            if is_macro:
                ax.set_xlabel("Indicators")
            else:
                ax.set_xlabel("Columns")
            st.pyplot(fig_miss)
    except Exception as e:
        st.info("Could not render missing matrix.")

    if report['issues']:
        with st.expander("Details Issues List", expanded=True):
            for issue in report['issues']:
                st.write(f"- {issue}")
    else:
        st.info("No active issues found.")

with tab_review:
    st.subheader("Outlier & Anomaly Review")
    st.markdown("Below are values flagged as potential outliers/anomalies.")
    
    if is_macro:
        # --- MACRO VISUAL: Boxplot Grid ---
        st.markdown("#### 🔎 Outlier Detection Scanner")
        numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Year', 'Month'], errors='ignore')
        
        if not numeric_df.empty:
            selected_box_cols = st.multiselect("Select Indicators to View Distribution", numeric_df.columns, default=numeric_df.columns[:3] if len(numeric_df.columns)>=3 else numeric_df.columns)
            if selected_box_cols:
                 fig_box, ax = plt.subplots(figsize=(10, 6))
                 melted_sub = numeric_df[selected_box_cols].melt(var_name='Indicator', value_name='Value')
                 sns.boxplot(data=melted_sub, x='Value', y='Indicator', palette="vlag", ax=ax)
                 sns.stripplot(data=melted_sub, x='Value', y='Indicator', size=4, color=".3", linewidth=0, ax=ax)
                 sns.despine(left=True)
                 ax.grid(True, axis='x', linestyle='--', alpha=0.5)
                 st.pyplot(fig_box)
    else:
        # --- TAX VISUAL: Scatter Anomaly ---
        st.markdown("#### 🔎 Revenue Anomaly Scanner")
        if 'Jenis Pajak' in df.columns and 'Nominal (Milyar)' in df.columns:
             fig_tax, ax = plt.subplots(figsize=(10, 6))
             sns.scatterplot(data=df, x='Jenis Pajak', y='Nominal (Milyar)', hue='Jenis Pajak', ax=ax, legend=False)
             ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
             st.pyplot(fig_tax)

    st.markdown("---")

    if not outliers.empty:
        # 1. Display Outliers
        st.dataframe(outliers, use_container_width=True)
        
        # 2. Interactive Editing
        st.markdown("#### 🔧 Fix Data")
        
        # Select outlier to fix
        outlier_options = []
        for idx, row in outliers.iterrows():
             try:
                # Robust label generation
                label = f"{row['Indicator']} (Idx: {row['Index']}) - Val: {row['Value']}"
                outlier_options.append(label)
             except: pass
             
        if outlier_options:
            selected_outlier_str = st.selectbox("Select Record to Edit", outlier_options)
            
            if selected_outlier_str:
                # Find matching row based on string (simple index mapping)
                idx_in_outliers = outlier_options.index(selected_outlier_str)
                record = outliers.iloc[idx_in_outliers]
                
                real_index = record['Index']
                # Determine column name: Macro uses 'Indicator' col, Tax uses named column if tax specific?
                # For Tax: Indicator was mapped to 'Jenis Pajak' in my generic catch logic? 
                # Wait, data_validator tax logic mapped 'Indicator' to 'Jenis Pajak', but 'Value' came from 'Nominal'.
                # We need to edit 'Nominal (Milyar)' column for that row.
                if not is_macro:
                    col_name = 'Nominal (Milyar)' 
                else:
                    col_name = record['Indicator']
                
                check_col1, check_col2 = st.columns(2)
                with check_col1:
                    st.info(f"**Current Value:** {record['Value']}")
                    st.write(f"**Issue:** {record['Issue']}")
                    st.write(f"**Suggested Range:** {record['Suggested_Min']} - {record['Suggested_Max']}")
                    
                with check_col2:
                    new_val = st.number_input(f"New Value for {col_name} (Row {real_index})", value=float(record['Value']), format="%.2f")
                    
                    if st.button("Apply Fix"):
                        # Update DataFrame
                        df.at[real_index, col_name] = new_val
                        # Save Back
                        df.to_csv(DATA_FILE, index=False)
                        st.toast("✅ Value updated!", icon='🎉')
                        # Note: Snapshots for Manual Fixes on Tax not yet fully wired in data_versioning but saving to CSV works.
                        st.rerun()
        else:
            st.info("Outliers detected but display format issue.")
    else:
        st.success("No outliers requiring review at this time.")

with tab_snapshots:
    st.subheader("Version Control")
    # Only support Snapshots for Macro Data currently as built in data_versioning.py
    if is_macro:
        st.markdown("Restore previous versions of your data if needed.")
        
        snapshots = data_versioning.list_snapshots()
        
        if snapshots:
            selected_snap = st.selectbox("Select Snapshot", snapshots)
            
            if st.button("Restore this Version"):
                data_versioning.restore_snapshot(selected_snap)
                st.toast(f"Restored {selected_snap}!", icon='↺')
                st.rerun()
                
            st.write("---")
            st.write("Preview of selected snapshot:")
            try:
                prev_df = data_versioning.load_snapshot(selected_snap)
                st.dataframe(prev_df.head())
            except:
                st.error("Could not load preview.")
        else:
            st.info("No snapshots available yet.")
    else:
        st.info("Snapshot versioning currently enabled for Macro Indicators only.")
