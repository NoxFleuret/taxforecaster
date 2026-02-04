import pandas as pd
import numpy as np

# --- Configuration: Validation Rules ---
VALIDATION_RULES = {
    'Pertumbuhan_Ekonomi': {'min': -15.0, 'max': 15.0, 'critical': True},
    'Inflasi': {'min': -5.0, 'max': 30.0, 'critical': True},
    'Kurs_USD': {'min': 8000.0, 'max': 25000.0, 'critical': True},
    'IHSG': {'min': 1000.0, 'max': 15000.0, 'critical': False},
    'Harga_Minyak_ICP': {'min': 10.0, 'max': 200.0, 'critical': False},
    'Harga_Batubara': {'min': 20.0, 'max': 600.0, 'critical': False},
    'Harga_CPO': {'min': 200.0, 'max': 3000.0, 'critical': False},
    'SBN_10Y': {'min': 1.0, 'max': 20.0, 'critical': True},
    'BI_Rate': {'min': 2.0, 'max': 15.0, 'critical': True},
    'Lifting_Minyak': {'min': 300.0, 'max': 1200.0, 'critical': False},
    'Lifting_Gas': {'min': 500.0, 'max': 2000.0, 'critical': False},
    'PMI_Manufaktur': {'min': 20.0, 'max': 80.0, 'critical': False},
    'Ekspor_Growth': {'min': -50.0, 'max': 100.0, 'critical': False},
    'Impor_Growth': {'min': -50.0, 'max': 100.0, 'critical': False},
    'Konsumsi_RT_Growth': {'min': -20.0, 'max': 20.0, 'critical': True},
}

# Score Weights
WEIGHTS = {
    'missing': 40,  # Penalty weight for missing data
    'outliers': 40, # Penalty weight for outliers
    'freshness': 20 # Penalty if recent data is missing
}

def validate_macro_data(df):
    """
    Validates the macro dataframe.
    Returns:
        report (dict): Detailed issues list.
        score (float): 0-100 quality score.
        outliers (pd.DataFrame): Subset of data containing outliers (for UI review).
    """
    issues = []
    outlier_indices = []
    total_cells = df.size
    total_rows = len(df)
    
    if df.empty:
        return {'status': 'ERROR', 'issues': ["Dataset is empty"]}, 0, pd.DataFrame()

    # 1. Missing Value Check
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / total_cells) * 100
    if missing_pct > 0:
        issues.append(f"Found {missing_count} missing values ({missing_pct:.1f}%).")

    # 2. Range/Outlier Check
    outlier_count = 0
    df_outliers_list = []

    for col, rules in VALIDATION_RULES.items():
        if col in df.columns:
            # Check Min
            mask_min = df[col] < rules['min']
            if mask_min.any():
                cnt = mask_min.sum()
                outlier_count += cnt
                issues.append(f"⚠️ {col}: {cnt} values below {rules['min']}")
                
                # Add to outlier detail list
                bad_rows = df[mask_min].copy()
                for idx, row in bad_rows.iterrows():
                    df_outliers_list.append({
                        'Index': idx,
                        'Tanggal': row.get('Tanggal', idx), # Fallback if Tanggal not col
                        'Indicator': col,
                        'Value': row[col],
                        'Issue': f"Too Low (< {rules['min']})",
                        'Suggested_Min': rules['min'],
                        'Suggested_Max': rules['max']
                    })

            # Check Max
            mask_max = df[col] > rules['max']
            if mask_max.any():
                cnt = mask_max.sum()
                outlier_count += cnt
                issues.append(f"⚠️ {col}: {cnt} values above {rules['max']}")

                bad_rows = df[mask_max].copy()
                for idx, row in bad_rows.iterrows():
                    df_outliers_list.append({
                        'Index': idx,
                        'Tanggal': row.get('Tanggal', idx),
                        'Indicator': col,
                        'Value': row[col],
                        'Issue': f"Too High (> {rules['max']})",
                        'Suggested_Min': rules['min'],
                        'Suggested_Max': rules['max']
                    })
    
    outlier_pct = (outlier_count / total_cells) * 100

    # 3. Freshness Check (Assuming 'Tanggal' exists or index is date-like)
    freshness_penalty = 0
    # Try to find date column
    date_col = None
    if 'Tanggal' in df.columns:
        date_col = df['Tanggal']
    
    if date_col is not None:
        try:
            # Handle DD/MM/YYYY format if string
            if df['Tanggal'].dtype == 'object':
                last_date = pd.to_datetime(df['Tanggal'], dayfirst=True).max()
            else:
                last_date = pd.to_datetime(df['Tanggal']).max()
            
            days_diff = (pd.Timestamp.now() - last_date).days
            if days_diff > 90:
                issues.append(f"⚠️ Data is stale. Last update: {last_date.date()} ({days_diff} days ago).")
                freshness_penalty = 20
            elif days_diff > 35:
                issues.append(f"ℹ️ Data slightly old. Last update: {last_date.date()}.")
                freshness_penalty = 5
        except:
            issues.append("Could not parse 'Tanggal' for freshness check.")
    
    # 4. Score Calculation
    # Cap missing/outlier penalties at assigned weights
    score = 100
    score -= min(missing_pct * 2, WEIGHTS['missing']) # weight missing heavily
    score -= min(outlier_pct * 5, WEIGHTS['outliers']) # weight outliers heavily
    score -= freshness_penalty
    
    score = max(0, round(score, 1))
    
    # 5. Construct Result
    status = "OK"
    if score < 50: status = "CRITICAL"
    elif score < 80: status = "WARNING"
    
    report = {
        'status': status,
        'issues': issues,
        'missing_count': int(missing_count),
        'outlier_count': int(outlier_count),
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    outliers_df = pd.DataFrame(df_outliers_list)
    
    return report, score, outliers_df

def validate_tax_data(df):
    """
    Validates the tax history dataframe.

    Args:
        df (pd.DataFrame): Tax history dataframe check.

    Returns:
        report (dict): Detailed issues list.
        score (float): 0-100 quality score.
        outliers (pd.DataFrame): Subset of data containing outliers.
    """
    issues = []
    outliers_list = []
    
    if df.empty:
        return {'status': 'ERROR', 'issues': ["Tax Dataset is empty"], 'missing_count': 0, 'outlier_count': 0}, 0, pd.DataFrame()
        
    required_cols = ['Tanggal', 'Jenis Pajak', 'Nominal (Milyar)']
    for c in required_cols:
        if c not in df.columns:
            return {'status': 'CRITICAL', 'issues': [f"Missing required column: {c}"], 'missing_count': 0, 'outlier_count': 0}, 0, pd.DataFrame()
            
    # 1. Missing Values
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / df.size) * 100
    if missing_count > 0:
        issues.append(f"Found {missing_count} missing cells.")
        
    # 2. Negative/Zero Revenue (Anomaly Check)
    # Revenue is typically positive. Negative might imply restitution > revenue, but often is data error.
    neg_mask = df['Nominal (Milyar)'] <= 0
    outlier_count = neg_mask.sum()
    
    if outlier_count > 0:
        issues.append(f"⚠️ Found {outlier_count} records with Zero or Negative revenue.")
        neg_rows = df[neg_mask].copy()
        for idx, row in neg_rows.iterrows():
            outliers_list.append({
                'Index': idx,
                'Tanggal': row['Tanggal'],
                'Indicator': row['Jenis Pajak'], # Use Tax Type as Indicator
                'Value': row['Nominal (Milyar)'],
                'Issue': "Revenue <= 0 (Potential Anomaly/Restitution)",
                'Suggested_Min': 0.01,
                'Suggested_Max': "N/A"
            })
            
    # 3. Freshness
    try:
        last_date = pd.to_datetime(df['Tanggal'], dayfirst=True).max()
        days_diff = (pd.Timestamp.now() - last_date).days
        if days_diff > 45:
             issues.append(f"ℹ️ Tax Data might be outdated. Last entry: {last_date.date()}.")
    except:
        issues.append("Could not parse Dates.")

    # Score
    score = 100 - min(missing_pct * 3, 40) - min((outlier_count/len(df))*100, 40)
    score = max(0, round(score, 1))
    
    status = "OK"
    if score < 60: status = "WARNING"
    if score < 40: status = "CRITICAL"
    
    report = {
        'status': status,
        'issues': issues,
        'missing_count': int(missing_count),
        'outlier_count': int(outlier_count),
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return report, score, pd.DataFrame(outliers_list)
