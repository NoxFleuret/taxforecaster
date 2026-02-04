import os
import pandas as pd
import datetime
import shutil

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), 'snapshots')
DATA_FILE = os.path.join(os.path.dirname(__file__), 'macro_data_auto.csv')

def _ensure_snapshot_dir():
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

def save_snapshot(df, description="auto_save"):
    """Saves a dataframe as a timestamped CSV snapshot."""
    _ensure_snapshot_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{description}.csv"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    
    # Save metadata/data
    df.to_csv(filepath, index=False)
    print(f"[Versioning] Snapshot saved: {filename}")
    return filename

def list_snapshots():
    """Returns a list of available snapshots sorted by date (newest first)."""
    _ensure_snapshot_dir()
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.csv')]
    files.sort(reverse=True)
    return files

def load_snapshot(filename):
    """Loads a specific snapshot file into a DataFrame."""
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Snapshot {filename} not found.")
    return pd.read_csv(filepath)

def restore_snapshot(filename):
    """Restores a snapshot to be the active macro_data_auto.csv."""
    source = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Snapshot {filename} not found.")
    
    shutil.copy(source, DATA_FILE)
    print(f"[Versioning] Restored {filename} to {DATA_FILE}")
    return True
