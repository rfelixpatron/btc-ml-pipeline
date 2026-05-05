import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_ml.config import load_config
from btc_ml.utils.io import load_parquet

cfg = load_config()
print(f"Checking path: {cfg.paths.btc_1min_file}")
try:
    df = load_parquet(cfg.paths.btc_1min_file)
    print(f"Success! Loaded {len(df)} rows.")
    last_ts = int(df.index.max().timestamp())
    print(f"Last timestamp: {last_ts}")
except Exception as e:
    print(f"Error: {e}")
