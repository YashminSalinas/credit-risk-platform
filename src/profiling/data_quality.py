import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from src.common.io_utils import load_all_accepted

BRONZE = Path("data/bronze")
LOGS = Path("logs")

LOGS.mkdir(exist_ok=True)

# -----------------------
# LOGGING
# -----------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS / f"data_quality_{RUN_ID}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# -----------------------
# LOAD DATA
# -----------------------
def load_data():
    
    log("Loading dataset...")

    df = load_all_accepted()

    log(f"Raw data shape: {df.shape}")
    return df

# -----------------------
# DATA QUALITY
# -----------------------
def data_quality_report(df: pd.DataFrame):

    log("STARTING DATA QUALITY REPORT")

    # Shape
    log(f"Dataset shape: {df.shape}")

    # Missing values
    nulls = df.isnull().mean().sort_values(ascending=False)
    top_nulls = (nulls.head(15) * 100).round(2)

    log("Top missing values (%):")
    for col, val in top_nulls.items():
        log(f"{col}: {val}%")

    # Duplicates
    dup_count = df.duplicated().sum()
    log(f"Duplicate rows: {dup_count}")

    # Data types
    dtype_summary = df.dtypes.value_counts()
    log("Data type distribution:")
    for dtype, count in dtype_summary.items():
        log(f"{dtype}: {count}")

    # Numeric summary
    log("Generating numeric summary...")
    desc = df.describe().T.head(10)

    log("DATA QUALITY REPORT COMPLETED")

    return desc

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    log(f"DATA QUALITY PIPELINE STARTED | RUN_ID={RUN_ID}")

    df = load_data()
    report = data_quality_report(df)

    log("PIPELINE FINISHED SUCCESSFULLY")