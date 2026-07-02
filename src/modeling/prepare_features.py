import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from src.common.io_utils import load_all_accepted

# =====================================================
# PATHS
# =====================================================
BRONZE = Path("data/bronze")
SILVER = Path("data/silver")
SILVER.mkdir(parents=True, exist_ok=True)

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=LOGS / f"sprint3_1_{RUN_ID}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# =====================================================
# LOAD DATA
# =====================================================
def load_data():

    log("Loading bronze data...")

    df = load_all_accepted()

    log(f"Data loaded | shape={df.shape}")

    return df

# =====================================================
# CREATE TARGET
# =====================================================
def create_target(df):

    df["loan_status"] = df["loan_status"].astype(str).str.lower()

    bad = ["charged off", "default", "late (31-120 days)", "late (16-30 days)"]
    good = ["fully paid"]

    df = df[df["loan_status"].isin(bad + good)].copy()

    df["target_default"] = df["loan_status"].apply(
        lambda x: 1 if x in bad else 0
    )

    log("Target created")

    return df

# =====================================================
# CLEAN DATASET
# =====================================================
def clean_data(df):

    log("Cleaning dataset...")

    # eliminar columnas con +80% missing
    threshold = 0.8
    missing_ratio = df.isnull().mean()

    cols_to_keep = missing_ratio[missing_ratio < threshold].index
    df = df[cols_to_keep]

    log(f"Columns after missing filter: {df.shape[1]}")

    # convertir numéricos básicos
    numeric_cols = [
        "fico_range_low",
        "fico_range_high",
        "int_rate",
        "annual_inc",
        "dti",
        "loan_amnt",
        "revol_util"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# =====================================================
# SAVE SILVER
# =====================================================
def save_silver(df):

    output_path = SILVER / "credit_risk_dataset.parquet"

    df.to_parquet(output_path, index=False)

    log(f"Saved silver dataset → {output_path}")
    log(f"Final shape: {df.shape}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    log(f"SPRINT 3.1 STARTED | RUN_ID={RUN_ID}")

    df = load_data()
    df = create_target(df)
    df = clean_data(df)

    save_silver(df)

    log("SPRINT 3.1 COMPLETED SUCCESSFULLY")