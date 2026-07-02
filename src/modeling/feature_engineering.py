import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# =====================================================
# PATHS
# =====================================================
SILVER = Path("data/silver")
GOLD = Path("data/gold")
GOLD.mkdir(parents=True, exist_ok=True)

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=LOGS / f"sprint3_2_{RUN_ID}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# =====================================================
# LOAD SILVER
# =====================================================
def load_data():

    log("Loading silver dataset...")

    df = pd.read_parquet(SILVER / "credit_risk_dataset.parquet")

    log(f"Shape: {df.shape}")

    return df

# =====================================================
# FEATURE ENGINEERING
# =====================================================
def create_features(df):

    log("Creating features...")

    # -------------------------
    # Income features
    # -------------------------
    if "annual_inc" in df.columns:
        df["monthly_income"] = df["annual_inc"] / 12

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["income_to_loan_ratio"] = df["annual_inc"] / (df["loan_amnt"] + 1)

    # -------------------------
    # Risk flags
    # -------------------------
    if "dti" in df.columns:
        df["high_dti_flag"] = (df["dti"] > 20).astype(int)

    if "revol_util" in df.columns:
        df["high_revol_util_flag"] = (df["revol_util"] > 70).astype(int)

    # -------------------------
    # FICO buckets
    # -------------------------
    if "fico_range_low" in df.columns:

        df["fico_bucket"] = pd.cut(
            df["fico_range_low"],
            bins=[0, 580, 670, 740, 850],
            labels=["poor", "fair", "good", "excellent"]
        )

    log("Features created successfully")

    return df

# =====================================================
# SAVE GOLD
# =====================================================
def save_gold(df):

    output_path = GOLD / "credit_risk_features.parquet"

    df.to_parquet(output_path, index=False)

    log(f"Saved GOLD dataset → {output_path}")
    log(f"Final shape: {df.shape}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    log(f"SPRINT 3.2 STARTED | RUN_ID={RUN_ID}")

    df = load_data()
    df = create_features(df)
    save_gold(df)

    log("SPRINT 3.2 COMPLETED SUCCESSFULLY")