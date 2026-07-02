import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from src.common.io_utils import load_all_accepted


# -----------------------
# PATHS
# -----------------------
BRONZE = Path("data/bronze")
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

# -----------------------
# LOGGING
# -----------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS / f"sprint2_2_{RUN_ID}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# -----------------------
# LOAD + TARGET
# -----------------------
def load_data():
    log("Loading dataset...")

    df = load_all_accepted()

    log(f"Raw data shape: {df.shape}")

    # -----------------------
    # TARGET CREATION
    # -----------------------
    df["loan_status"] = df["loan_status"].str.lower()

    bad = ["charged off", "default", "late (31-120 days)", "late (16-30 days)"]
    good = ["fully paid"]

    df = df[df["loan_status"].isin(bad + good)]

    df["target_default"] = df["loan_status"].apply(
        lambda x: 1 if x in bad else 0
    )

    log(f"After target filtering shape: {df.shape}")

    return df

# -----------------------
# TYPE CLEANING (MINI SILVER LAYER)
# -----------------------
def clean_types(df):

    log("Converting data types for analysis...")

    cols_numeric = [
        "fico_range_low",
        "fico_range_high",
        "int_rate",
        "annual_inc",
        "dti",
        "revol_util",
        "loan_amnt"
    ]

    for col in cols_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log("Type conversion completed")

    return df

# -----------------------
# EDA RISK ANALYSIS
# -----------------------
def risk_eda(df):

    log("Starting EDA analysis...")

    # -----------------------
    # FICO
    # -----------------------
    log("\nFICO analysis (mean by risk):")
    print(df.groupby("target_default")["fico_range_low"].mean())

    # -----------------------
    # INTEREST RATE
    # -----------------------
    log("\nInterest rate analysis:")
    print(df.groupby("target_default")["int_rate"].mean())

    # -----------------------
    # INCOME
    # -----------------------
    log("\nIncome analysis:")
    print(df.groupby("target_default")["annual_inc"].mean())

    # -----------------------
    # DTI
    # -----------------------
    log("\nDTI analysis:")
    print(df.groupby("target_default")["dti"].mean())

    log("EDA completed successfully")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    log(f"SPRINT 2.2 STARTED | RUN_ID={RUN_ID}")

    df = load_data()
    df = clean_types(df)
    risk_eda(df)

    log("SPRINT 2.2 COMPLETED SUCCESSFULLY")