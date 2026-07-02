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
# LOGGING SETUP
# -----------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS / f"sprint2_1_{RUN_ID}.log"

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

    log("Loading bronze dataset...")

    df = load_all_accepted()

    log(f"Data loaded | shape={df.shape}")

    return df

# -----------------------
# CREATE TARGET
# -----------------------
def create_target(df):
    log("Starting target creation (loan_status)")

    df["loan_status"] = df["loan_status"].str.lower()

    bad = ["charged off", "default", "late (31-120 days)", "late (16-30 days)"]
    good = ["fully paid"]

    df = df[df["loan_status"].isin(bad + good)]

    df["target_default"] = df["loan_status"].apply(
        lambda x: 1 if x in bad else 0
    )

    log("Target created successfully")

    log(f"Final shape after filtering: {df.shape}")

    log("Target distribution:")
    dist = df["target_default"].value_counts(normalize=True)

    for k, v in dist.items():
        log(f"{k}: {round(v*100,2)}%")

    return df

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    log(f"SPRINT 2.1 STARTED | RUN_ID={RUN_ID}")

    df = load_data()
    df = create_target(df)

    log("SPRINT 2.1 COMPLETED SUCCESSFULLY")