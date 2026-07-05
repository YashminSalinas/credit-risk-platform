from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import numpy as np

# =====================================================
# PATHS
# =====================================================

GOLD = Path("data/gold")
MODEL = Path("data/model/datasets")
REPORTS = Path("reports")
LOGS = Path("logs")

MODEL.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=LOGS / f"feature_selection_{RUN_ID}.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def log(msg):
    print(msg)
    logging.info(msg)


# =====================================================
# CONFIGURATION
# =====================================================

TARGET = "target_default"

MISSING_THRESHOLD = 0.80

LEAKAGE_COLUMNS = [

    "loan_status",

    "last_pymnt_d",

    "last_pymnt_amnt",

    "next_pymnt_d",

    "recoveries",

    "collection_recovery_fee",

    "total_rec_prncp",

    "total_rec_int",

    "total_rec_late_fee",

    "total_pymnt",

    "total_pymnt_inv",

    "out_prncp",

    "out_prncp_inv"

]


removed_features = []


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    path = GOLD / "credit_risk_features.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log("Loading Gold dataset...")

    df = pd.read_parquet(path)

    log(f"Shape: {df.shape}")

    return df


# =====================================================
# VALIDATION
# =====================================================

def validate_schema(df):

    log("Validating schema...")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    before = len(df)

    df = df.drop_duplicates()

    log(f"Duplicates removed: {before - len(df)}")

    before = len(df)

    df = df.dropna(subset=[TARGET])

    log(f"Rows without target removed: {before - len(df)}")

    return df


# =====================================================
# REMOVE LEAKAGE
# =====================================================

def remove_leakage(df):

    log("Removing leakage columns...")

    cols = []

    for col in LEAKAGE_COLUMNS:

        if col in df.columns:

            cols.append(col)

            removed_features.append({
                "feature": col,
                "reason": "Data Leakage"
            })

    df = df.drop(columns=cols)

    log(f"Leakage columns removed: {len(cols)}")

    return df


# =====================================================
# REMOVE CONSTANT COLUMNS
# =====================================================

def remove_constant_columns(df):

    log("Removing constant columns...")

    constant_cols = []

    for col in df.columns:

        if col == TARGET:
            continue

        if df[col].nunique(dropna=False) <= 1:

            constant_cols.append(col)

            removed_features.append({
                "feature": col,
                "reason": "Constant Value"
            })

    df = df.drop(columns=constant_cols)

    log(f"Constant columns removed: {len(constant_cols)}")

    return df


# =====================================================
# REMOVE HIGH MISSING
# =====================================================

def remove_high_missing(df):

    log("Removing high-missing columns...")

    missing_ratio = df.isna().mean()

    cols = missing_ratio[
        missing_ratio > MISSING_THRESHOLD
    ].index.tolist()

    cols = [c for c in cols if c != TARGET]

    for col in cols:

        removed_features.append({
            "feature": col,
            "reason": "High Missing"
        })

    df = df.drop(columns=cols)

    log(f"High-missing columns removed: {len(cols)}")

    return df

# =====================================================
# CORRELATION FILTER
# =====================================================
CORRELATION_THRESHOLD = 0.95

def remove_high_correlation(df):

    log("Removing highly correlated features...")

    numeric_df = df.select_dtypes(include="number").drop(columns=[TARGET], errors="ignore")

    corr_matrix = numeric_df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > CORRELATION_THRESHOLD)
    ]

    for col in to_drop:
        removed_features.append({
            "feature": col,
            "reason": "High Correlation"
        })

    df = df.drop(columns=to_drop)

    log(f"Highly correlated columns removed: {len(to_drop)}")

    return df

# =====================================================
# IV CALCULATION
# =====================================================

def calculate_iv_table(df, feature, target):

    eps = 1e-6

    grouped = df.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()

    grouped["woe"] = np.log(
        (grouped["dist_good"] + eps) / (grouped["dist_bad"] + eps)
    )

    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    return grouped

def calculate_iv(df):

    iv_dict = {}

    for col in df.columns:

        if col == TARGET:
            continue

        # solo categóricas o low-cardinality
        if df[col].dtype == "object" or df[col].nunique() < 15:

            try:
                table = calculate_iv_table(df, col, TARGET)
                iv_dict[col] = table["iv"].sum()

            except:
                continue

    return pd.Series(iv_dict).sort_values(ascending=False)

def select_by_iv(df, iv_series, threshold=0.02):

    log("Selecting features by IV...")

    selected = iv_series[iv_series >= threshold].index.tolist()

    removed = iv_series[iv_series < threshold].index.tolist()

    for col in removed:
        removed_features.append({
            "feature": col,
            "reason": "Low IV"
        })

    return df[selected + [TARGET]], iv_series


# =====================================================
# SAVE DATASET
# =====================================================

def save_dataset(df):

    output = MODEL / "model_features.parquet"

    df.to_parquet(output, index=False)

    log(f"Dataset saved: {output}")

    log(f"Final shape: {df.shape}")


# =====================================================
# SAVE REPORT
# =====================================================

def save_report(df):

    report = pd.DataFrame(removed_features)

    report.to_csv(
        REPORTS / "feature_selection_report.csv",
        index=False
    )

    log("Feature Selection report saved.")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    log("=" * 60)
    log("FEATURE SELECTION PIPELINE")
    log("=" * 60)

    df = load_data()

    df = validate_schema(df)

    df = remove_leakage(df)

    df = remove_constant_columns(df)

    df = remove_high_missing(df)

    df = remove_high_correlation(df)

    iv = calculate_iv(df)

    log("\nIV SUMMARY:\n")
    log(iv.head(20).to_string())

    df, iv = select_by_iv(df, iv)

    save_dataset(df)

    save_report(df)

    log("PIPELINE FINISHED SUCCESSFULLY")