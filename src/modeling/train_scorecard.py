import pandas as pd

from src.scorecard.pipeline import train_scorecard

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("🚀 START")

    df = pd.read_parquet("data/gold/credit_risk_features.parquet")

    print("DATA:", df.shape)

    train_scorecard(df)

    print("DONE")
