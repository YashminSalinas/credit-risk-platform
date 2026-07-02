import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# -----------------------
# PATHS
# -----------------------
LANDING = Path("data/landing")
BRONZE = Path("data/bronze")
LOGS = Path("logs")

BRONZE.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(exist_ok=True)

# -----------------------
# LOGGING
# -----------------------
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS / f"run_{RUN_ID}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# -----------------------
# INGESTION (ROBUST BRONZE)
# -----------------------
def ingest_file(file_name, output_name):
    file_path = LANDING / file_name

    log(f"STARTING INGESTION: {file_name}")

    chunksize = 100000
    first_chunk = True
    part = 0

    # limpiar archivos previos (overwrite real)
    output_file = BRONZE / f"{output_name}.parquet"
    if output_file.exists():
        output_file.unlink()

    for chunk in pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False,
        dtype=str  # 🔥 CLAVE: evita errores de memoria y tipos
    ):
        # escribir por partes (sin acumular en RAM)
        temp_file = BRONZE / f"{output_name}_part_{part}.parquet"

        chunk.to_parquet(temp_file, index=False)

        log(f"Saved {temp_file}")

        part += 1
        first_chunk = False

    log(f"COMPLETED: {file_name} | chunks={part}")

    return part

# -----------------------
# MAIN PIPELINE
# -----------------------
if __name__ == "__main__":
    start = datetime.now()

    log(f"PIPELINE STARTED | RUN_ID={RUN_ID}")

    acc_chunks = ingest_file(
        "accepted_2007_to_2018Q4.csv",
        "accepted"
    )

    rej_chunks = ingest_file(
        "rejected_2007_to_2018Q4.csv",
        "rejected"
    )

    log("PIPELINE COMPLETED SUCCESSFULLY")
    log(f"Accepted chunks: {acc_chunks}")
    log(f"Rejected chunks: {rej_chunks}")
    log(f"Duration: {datetime.now() - start}")