import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from src.common.io_utils import load_all_accepted


# =====================================================
# CONFIGURACIÓN EDITABLE
# =====================================================
LABEL_GOOD = "PAGADOR"
LABEL_BAD = "MOROSO"

# =====================================================
# PATHS
# =====================================================
BRONZE = Path("data/bronze")
LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS / f"sprint2_3_{RUN_ID}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log(msg):
    print(msg)
    logging.info(msg)

# =====================================================
# LOAD DATA + TARGET
# =====================================================
def load_data():

    log("Cargando dataset...")

    df = load_all_accepted()

    df["loan_status"] = df["loan_status"].astype(str).str.lower()

    bad = ["charged off", "default", "late (31-120 days)", "late (16-30 days)"]
    good = ["fully paid"]

    df = df[df["loan_status"].isin(bad + good)].copy()

    df["target_default"] = df["loan_status"].apply(
        lambda x: 1 if x in bad else 0
    )

    log(f"Dataset filtrado | shape={df.shape}")

    # validar distribución target
    dist = df["target_default"].value_counts(dropna=False)
    log(f"Distribución target:\n{dist}")

    if df.empty:
        raise ValueError("Dataset vacío después del filtrado de target")

    return df

# =====================================================
# VALIDACIÓN SEGURA DE COLUMNAS
# =====================================================
def safe_numeric(df, col):

    if col not in df.columns:
        log(f"WARNING: columna {col} no existe")
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[col], errors="coerce")

# =====================================================
# VALIDAR DATOS ANTES DE GRAFICAR
# =====================================================
def validate_series(name, s_good, s_bad):

    log(f"\nValidando {name}...")

    log(f"{LABEL_GOOD} size: {len(s_good)}")
    log(f"{LABEL_BAD} size: {len(s_bad)}")

    if len(s_good) == 0 and len(s_bad) == 0:
        raise ValueError(f"No hay datos para graficar en {name}")

# =====================================================
# HISTOGRAMAS
# =====================================================
def plot_distributions(df):

    df_good = df[df["target_default"] == 0]
    df_bad = df[df["target_default"] == 1]

    # =======================
    # FICO
    # =======================
    fico_good = safe_numeric(df_good, "fico_range_low").dropna()
    fico_bad = safe_numeric(df_bad, "fico_range_low").dropna()

    validate_series("FICO", fico_good, fico_bad)

    if len(fico_good) > 0 or len(fico_bad) > 0:

        plt.figure()
        plt.hist(fico_good, bins=30, alpha=0.5, label=LABEL_GOOD)
        plt.hist(fico_bad, bins=30, alpha=0.5, label=LABEL_BAD)

        plt.title("Distribución del Score FICO")
        plt.xlabel("FICO")
        plt.ylabel("Frecuencia")
        plt.legend()

        plt.show()
    else:
        log("SKIP FICO: sin datos válidos")

    # =======================
    # INTERÉS
    # =======================
    int_good = safe_numeric(df_good, "int_rate").dropna()
    int_bad = safe_numeric(df_bad, "int_rate").dropna()

    validate_series("INTEREST", int_good, int_bad)

    if len(int_good) > 0 or len(int_bad) > 0:

        plt.figure()
        plt.hist(int_good, bins=30, alpha=0.5, label=LABEL_GOOD)
        plt.hist(int_bad, bins=30, alpha=0.5, label=LABEL_BAD)

        plt.title("Distribución de la Tasa de Interés")
        plt.xlabel("Tasa de Interés")
        plt.ylabel("Frecuencia")
        plt.legend()

        plt.show()
    else:
        log("SKIP INTEREST: sin datos válidos")

    # =======================
    # INGRESO
    # =======================
    inc_good = safe_numeric(df_good, "annual_inc").dropna()
    inc_bad = safe_numeric(df_bad, "annual_inc").dropna()

    validate_series("INCOME", inc_good, inc_bad)

    if len(inc_good) > 0 or len(inc_bad) > 0:

        plt.figure()
        plt.hist(inc_good, bins=30, alpha=0.5, label=LABEL_GOOD)
        plt.hist(inc_bad, bins=30, alpha=0.5, label=LABEL_BAD)

        plt.title("Distribución del Ingreso Anual")
        plt.xlabel("Ingreso Anual")
        plt.ylabel("Frecuencia")
        plt.legend()

        plt.show()
    else:
        log("SKIP INCOME: sin datos válidos")

        

# =====================================================
# BOXPLOTS
# =====================================================
def boxplots(df):

    def safe_box(col, title):

        temp = pd.to_numeric(df[col], errors="coerce")

        if temp.dropna().empty:
            log(f"SKIP {col}: sin datos")
            return

        df_plot = df.copy()
        df_plot[col] = temp

        fig, ax = plt.subplots()

        df_plot.boxplot(column=col, by="target_default", ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Estado (0 = Pagador, 1 = Moroso)")
        ax.set_ylabel(col)

        # 👇 CLAVE: elimina figura automática duplicada de pandas
        plt.suptitle("")

        plt.show()

    safe_box("fico_range_low", "FICO vs Morosidad")
    safe_box("dti", "DTI vs Morosidad")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    log(f"SPRINT 2.3 INICIADO | RUN_ID={RUN_ID}")

    df = load_data()

    plot_distributions(df)
    boxplots(df)

    log("SPRINT 2.3 FINALIZADO")