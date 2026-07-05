import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone


# =========================================================
# CONFIG
# =========================================================

# Columnas conocidas SOLO después de originado el préstamo
# (o directamente el origen del target) -> deben excluirse siempre,
# de lo contrario el modelo tiene fuga de información (data leakage).
LEAKAGE_COLS = [
    # fechas post-origination
    "issue_d",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
    # origen literal del target
    "loan_status",
    # pagos / cobranza (solo se conocen durante/después de la vida del préstamo)
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",
    "last_fico_range_high",
    "last_fico_range_low",
    "collections_12_mths_ex_med",
    "policy_code",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
]

# Identificadores / texto libre de alta cardinalidad: no aportan señal
# generalizable y, calculados vía WOE por valor exacto, casi garantizan
# overfitting (cada valor casi-único "separa perfectamente" good/bad).
ID_LIKE_COLS = [
    "id",
    "member_id",
    "url",
    "desc",
    "emp_title",
    "title",
    "zip_code",
]

LOW_IV_THRESHOLD = 0.01

# -----------------------------------------------------------------------
# WHITELIST DE BINNING
# -----------------------------------------------------------------------
# Regla por defecto: una columna numérica con pocos valores únicos ya es
# "casi categórica" (flags 0/1, conteos como delinq_2yrs, pub_rec, etc.)
# y NO se beneficia de binning por percentiles -- con datos tan sesgados
# (muchos ceros) el binning por percentiles suele colapsar casi toda la
# masa en un solo bin. Se trata como categoría directa (WOE por valor
# exacto), igual que una columna object.
#
# MIN_UNIQUE_FOR_BINNING: por debajo de este umbral de valores únicos en
# train, NO se bineá (se asume ya discreta/ordinal de pocos niveles).
MIN_UNIQUE_FOR_BINNING = 10

# Overrides explícitos por si la regla de cardinalidad no basta:
FORCE_BIN = []        # forzar binning aunque tenga pocos únicos
FORCE_NO_BIN = []      # forzar NO binning aunque tenga muchos únicos

# -----------------------------------------------------------------------
# MONOTONICIDAD DEL WOE
# -----------------------------------------------------------------------
# En un scorecard bancario real cada variable debe tener WOE monotónico
# respecto al riesgo (p.ej. a mayor dti, WOE consistentemente peor, sin
# zigzags): es lo que hace el modelo interpretable y defendible ante un
# comité de riesgos o un regulador. Se logra empezando con bins finos y
# fusionando bins adyacentes que rompen la tendencia hasta que la
# secuencia de WOE queda monotónica (o se llega a MONOTONIC_MIN_BINS).
#
# NOTA (supervised binning, no leakage clásico): create_monotonic_bins_fit
# usa y_train para decidir dónde fusionar bins. Esto es feature engineering
# supervisado -- práctica estándar y aceptada en scorecards de crédito
# (monotonic/WOE binning) -- pero debe quedar documentado como tal ante
# un comité o auditoría: no es leakage porque y_test nunca se toca, pero
# tampoco es un binning "no supervisado" ingenuo.
#
# LIMITACIÓN CONOCIDA: la monotonicidad se logra y se verifica sobre la
# muestra de X_train/y_train en el momento del fit. Es monotonicidad
# "local" a esa muestra, no una garantía estadística: en otra muestra
# (test, out-of-time, producción) la curva WOE podría dejar de ser
# perfectamente monotónica. Se recomienda re-validar visualmente la
# curva WOE de cada variable sobre test/OOT antes de presentar el
# scorecard a comité.
ENFORCE_MONOTONIC_WOE = True
MONOTONIC_INITIAL_BINS = 10   # granularidad inicial antes de fusionar
MONOTONIC_MIN_BINS = 2        # no fusionar por debajo de esto

# -----------------------------------------------------------------------
# RARE CATEGORY GROUPING
# -----------------------------------------------------------------------
# Categorías (o bins) con muy pocas observaciones en train tienen WOE
# ruidoso/inestable incluso con smoothing, y generalizan mal a test.
# Se agrupan en "OTHER" antes de calcular IV/WOE. "MISSING" nunca se
# agrupa, aunque sea rara: la ausencia de dato es semánticamente distinta
# de "categoría poco frecuente".
#
# IMPORTANTE (orden de operaciones vs. monotonicidad): el rare-grouping
# corre DESPUÉS de create_monotonic_bins_fit. Eso significa que la
# monotonicidad lograda durante el binning se calcula sobre los bins
# ANTES de que existan categorías "OTHER", pero el WOE final que ve el
# modelo (fit_woe, más abajo) se calcula DESPUÉS del rare-grouping. Un
# "OTHER" que agrupa dos o más bins con WOE distinto puede introducir un
# valor que no respeta el orden de sus vecinos, rompiendo la monotonicidad
# "final" aunque el binning haya sido monotónico en su momento. Por eso,
# más abajo en el pipeline se vuelve a verificar la monotonicidad del WOE
# final (post rare-grouping) con check_final_monotonicity() y se loguea
# un warning por columna donde se detecte una violación, en vez de asumir
# que el binning ya lo garantizó.
MIN_CATEGORY_COUNT = 30

# FIX (punto 2): antes solo se logueaba un warning si una columna perdía
# monotonicidad después del rare-grouping, pero el modelo la entrenaba
# igual -- "scorecard aparentemente monotónico pero no defendible
# operacionalmente". Con este flag en True, cualquier columna que falle
# check_final_monotonicity() se descarta del feature set final en vez de
# solo loguear. Es la opción "drop" (la más simple y segura de las tres
# sugeridas -- drop / rebin / fallback a binning simple); rebin
# automático post-grouping no se implementa acá porque volvería a correr
# el riesgo de que el rare-grouping rompa el nuevo binning otra vez.
DROP_NON_MONOTONIC_AFTER_GROUPING = True

# -----------------------------------------------------------------------
# CORRELATION PRUNING
# -----------------------------------------------------------------------
# Filtro simple de redundancia entre features YA transformadas a WOE:
# de cada par con correlación > CORR_THRESHOLD, se descarta la de menor
# IV. Es una aproximación más barata que VIF (Variance Inflation Factor)
# -- VIF detecta multicolinealidad multivariada real y sería más riguroso,
# pero requiere una regresión auxiliar por feature (statsmodels) y no
# está implementado aquí para no sumar dependencias.
#
# LIMITACIÓN CONOCIDA: el desempate dentro de cada par correlacionado usa
# el IV ya calculado sobre train (post filtro LOW_IV_THRESHOLD). El IV es
# una medida univariada de poder predictivo en esa muestra, no de
# estabilidad: dos variables con IV parecido pueden tener estabilidad muy
# distinta en el tiempo. Mejora futura sugerida: combinar IV con una
# medida de estabilidad (PSI de la variable entre sub-períodos de train,
# o coeficiente de variación del IV entre folds) antes de usar el IV como
# único criterio de desempate en el pruning por correlación.
CORR_THRESHOLD = 0.85

# FIX (punto 7): X.corr() sobre TODAS las filas de train es O(n_features^2)
# en memoria/tiempo pero también recorre las n_rows filas para cada par;
# con datasets grandes (300k+ filas y muchas features WOE) puede ser caro.
# Si len(X) supera este umbral, prune_correlated_features calcula la
# matriz de correlación sobre una muestra aleatoria en vez de todo train.
# La correlación entre features YA en escala WOE no debería cambiar
# demasiado con un sample representativo de este tamaño.
CORR_SAMPLE_MAX_ROWS = 50_000

# -----------------------------------------------------------------------
# TUNING ADAPTATIVO SEGÚN TAMAÑO DEL DATASET
# -----------------------------------------------------------------------
LARGE_DATASET_ROWS = 300_000

# En banca, un C elegido "automáticamente por CV" es menos auditable que
# uno fijo y justificado a mano (un regulador puede preguntar por qué ese
# valor). USE_CV_TUNING=True usa LogisticRegressionCV para EXPLORAR el
# espacio de C (útil en desarrollo/experimentación); una vez que sepas
# qué C funciona bien, poné USE_CV_TUNING=False y fijá FIXED_C con ese
# valor para tener un modelo reproducible y justificable en un solo número.
USE_CV_TUNING = True
FIXED_C = 1.0

# FIX: en modo auditable (USE_CV_TUNING=False) el objetivo es que el
# modelo sea 100% reproducible con un puñado de números fijos y
# justificados a mano. Antes, el solver se tomaba de get_cv_settings(),
# que depende del TAMAÑO del dataset -- si el dataset crecía/decrecía
# entre corridas (p.ej. cruzaba el umbral LARGE_DATASET_ROWS), el solver
# cambiaba solo (lbfgs <-> saga) y distintos solvers pueden converger a
# coeficientes ligeramente distintos con el mismo C, rompiendo la
# reproducibilidad que el modo FIXED_C pretende garantizar. Ahora el
# solver y max_iter también se fijan a mano en modo auditable.
FIXED_SOLVER = "lbfgs"
FIXED_MAX_ITER = 1000

# -----------------------------------------------------------------------
# IV STABILITY CHECK (punto 3)
# -----------------------------------------------------------------------
# El IV mide poder predictivo en UNA muestra, no estabilidad: una feature
# puede tener IV alto por sobreajuste a ruido de esa muestra puntual y
# perder poder predictivo en otro corte de los datos. Chequeo barato de
# estabilidad: partir X_train (post rare-grouping, pre-WOE) en dos
# mitades aleatorias, calcular IV de cada feature en cada mitad, y
# marcar como inestable cualquier feature cuya diferencia absoluta de IV
# entre mitades supere IV_STABILITY_MAX_DIFF. No reemplaza un verdadero
# análisis temporal/out-of-time, pero detecta el caso más obvio de
# feature que "gana" IV por ruido muestral.
CHECK_IV_STABILITY = True
IV_STABILITY_MAX_DIFF = 0.03
DROP_UNSTABLE_IV_FEATURES = False  # por defecto solo reporta, no descarta

# -----------------------------------------------------------------------
# CALIBRACIÓN DE PROBABILIDAD (punto 5)
# -----------------------------------------------------------------------
# class_weight="balanced" hace que predict_proba ya NO refleje la tasa
# de default real de la población (sobre-representa la clase minoritaria
# en el ajuste), así que "prob_default" es un pseudo-score útil para
# ranking/AUC pero no una PD calibrada. Con APPLY_CALIBRATION=True se
# entrena un CalibratedClassifierCV (Platt scaling por defecto) sobre el
# mismo X_train/y_train para llevar las probabilidades de vuelta a una
# escala interpretable como PD; se usa solo para las probabilidades de
# salida (predict_proba) -- los coeficientes/pesos que se reportan como
# "feature importance" siguen siendo los del modelo base sin calibrar,
# que es el que tiene la interpretación WOE-lineal directa.
APPLY_CALIBRATION = True
CALIBRATION_METHOD = "sigmoid"   # "sigmoid" (Platt) recomendado con pocos
                                   # datos; "isotonic" si hay volumen grande
CALIBRATION_CV_FOLDS = 5


def get_cv_settings(n_rows):
    """
    Ajusta la búsqueda de hiperparámetros según el tamaño de X_train:
    - datasets grandes (Lending Club: ~1M+ filas): menos valores de C y
      menos folds para no disparar el tiempo de cómputo, y solver 'saga'
      (escala mejor que 'lbfgs' con muchas filas, aunque necesita más
      iteraciones para converger).
    - datasets más chicos: búsqueda más fina; 'lbfgs' converge rápido y
      es suficientemente preciso.

    NOTA: estos settings son solo para USE_CV_TUNING=True (exploración).
    En modo auditable (USE_CV_TUNING=False) se usan FIXED_SOLVER y
    FIXED_MAX_ITER en su lugar, fijos independientemente del tamaño del
    dataset -- ver comentario en la sección de CONFIG.
    """
    if n_rows > LARGE_DATASET_ROWS:
        return {"Cs": 8, "cv": 3, "solver": "saga", "max_iter": 3000}
    else:
        return {"Cs": 15, "cv": 5, "solver": "lbfgs", "max_iter": 1000}


# =========================================================
# 1. VALIDATION
# =========================================================

def validate_schema(df, target):

    df = df.copy()

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    for col in df.columns:
        if col != target and (
            df[col].dtype == "object" or str(df[col].dtype).startswith("string")
        ):
            df[col] = df[col].astype(str).replace("nan", "MISSING")

    return df


# =========================================================
# 2. BINNING (FIT + APPLY CONSISTENTE)
# =========================================================

def create_bins_fit(series, bins=10):
    series = pd.to_numeric(series, errors="coerce")

    edges = np.nanpercentile(series, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)

    # fallback safety
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    else:
        # extender los bordes para que valores de test fuera del rango
        # visto en train (mínimo/máximo) caigan en el bin extremo en
        # lugar de convertirse en NaN silencioso.
        edges[0] = -np.inf
        edges[-1] = np.inf

    return edges


def apply_bins(series, edges):
    series = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(series, bins=edges, include_lowest=True).astype(str)
    # los NaN reales quedan como el string "nan" tras el astype(str);
    # se renombran a "MISSING" para que quede explícito en logs/auditoría
    # que es una categoría de ausencia de dato, no un bin numérico más.
    return binned.replace("nan", "MISSING")


def get_ordered_bin_labels(edges):
    """
    Devuelve las etiquetas de bin (como las genera apply_bins/pd.cut) en
    su orden ordinal natural, de menor a mayor. Se usa para verificar
    monotonicidad del WOE final en el orden real de los bins, sin
    depender del orden en que aparecen en woe_maps (que es orden de
    aparición/groupby, no orden ordinal).
    """
    intervals = pd.IntervalIndex.from_breaks(edges)
    return [str(iv) for iv in intervals]


def check_final_monotonicity(woe_map: dict, ordered_labels: list,
                              expected_direction: str = None) -> tuple:
    """
    Verifica si el WOE final (post rare-grouping, el que efectivamente
    usa el modelo vía apply_woe) es monotónico en el orden ordinal real
    de los bins, y en la misma dirección (creciente/decreciente) que se
    fijó durante el binning original.

    "OTHER" y "MISSING" no participan de esta secuencia porque no poseen
    orden natural dentro del continuo de la variable: "OTHER" es una
    agregación artificial de múltiples bins originales (potencialmente
    con WOE distinto entre sí, ver nota en MIN_CATEGORY_COUNT) y por
    definición carece de una posición ordinal única entre sus vecinos, y
    "MISSING" es una categoría de ausencia de dato, no un punto del eje
    numérico de la variable. Ambas SÍ tienen su propio WOE en woe_map
    (consultable directamente ahí, o en monotonicity_audit una vez
    persistido) -- lo único que queda excluido es su participación en el
    chequeo de tendencia ordinal.

    Si se pasa expected_direction ("increasing" o "decreasing", tal como
    la determina create_monotonic_bins_fit), no alcanza con que la
    secuencia sea monotónica en cualquier sentido: tiene que serlo en esa
    dirección específica. Esto detecta el caso en que el rare-grouping
    invirtió el sentido de la curva (p.ej. de creciente a decreciente),
    que un chequeo de "monotónico en algún sentido" dejaría pasar. Si
    expected_direction es None (dirección no determinada en el fit),
    se acepta monotonicidad en cualquier sentido, como antes.

    Devuelve (es_monotonico: bool, secuencia_woe_evaluada: list).
    """
    ordered_woe = [woe_map[b] for b in ordered_labels if b in woe_map]

    if len(ordered_woe) < 2:
        # con 0 o 1 bin ordinal restante (el resto cayó en OTHER/MISSING)
        # no hay secuencia que pueda violar monotonicidad.
        return True, ordered_woe

    diffs = np.diff(ordered_woe)

    if expected_direction == "increasing":
        is_monotonic = bool(np.all(diffs >= 0))
    elif expected_direction == "decreasing":
        is_monotonic = bool(np.all(diffs <= 0))
    else:
        is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    return is_monotonic, ordered_woe


def _bin_woe_sequence(series, y, edges, smoothing=0.5):
    """
    WOE por bin (en orden de bin, 0..n-1) para un set de edges dado.
    Se usa solo internamente durante la fusión de bins -- es un cálculo
    más liviano que pasar por woe_table/DataFrames completos.
    """
    codes = pd.cut(pd.to_numeric(series, errors="coerce"), bins=edges,
                    include_lowest=True, labels=False)

    tmp = pd.DataFrame({"bin": codes, "y": y.values})
    tmp = tmp.dropna(subset=["bin"])

    grouped = tmp.groupby("bin")["y"].agg(["count", "sum"])
    good = grouped["count"] - grouped["sum"]
    bad = grouped["sum"]

    good_adj = good + smoothing
    bad_adj = bad + smoothing

    woe = np.log((good_adj / good_adj.sum()) / (bad_adj / bad_adj.sum()))

    return woe.sort_index()


def create_monotonic_bins_fit(series, y, initial_bins=MONOTONIC_INITIAL_BINS,
                               min_bins=MONOTONIC_MIN_BINS):
    """
    Bins percentiles finos + fusión iterativa de bins adyacentes que
    rompen la tendencia de riesgo, hasta que el WOE quede monotónico.

    La dirección deseada (creciente/decreciente) se determina una sola
    vez vía correlación de Spearman entre la variable cruda y el target,
    y se mantiene fija durante toda la fusión.

    Devuelve (edges, direction), donde direction es "increasing",
    "decreasing", o None si no se pudo determinar (muy pocos bins o muy
    pocos datos no-nulos). Esa dirección se guarda y se reutiliza después
    en check_final_monotonicity: no alcanza con que el WOE final sea
    monotónico en CUALQUIER dirección, tiene que serlo en la misma
    dirección que se fijó acá -- si el rare-grouping invirtiera el
    sentido de la curva, un chequeo que solo pida "monotónico en algún
    sentido" no lo detectaría.

    NOTA: esta monotonicidad se verifica en el momento del fit, sobre
    los bins "crudos" (antes de rare-grouping). Ver check_final_monotonicity
    para la verificación posterior, sobre el WOE que efectivamente usa
    el modelo.
    """

    edges = create_bins_fit(series, bins=initial_bins)

    if len(edges) - 1 <= min_bins:
        return edges, None

    numeric_series = pd.to_numeric(series, errors="coerce")
    mask = numeric_series.notna()

    if mask.sum() < 2:
        return edges, None

    corr = numeric_series[mask].corr(y[mask].astype(float), method="spearman")
    direction = "increasing" if (corr is not None and corr >= 0) else "decreasing"

    while len(edges) - 1 > min_bins:

        woe_seq = _bin_woe_sequence(series, y, edges)

        if len(woe_seq) < 2:
            break

        diffs = woe_seq.diff().dropna()

        violations = diffs[diffs < 0] if direction == "increasing" else diffs[diffs > 0]

        if violations.empty:
            break

        # fusionar primero la violación más chica (la que probablemente
        # es ruido, no señal real) para perder la menor información posible
        worst_bin = violations.abs().idxmin()

        # el borde a eliminar es el que separa el bin (worst_bin - 1) del
        # bin worst_bin; como el bin 0 empieza en edges[0], ese borde es
        # edges[worst_bin]
        edge_pos_to_remove = int(worst_bin)
        edges = np.delete(edges, edge_pos_to_remove)

    return edges, direction


def group_rare_categories(train_series, test_series, min_count=MIN_CATEGORY_COUNT):
    """
    Colapsa en "OTHER" las categorías con menos de min_count observaciones
    en TRAIN. Cualquier valor de test no visto lo suficiente en train
    (incluyendo categorías nuevas) también cae en "OTHER".
    """

    # FIX (punto 1): en el pipeline actual, todas las columnas ya llegan
    # como string explícito antes de esta función (apply_bins y la rama
    # "no bin" en train_scorecard hacen .astype(str)). Se fuerza igual acá
    # de forma defensiva: si esta función se reutiliza en otro contexto
    # (p.ej. un notebook de exploración) con una columna dtype "category"
    # o con tipos mixtos, `.isin()` puede comportarse de forma inesperada
    # al comparar categorías no declaradas en el catálogo de la Categorical
    # contra strings sueltos como "OTHER"/"MISSING".
    train_series = train_series.astype(str)
    test_series = test_series.astype(str)

    counts = train_series.value_counts(dropna=False)

    keep_values = set(counts[counts >= min_count].index)
    keep_values.add("MISSING")  # nunca se agrupa, aunque sea rara

    train_grouped = train_series.where(train_series.isin(keep_values), "OTHER")
    test_grouped = test_series.where(test_series.isin(keep_values), "OTHER")

    return train_grouped, test_grouped


# =========================================================
# CORRELATION PRUNING
# =========================================================

def prune_correlated_features(X, iv, threshold=CORR_THRESHOLD,
                               sample_max_rows=CORR_SAMPLE_MAX_ROWS):
    """
    Filtro de redundancia sobre features YA en escala WOE (numéricas).
    Para cada par con correlación absoluta > threshold, descarta la de
    menor IV (se asume que iv está indexado por el mismo nombre de
    columna que X). Ver nota sobre VIF vs. correlación pairwise, y sobre
    la limitación de usar solo IV como criterio de desempate, en CONFIG.

    FIX (punto 6 - determinismo): antes se iteraba en el orden natural
    de X.columns. La decisión de qué feature descartar en cada par ya
    usaba IV, pero el orden de iteración igual afectaba el resultado en
    cadenas de 3+ features correlacionadas entre sí (una vez que una
    columna cae en `to_drop`, se salta sin reconsiderar). Se ordenan las
    columnas por IV descendente antes de armar la matriz de correlación,
    para que el algoritmo greedy siempre priorice conservar primero las
    features de mayor IV, sin importar el orden original de X.

    FIX (punto 7 - performance): X.corr() sobre TODAS las filas puede ser
    caro en RAM/tiempo con datasets grandes. Si len(X) > sample_max_rows,
    se calcula la correlación sobre una muestra aleatoria reproducible.
    """

    sorted_cols = sorted(X.columns, key=lambda c: iv.get(c, 0), reverse=True)
    X_sorted = X[sorted_cols]

    if len(X_sorted) > sample_max_rows:
        X_for_corr = X_sorted.sample(n=sample_max_rows, random_state=42)
    else:
        X_for_corr = X_sorted

    corr = X_for_corr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop = set()

    for col in upper.columns:
        for other in upper.index[upper[col] > threshold]:
            if col in to_drop or other in to_drop:
                continue

            iv_col = iv.get(col, 0)
            iv_other = iv.get(other, 0)

            to_drop.add(other if iv_col >= iv_other else col)

    return [c for c in X.columns if c not in to_drop]


# =========================================================
# 3. WOE / IV
# =========================================================

def woe_table(df, col, target, smoothing=0.5):

    grouped = df.groupby(col)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    # Laplace / add-k smoothing sobre los conteos (no sobre la proporción
    # final): evita WOE extremos/inestables en categorías con pocas
    # observaciones (p.ej. bins con 3-4 casos), más robusto que sumar un
    # epsilon plano al final.
    good_adj = grouped["good"] + smoothing
    bad_adj = grouped["bad"] + smoothing

    grouped["dist_good"] = good_adj / good_adj.sum()
    grouped["dist_bad"] = bad_adj / bad_adj.sum()

    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])

    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

    return grouped


def iv_stability_check(X_binned, y, target, max_diff=IV_STABILITY_MAX_DIFF,
                        random_state=42):
    """
    FIX (punto 3): chequeo barato de estabilidad del IV. Parte X_binned
    (post rare-grouping, pre-WOE) en dos mitades aleatorias del mismo
    tamaño, calcula IV de cada feature en cada mitad por separado, y
    devuelve un DataFrame con el IV en cada mitad y la diferencia
    absoluta entre ambas. Una diferencia grande sugiere que el IV total
    (calculado sobre todo train) está inflado por ruido muestral en una
    sub-región de los datos, no por señal real y estable.

    No reemplaza una validación temporal/out-of-time real (sigue siendo
    train vs. train, no train vs. futuro), pero es una señal de alerta
    barata de calcular en el mismo pipeline.
    """
    idx = X_binned.index.to_series().sample(frac=1.0, random_state=random_state).index
    half = len(idx) // 2
    idx_a, idx_b = idx[:half], idx[half:]

    df_a = X_binned.loc[idx_a].copy()
    df_a[target] = y.loc[idx_a]
    df_b = X_binned.loc[idx_b].copy()
    df_b[target] = y.loc[idx_b]

    iv_a = iv_summary(df_a, target)
    iv_b = iv_summary(df_b, target)

    stability = pd.DataFrame({"iv_half_a": iv_a, "iv_half_b": iv_b}).fillna(0)
    stability["abs_diff"] = (stability["iv_half_a"] - stability["iv_half_b"]).abs()
    stability["unstable"] = stability["abs_diff"] > max_diff

    return stability.sort_values("abs_diff", ascending=False)


def iv_summary(df, target):
    """
    IMPORTANTE: esto debe llamarse SOLO sobre datos ya "binneados"
    (columnas numéricas discretizadas + columnas categóricas crudas).
    Calcular IV sobre una columna numérica continua sin binning agrupa
    casi cada fila en su propia categoría, lo que infla artificialmente
    el IV (sobreajuste) y hace que el filtro de IV no filtre nada.
    """

    iv = {}

    for col in df.columns:
        if col == target:
            continue

        try:
            iv[col] = woe_table(df, col, target)["iv"].sum()
        except (TypeError, ValueError, KeyError) as e:
            warnings.warn(f"No se pudo calcular IV para '{col}': {e}")
            continue

    return pd.Series(iv).sort_values(ascending=False)


# =========================================================
# 4. WOE FIT / APPLY
# =========================================================

def fit_woe(df, target):

    maps = {}

    for col in df.columns:
        if col == target:
            continue

        maps[col] = woe_table(df, col, target)["woe"].to_dict()

    return maps


def apply_woe(df, maps):
    """
    FIX (punto 4): antes, tanto un "MISSING" real como una categoría
    nunca vista en train (fuera de "OTHER"/"MISSING", p.ej. un edge case
    de binning) caían en el mismo fillna(0) genérico, mezclando dos
    significados distintos (ausencia de dato vs. valor no visto). Ahora:
      - Si la columna tiene una categoría "MISSING" con su propio WOE
        calculado en train (caso normal: MISSING está en `m`), el
        .map(m) ya la resuelve correctamente sin pasar por fillna.
      - El fillna solo cubre el caso residual de un valor que ni siquiera
        aparece como key en `m` (verdaderamente no visto en train): en
        ese caso se usa el WOE de "MISSING" como aproximación neutra si
        existe, y 0 solo como último fallback si ni siquiera hay WOE de
        "MISSING" para esa columna.
    """

    df = df.copy()

    for col, m in maps.items():
        if col in df.columns:
            m = maps.get(col, {})
            missing_woe = m.get("MISSING", 0)

            mapped = df[col].map(m)
            mapped = mapped.fillna(missing_woe)

            df[col] = mapped.astype(float).fillna(0)

    return df


# =========================================================
# 5. SCORE
# =========================================================

# NOTA (calibración de probabilidad): score() asume que la salida de
# predict_proba de la regresión logística es directamente la PD real
# ("logistic output = PD"). En scorecards bancarios de producción esto
# rara vez es cierto tal cual -- especialmente si se usó
# class_weight="balanced" en el entrenamiento (como acá), que distorsiona
# las probabilidades predichas respecto a la tasa de default real de la
# población. Lo estándar es calibrar la salida del modelo antes de
# convertirla a score (p.ej. Platt scaling / regresión logística de
# calibración, isotonic regression, o un mapping empírico score-to-bad-rate
# por banda usando datos de validación). No se implementa acá para no
# sumar dependencias/alcance, pero debe quedar documentado como una
# limitación conocida antes de usar estos scores para decisiones reales
# de crédito.
def score(prob, base=600, pdo=50, odds=50):

    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds)

    return offset - factor * np.log((1 - prob) / prob)


def band(s):

    if s >= 750: return "A"
    if s >= 700: return "B"
    if s >= 650: return "C"
    if s >= 600: return "D"
    return "E"


# =========================================================
# 6. PIPELINE
# =========================================================

def train_scorecard(df):

    target = "target_default"

    df = validate_schema(df, target)

    # =========================
    # PARSEO DE FECHA
    # =========================
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])

    # ordenar por tiempo (CRÍTICO)
    df = df.sort_values("issue_d", kind="mergesort")

    # =========================
    # CREAR COHORTES TEMPORALES REALES
    # =========================

    # convertir a mes (cohortes reales tipo crédito)
    df["year_month"] = df["issue_d"].dt.to_period("M")

    # lista ordenada de meses reales
    months = sorted(df["year_month"].unique())

    # =========================
    # DEFINIR SPLIT POR TIEMPO REAL
    # =========================

    n_months = len(months)

    train_months = months[:int(n_months * 0.70)]
    calib_months = months[int(n_months * 0.70):int(n_months * 0.85)]
    test_months = months[int(n_months * 0.85):]

    train_df = df[df["year_month"].isin(train_months)]
    calib_df = df[df["year_month"].isin(calib_months)]
    test_df = df[df["year_month"].isin(test_months)]

    # =========================
    # DROP AUX COLUMN (IMPORTANTE)
    # =========================

    train_df = train_df.drop(columns=["year_month"])
    calib_df = calib_df.drop(columns=["year_month"])
    test_df = test_df.drop(columns=["year_month"])

    # =========================
    # LEAKAGE + ID CLEANING
    # =========================

    drop_cols = [c for c in (LEAKAGE_COLS + ID_LIKE_COLS) if c in df.columns]

    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    calib_df = calib_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # =========================
    # FEATURES / TARGET
    # =========================

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_calib = calib_df.drop(columns=[target])
    y_calib = calib_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    candidate_cols = list(X_train.columns)

    # =====================================================
    # BINNING FIT (SOBRE TODAS LAS CANDIDATAS, ANTES DEL IV)
    # =====================================================
    # El binning debe ocurrir ANTES de calcular IV: el IV se calcula
    # por categoría/bin, no por valor crudo. Si se calcula IV sobre
    # columnas numéricas continuas sin binning, el resultado queda
    # inflado y el filtro de IV deja de tener sentido (ver nota en
    # iv_summary).

    bin_maps = {}
    # columnas donde se intentó forzar monotonicidad -- se usa más abajo
    # para saber a qué columnas aplicarles check_final_monotonicity
    # (no tiene sentido chequear monotonicidad en columnas categóricas
    # o en columnas binneadas sin ENFORCE_MONOTONIC_WOE).
    monotonic_bin_cols = []
    # dirección esperada ("increasing"/"decreasing"/None) fijada en el
    # momento del binning -- se reutiliza en la verificación final para
    # detectar no solo si el WOE dejó de ser monotónico, sino si además
    # invirtió el sentido de la curva.
    monotonic_directions = {}

    X_train_binned = X_train.copy()
    X_calib_binned = X_calib.copy()
    X_test_binned = X_test.copy()

    for col in candidate_cols:

        # Usar is_numeric_dtype en vez de comparar contra "object":
        # columnas creadas con pd.cut(..., labels=[...]) (p.ej. fico_bucket)
        # tienen dtype "category", no "object" ni numérico. Comparar contra
        # "object" las trataba como numéricas por error, las convertía todas
        # a NaN vía pd.to_numeric, y las colapsaba en un solo bin sin señal.
        is_numeric = pd.api.types.is_numeric_dtype(X_train[col])
        n_unique = X_train[col].nunique(dropna=True) if is_numeric else 0

        should_bin = (
            is_numeric
            and col not in FORCE_NO_BIN
            and (col in FORCE_BIN or n_unique >= MIN_UNIQUE_FOR_BINNING)
        )

        if should_bin:

            if ENFORCE_MONOTONIC_WOE:
                edges, direction = create_monotonic_bins_fit(X_train[col], y_train)
                monotonic_bin_cols.append(col)
                monotonic_directions[col] = direction
            else:
                edges = create_bins_fit(X_train[col])

            bin_maps[col] = edges

            X_train_binned[col] = apply_bins(X_train[col], edges)
            X_calib_binned[col] = apply_bins(X_calib[col], edges)
            X_test_binned[col] = apply_bins(X_test[col], edges)

        else:
           X_train_binned[col] = X_train[col].astype(str).replace("nan", "MISSING")
           X_calib_binned[col] = X_calib[col].astype(str).replace("nan", "MISSING")
           X_test_binned[col] = X_test[col].astype(str).replace("nan", "MISSING")

    X_train = X_train_binned
    X_calib = X_calib_binned
    X_test = X_test_binned

    # =====================================================
    # RARE CATEGORY GROUPING (TRAIN ONLY DEFINE, ANTES DEL IV)
    # =====================================================
    # Igual razón que el binning: debe pasar ANTES de calcular IV, porque
    # una categoría con 2-3 casos infla el IV artificialmente (separa
    # "perfecto" por azar) y produce un WOE inestable que no generaliza.

    for col in X_train.columns:

        X_train[col], X_calib[col] = group_rare_categories(
            X_train[col],
            X_calib[col]
        )

        _, X_test[col] = group_rare_categories(
            X_train[col],
            X_test[col]
        )

    # =====================================================
    # IV FILTER (TRAIN ONLY, SOBRE DATOS YA BINNEADOS)
    # =====================================================

    train_df = X_train.copy()
    train_df[target] = y_train

    iv = iv_summary(train_df, target)

    keep_cols = iv[iv > LOW_IV_THRESHOLD].index.tolist()
    keep_cols = [c for c in keep_cols if c in X_train.columns]

    X_train = X_train[keep_cols]
    X_test = X_test[keep_cols]
    X_calib = X_calib[keep_cols]

    # =====================================================
    # FIX (punto 3): CHEQUEO DE ESTABILIDAD DEL IV
    # =====================================================
    # El IV usado arriba mide poder predictivo en TODO train, pero no
    # dice nada sobre estabilidad: una feature puede "ganar" IV por ruido
    # en una sub-región de la muestra. Se calcula el IV por separado en
    # dos mitades aleatorias de train (post rare-grouping, mismo dataset
    # que ya se usó para el filtro de arriba) y se reporta qué features
    # tienen una diferencia de IV grande entre mitades. Por defecto solo
    # se reporta (DROP_UNSTABLE_IV_FEATURES=False); esto es intencional,
    # ya que un split aleatorio no es un verdadero chequeo temporal y
    # descartar automáticamente en base a él podría ser demasiado
    # agresivo -- queda como señal para revisión manual, con la opción de
    # activar el descarte automático si se prefiere.
    if CHECK_IV_STABILITY and len(X_train.columns) > 0:
        train_df_for_stability = X_train.copy()
        stability = iv_stability_check(train_df_for_stability, y_train, target)

        unstable_feats = stability[stability["unstable"]].index.tolist()
        if unstable_feats:
            print(
                f"\n⚠️  IV inestable (diff > {IV_STABILITY_MAX_DIFF}) entre "
                f"mitades de train en {len(unstable_feats)} feature(s):"
            )
            print(stability.loc[unstable_feats])

            if DROP_UNSTABLE_IV_FEATURES:
                keep_cols = [c for c in X_train.columns if c not in unstable_feats]
                X_train = X_train[keep_cols]
                X_test = X_test[keep_cols]
                print(f"Descartadas por IV inestable: {unstable_feats}")

    # =====================================================
    # WOE FIT (TRAIN ONLY)
    # =====================================================

    train_df = X_train.copy()
    train_df[target] = y_train

    woe_maps = fit_woe(train_df, target)

    # =====================================================
    # FIX (puntos 4/6): VERIFICACIÓN DE MONOTONICIDAD FINAL
    # =====================================================
    # create_monotonic_bins_fit garantiza monotonicidad sobre los bins
    # CRUDOS, antes del rare-grouping. Pero el WOE que efectivamente usa
    # el modelo (woe_maps, recién calculado arriba) es POSTERIOR al
    # rare-grouping, que puede haber colapsado bins con WOE distinto en
    # un único "OTHER" y roto -- o incluso invertido -- el orden. Se
    # verifica acá, sobre el WOE final, en el orden ordinal real de los
    # bins, y en la misma dirección fijada durante el binning
    # (monotonic_directions). No se aborta el entrenamiento por esto,
    # pero queda registrado para revisión antes de presentar el
    # scorecard a comité.
    #
    # FIX (doble aviso): antes se logueaba el mismo evento dos veces
    # (un print manual + warnings.warn). Se deja únicamente warnings.warn
    # como mecanismo de alerta -- además de evitar la duplicación, permite
    # filtrarlo/redirigirlo con la maquinaria estándar de `warnings`
    # (warnings.filterwarnings, capturar en tests, silenciar en batch,
    # etc.), cosa que un print no ofrece.
    #
    # AUDITABILIDAD (OTHER/MISSING): check_final_monotonicity excluye
    # "OTHER"/"MISSING" de la secuencia ordinal porque no poseen orden
    # natural dentro del continuo de la variable (ver docstring de esa
    # función). Eso no significa que su WOE deba desaparecer de la vista:
    # se registra explícitamente para cada columna con binning
    # monotónico -- pase o no el chequeo -- en `monotonicity_audit`,
    # persistido en el pipeline_bundle para consulta posterior sin
    # recalcular nada, aunque ya no se imprima por consola en cada corrida.
    cols_to_drop_monotonic = []
    monotonicity_audit = {}

    for col in monotonic_bin_cols:
        if col not in X_train.columns or col not in woe_maps:
            # la columna pudo haber sido descartada por el filtro de IV
            continue

        expected_direction = monotonic_directions.get(col)
        ordered_labels = get_ordered_bin_labels(bin_maps[col])
        is_monotonic, evaluated_seq = check_final_monotonicity(
            woe_maps[col], ordered_labels, expected_direction=expected_direction
        )

        other_woe = woe_maps[col].get("OTHER")
        missing_woe = woe_maps[col].get("MISSING")

        monotonicity_audit[col] = {
            "is_monotonic": is_monotonic,
            "expected_direction": expected_direction,
            "ordinal_woe_sequence": evaluated_seq,
            "other_woe": other_woe,
            "missing_woe": missing_woe,
        }

        if not is_monotonic:
            warnings.warn(
                f"WOE final de '{col}' NO es monotónico después de "
                f"rare-grouping. Dirección esperada: "
                f"{expected_direction or 'desconocida'}. Secuencia WOE "
                f"obtenida (orden ordinal, sin OTHER/MISSING): "
                f"{evaluated_seq}. WOE(OTHER)={other_woe}, "
                f"WOE(MISSING)={missing_woe}. Revisar antes de presentar "
                f"a comité de riesgos."
            )

            if DROP_NON_MONOTONIC_AFTER_GROUPING:
                cols_to_drop_monotonic.append(col)

    # FIX (punto 2): antes solo se avisaba y el modelo entrenaba igual con
    # la columna no-monotónica. Si DROP_NON_MONOTONIC_AFTER_GROUPING está
    # activo, se descartan acá esas columnas del feature set final -- antes
    # de calcular baseline_distributions, para que el snapshot de drift
    # monitoring y el modelo queden consistentes con el mismo set de
    # features.
    if cols_to_drop_monotonic:
        print(
            f"\nDescartadas por perder monotonicidad tras rare-grouping: "
            f"{cols_to_drop_monotonic}"
        )
        X_train = X_train.drop(columns=cols_to_drop_monotonic, errors="ignore")
        X_test = X_test.drop(columns=cols_to_drop_monotonic, errors="ignore")
        for c in cols_to_drop_monotonic:
            woe_maps.pop(c, None)

    # snapshot de la distribución categórica/binneada de train ANTES de
    # convertir a WOE -- es la línea base contra la que un script de
    # monitoreo de drift (PSI) compara distribuciones futuras en
    # producción. Sin esto no hay con qué comparar más adelante.
    baseline_distributions = {
        col: X_train[col].value_counts(normalize=True).to_dict()
        for col in X_train.columns
    }

    X_train = apply_woe(X_train, woe_maps)
    X_calib = apply_woe(X_calib, woe_maps)
    X_test = apply_woe(X_test, woe_maps)

    # =====================================================
    # CLEAN + ALIGN
    # =====================================================

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_calib = X_calib.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # remove constant columns
    nunique = X_train.nunique()
    keep_final = nunique[nunique > 1].index

    X_train = X_train[keep_final]

    X_calib = X_calib.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # =====================================================
    # CORRELATION PRUNING (REDUNDANCIA ENTRE FEATURES)
    # =====================================================
    # El filtro de IV es univariado: no detecta que dos features midan
    # esencialmente lo mismo (p.ej. fico_range_low y fico_bucket ya
    # binneado). De cada par muy correlacionado se descarta la de menor IV.
    # (Ver nota en CORR_THRESHOLD sobre la limitación de usar solo IV
    # como criterio de desempate.)

    keep_uncorrelated = prune_correlated_features(X_train, iv)

    dropped_by_corr = [c for c in X_train.columns if c not in keep_uncorrelated]
    if dropped_by_corr:
        print(f"\nDescartadas por correlación alta (> {CORR_THRESHOLD}): {dropped_by_corr}")

    X_train = X_train[keep_uncorrelated]

    X_calib = X_calib[keep_uncorrelated]

    X_test = X_test[keep_uncorrelated]

    print("\nFINAL SHAPE:", X_train.shape)

    # =====================================================
    # MODEL
    # =====================================================

    # LogisticRegressionCV busca el mejor C vía validación cruzada SOLO
    # sobre X_train -- no toca X_test, así que no hay leakage por el
    # tuning. Cs/cv/solver se ajustan según el tamaño de X_train (ver
    # get_cv_settings): con datasets grandes se prueban menos valores y
    # se usa un solver que escala mejor, para no disparar el tiempo de
    # entrenamiento. scoring="roc_auc" porque es la métrica que reportamos
    # al final. class_weight="balanced" compensa el desbalance típico
    # (~15-20% default).
    cv_settings = get_cv_settings(len(X_train))

    if USE_CV_TUNING:
        print(f"\nAjuste de CV según tamaño de dataset ({len(X_train)} filas): {cv_settings}")

        model = LogisticRegressionCV(
            Cs=cv_settings["Cs"],
            cv=cv_settings["cv"],
            scoring="roc_auc",
            class_weight="balanced",
            solver=cv_settings["solver"],
            max_iter=cv_settings["max_iter"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        print(f"\nMejor C (regularización, vía CV): {model.C_[0]:.5f}")

    else:
        # FIX: modo "auditable" -- C, solver y max_iter fijos y
        # justificados a mano (ya no dependen de get_cv_settings, que
        # varía según el tamaño del dataset). Usar esta rama para el
        # modelo que efectivamente se documenta/presenta, una vez que ya
        # exploraste el espacio de C con USE_CV_TUNING=True.
        print(
            f"\nUsando configuración fija (auditable): "
            f"C={FIXED_C}, solver={FIXED_SOLVER}, max_iter={FIXED_MAX_ITER}"
        )

        model = LogisticRegression(
            C=FIXED_C,
            class_weight="balanced",
            solver=FIXED_SOLVER,
            max_iter=FIXED_MAX_ITER,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # =====================================================
    # FIX (punto 5): CALIBRACIÓN DE PROBABILIDAD
    # =====================================================
    # class_weight="balanced" distorsiona predict_proba respecto a la
    # tasa de default real de la población -- es correcto para que el
    # modelo aprenda bien con clases desbalanceadas, pero el resultado
    # deja de ser una PD interpretable tal cual. El flujo real de esta
    # sección, explícito para que quede sin ambigüedad en el código y en
    # los logs, es el mismo que se usa en un banco:
    #
    #   predict_proba (del modelo, sesgado por class_weight)
    #         -> Platt scaling / CalibratedClassifierCV
    #         -> PD calibrada (prob_default_calibrated)
    #         -> score()
    #
    # Si APPLY_CALIBRATION está activo, se entrena un CalibratedClassifierCV
    # (mismos hiperparámetros que `model`, sin data leakage porque hace su
    # propio CV interno solo sobre X_train/y_train) y su salida
    # (prob_default_calibrated) es la que efectivamente alimenta score()
    # más abajo -- NO la salida cruda de `model.predict_proba`. El `model`
    # original (sin calibrar) se mantiene intacto solo para reportar
    # coeficientes/feature importance, donde la escala WOE-lineal original
    # es la interpretación correcta; calibrar no tiene sentido a nivel de
    # coeficiente. La probabilidad cruda (sin calibrar) se conserva también
    # en el resultado final para trazabilidad/comparación, pero no se usa
    # para el cálculo del score.
    prob_default_raw = model.predict_proba(X_test)[:, 1]

    if APPLY_CALIBRATION:
        print(
            f"\nCalibrando probabilidades: predict_proba -> "
            f"{CALIBRATION_METHOD} scaling -> PD calibrada "
            f"(CalibratedClassifierCV, cv={CALIBRATION_CV_FOLDS})..."
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=clone(model),
            method=CALIBRATION_METHOD,
            cv=5
        )

        calibrated_model.fit(
            pd.concat([X_train, X_calib]),
            pd.concat([y_train, y_calib])
        )
        prob_default_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        print(
            "\nAPPLY_CALIBRATION=False: usando predict_proba cruda "
            "(sesgada por class_weight='balanced') directamente para el "
            "score. No recomendado para PD reportada a comité/regulador."
        )
        prob_default_calibrated = prob_default_raw

    # A partir de acá, `probs` es SIEMPRE la PD que efectivamente se usa
    # para score() y para las métricas reportadas -- calibrada si
    # APPLY_CALIBRATION=True, cruda si no.
    probs = prob_default_calibrated

    print("\nREPORT:")
    print(classification_report(y_test, (probs > 0.5).astype(int)))

    print("AUC (sobre PD usada para score):", roc_auc_score(y_test, probs))
    if APPLY_CALIBRATION:
        print(
            "AUC (sobre predict_proba cruda, sin calibrar):",
            roc_auc_score(y_test, prob_default_raw),
        )

    # =====================================================
    # SCORE OUTPUT
    # =====================================================

    scores = score(probs)

    results = pd.DataFrame({
        "prob_default_raw": prob_default_raw,
        "prob_default": probs,
        "score": scores
    }, index=X_test.index)

    results["band"] = results["score"].apply(band)

    print("\nDISTRIBUTION:")
    print(results["band"].value_counts(normalize=True))

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    fi = pd.DataFrame({
        "feature": X_train.columns,
        "weight": model.coef_[0]
    }).sort_values("weight", ascending=False)

    print("\nTOP FEATURES:")
    print(fi.head(15))

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs("data/model", exist_ok=True)

    joblib.dump(model, "data/model/scorecard.pkl")

    if APPLY_CALIBRATION:
        joblib.dump(
            calibrated_model,
            "data/model/calibrated_model.pkl"
        )

    joblib.dump(woe_maps, "data/model/woe_maps.pkl")
    joblib.dump(bin_maps, "data/model/bin_maps.pkl")
    joblib.dump(list(X_train.columns), "data/model/features.pkl")
    joblib.dump(results, "data/model/scored_clients.pkl")
    joblib.dump(iv, "data/model/iv_summary.pkl")
    joblib.dump(baseline_distributions, "data/model/baseline_distributions.pkl")
    joblib.dump(monotonicity_audit, "data/model/monotonicity_audit.pkl")

    # Artefacto único con todo lo necesario para reproducir el scoring de
    # un dataset nuevo (bin_maps -> woe_maps -> features -> model) sin
    # tener que recordar el orden de los .pkl individuales. Incluye
    # baseline_distributions para que src/monitoring/monitor_drift.py
    # pueda comparar producción vs. train sin depender de este script.
    # Incluye también monotonicity_audit: por cada columna con binning
    # monotónico, su secuencia ordinal de WOE, si pasó el chequeo final,
    # y el WOE explícito de "OTHER"/"MISSING" -- para poder responder
    # cualquier pregunta de auditoría sobre esas categorías sin tener
    # que reentrenar ni recalcular nada.
    pipeline_bundle = {
        "leakage_cols": LEAKAGE_COLS,
        "id_like_cols": ID_LIKE_COLS,
        "bin_maps": bin_maps,
        "woe_maps": woe_maps,
        "features": list(X_train.columns),
        "model": model,
        "calibrated_model": (
            calibrated_model
            if APPLY_CALIBRATION
            else None
        ),
        "baseline_distributions": baseline_distributions,
        "monotonicity_audit": monotonicity_audit,
    }
    joblib.dump(pipeline_bundle, "data/model/scorecard_pipeline.pkl")

    print("\nSAVED OK")

    return model, results, iv, woe_maps


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("🚀 START")

    df = pd.read_parquet("data/gold/credit_risk_features.parquet")

    print("DATA:", df.shape)

    train_scorecard(df)

    print("DONE")