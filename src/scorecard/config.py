"""
Configuración central del pipeline de scorecard: columnas de leakage,
thresholds de IV/correlación, flags de comportamiento (monotonicidad,
calibración, tuning) y sus justificaciones.

Este módulo NO importa nada de los otros módulos del paquete -- todos
los demás (binning, woe, correlation, scoring, pipeline) importan sus
constantes desde acá. Cambiar un threshold o un flag se hace en un solo
lugar, sin tocar lógica.
"""

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

