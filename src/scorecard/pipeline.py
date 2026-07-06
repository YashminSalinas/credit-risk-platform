"""
Orquestador del entrenamiento del scorecard: split temporal, binning +
monotonicidad, rare-category grouping, filtro de IV, fit de WOE,
verificación de monotonicidad final, poda por correlación, entrenamiento
del modelo (con o sin CV), calibración de probabilidad, scoring y
persistencia de todos los artefactos (modelo, mapas, reportes).

Las funciones reutilizables (binning, WOE, correlación, scoring) viven
en sus propios módulos -- acá solo queda la lógica específica de ESTE
entrenamiento: `validate_schema` y `get_cv_settings` son helpers que no
tiene sentido reusar fuera de este pipeline, y `train_scorecard` es la
función que los orquesta a todos.
"""
import warnings
import os
import warnings
from pathlib import Path

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from .config import (
    LEAKAGE_COLS,
    ID_LIKE_COLS,
    FORCE_NO_BIN,
    FORCE_BIN,
    MIN_UNIQUE_FOR_BINNING,
    ENFORCE_MONOTONIC_WOE,
    LOW_IV_THRESHOLD,
    CHECK_IV_STABILITY,
    IV_STABILITY_MAX_DIFF,
    DROP_UNSTABLE_IV_FEATURES,
    DROP_NON_MONOTONIC_AFTER_GROUPING,
    CORR_THRESHOLD,
    USE_CV_TUNING,
    FIXED_C,
    FIXED_SOLVER,
    FIXED_MAX_ITER,
    LARGE_DATASET_ROWS,
    APPLY_CALIBRATION,
    CALIBRATION_METHOD,
    CALIBRATION_CV_FOLDS,
)
from .binning import (
    create_bins_fit,
    apply_bins,
    get_ordered_bin_labels,
    check_final_monotonicity,
    create_monotonic_bins_fit,
    group_rare_categories,
)
from .woe import iv_summary, iv_stability_check, fit_woe, apply_woe
from .correlation import prune_correlated_features
from .scoring import score, band
from .metrics import (
    evaluate_model,
    print_metrics,
    save_metrics,
)
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

MODEL_DATASETS = Path("data/model/datasets")
MODEL_DATASETS.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)

# Registro de auditoría: una entrada por cada columna descartada en
# cualquier punto del pipeline, con el motivo puntual. Reemplaza al CSV
# que generaba el feature_selection.py viejo (ahora archivado) -- la
# diferencia es que acá se registra el motivo REAL por el que este
# pipeline descarta cada columna (incluye pasos que feature_selection.py
# no tenía: no-monotonicidad post rare-grouping, IV inestable, etc.), no
# una lista aparte y desincronizada del pipeline que efectivamente
# entrena el modelo.
removed_features = []


def _log_removed(cols, reason):
    """Agrega columnas descartadas al registro de auditoría."""
    for col in cols:
        removed_features.append({"feature": col, "reason": reason})


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
# PIPELINE
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

    _log_removed(
        [c for c in drop_cols if c in LEAKAGE_COLS], "Data Leakage"
    )
    _log_removed(
        [c for c in drop_cols if c in ID_LIKE_COLS], "ID-like / Alta Cardinalidad"
    )

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

    _log_removed(
        [c for c in X_train.columns if c not in keep_cols],
        f"Low IV (<= {LOW_IV_THRESHOLD})",
    )

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
                _log_removed(unstable_feats, "Unstable IV (train vs. mitades)")
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
        _log_removed(cols_to_drop_monotonic, "Non-monotonic WOE post rare-grouping")
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

    _log_removed(
        [c for c in X_train.columns if c not in keep_final],
        "Constant Value (post-WOE)",
    )

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
        _log_removed(dropped_by_corr, f"High Correlation (> {CORR_THRESHOLD})")

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
    # MODEL METRICS
    # =====================================================

    metrics = evaluate_model(
        y_true=y_test,
        probabilities=probs
    )

    print_metrics(metrics)

    save_metrics(
        metrics,
        REPORTS / f"model_metrics_{RUN_ID}.csv"
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

    # =====================================================
    # FEATURE SELECTION REPORT (CSV) + MODEL FEATURES (PARQUET)
    # =====================================================
    # Reemplaza al feature_selection.py viejo (ahora archivado): ese
    # script generaba un CSV de "columna eliminada + motivo" y un parquet
    # con el dataset ya filtrado, pero corriendo su PROPIA lógica de
    # leakage/correlación/IV, desconectada de este pipeline y con al
    # menos un bug conocido (nunca calculaba IV para columnas numéricas
    # continuas de alta cardinalidad, así que las descartaba igual, pero
    # por el motivo equivocado). Acá el CSV registra los motivos REALES
    # por los que este pipeline descarta cada columna -- se arma en
    # paralelo, en cada punto donde ya se decide un descarte (ver
    # removed_features / _log_removed a lo largo de la función) -- y el
    # parquet es el dataset que efectivamente entrena el modelo, no una
    # reconstrucción aparte.
    report = pd.DataFrame(removed_features)
    report.to_csv(REPORTS / "feature_selection_report.csv", index=False)
    print(f"\nFeature selection report guardado: {REPORTS / 'feature_selection_report.csv'}")
    print(f"Columnas descartadas en total: {len(report)}")

    model_features_df = X_train.copy()
    model_features_df[target] = y_train.values
    model_features_df.to_parquet(
        MODEL_DATASETS / "model_features.parquet", index=False
    )
    print(
        f"Dataset final de features (WOE, post todos los filtros) guardado: "
        f"{MODEL_DATASETS / 'model_features.parquet'} | shape={model_features_df.shape}"
    )

    print("\nSAVED OK")

    return model, results, iv, woe_maps

