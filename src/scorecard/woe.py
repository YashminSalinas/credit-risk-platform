"""
Weight of Evidence (WOE) e Information Value (IV): cálculo por columna,
chequeo de estabilidad del IV entre mitades de train, y fit/apply del
mapeo WOE (train-only fit, aplicado luego a calib/test/producción).
"""

import warnings

import numpy as np
import pandas as pd

from .config import IV_STABILITY_MAX_DIFF

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
