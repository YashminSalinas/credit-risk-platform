"""
Poda de redundancia entre features ya en escala WOE: de cada par muy
correlacionado, se descarta la de menor IV.
"""

import numpy as np

from .config import CORR_THRESHOLD, CORR_SAMPLE_MAX_ROWS

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
