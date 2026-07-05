"""
Binning de variables numéricas (percentiles + fusión iterativa para
forzar monotonicidad de WOE), verificación de monotonicidad final, y
agrupamiento de categorías raras ("OTHER"/"MISSING").

Todas las funciones acá son fit/apply explícitas: "*_fit" se llama solo
sobre train y devuelve algo (edges, dirección) que se persiste y se
reaplica igual sobre calib/test/producción vía apply_bins.
"""

import numpy as np
import pandas as pd

from .config import MONOTONIC_INITIAL_BINS, MONOTONIC_MIN_BINS, MIN_CATEGORY_COUNT

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
