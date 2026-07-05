"""
Transformación de probabilidad de default (PD) a score estilo scorecard
bancario (base/PDO/odds) y bandeo del score en categorías A-E.
"""

import numpy as np

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
