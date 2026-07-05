"""
Paquete del scorecard de credit risk: binning/monotonicidad, WOE/IV,
poda por correlación, scoring y el pipeline de entrenamiento completo.

Uso típico:

    from src.scorecard.pipeline import train_scorecard
    model, results, iv, woe_maps = train_scorecard(df)

Las funciones de bajo nivel (apply_bins, apply_woe, score, band, etc.)
se importan directamente desde su módulo cuando se necesiten fuera del
entrenamiento -- por ejemplo, en un futuro script de inferencia
(predict.py) o en src/monitoring/monitor_drift.py, sin tener que
importar ni ejecutar el módulo de entrenamiento completo.
"""

from .pipeline import train_scorecard

__all__ = ["train_scorecard"]
