"""
descriptiva.py — Medidas descriptivas (tendencia central, dispersión, posición y forma).
"""

import numpy as np
import pandas as pd
from scipy import stats


def resumen(x) -> dict:
    """Devuelve las medidas descriptivas de una muestra (desviación y varianza MUESTRALES, n-1)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    media = x.mean()
    s = x.std(ddof=1) if n > 1 else np.nan
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "n": n,
        "Media": media,
        "Mediana": q2,
        "Desv. estándar (s)": s,
        "Varianza (s²)": s ** 2,
        "Coef. de variación (%)": 100 * s / media if media != 0 else np.nan,
        "Mínimo": x.min(),
        "Q1": q1,
        "Q3": q3,
        "Máximo": x.max(),
        "Rango intercuartílico": iqr,
        "Asimetría": stats.skew(x, bias=False) if n > 2 else np.nan,
        "Curtosis (exceso)": stats.kurtosis(x, bias=False) if n > 3 else np.nan,
        "Atípicos (regla 1.5·RIC)": int(((x < li) | (x > ls)).sum()),
    }


def tabla_resumen(columnas: dict) -> pd.DataFrame:
    """columnas = {"nombre": datos, ...} → tabla con una columna por grupo."""
    return pd.DataFrame({nombre: resumen(v) for nombre, v in columnas.items()})


def numero_clases(n: int) -> int:
    """Regla de la guía: k es el menor entero positivo tal que 2^k > n."""
    k = 1
    while 2 ** k <= n:
        k += 1
    return k
