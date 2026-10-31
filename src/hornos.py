"""
hornos.py — Punto 1: estudio de mercado de 5 marcas de hornos de cocción a gas.

Especificaciones del enunciado:
  • Potencia nominal 2.7 ± 0.22 kW  →  LIE = 2.48 kW,  LSE = 2.92 kW
  • Emisiones: la norma permite hasta 1000 ppm; por seguridad se exige ≤ 400 ppm
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import inferencia as inf

RAIZ = Path(__file__).resolve().parents[1]
LIE, LSE, NOMINAL = 2.48, 2.92, 2.70
PPM_SEGURIDAD, PPM_NORMA = 400, 1000


def cargar(ruta: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(ruta or RAIZ / "data" / "hornos.csv")


def marcas(df: pd.DataFrame) -> list[str]:
    return sorted(df["marca"].unique())


# -----------------------------------------------------------------------------
# a) Emisiones
# -----------------------------------------------------------------------------
def resumen_ppm(df: pd.DataFrame) -> pd.DataFrame:
    filas = {}
    for m, g in df.groupby("marca"):
        x = g["ppm"]
        q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
        ric = q3 - q1
        filas[m] = {
            "n": len(x), "Media": x.mean(), "Mediana": q2, "Q1": q1, "Q3": q3, "RIC": ric,
            "Bigote sup.": x[x <= q3 + 1.5 * ric].max(),
            "Atípicos": int((x > q3 + 1.5 * ric).sum() + (x < q1 - 1.5 * ric).sum()),
            "% ≤ 400 ppm": 100 * (x <= PPM_SEGURIDAD).mean(),
            "% > 1000 ppm": 100 * (x > PPM_NORMA).mean(),
        }
    return pd.DataFrame(filas).T


def comparar_ppm(df: pd.DataFrame) -> dict:
    """Kruskal–Wallis (¿alguna marca difiere?) y Mann–Whitney entre las dos mejores."""
    grupos = [g["ppm"].values for _, g in df.groupby("marca")]
    kw = stats.kruskal(*grupos)
    r = resumen_ppm(df).sort_values("Mediana")
    m1, m2 = r.index[0], r.index[1]
    mw = inf.mann_whitney(df.loc[df.marca == m1, "ppm"], df.loc[df.marca == m2, "ppm"])
    return {"H": float(kw.statistic), "p_kw": float(kw.pvalue), "mejor": m1, "segunda": m2,
            "mw": mw}


# -----------------------------------------------------------------------------
# b) Potencia
# -----------------------------------------------------------------------------
def resumen_potencia(df: pd.DataFrame) -> pd.DataFrame:
    filas = {}
    for m, g in df.groupby("marca"):
        c = inf.capacidad(g["potencia_kw"], LIE, LSE)
        filas[m] = {"n": len(g), "Media": c["media"], "Desv. (s)": c["s"],
                    "|x̄ − 2.70|": abs(c["media"] - NOMINAL),
                    "% dentro de espec.": 100 * c["dentro"], "Cp": c["Cp"], "Cpk": c["Cpk"],
                    "ppm fuera (teórico)": c["ppm_fuera_teorico"]}
    return pd.DataFrame(filas).T


# -----------------------------------------------------------------------------
# c) Desempeño global
# -----------------------------------------------------------------------------
def tabla_desempeno(df: pd.DataFrame) -> pd.DataFrame:
    """Cada marca se ordena (1 = mejor) en 5 criterios; gana la de menor suma de puestos."""
    rp = resumen_ppm(df)
    rw = resumen_potencia(df)
    rep = df.groupby("marca")["reparaciones"].mean()
    tasa = (df.groupby("marca")["reparaciones"].sum()
            / df.groupby("marca")["tiempo_uso_anios"].sum())
    t = pd.DataFrame({
        "Mediana ppm": rp["Mediana"],
        "% ≤ 400 ppm": rp["% ≤ 400 ppm"],
        "Cpk potencia": rw["Cpk"],
        "% potencia en espec.": rw["% dentro de espec."],
        "Reparaciones / año de uso": tasa,
    })
    puestos = pd.DataFrame({
        "Mediana ppm": t["Mediana ppm"].rank(),
        "% ≤ 400 ppm": t["% ≤ 400 ppm"].rank(ascending=False),
        "Cpk potencia": t["Cpk potencia"].rank(ascending=False),
        "% potencia en espec.": t["% potencia en espec."].rank(ascending=False),
        "Reparaciones / año de uso": t["Reparaciones / año de uso"].rank(),
    })
    t["Reparaciones promedio"] = rep
    t["Suma de puestos"] = puestos.sum(axis=1)
    t["Puesto final"] = t["Suma de puestos"].rank(method="min").astype(int)
    return t.sort_values("Puesto final")


def mejor_marca(df: pd.DataFrame) -> str:
    return tabla_desempeno(df).index[0]


# -----------------------------------------------------------------------------
# d) Intervalos para la mejor marca
# -----------------------------------------------------------------------------
def intervalos(df: pd.DataFrame, marca: str, confianza: float = 0.95) -> dict:
    g = df[df.marca == marca]
    cumple = int((g["ppm"] <= PPM_SEGURIDAD).sum())
    return {"media": inf.ic_media(g["potencia_kw"], confianza),
            "varianza": inf.ic_varianza(g["potencia_kw"], confianza),
            "proporcion": inf.ic_proporcion(cumple, len(g), confianza)}


# -----------------------------------------------------------------------------
# e) Variabilidad marcas 3 y 4
# -----------------------------------------------------------------------------
def variabilidad(df: pd.DataFrame, m1="Marca 3", m2="Marca 4", alpha=0.05) -> dict:
    x1 = df.loc[df.marca == m1, "potencia_kw"]
    x2 = df.loc[df.marca == m2, "potencia_kw"]
    return {"F": inf.prueba_f(x1, x2, alpha), "levene": inf.prueba_levene(x1, x2, alpha),
            "s1": x1.std(ddof=1), "s2": x2.std(ddof=1), "m1": m1, "m2": m2}


# -----------------------------------------------------------------------------
# f) Afirmación del fabricante: 80 de cada 100 cumplen
# -----------------------------------------------------------------------------
def afirmacion(df: pd.DataFrame, marca: str, p0: float = 0.80, alpha=0.05) -> dict:
    g = df[df.marca == marca]
    x = int((g["ppm"] <= PPM_SEGURIDAD).sum())
    return {"dos": inf.prueba_proporcion(x, len(g), p0, alpha, "dos"),
            "menor": inf.prueba_proporcion(x, len(g), p0, alpha, "menor")}


# -----------------------------------------------------------------------------
# Relaciones entre variables
# -----------------------------------------------------------------------------
VARIABLES = {"tiempo_uso_anios": "Tiempo de uso (años)", "reparaciones": "Reparaciones",
             "ppm": "Emisiones (ppm)", "potencia_kw": "Potencia (kW)"}


def matriz_correlacion(df: pd.DataFrame, metodo="pearson") -> pd.DataFrame:
    m = df[list(VARIABLES)].corr(method=metodo)
    m.index = m.columns = list(VARIABLES.values())
    return m


def regresion_ppm_uso(df: pd.DataFrame, marca: str | None = None) -> dict:
    """ln(ppm) = b₀ + b₁·uso  →  cada año de uso multiplica las emisiones por e^{b₁}."""
    g = df if marca is None else df[df.marca == marca]
    r = inf.regresion(g["tiempo_uso_anios"], np.log(g["ppm"]))
    r["pct_por_anio"] = 100 * (np.exp(r["b1"]) - 1)
    return r
