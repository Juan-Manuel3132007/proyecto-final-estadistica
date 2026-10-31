"""
aplicaciones.py — Punto 3: la Estadística aplicada con DATOS REALES en las dos carreras
del grupo.

  A. Ingeniería de Sistemas — "¿Cuánto trabajo llega al repositorio y de qué tamaño?"
     Historial real de la rama principal de Streamlit (github.com/streamlit/streamlit),
     2024–2025: cada commit con su fecha, autor, archivos y líneas modificadas.

  B. Ingeniería Industrial — "¿El proceso es capaz y está bajo control?"
     Diámetros de anillos de pistón y latas de jugo no conformes: datos reales de
     Montgomery, *Introduction to Statistical Quality Control*, distribuidos en el
     paquete qcc de R.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import bondad as b
from . import inferencia as inf

RAIZ = Path(__file__).resolve().parents[1]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# =============================================================================
# A. Ingeniería de Sistemas
# =============================================================================
def cargar_sistemas():
    c = pd.read_csv(RAIZ / "data" / "sistemas_commits.csv")
    d = pd.read_csv(RAIZ / "data" / "sistemas_diario.csv")
    return c, d


def llegadas_diarias(diario: pd.DataFrame, alpha=0.05) -> dict:
    """¿El número de commits por día hábil sigue una Poisson? ¿Y una binomial negativa?"""
    x = diario.loc[diario["habil"], "commits"].to_numpy()
    media, var = x.mean(), x.var(ddof=1)
    tope = int(np.percentile(x, 99))
    poi = b.chi_cuadrado_datos_discretos(x, "Poisson", alpha, max_valor=tope)
    bn = b.chi_cuadrado_datos_discretos(x, "Binomial negativa", alpha, max_valor=tope)
    r, p = bn["params"]
    nb = stats.nbinom(r, p)
    return {"x": x, "n": len(x), "media": media, "var": var, "indice_dispersion": var / media,
            "poisson": poi, "binneg": bn,
            "p95_poisson": int(stats.poisson(media).ppf(0.95)), "p95_binneg": int(nb.ppf(0.95)),
            "p95_real": float(np.percentile(x, 95)),
            "ic_media": inf.ic_media(x)}


def tamano_cambios(commits: pd.DataFrame, alpha=0.05) -> dict:
    """Las líneas modificadas por commit: ¿siguen una lognormal?"""
    x = commits["lineas_cambiadas"].to_numpy(dtype=float)
    ranking = b.ranking(x, ("Lognormal", "Gamma", "Weibull", "Exponencial"))
    logn = b.pruebas_bondad(x, "Lognormal", alpha=alpha, n_mc=499, clases="equiprobables")
    return {"x": x, "ranking": ranking, "lognormal": logn,
            "mediana": float(np.median(x)), "media": float(x.mean()),
            "p90": float(np.percentile(x, 90))}


def correccion_vs_funcionalidad(commits: pd.DataFrame, alpha=0.05) -> dict:
    """¿Los commits de nuevas funcionalidades son más grandes que las correcciones?"""
    f = commits.loc[commits.tipo == "Funcionalidad", "lineas_cambiadas"]
    c = commits.loc[commits.tipo == "Corrección", "lineas_cambiadas"]
    return {"mw": inf.mann_whitney(f, c, alpha), "t_log": inf.prueba_t_dos(np.log(f), np.log(c), alpha),
            "n_f": len(f), "n_c": len(c), "med_f": float(f.median()), "med_c": float(c.median())}


def archivos_vs_lineas(commits: pd.DataFrame) -> dict:
    x, y = commits["archivos"], commits["lineas_cambiadas"]
    cor = inf.correlacion(x, y)
    cor_log = inf.correlacion(np.log10(x), np.log10(y))
    reg = inf.regresion(np.log10(x), np.log10(y))
    return {"cor": cor, "cor_log": cor_log, "reg": reg}


def proporcion_correcciones(commits: pd.DataFrame, alpha=0.05) -> dict:
    """Proporción de commits que son correcciones: 2024 vs 2025."""
    a = commits[commits.anio == 2024]
    b_ = commits[commits.anio == 2025]
    x1, n1 = int((a.tipo == "Corrección").sum()), len(a)
    x2, n2 = int((b_.tipo == "Corrección").sum()), len(b_)
    tot = int((commits.tipo == "Corrección").sum())
    return {"ic": inf.ic_proporcion(tot, len(commits)),
            "prueba": inf.prueba_dos_proporciones(x1, n1, x2, n2, alpha),
            "x1": x1, "n1": n1, "x2": x2, "n2": n2}


# =============================================================================
# B. Ingeniería Industrial
# =============================================================================
ANILLO_NOMINAL, ANILLO_TOL = 74.000, 0.050       # especificación 74.000 ± 0.050 mm
A2, D3, D4 = 0.577, 0.0, 2.114                    # constantes de cartas X̄–R para n = 5


def cargar_industrial():
    a = pd.read_csv(RAIZ / "data" / "industrial_anillos.csv")
    j = pd.read_csv(RAIZ / "data" / "industrial_jugo.csv")
    return a, j


def cartas_xr(anillos: pd.DataFrame) -> dict:
    """Límites de las cartas X̄ y R con la fase I (muestras 1–25) y se aplican a todas."""
    g = anillos.groupby("muestra")["diametro_mm"]
    tabla = pd.DataFrame({"xbar": g.mean(), "R": g.max() - g.min()})
    tabla["fase_I"] = anillos.groupby("muestra")["fase_I"].first()
    f1 = tabla[tabla.fase_I]
    xbb, rb = f1["xbar"].mean(), f1["R"].mean()
    lim = {"LC_x": xbb, "LSC_x": xbb + A2 * rb, "LIC_x": xbb - A2 * rb,
           "LC_r": rb, "LSC_r": D4 * rb, "LIC_r": D3 * rb}
    tabla["fuera_x"] = (tabla.xbar > lim["LSC_x"]) | (tabla.xbar < lim["LIC_x"])
    tabla["fuera_r"] = (tabla.R > lim["LSC_r"]) | (tabla.R < lim["LIC_r"])
    sigma_r = rb / 2.326                                  # d₂ = 2.326 para n = 5
    return {"tabla": tabla, "lim": lim, "sigma_r": sigma_r}


def analisis_anillos(anillos: pd.DataFrame, alpha=0.05) -> dict:
    f1 = anillos.loc[anillos.fase_I, "diametro_mm"]
    f2 = anillos.loc[~anillos.fase_I, "diametro_mm"]
    lie, lse = ANILLO_NOMINAL - ANILLO_TOL, ANILLO_NOMINAL + ANILLO_TOL
    return {
        "n1": len(f1), "n2": len(f2),
        "normal": b.pruebas_bondad(f1, "Normal", alpha=alpha),
        "cap": inf.capacidad(f1, lie, lse),
        "ic_media": inf.ic_media(f1), "t_nominal": inf.prueba_t_una(f1, ANILLO_NOMINAL, alpha),
        "t_fases": inf.prueba_t_dos(f1, f2, alpha), "f_fases": inf.prueba_f(f1, f2, alpha),
        "cartas": cartas_xr(anillos), "lie": lie, "lse": lse,
    }


def analisis_jugo(jugo: pd.DataFrame, alpha=0.05) -> dict:
    """Latas no conformes en muestras de 50: carta p y prueba de ajuste binomial."""
    n = int(jugo["tamano"].iloc[0])
    # Montgomery identifica causas asignables en las muestras 15 y 23 (fase I) → se excluyen
    excl = [15, 23]
    f1 = jugo[jugo.fase_I & ~jugo.muestra.isin(excl)]
    f2 = jugo[~jugo.fase_I]
    pbar = f1["no_conformes"].sum() / (n * len(f1))
    lsc = pbar + 3 * np.sqrt(pbar * (1 - pbar) / n)
    lic = max(0.0, pbar - 3 * np.sqrt(pbar * (1 - pbar) / n))
    binom = b.chi_cuadrado_datos_discretos(f1["no_conformes"], "Binomial", alpha, ensayos=n)
    prueba = inf.prueba_dos_proporciones(int(f1.no_conformes.sum()), n * len(f1),
                                         int(f2.no_conformes.sum()), n * len(f2), alpha)
    return {"n": n, "pbar": pbar, "lsc": lsc, "lic": lic, "excluidas": excl,
            "binomial": binom, "prueba": prueba,
            "p_fase2": f2["no_conformes"].sum() / (n * len(f2)),
            "k1": len(f1), "k2": len(f2)}
