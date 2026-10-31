"""
montecarlo.py — Estudio de Monte Carlo: ¿qué prueba detecta mejor la diferencia?

Se repite MUCHAS veces el experimento del punto 2 (generar 60 datos por equipo y
aplicar las pruebas). La fracción de veces que cada prueba rechaza H₀ es:

  • su POTENCIA, si las poblaciones realmente son distintas, o
  • su ERROR TIPO I real, si las poblaciones son iguales.

Así se conecta este proyecto con el trabajo anterior de Monte Carlo y con el TLC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import bondad as b

ESCENARIOS = {
    "Escenario 1 (normales)": (
        lambda rng, n: rng.normal(175, 30, n),
        lambda rng, n: rng.normal(170, 50, n),
    ),
    "Escenario 2 (gamma)": (
        lambda rng, n: rng.gamma(1.2, 150, n),
        lambda rng, n: rng.gamma(0.95, 190, n),
    ),
}


def estudio_potencia(generar_a, generar_b, n: int = 60, repeticiones: int = 4000,
                     alpha: float = 0.05, semilla: int = 2026) -> dict:
    """Tasa de rechazo de t de Welch, Z y Mann–Whitney (vectorizado por bloques)."""
    rng = np.random.default_rng(semilla)
    a = np.array([generar_a(rng, n) for _ in range(repeticiones)])
    b = np.array([generar_b(rng, n) for _ in range(repeticiones)])
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    ee = np.sqrt(va / n + vb / n)
    dif = a.mean(axis=1) - b.mean(axis=1)
    est = dif / ee
    gl = (va / n + vb / n) ** 2 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
    p_t = 2 * stats.t.sf(np.abs(est), gl)
    p_z = 2 * stats.norm.sf(np.abs(est))
    p_mw = stats.mannwhitneyu(a, b, axis=1, alternative="two-sided", method="asymptotic").pvalue
    return {"t de Welch": float((p_t < alpha).mean()),
            "Z (normal)": float((p_z < alpha).mean()),
            "Mann–Whitney": float((p_mw < alpha).mean()),
            "repeticiones": repeticiones}


def prob_superioridad(generar_a, generar_b, tam: int = 400_000, semilla: int = 7) -> float:
    """P(X_A > X_B) estimada por simulación. Mann–Whitney contrasta H₀: P(X_A > X_B) = 0.5."""
    rng = np.random.default_rng(semilla)
    return float((generar_a(rng, tam) > generar_b(rng, tam)).mean())


def resumen_escenarios(repeticiones: int = 4000, semilla: int = 2026) -> dict:
    salida = {}
    for nombre, (ga, gb) in ESCENARIOS.items():
        r = estudio_potencia(ga, gb, repeticiones=repeticiones, semilla=semilla)
        r["P(A > B)"] = prob_superioridad(ga, gb)
        salida[nombre] = r
    # Control: dos poblaciones IGUALES (H₀ verdadera) → la tasa debe ser ≈ α = 5 %
    g = ESCENARIOS["Escenario 2 (gamma)"][0]
    r = estudio_potencia(g, g, repeticiones=repeticiones, semilla=semilla + 1)
    r["P(A > B)"] = 0.5
    salida["Control: gamma A vs gamma A (H₀ cierta)"] = r
    return salida


# =============================================================================
# Estudios de Monte Carlo sobre las pruebas de bondad de ajuste
# =============================================================================
def lilliefors(n=60, rep=2000, semilla=11):
    """¿Qué pasa con el valor p clásico de KS si μ y σ se estiman con la misma muestra?"""
    rng = np.random.default_rng(semilla)
    rech_clasico = rech_conocido = 0
    ds = []
    for _ in range(rep):
        x = rng.normal(100, 15, n)
        d_est = b.estadistico_ks(x, stats.norm(x.mean(), x.std(ddof=1)))["D"]
        ds.append(d_est)
        rech_clasico += stats.kstwo.sf(d_est, n) < 0.05
        rech_conocido += stats.kstwo.sf(b.estadistico_ks(x, stats.norm(100, 15))["D"], n) < 0.05
    return {"rech_clasico": rech_clasico / rep, "rech_conocido": rech_conocido / rep,
            "d95_est": float(np.percentile(ds, 95)), "d95_clasico": float(stats.kstwo.ppf(.95, n))}


def potencias_bondad(n=60, rep=600, semilla=5):
    """Potencia de χ², KS (Lilliefors) y AD para rechazar la NORMAL cuando los datos vienen de otra."""
    rng = np.random.default_rng(semilla)
    gen = {"t de Student (gl = 4): colas pesadas": lambda: rng.standard_t(4, n),
           "Gamma (α = 4): asimetría moderada": lambda: rng.gamma(4, 1, n),
           "Uniforme: sin colas": lambda: rng.uniform(0, 1, n)}
    # Valores críticos de KS y AD (con μ, σ estimados) por simulación bajo H₀
    nul_ks, nul_ad = [], []
    for _ in range(2000):
        z = rng.normal(0, 1, n)
        d = stats.norm(z.mean(), z.std(ddof=1))
        nul_ks.append(b.estadistico_ks(z, d)["D"])
        nul_ad.append(b.estadistico_ad(z, d))
    c_ks, c_ad = np.percentile(nul_ks, 95), np.percentile(nul_ad, 95)
    filas = []
    for nombre, fg in gen.items():
        r_chi = r_ks = r_ad = 0
        for _ in range(rep):
            x = fg()
            d = stats.norm(x.mean(), x.std(ddof=1))
            ch = b.chi_cuadrado_continua(x, d, 2, clases="equiprobables")
            r_chi += bool(ch["rechaza"])
            r_ks += b.estadistico_ks(x, d)["D"] > c_ks
            r_ad += b.estadistico_ad(x, d) > c_ad
        filas.append({"Datos generados desde": nombre, "χ² (equiprobables)": r_chi / rep,
                      "Kolmogorov–Smirnov": r_ks / rep, "Anderson–Darling": r_ad / rep})
    return pd.DataFrame(filas)


