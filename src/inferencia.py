"""
inferencia.py — Intervalos de confianza y pruebas de hipótesis.

Cada función devuelve un diccionario con TODOS los números intermedios (estadístico,
valor crítico, valor p, decisión) para poder mostrarlos paso a paso en la app y en
el informe, igual que se hace a mano en clase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _limpia(x):
    return np.asarray(pd.Series(x).dropna(), dtype=float)


# =============================================================================
# Intervalos de confianza
# =============================================================================
def ic_media(x, confianza=0.95) -> dict:
    """x̄ ± t_{α/2, n−1} · s/√n   (σ desconocida)."""
    x = _limpia(x)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    t = stats.t.ppf(1 - (1 - confianza) / 2, n - 1)
    e = t * s / np.sqrt(n)
    return {"n": n, "media": m, "s": s, "t": t, "gl": n - 1, "error": e,
            "li": m - e, "ls": m + e, "confianza": confianza}


def ic_varianza(x, confianza=0.95) -> dict:
    """[(n−1)s² / χ²_{α/2},  (n−1)s² / χ²_{1−α/2}]   (requiere normalidad)."""
    x = _limpia(x)
    n, s2 = len(x), x.var(ddof=1)
    a = 1 - confianza
    chi_der = stats.chi2.ppf(1 - a / 2, n - 1)   # deja α/2 a la derecha
    chi_izq = stats.chi2.ppf(a / 2, n - 1)       # deja α/2 a la izquierda
    return {"n": n, "s2": s2, "gl": n - 1, "chi_der": chi_der, "chi_izq": chi_izq,
            "li": (n - 1) * s2 / chi_der, "ls": (n - 1) * s2 / chi_izq, "confianza": confianza}


def ic_proporcion(exitos: int, n: int, confianza=0.95) -> dict:
    """p̂ ± z_{α/2} √(p̂(1−p̂)/n)  (Wald) y, como respaldo, el intervalo de Wilson."""
    p = exitos / n
    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    e = z * np.sqrt(p * (1 - p) / n)
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    radio = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return {"n": n, "x": exitos, "p": p, "z": z, "error": e, "li": p - e, "ls": p + e,
            "wilson_li": centro - radio, "wilson_ls": centro + radio, "confianza": confianza}


# =============================================================================
# Pruebas de hipótesis
# =============================================================================
def _decision(p, alpha):
    return bool(p < alpha)


def prueba_proporcion(exitos: int, n: int, p0: float, alpha=0.05, cola="dos") -> dict:
    """Z = (p̂ − p₀) / √(p₀(1−p₀)/n)."""
    p = exitos / n
    z = (p - p0) / np.sqrt(p0 * (1 - p0) / n)
    if cola == "dos":
        pval, crit = 2 * stats.norm.sf(abs(z)), stats.norm.ppf(1 - alpha / 2)
    elif cola == "menor":
        pval, crit = stats.norm.cdf(z), stats.norm.ppf(alpha)
    else:
        pval, crit = stats.norm.sf(z), stats.norm.ppf(1 - alpha)
    return {"p_hat": p, "p0": p0, "n": n, "x": exitos, "z": z, "critico": crit, "p": pval,
            "rechaza": _decision(pval, alpha), "cola": cola, "alpha": alpha}


def prueba_dos_proporciones(x1: int, n1: int, x2: int, n2: int, alpha=0.05) -> dict:
    """Z = (p̂₁ − p̂₂) / √(p̄(1−p̄)(1/n₁ + 1/n₂)),  con p̄ = (x₁ + x₂)/(n₁ + n₂)."""
    p1, p2 = x1 / n1, x2 / n2
    pc = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pc * (1 - pc) * (1 / n1 + 1 / n2))
    pval = 2 * stats.norm.sf(abs(z))
    ee = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"p1": p1, "p2": p2, "p_comb": pc, "z": z, "p": pval, "critico": zc,
            "rechaza": _decision(pval, alpha), "ic": (p1 - p2 - zc * ee, p1 - p2 + zc * ee)}


def prueba_f(x1, x2, alpha=0.05) -> dict:
    """H₀: σ₁² = σ₂²  vs  H₁: σ₁² ≠ σ₂².  F = s₁²/s₂² ~ F(n₁−1, n₂−1). Pone la mayor arriba."""
    x1, x2 = _limpia(x1), _limpia(x2)
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    invertido = v2 > v1
    if invertido:
        x1, x2, v1, v2 = x2, x1, v2, v1
    gl1, gl2 = len(x1) - 1, len(x2) - 1
    F = v1 / v2
    pval = min(1.0, 2 * stats.f.sf(F, gl1, gl2))
    return {"F": F, "s2_num": v1, "s2_den": v2, "gl1": gl1, "gl2": gl2,
            "critico": stats.f.ppf(1 - alpha / 2, gl1, gl2), "p": pval,
            "rechaza": _decision(pval, alpha), "invertido": invertido,
            "ic_razon": (F / stats.f.ppf(1 - alpha / 2, gl1, gl2),
                         F * stats.f.ppf(1 - alpha / 2, gl2, gl1))}


def prueba_levene(x1, x2, alpha=0.05) -> dict:
    """Levene (centrada en la mediana, Brown–Forsythe): no exige normalidad."""
    r = stats.levene(_limpia(x1), _limpia(x2), center="median")
    return {"W": float(r.statistic), "p": float(r.pvalue), "rechaza": _decision(r.pvalue, alpha)}


def prueba_t_dos(x1, x2, alpha=0.05, varianzas_iguales: bool | None = None) -> dict:
    """Comparación de medias con t-Student.

    Si varianzas_iguales es None se decide con la prueba F (como se ve en clase):
      • iguales  → t combinada (pooled), gl = n₁ + n₂ − 2
      • distintas → t de Welch, gl de Satterthwaite
    """
    x1, x2 = _limpia(x1), _limpia(x2)
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    if varianzas_iguales is None:
        varianzas_iguales = not prueba_f(x1, x2, alpha)["rechaza"]
    if varianzas_iguales:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        ee = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        gl = n1 + n2 - 2
    else:
        sp2 = np.nan
        ee = np.sqrt(v1 / n1 + v2 / n2)
        gl = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = (m1 - m2) / ee
    pval = 2 * stats.t.sf(abs(t), gl)
    crit = stats.t.ppf(1 - alpha / 2, gl)
    return {"tipo": "combinada" if varianzas_iguales else "Welch", "m1": m1, "m2": m2,
            "dif": m1 - m2, "ee": ee, "sp2": sp2, "gl": gl, "t": t, "critico": crit, "p": pval,
            "rechaza": _decision(pval, alpha), "ic": (m1 - m2 - crit * ee, m1 - m2 + crit * ee)}


def prueba_z_dos(x1, x2, alpha=0.05) -> dict:
    """Z = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)  — válida por el TLC cuando n₁, n₂ ≥ 30."""
    x1, x2 = _limpia(x1), _limpia(x2)
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    ee = np.sqrt(x1.var(ddof=1) / n1 + x2.var(ddof=1) / n2)
    z = (m1 - m2) / ee
    pval = 2 * stats.norm.sf(abs(z))
    crit = stats.norm.ppf(1 - alpha / 2)
    return {"m1": m1, "m2": m2, "dif": m1 - m2, "ee": ee, "z": z, "critico": crit, "p": pval,
            "rechaza": _decision(pval, alpha), "ic": (m1 - m2 - crit * ee, m1 - m2 + crit * ee)}


def prueba_t_una(x, mu0, alpha=0.05) -> dict:
    """t = (x̄ − μ₀) / (s/√n)."""
    x = _limpia(x)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    t = (m - mu0) / (s / np.sqrt(n))
    pval = 2 * stats.t.sf(abs(t), n - 1)
    return {"n": n, "media": m, "s": s, "mu0": mu0, "t": t, "gl": n - 1,
            "critico": stats.t.ppf(1 - alpha / 2, n - 1), "p": pval,
            "rechaza": _decision(pval, alpha)}


def mann_whitney(x1, x2, alpha=0.05) -> dict:
    """Prueba de Mann–Whitney (Wilcoxon de suma de rangos), paso a paso.

    1. Se juntan las dos muestras y se ordenan de menor a mayor; a cada dato se le asigna
       su rango (1, 2, …, N). Los empates reciben el rango promedio.
    2. R₁ = suma de los rangos de la muestra 1.
    3. U₁ = R₁ − n₁(n₁+1)/2  (número de pares (x₁, x₂) en los que x₁ > x₂).
    4. Bajo H₀, E[U] = n₁n₂/2 y Var[U] = n₁n₂(N+1)/12 (con corrección por empates).
    5. Z = (U₁ − n₁n₂/2) / √Var[U]  → valor p con la normal estándar.
    """
    x1, x2 = _limpia(x1), _limpia(x2)
    n1, n2 = len(x1), len(x2)
    N = n1 + n2
    todos = np.concatenate([x1, x2])
    rangos = stats.rankdata(todos)
    R1, R2 = rangos[:n1].sum(), rangos[n1:].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    mu_u = n1 * n2 / 2
    _, cuentas = np.unique(todos, return_counts=True)
    corr_empates = np.sum(cuentas ** 3 - cuentas) / (N * (N - 1))
    var_u = n1 * n2 / 12 * ((N + 1) - corr_empates)
    z = (U1 - mu_u) / np.sqrt(var_u)
    pval = 2 * stats.norm.sf(abs(z))
    exacta = stats.mannwhitneyu(x1, x2, alternative="two-sided", method="auto")
    return {"n1": n1, "n2": n2, "R1": R1, "R2": R2, "U1": U1, "U2": U2, "U": min(U1, U2),
            "mu_U": mu_u, "sigma_U": np.sqrt(var_u), "z": z, "p": pval,
            "p_scipy": float(exacta.pvalue), "rechaza": _decision(pval, alpha),
            "prob_superioridad": U1 / (n1 * n2),   # estima P(X₁ > X₂)
            "med1": np.median(x1), "med2": np.median(x2)}


# =============================================================================
# Capacidad de proceso (control de calidad)
# =============================================================================
def capacidad(x, lie, lse) -> dict:
    """Cp = (LSE − LIE)/(6s);  Cpk = mín(LSE − x̄, x̄ − LIE)/(3s)."""
    x = _limpia(x)
    m, s = x.mean(), x.std(ddof=1)
    cp = (lse - lie) / (6 * s)
    cpk = min(lse - m, m - lie) / (3 * s)
    dentro = float(((x >= lie) & (x <= lse)).mean())
    teorico_fuera = stats.norm.cdf(lie, m, s) + stats.norm.sf(lse, m, s)
    return {"media": m, "s": s, "Cp": cp, "Cpk": cpk, "dentro": dentro,
            "ppm_fuera_teorico": teorico_fuera * 1e6}


# =============================================================================
# Correlación y regresión
# =============================================================================
def correlacion(x, y) -> dict:
    """Pearson (relación lineal) y Spearman (relación monótona, robusta a atípicos)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return {"n": n, "r": float(r), "p_r": float(p_r), "t": float(t), "rho": float(rho),
            "p_rho": float(p_rho), "r2": float(r ** 2)}


def regresion(x, y) -> dict:
    """Regresión lineal simple por mínimos cuadrados: ŷ = b₀ + b₁x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    r = stats.linregress(x, y)
    return {"b0": float(r.intercept), "b1": float(r.slope), "r": float(r.rvalue),
            "r2": float(r.rvalue ** 2), "p": float(r.pvalue), "ee_b1": float(r.stderr)}


def interpretar_r(r: float) -> str:
    """Escala usual de interpretación del coeficiente de correlación."""
    a = abs(r)
    if a < 0.10:
        fuerza = "nula"
    elif a < 0.30:
        fuerza = "débil"
    elif a < 0.50:
        fuerza = "moderada"
    elif a < 0.70:
        fuerza = "fuerte"
    else:
        fuerza = "muy fuerte"
    return f"{fuerza} {'positiva' if r > 0 else 'negativa'}" if a >= 0.10 else "nula"
