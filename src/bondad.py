"""
bondad.py — Pruebas de bondad de ajuste.

  • Chi-cuadrado (método de la guía del profesor): clases con 2^k > n, frecuencias
    esperadas E_i = n·P_i, agrupación de clases adyacentes hasta que E_i ≥ 5 y
    grados de libertad v = k − p − 1.
  • Kolmogorov–Smirnov (valor agregado): máxima distancia vertical entre la
    función de distribución empírica F_n(x) y la teórica F(x).
  • Anderson–Darling (valor agregado): distancia cuadrática ponderada que le da más
    peso a las colas.

Cuando los parámetros se estiman con la misma muestra, los valores p "clásicos" de
KS y AD dejan de ser válidos (salen demasiado grandes). Por eso el valor p se obtiene
con un bootstrap paramétrico (Monte Carlo): se simulan muchas muestras del modelo
ajustado, se re-estiman los parámetros en cada una y se mira qué tan raro es el
estadístico observado. Esto es la corrección de Lilliefors generalizada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .descriptiva import numero_clases

# =============================================================================
# Catálogo de distribuciones
# =============================================================================
CONTINUAS = {
    # nombre: (objeto scipy, etiquetas de parámetros, parámetros fijos para goodness_of_fit)
    "Normal": (stats.norm, ["μ", "σ"], {}),
    "Exponencial": (stats.expon, ["β (media)"], {"loc": 0}),
    "Gamma": (stats.gamma, ["α (forma)", "β (escala)"], {"loc": 0}),
    "Lognormal": (stats.lognorm, ["μ (de ln X)", "σ (de ln X)"], {"loc": 0}),
    "Weibull": (stats.weibull_min, ["k (forma)", "λ (escala)"], {"loc": 0}),
    "Uniforme": (stats.uniform, ["a", "b"], {}),
}
DISCRETAS = ["Poisson", "Binomial", "Binomial negativa"]


def ajustar(x, nombre: str, ensayos: int | None = None) -> tuple:
    """Estima los parámetros de la distribución con la muestra.

    Normal → x̄ y s (como en clase).  Gamma, Lognormal, Weibull → máxima verosimilitud.
    Poisson → λ = x̄ (es el estimador de máxima verosimilitud).
    Binomial → p = x̄ / n.   Binomial negativa → método de momentos.
    """
    x = np.asarray(x, dtype=float)
    if nombre == "Normal":
        return (x.mean(), x.std(ddof=1))
    if nombre == "Exponencial":
        return (x.mean(),)
    if nombre == "Gamma":
        a, _, esc = stats.gamma.fit(x, floc=0)
        return (a, esc)
    if nombre == "Lognormal":
        lx = np.log(x)
        return (lx.mean(), lx.std(ddof=0))
    if nombre == "Weibull":
        c, _, esc = stats.weibull_min.fit(x, floc=0)
        return (c, esc)
    if nombre == "Uniforme":
        return (x.min(), x.max())
    if nombre == "Poisson":
        return (x.mean(),)
    if nombre == "Binomial":
        return (ensayos, x.mean() / ensayos)
    if nombre == "Binomial negativa":
        m, v = x.mean(), x.var(ddof=1)
        if v <= m:  # sin sobredispersión no tiene sentido; se devuelve algo casi Poisson
            v = m * 1.0001
        r = m ** 2 / (v - m)
        return (r, r / (r + m))
    raise ValueError(nombre)


def congelar(nombre: str, params: tuple):
    """Devuelve la distribución de scipy con los parámetros fijos (objeto 'congelado')."""
    if nombre == "Normal":
        return stats.norm(params[0], params[1])
    if nombre == "Exponencial":
        return stats.expon(scale=params[0])
    if nombre == "Gamma":
        return stats.gamma(params[0], scale=params[1])
    if nombre == "Lognormal":
        return stats.lognorm(params[1], scale=np.exp(params[0]))
    if nombre == "Weibull":
        return stats.weibull_min(params[0], scale=params[1])
    if nombre == "Uniforme":
        return stats.uniform(params[0], params[1] - params[0])
    if nombre == "Poisson":
        return stats.poisson(params[0])
    if nombre == "Binomial":
        return stats.binom(int(params[0]), params[1])
    if nombre == "Binomial negativa":
        return stats.nbinom(params[0], params[1])
    raise ValueError(nombre)


def texto_parametros(nombre: str, params: tuple, dec: int = 4) -> str:
    if nombre in CONTINUAS:
        etiquetas = CONTINUAS[nombre][1]
    elif nombre == "Poisson":
        etiquetas = ["λ"]
    elif nombre == "Binomial":
        etiquetas = ["n", "p"]
    else:
        etiquetas = ["r", "p"]
    partes = []
    for e, v in zip(etiquetas, params):
        partes.append(f"{e} = {int(v)}" if e == "n" else f"{e} = {v:.{dec}f}")
    return ",  ".join(partes)


# =============================================================================
# Chi-cuadrado
# =============================================================================
def _agrupar(etiquetas, O, P, n, minimo=5.0):
    """Une clases adyacentes hasta que todas tengan E_i ≥ 5 (nota de la guía)."""
    grupos = [[e, o, p] for e, o, p in zip(etiquetas, O, P)]
    # De izquierda a derecha
    i = 0
    while i < len(grupos) - 1:
        if grupos[i][2] * n < minimo:
            sig = grupos.pop(i + 1)
            grupos[i] = [_unir(grupos[i][0], sig[0]), grupos[i][1] + sig[1], grupos[i][2] + sig[2]]
        else:
            i += 1
    # La última clase, si quedó pequeña, se une con la anterior
    while len(grupos) > 1 and grupos[-1][2] * n < minimo:
        ult = grupos.pop()
        grupos[-1] = [_unir(grupos[-1][0], ult[0]), grupos[-1][1] + ult[1], grupos[-1][2] + ult[2]]
    return grupos


def _unir(a: str, b: str) -> str:
    """Une dos etiquetas de clase: '0' y '1' → '0 y 1';  '(a, b]' y '(b, c]' → '(a, c]'."""
    if a.startswith(("(", "x ≤")) or b.startswith(("(", "x >")):
        return _rango(a, b)
    # Clases discretas: "8" + "9 o más" → "8 o más";  "0 o menos" + "1" → "1 o menos"
    if a.endswith("o menos"):
        return f"{b} o menos"
    ini = a.replace(" y ", "–").split("–")[0]
    if b.endswith("o más"):
        return f"{ini} o más"
    fin = b.replace(" y ", "–").split("–")[-1]
    try:
        consecutivos = int(fin) - int(ini) == 1
    except ValueError:
        consecutivos = False
    return f"{ini} y {fin}" if consecutivos else f"{ini}–{fin}"


def _rango(a: str, b: str) -> str:
    # Formatos posibles: "x ≤ 1.2", "(1.2, 1.5]", "x > 3.1"
    if a.startswith("x ≤"):
        if b.startswith("x >"):
            return "Todo ℝ"
        return "x ≤ " + b.split(",")[1].strip(" ]")
    ini = a.split(",")[0].strip("( ")
    if b.startswith("x >"):
        return "x > " + ini
    return f"({ini}, {b.split(',')[1].strip(' ]')}]"


def _resultado_chi(grupos, n, p_est, alpha, k_inicial, nota=""):
    etiquetas = [g[0] for g in grupos]
    O = np.array([g[1] for g in grupos], dtype=float)
    P = np.array([g[2] for g in grupos], dtype=float)
    E = n * P
    contrib = (O - E) ** 2 / E
    ep = contrib.sum()
    k = len(grupos)
    v = k - p_est - 1
    tabla = pd.DataFrame({"Clase": etiquetas, "Oᵢ": O.astype(int), "Pᵢ": P, "Eᵢ = nPᵢ": E,
                          "(Oᵢ − Eᵢ)²/Eᵢ": contrib})
    if v < 1:
        return {"tabla": tabla, "EP": ep, "v": v, "critico": np.nan, "p": np.nan, "k": k,
                "k_inicial": k_inicial, "p_est": p_est, "rechaza": None, "alpha": alpha,
                "nota": "No quedan grados de libertad suficientes (v < 1) tras agrupar."}
    critico = stats.chi2.ppf(1 - alpha, v)
    pval = stats.chi2.sf(ep, v)
    return {"tabla": tabla, "EP": ep, "v": v, "critico": critico, "p": pval, "k": k,
            "k_inicial": k_inicial, "p_est": p_est, "rechaza": bool(ep > critico),
            "alpha": alpha, "nota": nota}


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def chi_cuadrado_continua(x, dist, p_est: int, alpha: float = 0.05, k: int | None = None,
                          bordes: list | None = None, observados: list | None = None,
                          clases: str = "ancho"):
    """Chi-cuadrado para una variable continua.

    x        : muestra (se ignora si se pasan 'bordes' y 'observados', datos ya agrupados)
    dist     : distribución congelada de scipy (la hipótesis nula)
    p_est    : número de parámetros estimados con la muestra
    k        : número de clases (por defecto la regla de la guía 2^k > n)
    bordes   : bordes INTERIORES de las clases; la primera es x ≤ b₁ y la última x > b_k
    clases   : "ancho"         → k clases de igual ancho entre el mínimo y el máximo (guía)
               "equiprobables" → k clases con la misma probabilidad 1/k bajo H₀, de modo
                                 que todas tienen Eᵢ = n/k (no hace falta agrupar)
    """
    if bordes is None:
        x = np.asarray(x, dtype=float)
        n = len(x)
        k = k or numero_clases(n)
        if clases == "equiprobables":
            b = dist.ppf(np.arange(1, k) / k)
        else:
            b = np.linspace(x.min(), x.max(), k + 1)[1:-1]
        O = np.bincount(np.searchsorted(b, x, side="left"), minlength=k)
    else:
        b = np.asarray(bordes, dtype=float)
        O = np.asarray(observados)
        n = int(O.sum())
        k = len(O)
    F = dist.cdf(b)
    P = np.diff(np.concatenate([[0.0], F, [1.0]]))
    etiquetas = [f"x ≤ {_fmt(b[0])}"] + [f"({_fmt(b[i])}, {_fmt(b[i+1])}]" for i in range(len(b) - 1)] \
        + [f"x > {_fmt(b[-1])}"]
    grupos = _agrupar(etiquetas, O, P, n)
    return _resultado_chi(grupos, n, p_est, alpha, k)


def chi_cuadrado_discreta(valores, observados, dist, p_est: int, alpha: float = 0.05,
                          ultima_abierta: bool = True, decimales_p: int | None = None):
    """Chi-cuadrado para una variable discreta (como el ejemplo de ASTON de la guía).

    valores    : valores de la variable, en orden (0, 1, 2, …)
    observados : frecuencia observada de cada valor
    La última clase se toma como 'valor o más' (P = 1 − F(valor − 1)) para que las
    probabilidades sumen 1, igual que en la tabla de la guía.
    """
    valores = list(valores)
    O = np.asarray(observados)
    n = int(O.sum())
    P = np.array([dist.pmf(v) for v in valores], dtype=float)
    etiquetas = [str(v) for v in valores]
    if ultima_abierta:
        P[-1] = dist.sf(valores[-1] - 1)
        etiquetas[-1] = f"{valores[-1]} o más"
    if valores[0] > 0 and dist.cdf(valores[0] - 1) > 1e-12:
        P[0] = dist.cdf(valores[0])
        etiquetas[0] = f"{valores[0]} o menos"
    if decimales_p is not None:      # para replicar tablas hechas a mano con Pᵢ redondeadas
        P = np.round(P, decimales_p)
        if ultima_abierta:           # la última se completa para que ΣPᵢ = 1, como en la guía
            P[-1] = round(1 - P[:-1].sum(), decimales_p)
    grupos = _agrupar(etiquetas, O, P, n)
    return _resultado_chi(grupos, n, p_est, alpha, len(valores))


def chi_cuadrado_datos_discretos(x, nombre: str, alpha=0.05, params=None, ensayos=None,
                                 max_valor=None):
    """Atajo: recibe los datos crudos (enteros), arma la tabla y aplica la prueba."""
    x = np.asarray(x).astype(int)
    p_est = 0
    if params is None:
        params = ajustar(x, nombre, ensayos)
        p_est = {"Poisson": 1, "Binomial": 1, "Binomial negativa": 2}[nombre]
    dist = congelar(nombre, params)
    tope = max_valor if max_valor is not None else int(x.max())
    if nombre == "Binomial":
        tope = int(params[0])
    valores = list(range(0, tope + 1))
    O = [int((x == v).sum()) if v < tope else int((x >= v).sum()) for v in valores]
    res = chi_cuadrado_discreta(valores, O, dist, p_est, alpha, ultima_abierta=nombre != "Binomial")
    res["params"] = params
    res["dist"] = nombre
    return res


# =============================================================================
# Kolmogorov–Smirnov y Anderson–Darling
# =============================================================================
def estadistico_ks(x, dist) -> dict:
    """D = máx |F_n(x) − F(x)|, revisando a ambos lados de cada salto de la escalera."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    F = dist.cdf(x)
    i = np.arange(1, n + 1)
    d_mas = i / n - F            # la escalera por encima de la curva
    d_menos = F - (i - 1) / n    # la curva por encima de la escalera
    j_mas, j_menos = int(np.argmax(d_mas)), int(np.argmax(d_menos))
    if d_mas[j_mas] >= d_menos[j_menos]:
        D, j, lado = d_mas[j_mas], j_mas, "+"
    else:
        D, j, lado = d_menos[j_menos], j_menos, "−"
    return {"D": float(D), "D+": float(d_mas.max()), "D-": float(d_menos.max()),
            "x_max": float(x[j]), "Fn_antes": (j) / n, "Fn_despues": (j + 1) / n,
            "F_en_x": float(F[j]), "lado": lado}


def estadistico_ad(x, dist) -> float:
    """A² = −n − (1/n) Σ (2i − 1)[ln F(x₍ᵢ₎) + ln(1 − F(x₍ₙ₊₁₋ᵢ₎))]."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    F = np.clip(dist.cdf(x), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n)


def _p_montecarlo(x, nombre, params, estimados: bool, estadistico: str, n_mc: int, semilla: int):
    """Valor p por bootstrap paramétrico (Monte Carlo) con scipy.stats.goodness_of_fit."""
    obj, _, fijos = CONTINUAS[nombre]
    if estimados:
        kw = {} if nombre == "Normal" else {"known_params": fijos}
    else:
        kw = {"known_params": _a_scipy(nombre, params)}
    try:   # SciPy ≥ 1.15 usa 'rng'; versiones anteriores, 'random_state'
        res = stats.goodness_of_fit(obj, x, statistic=estadistico, n_mc_samples=n_mc,
                                    rng=np.random.default_rng(semilla), **kw)
    except TypeError:
        res = stats.goodness_of_fit(obj, x, statistic=estadistico, n_mc_samples=n_mc,
                                    random_state=np.random.default_rng(semilla), **kw)
    return float(res.pvalue)


def _a_scipy(nombre, params) -> dict:
    if nombre == "Normal":
        return {"loc": params[0], "scale": params[1]}
    if nombre == "Exponencial":
        return {"loc": 0, "scale": params[0]}
    if nombre == "Gamma":
        return {"a": params[0], "loc": 0, "scale": params[1]}
    if nombre == "Lognormal":
        return {"s": params[1], "loc": 0, "scale": np.exp(params[0])}
    if nombre == "Weibull":
        return {"c": params[0], "loc": 0, "scale": params[1]}
    if nombre == "Uniforme":
        return {"loc": params[0], "scale": params[1] - params[0]}
    raise ValueError(nombre)


# Valores críticos de Anderson–Darling para la NORMAL con μ y σ estimados (Stephens, 1974),
# aplicados al estadístico modificado A* = A²(1 + 0.75/n + 2.25/n²).
AD_NORMAL_CRITICOS = {0.10: 0.631, 0.05: 0.752, 0.025: 0.873, 0.01: 1.035}
# Caso 0 (todos los parámetros conocidos), válido para cualquier distribución continua.
AD_CASO0_CRITICOS = {0.10: 1.933, 0.05: 2.492, 0.025: 3.070, 0.01: 3.857}


def pruebas_bondad(x, nombre: str, params: tuple | None = None, alpha: float = 0.05,
                   n_mc: int = 999, semilla: int = 2026, k: int | None = None,
                   clases: str = "ancho") -> dict:
    """Aplica chi-cuadrado, Kolmogorov–Smirnov y Anderson–Darling a una muestra continua.

    Si params es None, los parámetros se estiman con la muestra (p = nº de parámetros).
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    estimados = params is None
    if estimados:
        params = ajustar(x, nombre)
    p_est = len(params) if estimados else 0
    dist = congelar(nombre, params)

    chi = chi_cuadrado_continua(x, dist, p_est, alpha, k=k, clases=clases)
    chi_eq = chi_cuadrado_continua(x, dist, p_est, alpha, k=k, clases="equiprobables")

    ks = estadistico_ks(x, dist)
    ks["critico_clasico"] = float(stats.kstwo.ppf(1 - alpha, n))
    ks["p_clasico"] = float(stats.kstwo.sf(ks["D"], n))
    ks["p"] = _p_montecarlo(x, nombre, params, estimados, "ks", n_mc, semilla) if estimados \
        else ks["p_clasico"]
    ks["rechaza"] = ks["p"] < alpha

    a2 = estadistico_ad(x, dist)
    ad = {"A2": a2}
    if nombre == "Normal" and estimados:
        ad["A_mod"] = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
        ad["critico_tabla"] = AD_NORMAL_CRITICOS.get(alpha)
    elif not estimados:
        ad["critico_tabla"] = AD_CASO0_CRITICOS.get(alpha)
    ad["p"] = _p_montecarlo(x, nombre, params, estimados, "ad", n_mc, semilla + 1)
    ad["rechaza"] = ad["p"] < alpha

    return {"dist": nombre, "params": params, "estimados": estimados, "n": n, "alpha": alpha,
            "chi": chi, "chi_eq": chi_eq, "ks": ks, "ad": ad, "congelada": dist}


def ranking(x, candidatas=("Normal", "Lognormal", "Gamma", "Weibull", "Exponencial")) -> pd.DataFrame:
    """Ajusta varias distribuciones y las ordena por el criterio de información de Akaike (AIC).

    Es rápido (no usa Monte Carlo): sirve para escoger candidatas antes de las pruebas formales.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    filas = []
    for nombre in candidatas:
        if nombre in ("Lognormal", "Gamma", "Weibull", "Exponencial") and x.min() <= 0:
            continue
        try:
            params = ajustar(x, nombre)
            d = congelar(nombre, params)
            logv = float(np.sum(d.logpdf(x)))
            filas.append({"Distribución": nombre, "Parámetros": texto_parametros(nombre, params),
                          "log-verosimilitud": logv,
                          "AIC": 2 * len(params) - 2 * logv,
                          "D (KS)": estadistico_ks(x, d)["D"],
                          "A² (AD)": estadistico_ad(x, d)})
        except Exception:  # noqa: BLE001 — una distribución que no ajusta no detiene el resto
            continue
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    tabla.index = tabla.index + 1
    return tabla
