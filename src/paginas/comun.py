"""
comun.py — Bloques reutilizables: el análisis completo de bondad de ajuste (χ², KS y AD)
y los cálculos en caché.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src import bondad as b
from src import graficos as g
from src import ui
from src.descriptiva import numero_clases


@st.cache_data(show_spinner="Calculando las pruebas de bondad de ajuste…")
def bondad_cache(x: tuple, nombre: str, params: tuple | None, alpha: float, n_mc: int,
                 clases: str, semilla: int = 2026):
    return b.pruebas_bondad(np.array(x), nombre, params, alpha, n_mc=n_mc, clases=clases,
                            semilla=semilla)


def hipotesis(variable: str, dist_txt: str):
    st.latex(rf"\begin{{aligned}}H_0 &: \text{{{variable} sigue una distribución {dist_txt}}}\\"
             rf"H_a &: \text{{{variable} NO sigue una distribución {dist_txt}}}\end{{aligned}}")


def resumen_bondad(res: dict) -> pd.DataFrame:
    chi, eq, ks, ad = res["chi"], res["chi_eq"], res["ks"], res["ad"]

    def dec(r):
        return "—" if r is None else ("Rechaza H₀" if r else "No rechaza H₀")

    filas = [
        {"Prueba": f"χ² (guía: {chi['k_inicial']} clases de igual ancho → {chi['k']})",
         "Estadístico": chi["EP"], "Valor crítico": chi["critico"], "Valor p": chi["p"],
         "Decisión": dec(chi["rechaza"])},
        {"Prueba": f"χ² (clases equiprobables, k = {eq['k']})", "Estadístico": eq["EP"],
         "Valor crítico": eq["critico"], "Valor p": eq["p"], "Decisión": dec(eq["rechaza"])},
        {"Prueba": "Kolmogorov–Smirnov (D)", "Estadístico": ks["D"],
         "Valor crítico": ks["critico_clasico"] if not res["estimados"] else np.nan,
         "Valor p": ks["p"], "Decisión": dec(ks["rechaza"])},
        {"Prueba": "Anderson–Darling (A²)", "Estadístico": ad["A2"],
         "Valor crítico": ad.get("critico_tabla") if not (res["dist"] == "Normal" and res["estimados"]) else np.nan,
         "Valor p": ad["p"], "Decisión": dec(ad["rechaza"])},
    ]
    return pd.DataFrame(filas)


def bloque_bondad(x, nombre: str, alpha: float, variable: str = "la variable", params=None,
                  clave: str = "", n_mc: int = 999, clases: str = "ancho", color=g.PIZARRA,
                  mostrar_resumen: bool = True):
    """Dibuja el análisis completo de bondad de ajuste de una muestra continua."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    res = bondad_cache(tuple(x), nombre, params, alpha, n_mc, clases)
    d = res["congelada"]
    texto_p = b.texto_parametros(nombre, res["params"])

    if mostrar_resumen:
        ui.tabla(resumen_bondad(res), dec=4, indice=False)
        votos = [r for r in (res["chi"]["rechaza"], res["ks"]["rechaza"], res["ad"]["rechaza"]) if r is not None]
        if votos and all(votos):
            ui.veredicto(True, f"Las pruebas coinciden: los datos NO se ajustan a una {nombre}.")
        elif votos and not any(votos):
            ui.veredicto(False, f"Ninguna prueba encuentra evidencia en contra de la {nombre}.")
        else:
            ui.lectura("Las pruebas **no coinciden**. Esto pasa cuando la diferencia es pequeña o está "
                       "concentrada en una zona: χ² pierde información al agrupar, KS mira sobre todo el "
                       "centro y AD es la más sensible en las colas. Se le da más peso a AD.", "alerta")

    t1, t2, t3, t4 = st.tabs(["χ² paso a paso", "Kolmogorov–Smirnov", "Anderson–Darling", "Gráfico Q-Q"])
    with t1:
        chi = res["chi"]
        hipotesis(variable, nombre)
        origen = "estimados con la muestra" if res["estimados"] else "dados (no se estiman)"
        st.markdown(f"Parámetros {origen}: **{texto_p}**  ·  n = **{res['n']}**  ·  "
                    f"clases iniciales con la regla $2^k>n$: $2^{{{numero_clases(res['n'])}}} = "
                    f"{2 ** numero_clases(res['n'])} > {res['n']}$ → **k = {chi['k_inicial']}**"
                    + (" (clases equiprobables)" if clases == "equiprobables" else ""))
        c1, c2 = st.columns([1.15, 1])
        with c1:
            ui.tabla(chi["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 3, "(Oᵢ − Eᵢ)²/Eᵢ": 4}, indice=False)
        with c2:
            st.plotly_chart(g.barras_chi(chi["tabla"], alto=300), use_container_width=True,
                            key=f"chi_{clave}")
        if chi["k"] < chi["k_inicial"]:
            st.caption(f"Se agruparon clases adyacentes hasta que todas tuvieran Eᵢ ≥ 5 "
                       f"({chi['k_inicial']} → {chi['k']} clases), como indica la nota de la guía.")
        if chi["v"] >= 1:
            st.latex(rf"EP=\sum_{{i=1}}^{{{chi['k']}}}\frac{{(O_i-E_i)^2}}{{E_i}} = {chi['EP']:.4f}"
                     rf"\qquad v = k-p-1 = {chi['k']}-{chi['p_est']}-1 = {chi['v']}")
            st.latex(rf"\chi^2_{{({alpha},\,{chi['v']})}} = {chi['critico']:.4f}\qquad"
                     rf"\text{{Valor }}P = P\left(\chi^2_{{({chi['v']})}} > {chi['EP']:.4f}\right) = {chi['p']:.4f}")
            ui.veredicto(chi["rechaza"], f"EP = {chi['EP']:.4f} {'>' if chi['rechaza'] else '<'} "
                                         f"{chi['critico']:.4f}")
        else:
            ui.lectura(f"Con clases de igual ancho y n = {res['n']}, al agrupar para cumplir Eᵢ ≥ 5 quedan "
                       f"solo {chi['k']} clases y **v = {chi['v']}**: no hay grados de libertad para decidir. "
                       "Esto ocurre con datos muy asimétricos. La solución es usar **clases equiprobables** "
                       "(cada una con probabilidad 1/k bajo H₀) o las pruebas KS y AD, que no agrupan.",
                       "alerta")
        eq = res["chi_eq"]
        with st.expander(f"Variante con clases equiprobables (k = {eq['k']}, todas con Eᵢ = n/k)"):
            ui.tabla(eq["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 3, "(Oᵢ − Eᵢ)²/Eᵢ": 4}, indice=False)
            if eq["v"] >= 1:
                st.markdown(f"EP = **{eq['EP']:.4f}**, v = {eq['v']}, χ² crítico = {eq['critico']:.4f}, "
                            f"valor p = **{ui.pval(eq['p'])}**")
    with t2:
        ks = res["ks"]
        st.latex(r"D=\max_x\left|F_n(x)-F(x)\right| = \max\left(D^+,D^-\right),\quad "
                 r"D^+=\max_i\left(\tfrac{i}{n}-F(x_{(i)})\right),\ D^-=\max_i\left(F(x_{(i)})-\tfrac{i-1}{n}\right)")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(g.ks_grafico(x, d, ks, alto=360), use_container_width=True, key=f"ks_{clave}")
        with c2:
            ui.kpis([(f"{ks['D']:.4f}", "D observado"), (f"{ks['x_max']:.4g}", "dónde ocurre")])
            st.markdown(f"D⁺ = {ks['D+']:.4f} · D⁻ = {ks['D-']:.4f}")
            if res["estimados"]:
                st.markdown(f"Valor p **corregido** (Lilliefors, Monte Carlo con {n_mc} réplicas): "
                            f"**{ui.pval(ks['p'])}**  \nValor p clásico (sería válido solo si los parámetros "
                            f"NO se estimaran): {ui.pval(ks['p_clasico'])}")
            else:
                st.markdown(f"Valor crítico ($\\alpha$ = {alpha}): **{ks['critico_clasico']:.4f}**  \n"
                            f"Valor p: **{ui.pval(ks['p'])}**")
            ui.veredicto(ks["rechaza"])
        ui.lectura(f"La escalera azul es la distribución de los datos y la curva roja la del modelo {nombre}. "
                   f"La mayor separación vertical es **D = {ks['D']:.4f}** (en x ≈ {ks['x_max']:.4g}): "
                   f"en ese punto el {100 * ks['D']:.1f} % de la probabilidad está \"mal repartida\".", "teoria")
    with t3:
        ad = res["ad"]
        st.latex(r"A^2=-n-\frac{1}{n}\sum_{i=1}^{n}(2i-1)\left[\ln F(x_{(i)})+\ln\left(1-F(x_{(n+1-i)})\right)\right]")
        c1, c2 = st.columns([1, 1])
        with c1:
            items = [(f"{ad['A2']:.4f}", "A² observado")]
            if "A_mod" in ad:
                items.append((f"{ad['A_mod']:.4f}", "A* = A²(1+0.75/n+2.25/n²)"))
            if ad.get("critico_tabla"):
                items.append((f"{ad['critico_tabla']:.3f}", f"crítico tabla α={alpha}"))
            ui.kpis(items)
            st.markdown(f"Valor p (Monte Carlo, {n_mc} réplicas): **{ui.pval(ad['p'])}**")
            ui.veredicto(ad["rechaza"])
        with c2:
            st.plotly_chart(g.peso_ad(alto=280), use_container_width=True, key=f"ad_{clave}")
        ui.lectura("Anderson–Darling compara las mismas dos curvas que KS, pero **suma** las diferencias al "
                   "cuadrado en todo el rango y las multiplica por el peso 1/[F(1−F)], que crece en las colas. "
                   "Por eso detecta colas mal modeladas que KS no ve.", "teoria")
    with t4:
        st.plotly_chart(g.qq(x, d, alto=380), use_container_width=True, key=f"qq_{clave}")
        st.caption("Si el modelo es correcto, los puntos siguen la diagonal. Las desviaciones en los extremos "
                   "son justamente las que Anderson–Darling castiga.")
    return res


def conclusion_bondad(res: dict, variable: str) -> str:
    nombre = res["dist"]
    p_ad, p_ks, p_chi = res["ad"]["p"], res["ks"]["p"], res["chi"]["p"]
    if res["ad"]["rechaza"]:
        return (f"Con α = {res['alpha']}, {variable} **no** se ajusta a una {nombre} "
                f"(AD: p = {ui.pval(p_ad)}).")
    return (f"Con α = {res['alpha']} no hay evidencia para rechazar que {variable} siga una {nombre} "
            f"(AD: p = {ui.pval(p_ad)}; KS: p = {ui.pval(p_ks)}"
            + (f"; χ²: p = {ui.pval(p_chi)}" if not np.isnan(p_chi) else "") + ").")
