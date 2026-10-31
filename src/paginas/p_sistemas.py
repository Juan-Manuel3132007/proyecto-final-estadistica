"""Punto 3A — Ingeniería de Sistemas con datos reales: el flujo de cambios de un proyecto de software."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import aplicaciones as ap
from src import bondad as b
from src import graficos as g
from src import ui
from src.paginas.comun import bloque_bondad


@st.cache_data
def _datos():
    return ap.cargar_sistemas()


@st.cache_data(show_spinner="Analizando 2 años de commits…")
def _analisis(alpha):
    c, d = ap.cargar_sistemas()
    return {"lleg": ap.llegadas_diarias(d, alpha), "corr": ap.correccion_vs_funcionalidad(c, alpha),
            "arch": ap.archivos_vs_lineas(c), "prop": ap.proporcion_correcciones(c, alpha)}


def pagina(alpha: float):
    c, d = _datos()
    A = _analisis(alpha)
    L = A["lleg"]
    ui.cabecera("03", "Punto 3 · Ingeniería de Sistemas · datos reales",
                "El <em>pulso</em> de un repositorio de software",
                "Las métricas de entrega de software (DevOps/DORA) se basan en estadística: cuántos cambios llegan, "
                "de qué tamaño y con qué variabilidad. Se analizó el historial <b>real</b> de la rama principal de "
                "<b>Streamlit</b>, la librería de código abierto con la que está construida esta misma app.")
    ui.kpis([(f"{len(c):,}", "commits analizados", "2024–2025, sin bots"), (f"{L['n']}", "días hábiles"),
             (f"{L['media']:.2f}", "commits / día hábil"), (f"{np.median(c.lineas_cambiadas):.0f}",
                                                             "líneas por commit (mediana)"),
             (f"{c.autor.nunique()}", "autores distintos")])

    tabs = st.tabs(["a · Problema", "b · Datos", "c.1 Llegadas por día", "c.2 Tamaño de los cambios",
                    "c.3 Corrección vs funcionalidad", "c.4 Archivos vs líneas", "c.5 Proporción de correcciones",
                    "d · Conclusiones"])
    with tabs[0]:
        st.markdown("### a) Aplicación y problema")
        ui.lectura(
            "**Aplicación de la Estadística en Ingeniería de Sistemas:** planeación de capacidad y gestión de la "
            "calidad del proceso de desarrollo (integración continua, revisión de código, DevOps).\n\n"
            "**Problema:** un equipo que mantiene un proyecto grande necesita dimensionar su infraestructura de "
            "integración continua (CI) y su capacidad de revisión. Para eso debe responder: (1) ¿cuántos cambios "
            "llegan por día y con qué variabilidad? ¿sirve el modelo de Poisson que se usa en teoría de colas? "
            "(2) ¿qué distribución tiene el tamaño de un cambio? (3) ¿los cambios de nuevas funcionalidades son "
            "más grandes que las correcciones? (4) ¿más archivos implica más líneas? (5) ¿cambió la proporción "
            "de correcciones entre 2024 y 2025?\n\n**Datos requeridos:** fecha, autor, tipo y tamaño (archivos y "
            "líneas) de cada cambio integrado. **Cómo se obtienen:** del historial de Git, que es público en los "
            "proyectos de código abierto, con `git log --first-parent --numstat`.", "teoria")
    with tabs[1]:
        st.markdown("### b) Recolección y organización")
        st.markdown(
            "1. Se clonó `github.com/streamlit/streamlit` y se extrajo `git log --first-parent --numstat` desde el "
            "1 de enero de 2024 hasta el 31 de diciembre de 2025 (**2 823 commits**).\n"
            "2. **Limpieza:** se quitaron los commits de bots (Dependabot, Snyk), las publicaciones de versión y "
            "los commits sin cambios de texto → **2 238 commits**.\n"
            "3. **Clasificación** por palabras clave del mensaje: corrección, funcionalidad, pruebas, "
            "refactorización, documentación y mantenimiento.\n"
            "4. Se contaron los commits de **cada día** (incluidos los días con 0) y se separaron los días hábiles.")
        ui.tabla(c.groupby("tipo").agg(commits=("commit", "size"), mediana_lineas=("lineas_cambiadas", "median"),
                                       media_lineas=("lineas_cambiadas", "mean"),
                                       mediana_archivos=("archivos", "median"))
                 .sort_values("commits", ascending=False), dec=1)
        st.dataframe(c.drop(columns=["commit"]), use_container_width=True, height=300, hide_index=True)
        st.download_button("Descargar sistemas_commits.csv", c.to_csv(index=False).encode("utf-8"),
                           "sistemas_commits.csv", "text/csv")

    with tabs[2]:
        st.markdown("### c.1) ¿Los commits por día siguen una Poisson?")
        st.plotly_chart(g.conteos_diarios(L["binneg"]["tabla"], L["media"]), use_container_width=True,
                        key="lleg")
        poi, bn = L["poisson"], L["binneg"]
        st.latex(rf"\bar x={L['media']:.4f},\qquad s^2={L['var']:.4f},\qquad "
                 rf"\frac{{s^2}}{{\bar x}}={L['indice_dispersion']:.3f}\ \ (\text{{Poisson exige}}\approx 1)")
        c1, c2 = st.columns(2)
        with c1:
            ui.paso("H₀", f"Poisson (λ = {poi['params'][0]:.4f})")
            ui.tabla(poi["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 2, "(Oᵢ − Eᵢ)²/Eᵢ": 3}, indice=False)
            st.markdown(f"EP = **{poi['EP']:.2f}**, v = {poi['v']}, χ² crítico = {poi['critico']:.3f}, "
                        f"p {ui.pval(poi['p'])}")
            ui.veredicto(poi["rechaza"])
        with c2:
            ui.paso("H₀", f"Binomial negativa ({b.texto_parametros('Binomial negativa', bn['params'], 3)})")
            ui.tabla(bn["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 2, "(Oᵢ − Eᵢ)²/Eᵢ": 3}, indice=False)
            st.markdown(f"EP = **{bn['EP']:.2f}**, v = {bn['v']}, χ² crítico = {bn['critico']:.3f}, "
                        f"p = {bn['p']:.4f}")
            ui.veredicto(bn["rechaza"])
        ui.lectura(
            f"La Poisson exige que la varianza sea igual a la media, pero aquí la varianza es "
            f"**{L['indice_dispersion']:.2f} veces la media** (sobredispersión): hay muchos más días sin commits "
            f"({int(poi['tabla'].iloc[0]['Oᵢ'])} observados vs {poi['tabla'].iloc[0]['Eᵢ = nPᵢ']:.1f} esperados) y "
            f"muchos más días de mucha actividad. La χ² la rechaza con contundencia. La **binomial negativa** "
            f"(una Poisson cuya tasa varía de un día a otro) sí se ajusta (p = {bn['p']:.4f}).\n\n**Consecuencia "
            f"práctica:** si se dimensiona la CI con Poisson, el percentil 95 diría {L['p95_poisson']} commits/día; "
            f"con la binomial negativa son **{L['p95_binneg']}**, y el real fue {L['p95_real']:.0f}. Usar Poisson "
            f"subestimaría la capacidad necesaria en los días pico.")

    with tabs[3]:
        st.markdown("### c.2) ¿Qué distribución tiene el tamaño de un commit?")
        T = ap.tamano_cambios(c, alpha)
        st.markdown("Ranking de distribuciones por AIC (criterio de información de Akaike, menor = mejor):")
        ui.tabla(T["ranking"], dec=3)
        st.plotly_chart(g.histograma_ajuste(T["x"], {"Lognormal": T["lognormal"]["congelada"]}, nbins=40,
                                            log_x=True), use_container_width=True, key="hist_log")
        bloque_bondad(T["x"], "Lognormal", alpha, "el número de líneas modificadas", clave="logn", n_mc=499,
                      clases="equiprobables")
        ui.lectura(
            f"Los tamaños van de 1 a {T['x'].max():,.0f} líneas, con mediana {T['mediana']:.0f} y media "
            f"{T['media']:.0f}: una asimetría enorme. En escala logarítmica el histograma es casi una campana, "
            f"por eso la **lognormal** es la mejor candidata (menor AIC). Aun así, con n = {len(T['x']):,} las "
            f"pruebas la rechazan: hay **más commits diminutos** (≤ 6 líneas) de los que el modelo predice "
            f"(ajustes de una línea, typos). Con muestras tan grandes cualquier diferencia pequeña resulta "
            f"significativa; el modelo sigue siendo útil para el centro y la cola derecha, que es lo que "
            f"importa para planear revisiones. El 90 % de los commits modifica menos de {T['p90']:.0f} líneas.")

    with tabs[4]:
        Cf = A["corr"]
        mw = Cf["mw"]
        st.markdown("### c.3) ¿Las funcionalidades son más grandes que las correcciones?")
        st.plotly_chart(g.cajas_tipo(c), use_container_width=True, key="cajas_tipo")
        st.latex(r"H_0:\ P(X_{\text{func}}>X_{\text{corr}})=0.5\qquad H_a:\ P(X_{\text{func}}>X_{\text{corr}})\neq0.5")
        st.latex(rf"U={mw['U1']:,.0f},\quad Z=\frac{{U-\mu_U}}{{\sigma_U}}=\frac{{{mw['U1']:,.0f}-{mw['mu_U']:,.0f}}}"
                 rf"{{{mw['sigma_U']:,.1f}}}={mw['z']:.3f}")
        ui.veredicto(mw["rechaza"], f"p {ui.pval(mw['p'])}")
        ui.lectura(
            f"Se usa **Mann–Whitney** porque los tamaños son muy asimétricos (c.2), no normales. Mediana de "
            f"funcionalidades: **{Cf['med_f']:.0f} líneas** (n = {Cf['n_f']}); de correcciones: "
            f"**{Cf['med_c']:.0f} líneas** (n = {Cf['n_c']}). En el {100 * mw['prob_superioridad']:.1f} % de los "
            f"pares (funcionalidad, corrección) la funcionalidad es más grande. Una t sobre ln(líneas) confirma "
            f"(t = {Cf['t_log']['t']:.2f}, p {ui.pval(Cf['t_log']['p'])}). Las correcciones son cambios "
            f"pequeños y localizados; las funcionalidades, unas **{Cf['med_f'] / Cf['med_c']:.0f} veces** más grandes.")

    with tabs[5]:
        Ar = A["arch"]
        st.markdown("### c.4) Correlación y regresión: archivos vs líneas")
        st.plotly_chart(g.dispersion_log(c, Ar["reg"]), use_container_width=True, key="disp_log")
        tab = pd.DataFrame({
            "Medida": ["Pearson r (datos originales)", "Spearman ρ (rangos)", "Pearson r (log₁₀ – log₁₀)"],
            "Valor": [Ar["cor"]["r"], Ar["cor"]["rho"], Ar["cor_log"]["r"]],
            "Valor p": [Ar["cor"]["p_r"], Ar["cor"]["p_rho"], Ar["cor_log"]["p_r"]]})
        ui.tabla(tab, dec=4, indice=False)
        st.latex(rf"\log_{{10}}(\widehat{{\text{{líneas}}}})={Ar['reg']['b0']:.4f}+{Ar['reg']['b1']:.4f}\,"
                 rf"\log_{{10}}(\text{{archivos}})\qquad R^2={Ar['reg']['r2']:.3f}")
        ui.lectura(
            f"Una lección importante: el **Pearson sobre los datos originales es apenas r = {Ar['cor']['r']:.3f}**, "
            f"porque unos pocos commits gigantes (decenas de miles de líneas) dominan el cálculo. Spearman, que usa "
            f"rangos, muestra la relación real: **ρ = {Ar['cor']['rho']:.3f}** (fuerte y positiva). En escala log-log "
            f"la relación es lineal con r = {Ar['cor_log']['r']:.3f}: la pendiente {Ar['reg']['b1']:.3f} indica que "
            f"duplicar los archivos multiplica las líneas por 2^{Ar['reg']['b1']:.3f} = "
            f"{2 ** Ar['reg']['b1']:.2f}. El número de archivos explica el {100 * Ar['reg']['r2']:.1f} % de la "
            f"variabilidad del log del tamaño.")

    with tabs[6]:
        P = A["prop"]
        pr = P["prueba"]
        st.markdown("### c.5) ¿Cambió la proporción de correcciones entre 2024 y 2025?")
        ic = P["ic"]
        st.latex(rf"\hat p_{{2024}}=\frac{{{P['x1']}}}{{{P['n1']}}}={pr['p1']:.4f}\qquad"
                 rf"\hat p_{{2025}}=\frac{{{P['x2']}}}{{{P['n2']}}}={pr['p2']:.4f}\qquad Z={pr['z']:.4f}")
        ui.veredicto(pr["rechaza"], f"p = {pr['p']:.4f}")
        ui.lectura(
            f"En total, el **{100 * ic['p']:.1f} %** de los commits son correcciones (IC 95 %: "
            f"{100 * ic['li']:.1f} %–{100 * ic['ls']:.1f} %). La proporción subió de {100 * pr['p1']:.1f} % en 2024 a "
            f"{100 * pr['p2']:.1f} % en 2025; la diferencia es significativa (IC 95 % para p₂₀₂₄ − p₂₀₂₅: "
            f"{100 * pr['ic'][0]:.1f} a {100 * pr['ic'][1]:.1f} puntos). Además, el ritmo total subió de "
            f"{P['n1']} a {P['n2']} commits. Un proyecto que crece acumula más código que mantener.")

    with tabs[7]:
        st.markdown("### d) Interpretación, conclusiones y recomendaciones")
        ui.lectura(
            f"1. **No asumir Poisson** para la llegada de trabajo: los días de un equipo humano no son "
            f"independientes ni idénticos. Dimensionar la CI con la binomial negativa (P95 = {L['p95_binneg']} "
            f"commits/día) en lugar de Poisson (P95 = {L['p95_poisson']}).\n"
            f"2. El tamaño de un cambio es **lognormal**: reportar medianas y percentiles, no promedios, en los "
            f"tableros del equipo. La media ({np.mean(c.lineas_cambiadas):.0f} líneas) es más de tres veces la mediana.\n"
            f"3. Las funcionalidades son mucho más grandes que las correcciones: conviene **dividirlas en "
            f"cambios pequeños** para que la revisión sea efectiva.\n"
            f"4. Para medir relaciones entre métricas de software, **preferir Spearman o transformar a log**: "
            f"Pearson se deja engañar por los valores extremos.\n"
            f"5. El aumento de la proporción de correcciones sugiere reforzar las **pruebas automatizadas**.")
    ui.pie()
