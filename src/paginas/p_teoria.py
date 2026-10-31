"""Teoría — χ², Kolmogorov–Smirnov y Anderson–Darling explicados a fondo (valor agregado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src import bondad as b
from src import graficos as g
from src import montecarlo as mc
from src import ui
from src.paginas.comun import bloque_bondad


@st.cache_data(show_spinner="Simulando 2000 muestras normales…")
def _lilliefors():
    return mc.lilliefors()


@st.cache_data(show_spinner="Comparando la potencia de las tres pruebas…")
def _potencias():
    return mc.potencias_bondad()


def pagina(alpha: float):
    ui.cabecera("06", "Valor agregado", "χ², <em>Kolmogorov–Smirnov</em> y Anderson–Darling",
                "Las tres pruebas contestan la misma pregunta —¿los datos vienen de la distribución propuesta?— "
                "pero miden la discrepancia de forma distinta. Entender esa diferencia es lo que permite escoger "
                "la prueba adecuada y explicar por qué a veces no coinciden.")
    st.latex(r"H_0:\ \text{la muestra proviene de la distribución } F_0 \qquad "
             r"H_a:\ \text{la muestra NO proviene de } F_0")

    tabs = st.tabs(["χ² de Pearson", "Kolmogorov–Smirnov", "Anderson–Darling", "Comparación",
                    "Laboratorio interactivo", "Mann–Whitney"])
    with tabs[0]:
        st.markdown("### Compara **frecuencias** por clases")
        st.latex(r"EP=\sum_{i=1}^{k}\frac{(O_i-E_i)^2}{E_i}\ \sim\ \chi^2_{(v)},\qquad E_i=nP_i,\qquad v=k-p-1")
        ui.lectura(
            "Se reparten los datos en k clases y se compara cuántos cayeron en cada una (Oᵢ) con cuántos deberían "
            "haber caído si H₀ fuera cierta (Eᵢ). Cada término (Oᵢ − Eᵢ)²/Eᵢ es una \"sorpresa relativa\". Si la "
            "suma es grande, el modelo no explica los datos. Se rechaza H₀ si EP > χ²(α, v).\n\n"
            "**Ventajas:** sirve para variables **discretas y continuas**; es la única de las tres que aplica a "
            "Poisson o binomial.\n**Limitaciones:** (1) al agrupar se pierde información (dos datos en la misma "
            "clase \"valen igual\"); (2) el resultado depende de cómo se elijan las clases; (3) necesita Eᵢ ≥ 5, "
            "lo que con muestras pequeñas obliga a juntar clases y deja pocos grados de libertad. Las clases "
            "**equiprobables** (cada una con probabilidad 1/k bajo H₀) reducen los problemas 2 y 3.", "teoria")
    with tabs[1]:
        st.markdown("### Compara **funciones de distribución acumulada**: la mayor distancia vertical")
        st.latex(r"F_n(x)=\frac{\#\{x_i\le x\}}{n}\qquad D=\sup_x\left|F_n(x)-F_0(x)\right|")
        st.latex(r"D^+=\max_{1\le i\le n}\left(\frac{i}{n}-F_0(x_{(i)})\right),\qquad "
                 r"D^-=\max_{1\le i\le n}\left(F_0(x_{(i)})-\frac{i-1}{n}\right),\qquad D=\max(D^+,D^-)")
        ui.lectura(
            "**Idea:** Fₙ(x) es una escalera que sube 1/n en cada dato ordenado. Si H₀ es cierta, por el teorema "
            "de Glivenko–Cantelli la escalera se pega a la curva F₀ cuando n crece. D mide el peor desacuerdo. "
            "Como la escalera salta en cada x₍ᵢ₎, el máximo se busca justo antes (i−1)/n y justo después i/n de "
            "cada salto: de ahí D⁺ y D⁻.\n\n**Distribución de D:** si F₀ es continua y **completamente "
            "especificada**, la distribución de D no depende de F₀ (es \"de distribución libre\"); para n grande, "
            "P(√n·D > t) → 2Σ(−1)^{k−1}e^{−2k²t²} (Kolmogorov). Aproximación del crítico al 5 %: 1.36/√n.\n\n"
            "**Ventajas:** usa cada dato sin agrupar; da un gráfico muy intuitivo. **Limitaciones:** solo para "
            "variables continuas; es más sensible en el **centro** que en las colas (allí F₀ es casi 0 o 1 y las "
            "diferencias son pequeñas por construcción).", "teoria")
        st.markdown("#### El detalle que casi nadie explica: estimar parámetros cambia el valor p (Lilliefors)")
        L = _lilliefors()
        ui.kpis([(f"{100 * L['rech_conocido']:.1f}%", "rechazos con μ, σ conocidos"),
                 (f"{100 * L['rech_clasico']:.1f}%", "rechazos si se estiman y se usa la tabla"),
                 (f"{L['d95_clasico']:.4f}", "crítico de la tabla (n = 60)"),
                 (f"{L['d95_est']:.4f}", "crítico real con μ, σ estimados")], acento=1)
        ui.lectura(
            f"Se simularon 2000 muestras **normales** de tamaño 60. Con μ y σ conocidos, KS rechaza el "
            f"{100 * L['rech_conocido']:.1f} % de las veces (≈ α, correcto). Pero si μ y σ se **estiman con la misma "
            f"muestra** y se usa la tabla de Kolmogorov, solo rechaza el {100 * L['rech_clasico']:.1f} %: la normal "
            f"ajustada \"se acomoda\" a los datos y D sale más pequeño. El crítico verdadero es "
            f"{L['d95_est']:.4f}, no {L['d95_clasico']:.4f}. **Por eso en este proyecto el valor p de KS y AD se "
            f"calcula por Monte Carlo** (bootstrap paramétrico): se simulan muchas muestras del modelo ajustado, se "
            f"re-estiman los parámetros en cada una y se ve qué tan extremo es el D observado. Para la normal, "
            f"esto es la prueba de Lilliefors.", "alerta")
    with tabs[2]:
        st.markdown("### También compara acumuladas, pero **suma** las diferencias y pesa más las colas")
        st.latex(r"A^2=n\int_{-\infty}^{\infty}\frac{\left[F_n(x)-F_0(x)\right]^2}{F_0(x)\left[1-F_0(x)\right]}\,dF_0(x)")
        st.latex(r"A^2=-n-\frac{1}{n}\sum_{i=1}^{n}(2i-1)\Big[\ln F_0(x_{(i)})+\ln\big(1-F_0(x_{(n+1-i)})\big)\Big]")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            ui.lectura(
                "**Idea:** en lugar de quedarse con la peor diferencia (KS), A² integra las diferencias al cuadrado "
                "en todo el rango. La clave es el **peso** ψ(F) = 1/[F(1−F)]: vale 4 en la mediana (F = 0.5) pero "
                "crece sin límite cuando F → 0 o F → 1, es decir, en las **colas**. Una diferencia de 0.02 en la "
                "cola pesa mucho más que la misma diferencia en el centro.\n\n**Por qué importa en ingeniería:** "
                "los problemas suelen estar en las colas: el horno que más emite, el tiempo de respuesta más lento, "
                "la pieza fuera de tolerancia. AD es la prueba más potente para detectar colas mal modeladas.\n\n"
                "**Valores críticos:** dependen de la distribución y de si los parámetros se estiman. Para la normal "
                "con μ y σ estimados, Stephens (1974) usa A* = A²(1 + 0.75/n + 2.25/n²) con crítico 0.752 al 5 %; "
                "con parámetros conocidos, 2.492. En la app el valor p se obtiene por Monte Carlo para cualquier "
                "distribución.", "teoria")
        with c2:
            st.plotly_chart(g.peso_ad(), use_container_width=True, key="peso_teo")
            st.caption("Escala logarítmica: el peso de AD en las colas es decenas de veces mayor que en el centro.")
    with tabs[3]:
        st.markdown("### ¿Cuál usar?")
        comp = pd.DataFrame({
            "χ² de Pearson": ["Frecuencias por clase", "Discretas y continuas", "Sí (pierde información)",
                              "Sí: Eᵢ ≥ 5", "Depende de las clases", "Conteos, datos ya agrupados"],
            "Kolmogorov–Smirnov": ["Máxima distancia entre acumuladas", "Continuas", "No", "No",
                                   "Centro de la distribución", "Gráfico intuitivo, n pequeño"],
            "Anderson–Darling": ["Distancia cuadrática ponderada", "Continuas", "No", "No",
                                 "Colas", "Normalidad, confiabilidad, riesgos extremos"],
        }, index=["¿Qué mide?", "Variables", "¿Agrupa?", "¿Requisito de frecuencias?", "Más sensible en",
                  "Ideal para"])
        st.dataframe(comp, use_container_width=True)
        st.markdown("#### Potencia simulada para rechazar la normal (n = 60, α = 0.05, 600 réplicas)")
        P = _potencias()
        ui.tabla(P, dec=3, indice=False)
        ui.lectura(
            "Cada celda es el porcentaje de veces que la prueba detectó que los datos **no** eran normales (más "
            "alto = mejor). **Anderson–Darling gana en los tres casos**, sobre todo con colas pesadas (t de "
            "Student) y sin colas (uniforme), que son justamente diferencias en los extremos. χ² es la menos "
            "potente porque agrupa. Esta es la razón técnica para presentar KS y AD como valor agregado.")
    with tabs[4]:
        st.markdown("### Pruébelo usted: genere datos de una distribución y póngalos a prueba con otra")
        c1, c2, c3, c4 = st.columns(4)
        real = c1.selectbox("Los datos vienen de", ["Normal(50, 10)", "Gamma(2, 10)", "Exponencial(20)",
                                                   "Lognormal(3, 0.5)", "Uniforme(0, 100)", "t de Student (3 gl)"],
                            key="teo_real")
        n = c2.select_slider("Tamaño n", [20, 30, 60, 100, 200, 500], value=60, key="teo_n")
        h0 = c3.selectbox("H₀: los datos son", ["Normal", "Gamma", "Lognormal", "Exponencial", "Weibull"],
                          key="teo_h0")
        sem = c4.number_input("Semilla", 0, 99999, 2026, key="teo_sem")
        rng = np.random.default_rng(int(sem))
        x = {"Normal(50, 10)": lambda: rng.normal(50, 10, n), "Gamma(2, 10)": lambda: rng.gamma(2, 10, n),
             "Exponencial(20)": lambda: rng.exponential(20, n), "Lognormal(3, 0.5)": lambda: rng.lognormal(3, .5, n),
             "Uniforme(0, 100)": lambda: rng.uniform(0, 100, n),
             "t de Student (3 gl)": lambda: 50 + 10 * rng.standard_t(3, n)}[real]()
        if h0 in ("Gamma", "Lognormal", "Exponencial", "Weibull") and x.min() <= 0:
            st.warning("Hay datos ≤ 0: esa distribución no puede generarlos (se rechazaría trivialmente). "
                       "Escoja otra H₀.")
        else:
            st.plotly_chart(g.histograma_ajuste(x, {h0: b.congelar(h0, b.ajustar(x, h0))}),
                            use_container_width=True, key="teo_hist")
            bloque_bondad(x, h0, alpha, "la muestra", clave=f"teo_{real}_{n}_{h0}_{sem}", n_mc=499)
    with tabs[5]:
        st.markdown("### Mann–Whitney (Wilcoxon de suma de rangos)")
        st.latex(r"U_1=R_1-\frac{n_1(n_1+1)}{2}=\#\{(i,j): x_{1i}>x_{2j}\}\qquad U_1+U_2=n_1n_2")
        st.latex(r"E[U]=\frac{n_1n_2}{2},\qquad Var[U]=\frac{n_1n_2}{12}\left[(N+1)-"
                 r"\frac{\sum_j(t_j^3-t_j)}{N(N-1)}\right],\qquad Z=\frac{U_1-E[U]}{\sqrt{Var[U]}}")
        ui.lectura(
            "Es la alternativa **no paramétrica** a la t de dos muestras independientes. Solo usa el **orden** de los "
            "datos, así que no necesita normalidad y no la afectan los atípicos. H₀: ambas muestras vienen de la "
            "misma distribución, equivalente a P(X₁ > X₂) = 0.5. Si además las dos distribuciones tienen la misma "
            "forma, la prueba compara medianas. El término tⱼ corrige por empates (tⱼ datos con el mismo valor). "
            "Para n₁, n₂ > 10 la aproximación normal de U es muy buena; para muestras pequeñas se usan tablas "
            "exactas.\n\n**Cuándo preferirla:** datos asimétricos, con atípicos, ordinales, o muestras pequeñas no "
            "normales. **Costo:** si los datos sí son normales, pierde un poco de potencia frente a la t "
            "(eficiencia relativa asintótica de 3/π ≈ 0.955).", "teoria")
    ui.pie()
