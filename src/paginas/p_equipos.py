"""Punto 2 — Tiempo de procesamiento de dos equipos A y B (datos simulados)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from src import graficos as g
from src import inferencia as inf
from src import montecarlo as mc
from src import ui
from src.descriptiva import tabla_resumen
from src.generacion import generar_simulacion
from src.paginas.comun import bloque_bondad, bondad_cache

GRUPOS = {
    "E1_equipo_A": ("Escenario 1 · Equipo A", "Normal(μ = 175, σ = 30)"),
    "E1_equipo_B": ("Escenario 1 · Equipo B", "Normal(μ = 170, σ = 50)"),
    "E2_equipo_A": ("Escenario 2 · Equipo A", "Gamma(α = 1.2, β = 150)"),
    "E2_equipo_B": ("Escenario 2 · Equipo B", "Gamma(α = 0.95, β = 190)"),
}


@st.cache_data
def _sim(semilla):
    return generar_simulacion(semilla)


@st.cache_data(show_spinner="Simulando miles de experimentos…")
def _mc(rep, semilla):
    return mc.resumen_escenarios(rep, semilla)


def pagina(alpha: float, semilla: int, rep_mc: int):
    ui.cabecera("02", "Punto 2 · Simulación", "Equipo <em>A</em> contra equipo <em>B</em>",
                "La variable es el <b>tiempo de procesamiento</b> (ms). Se generan 60 valores por equipo en dos "
                "escenarios: normales con distinta media y desviación, y gamma (asimétricas, con β como parámetro "
                "de <b>escala</b>: E[X] = αβ). La semilla se cambia en la barra lateral.")
    s = _sim(semilla)
    ui.kpis([("60", "datos por grupo"), ("4", "grupos simulados"),
             (f"{1.2 * 150:.0f} · {0.95 * 190:.1f}", "medias teóricas E2 (ms)"),
             (str(semilla), "semilla")])
    tabs = st.tabs(["a · Histogramas, descriptivas y ajuste", "b · t-Student vs Normal",
                    "c · Mann–Whitney", "Monte Carlo: ¿qué prueba es mejor?", "Datos"])

    # ------------------------------------------------------------------ a)
    with tabs[0]:
        fig = make_subplots(rows=2, cols=2, subplot_titles=[v[0] for v in GRUPOS.values()],
                            horizontal_spacing=.08, vertical_spacing=.16)
        for i, (col, (tit, teo)) in enumerate(GRUPOS.items()):
            r, c = i // 2 + 1, i % 2 + 1
            x = s[col]
            color = g.BERMELLON if "A" in col[-1] else g.PIZARRA
            fig.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=6,
                                       marker=dict(color=g.alfa(color, .35), line=dict(color=color, width=1.2)),
                                       showlegend=False), r, c)
            xs = np.linspace(max(0.01, x.min() * .9) if "E2" in col else x.min() - 10, x.max() * 1.05, 300)
            d_teo = _teorica(col)
            fig.add_trace(go.Scatter(x=xs, y=d_teo.pdf(xs), mode="lines", line=dict(color=g.TINTA, width=2,
                                     dash="dot"), name="Densidad teórica", showlegend=i == 0), r, c)
        fig.update_layout(height=560, legend=dict(y=1.08))
        for a in fig.layout.annotations:
            a.font = dict(family=g.FUENTE_TITULO, size=15)
        st.plotly_chart(fig, use_container_width=True, key="hist4")
        st.caption("Histogramas con k = 6 clases (regla 2^k > 60 de la guía). La línea punteada es la densidad "
                   "de la que se simularon los datos.")
        st.markdown("#### Medidas descriptivas")
        ui.tabla(tabla_resumen({v[0]: s[k] for k, v in GRUPOS.items()}), dec=3)

        st.markdown("#### Pruebas de bondad de ajuste de los cuatro grupos")
        filas = []
        for col, (tit, _) in GRUPOS.items():
            for dist in ("Normal", "Gamma"):
                r = bondad_cache(tuple(s[col]), dist, None, alpha, 999, "ancho")
                filas.append({"Grupo": tit, "Modelo": dist,
                              "χ² guía (p)": r["chi"]["p"], "χ² equiprob. (p)": r["chi_eq"]["p"],
                              "KS (p)": r["ks"]["p"], "AD (p)": r["ad"]["p"],
                              "Conclusión (AD)": "Rechaza" if r["ad"]["rechaza"] else "No rechaza"})
        ui.tabla(pd.DataFrame(filas), dec=4, indice=False)
        ui.lectura(
            "En el **escenario 1** ninguna prueba rechaza la normal, como se espera (los datos se generaron "
            "normales). Note que la gamma tampoco se rechaza: con α grande la gamma se parece mucho a una normal, "
            "y con solo 60 datos no se pueden distinguir.\n\nEn el **escenario 2**, KS y AD rechazan la normal con "
            "claridad, mientras que la χ² con clases de igual ancho se queda sin grados de libertad (v ≤ 1): la "
            "cola larga obliga a juntar casi todas las clases. Este es el argumento principal para usar KS y AD "
            "como complemento: **no necesitan agrupar y aprovechan cada dato**.")
        st.markdown("#### Detalle de un grupo")
        c1, c2 = st.columns(2)
        grupo = c1.selectbox("Grupo", list(GRUPOS), format_func=lambda k: GRUPOS[k][0], key="grp")
        dist = c2.selectbox("Distribución a probar", ["Normal", "Gamma", "Lognormal", "Weibull", "Exponencial"],
                            key="dst")
        bloque_bondad(s[grupo], dist, alpha, "el tiempo de procesamiento", clave=f"eq_{grupo}_{dist}")

    # ------------------------------------------------------------------ b)
    with tabs[1]:
        st.markdown("### ¿Los dos equipos tienen el mismo tiempo medio?")
        st.latex(r"H_0:\ \mu_A=\mu_B\qquad H_a:\ \mu_A\neq\mu_B")
        filas = []
        for e, nombre in (("E1", "Escenario 1 · normales"), ("E2", "Escenario 2 · gamma")):
            a, bb = s[f"{e}_equipo_A"], s[f"{e}_equipo_B"]
            f = inf.prueba_f(a, bb, alpha)
            t = inf.prueba_t_dos(a, bb, alpha)
            z = inf.prueba_z_dos(a, bb, alpha)
            ui.paso(e, nombre)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**1 · ¿Varianzas iguales? (F)**")
                st.latex(rf"F=\frac{{s^2_{{\max}}}}{{s^2_{{\min}}}}={f['F']:.4f}")
                st.markdown(f"crítico {f['critico']:.4f} · p {ui.pval(f['p'])} → "
                            f"**{'distintas' if f['rechaza'] else 'iguales'}** → t **{t['tipo']}**")
            with c2:
                st.markdown(f"**2 · t-Student ({t['tipo']})**")
                if t["tipo"] == "Welch":
                    st.latex(r"t=\frac{\bar x_A-\bar x_B}{\sqrt{s_A^2/n_A+s_B^2/n_B}}")
                else:
                    st.latex(r"t=\frac{\bar x_A-\bar x_B}{s_p\sqrt{1/n_A+1/n_B}}")
                st.latex(rf"t=\frac{{{t['m1']:.3f}-{t['m2']:.3f}}}{{{t['ee']:.4f}}}={t['t']:.4f},\ "
                         rf"v={t['gl']:.2f}")
                st.markdown(f"crítico ±{t['critico']:.4f} · p = **{t['p']:.4f}**")
            with c3:
                st.markdown("**3 · Normal (Z)**")
                st.latex(r"Z=\frac{\bar x_A-\bar x_B}{\sqrt{s_A^2/n_A+s_B^2/n_B}}")
                st.latex(rf"Z={z['z']:.4f}")
                st.markdown(f"crítico ±{z['critico']:.4f} · p = **{z['p']:.4f}**")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(g.curva_prueba(stats.t(t["gl"]), t["t"], t["critico"], "dos", "t", alto=240),
                                use_container_width=True, key=f"t_{e}")
                ui.veredicto(t["rechaza"], "t-Student")
            with cc2:
                st.plotly_chart(g.curva_prueba(stats.norm(), z["z"], z["critico"], "dos", "Z", alto=240),
                                use_container_width=True, key=f"z_{e}")
                ui.veredicto(z["rechaza"], "Normal")
            filas.append({"Escenario": nombre, "x̄_A": t["m1"], "x̄_B": t["m2"], "Diferencia": t["dif"],
                          "t": t["t"], "gl": t["gl"], "p (t)": t["p"], "Z": z["z"], "p (Z)": z["p"],
                          "IC 95 % (t)": f"[{t['ic'][0]:.2f}, {t['ic'][1]:.2f}]",
                          "IC 95 % (Z)": f"[{z['ic'][0]:.2f}, {z['ic'][1]:.2f}]"})
        st.markdown("#### Comparación")
        ui.tabla(pd.DataFrame(filas), dec=4, indice=False)
        r1, r2 = filas
        ui.lectura(
            f"El estadístico t y el Z **son el mismo número** (con varianzas distintas ambos dividen entre "
            f"√(s²_A/n_A + s²_B/n_B)); lo que cambia es la distribución con que se juzga. La t tiene colas más "
            f"pesadas, así que su valor p siempre es un poco mayor: es la prueba más **conservadora**. Con "
            f"n = 60 por grupo la diferencia es pequeña (gl ≈ {r1['gl']:.0f}), porque la t ya se parece a la "
            f"normal; por el **Teorema del Límite Central** la Z es válida aunque los datos no sean normales.\n\n"
            f"Escenario 1: p(t) = {r1['p (t)']:.4f} y p(Z) = {r1['p (Z)']:.4f} → "
            f"{'se rechaza' if r1['p (t)'] < alpha else 'no se rechaza'} H₀. "
            f"Escenario 2: p(t) = {r2['p (t)']:.4f} y p(Z) = {r2['p (Z)']:.4f} → "
            f"{'se rechaza' if r2['p (t)'] < alpha else 'no se rechaza'} H₀ (las medias teóricas son casi "
            f"iguales: 180 vs 180.5 ms).")

    # ------------------------------------------------------------------ c)
    with tabs[2]:
        st.markdown("### Prueba no paramétrica de Mann–Whitney (Wilcoxon de suma de rangos)")
        ui.lectura(
            "**¿En qué consiste?** En lugar de comparar medias, compara **posiciones**: se juntan los 120 datos, "
            "se ordenan de menor a mayor y a cada uno se le asigna su rango (1 = el más rápido). Si los dos "
            "equipos fueran iguales, sus rangos estarían bien mezclados y cada uno tendría una suma de rangos "
            "cercana a la mitad del total. U₁ cuenta en cuántos de los n_A·n_B pares posibles el dato de A es "
            "mayor que el de B. **No supone ninguna distribución** (por eso es no paramétrica) y es robusta a "
            "datos atípicos y a la asimetría.\n\nH₀: las dos poblaciones tienen la misma distribución "
            "(P(X_A > X_B) = 0.5)   ·   Hₐ: una tiende a dar valores mayores que la otra.", "teoria")
        st.latex(r"U_1=R_1-\frac{n_1(n_1+1)}{2},\qquad \mu_U=\frac{n_1n_2}{2},\qquad "
                 r"\sigma_U=\sqrt{\frac{n_1n_2(n_1+n_2+1)}{12}},\qquad Z=\frac{U_1-\mu_U}{\sigma_U}")
        filas = []
        for e, nombre in (("E1", "Escenario 1 · normales"), ("E2", "Escenario 2 · gamma")):
            a, bb = s[f"{e}_equipo_A"].to_numpy(), s[f"{e}_equipo_B"].to_numpy()
            m = inf.mann_whitney(a, bb, alpha)
            t = inf.prueba_t_dos(a, bb, alpha)
            ui.paso(e, nombre)
            c1, c2 = st.columns([1.4, 1])
            with c1:
                st.plotly_chart(g.rangos_mw(a, bb), use_container_width=True, key=f"mw_{e}")
            with c2:
                st.latex(rf"R_A={m['R1']:.1f},\ R_B={m['R2']:.1f}")
                st.latex(rf"U_A={m['R1']:.1f}-\frac{{60\cdot61}}{{2}}={m['U1']:.1f}")
                st.latex(rf"Z=\frac{{{m['U1']:.1f}-{m['mu_U']:.0f}}}{{{m['sigma_U']:.3f}}}={m['z']:.4f}")
                st.markdown(f"p = **{m['p']:.4f}** (exacto en SciPy: {m['p_scipy']:.4f})")
                ui.veredicto(m["rechaza"])
            filas.append({"Escenario": nombre, "Mediana A": m["med1"], "Mediana B": m["med2"],
                          "U_A": m["U1"], "Z (MW)": m["z"], "p (MW)": m["p"], "P̂(A > B)": m["prob_superioridad"],
                          "p (t)": t["p"]})
        ui.tabla(pd.DataFrame(filas), dec=4, indice=False)
        r1, r2 = filas
        ui.lectura(
            f"En el **escenario 1** (datos normales) Mann–Whitney y la t "
            f"{'llegan a la misma decisión' if (r1['p (MW)'] < alpha) == (r1['p (t)'] < alpha) else 'no coinciden'} "
            f"(p = {r1['p (MW)']:.4f} vs {r1['p (t)']:.4f}); la t es ligeramente más potente porque su supuesto "
            f"(normalidad) se cumple. En el **escenario 2** las medias teóricas son casi idénticas, pero las "
            f"distribuciones tienen formas distintas; "
            f"{'ninguna de las dos pruebas rechaza' if max(r2['p (MW)'], r2['p (t)']) >= alpha else 'ambas rechazan'} "
            f"(p = {r2['p (MW)']:.4f} vs {r2['p (t)']:.4f}). Aquí Mann–Whitney es la prueba más apropiada, "
            f"porque los datos son claramente no normales (ver literal a) y la media está dominada por pocos "
            f"tiempos muy largos. P̂(A > B) = {r2['P̂(A > B)']:.3f}: elegido un trabajo al azar de cada equipo, "
            f"el de A tarda más en el {100 * r2['P̂(A > B)']:.1f} % de los pares.")

    # ------------------------------------------------------------------ Monte Carlo
    with tabs[3]:
        st.markdown("### Una sola muestra puede engañar: repetir el experimento miles de veces")
        res = _mc(rep_mc, semilla)
        st.plotly_chart(g.barras_potencia(res), use_container_width=True, key="potmc")
        tab = pd.DataFrame(res).T.drop(columns="repeticiones")
        ui.tabla(tab, dec=4)
        e1, e2, c0 = res.values()
        ui.lectura(
            f"Se simularon **{rep_mc:,} experimentos** iguales al del punto 2 y se contó cuántas veces cada prueba "
            f"rechaza H₀. Si las poblaciones son distintas, ese porcentaje es la **potencia**; si son iguales, "
            f"es el **error tipo I** real.\n\n• **Control** (dos poblaciones iguales): las tres pruebas rechazan "
            f"entre {100 * min(c0['t de Welch'], c0['Z (normal)'], c0['Mann–Whitney']):.1f} % y "
            f"{100 * max(c0['t de Welch'], c0['Z (normal)'], c0['Mann–Whitney']):.1f} % de las veces, cerca de α = 5 %: "
            f"están bien calibradas incluso con datos gamma (TLC en acción).\n"
            f"• **Escenario 1**: con una diferencia real de 5 ms y σ de 30 y 50 ms, la potencia no pasa del "
            f"{100 * max(e1['t de Welch'], e1['Z (normal)'], e1['Mann–Whitney']):.1f} %. Detectar esa diferencia con 60 datos es casi cuestión de suerte; si la muestra de "
            f"esta semilla resultó significativa, fue una muestra \"afortunada\".\n"
            f"• **Escenario 2**: las medias son prácticamente iguales, así que t y Z rechazan ≈ 5 %; "
            f"Mann–Whitney rechaza algo más ({100 * e2['Mann–Whitney']:.1f} %) porque detecta la diferencia de "
            f"**forma** (P(A > B) = {e2['P(A > B)']:.3f} ≠ 0.5), que la media no ve.", "")

    with tabs[4]:
        st.dataframe(s, use_container_width=True, height=380)
        st.download_button("Descargar simulacion.csv", s.to_csv(index=False).encode("utf-8"),
                           f"simulacion_semilla_{semilla}.csv", "text/csv")
    ui.pie()


def _teorica(col):
    return {"E1_equipo_A": stats.norm(175, 30), "E1_equipo_B": stats.norm(170, 50),
            "E2_equipo_A": stats.gamma(1.2, scale=150), "E2_equipo_B": stats.gamma(0.95, scale=190)}[col]
