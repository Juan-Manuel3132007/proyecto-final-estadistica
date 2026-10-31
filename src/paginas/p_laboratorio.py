"""Laboratorio — el usuario sube CUALQUIER archivo y le aplica todas las herramientas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src import bondad as b
from src import graficos as g
from src import inferencia as inf
from src import ui
from src.datos import (EXTENSIONES, columnas_categoricas, columnas_numericas, es_discreta,
                       hojas_excel, leer_archivo, leer_texto)
from src.descriptiva import numero_clases, resumen
from src.paginas.comun import bloque_bondad

EJEMPLOS = {
    "Hornos (punto 1)": "data/hornos.csv",
    "Equipos A y B (punto 2)": "data/simulacion.csv",
    "Commits de Streamlit (punto 3, Sistemas)": "data/sistemas_commits.csv",
    "Commits por día (punto 3, Sistemas)": "data/sistemas_diario.csv",
    "Anillos de pistón (punto 3, Industrial)": "data/industrial_anillos.csv",
    "Latas de jugo (punto 3, Industrial)": "data/industrial_jugo.csv",
}


@st.cache_data(show_spinner="Leyendo el archivo…")
def _leer(nombre, contenido, hoja):
    return leer_archivo(nombre, contenido, hoja)


def _cargar() -> tuple[pd.DataFrame | None, str]:
    origen = st.segmented_control("Origen de los datos", ["Subir archivo", "Pegar datos", "Ejemplos del proyecto"],
                                  default="Subir archivo", key="origen")
    if origen == "Subir archivo":
        f = st.file_uploader("Arrastre aquí cualquier archivo de datos", type=EXTENSIONES, key="up",
                             help="CSV, TXT o TSV (separados por coma, punto y coma, tabulador o espacios; decimales "
                                  "con punto o con coma), Excel (.xlsx, .xls, .xlsm, .ods), JSON o Parquet.")
        if f is None:
            st.info("Todavía no hay archivo. También puede pegar datos o usar los ejemplos del proyecto.")
            return None, ""
        contenido = f.getvalue()
        hojas = hojas_excel(f.name, contenido)
        hoja = st.selectbox("Hoja", hojas, key="hoja") if len(hojas) > 1 else 0
        try:
            return _leer(f.name, contenido, hoja), f.name
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")
            return None, ""
    if origen == "Pegar datos":
        txt = st.text_area("Pegue una columna de números (o una tabla con encabezados) copiada de Excel",
                           height=180, key="pegar",
                           placeholder="tiempo\n175,3\n181,2\n169,8\n…")
        if not txt.strip():
            return None, ""
        try:
            return leer_texto(txt.encode("utf-8")), "datos pegados"
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudieron interpretar los datos: {e}")
            return None, ""
    ej = st.selectbox("Base de ejemplo", list(EJEMPLOS), key="ej")
    return pd.read_csv(EJEMPLOS[ej]), ej


def pagina(alpha: float, confianza: float):
    ui.cabecera("05", "Herramienta abierta", "Laboratorio: <em>sus</em> datos",
                "Suba cualquier archivo (CSV, Excel, TXT, JSON…) y aplíquele todas las herramientas del proyecto: "
                "descriptivas, χ², Kolmogorov–Smirnov, Anderson–Darling, intervalos, pruebas de hipótesis, "
                "Mann–Whitney, correlación y regresión.")
    df, nombre = _cargar()
    if df is None:
        ui.pie()
        return
    nums = columnas_numericas(df)
    ui.kpis([(f"{len(df):,}", "filas"), (f"{df.shape[1]}", "columnas"), (f"{len(nums)}", "columnas numéricas"),
             (f"{int(df.isna().sum().sum()):,}", "celdas vacías")])
    with st.expander(f"Vista previa · {nombre}", expanded=False):
        st.dataframe(df.head(200), use_container_width=True, height=260)
    if not nums:
        st.warning("El archivo no tiene columnas numéricas con al menos 3 datos.")
        ui.pie()
        return

    t1, t2, t3, t4, t5 = st.tabs(["Descriptiva", "Bondad de ajuste", "Intervalos y pruebas",
                                  "Comparar dos grupos", "Correlación y regresión"])
    with t1:
        cols = st.multiselect("Variables", nums, default=nums[:min(4, len(nums))], key="lab_desc")
        if cols:
            ui.tabla(pd.DataFrame({c: resumen(df[c]) for c in cols}), dec=4)
            var = st.selectbox("Histograma y caja de", cols, key="lab_hist")
            x = df[var].dropna()
            c1, c2 = st.columns([1.6, 1])
            with c1:
                fig = go.Figure(go.Histogram(x=x, nbinsx=numero_clases(len(x)), marker=dict(color=g.alfa(g.PIZARRA, .4),
                                                                    line=dict(color=g.PIZARRA, width=1))))
                fig.update_layout(height=340, title=None)
                fig.update_xaxes(title=var)
                fig.update_yaxes(title="Frecuencia")
                st.plotly_chart(fig, use_container_width=True, key="lab_h")
            with c2:
                fig = go.Figure(go.Box(y=x, name=var, marker_color=g.BERMELLON, boxmean=True,
                                       fillcolor=g.alfa(g.BERMELLON, .15)))
                fig.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="lab_b")

    with t2:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        var = c1.selectbox("Variable", nums, key="lab_gof_var")
        x = df[var].dropna().to_numpy(dtype=float)
        discreta = es_discreta(x)
        tipo = c2.segmented_control("Tipo", ["Continua", "Discreta"], default="Discreta" if discreta else "Continua",
                                    key="lab_tipo")
        if tipo == "Discreta":
            dist = c3.selectbox("Distribución", ["Poisson", "Binomial", "Binomial negativa"], key="lab_dd")
            ensayos = None
            if dist == "Binomial":
                ensayos = st.number_input("Número de ensayos n", min_value=1, value=int(max(1, x.max())), key="lab_n")
            conocer = st.checkbox("Los parámetros son conocidos (no se estiman con la muestra)", key="lab_dk")
            params = None
            if conocer:
                if dist == "Poisson":
                    params = (st.number_input("λ", value=float(np.mean(x)), key="lab_lam"),)
                elif dist == "Binomial":
                    params = (ensayos, st.number_input("p", 0.0, 1.0, float(np.mean(x) / ensayos), key="lab_p"))
                else:
                    r0, p0 = b.ajustar(x, dist)
                    params = (st.number_input("r", value=float(r0), key="lab_r"),
                              st.number_input("p", 0.0, 1.0, float(p0), key="lab_pp"))
            res = b.chi_cuadrado_datos_discretos(x.astype(int), dist, alpha, params=params, ensayos=ensayos)
            st.markdown(f"Parámetros: **{b.texto_parametros(dist, res['params'])}**")
            cc1, cc2 = st.columns([1.1, 1])
            with cc1:
                ui.tabla(res["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 3, "(Oᵢ − Eᵢ)²/Eᵢ": 4}, indice=False)
            with cc2:
                st.plotly_chart(g.barras_chi(res["tabla"], alto=300), use_container_width=True, key="lab_dchi")
            if res["v"] >= 1:
                st.latex(rf"EP={res['EP']:.4f},\quad v={res['k']}-{res['p_est']}-1={res['v']},\quad "
                         rf"\chi^2_{{({alpha},{res['v']})}}={res['critico']:.4f},\quad P={res['p']:.4f}")
                ui.veredicto(res["rechaza"])
            else:
                st.warning(res["nota"])
            st.caption("Para variables discretas se usa χ²: KS y AD están diseñadas para distribuciones continuas.")
        else:
            if (x <= 0).any():
                st.caption("Hay valores ≤ 0: solo se ofrecen distribuciones definidas en toda la recta.")
            st.markdown("**Ranking automático** (máxima verosimilitud + AIC; menor AIC = mejor ajuste):")
            cand = ("Normal", "Lognormal", "Gamma", "Weibull", "Exponencial", "Uniforme")
            rk = b.ranking(x, cand)
            ui.tabla(rk, dec=4)
            opciones = list(rk["Distribución"]) or ["Normal"]
            cc1, cc2, cc3 = st.columns(3)
            dist = cc1.selectbox("Distribución a probar", opciones, key="lab_dc")
            clases = cc2.segmented_control("Clases para χ²", ["ancho", "equiprobables"], default="ancho",
                                           key="lab_cl", format_func=lambda s: "Igual ancho (guía)"
                                           if s == "ancho" else "Equiprobables")
            nmc = cc3.select_slider("Réplicas Monte Carlo", [199, 499, 999, 1999], value=499, key="lab_mc")
            conocer = st.checkbox("Los parámetros son conocidos (caso 0: no se estiman)", key="lab_ck")
            params = None
            if conocer:
                est = b.ajustar(x, dist)
                etiquetas = b.CONTINUAS[dist][1]
                cols_p = st.columns(len(est))
                params = tuple(cp.number_input(e, value=float(v), format="%.6f", key=f"lab_par_{i}")
                               for i, (cp, e, v) in enumerate(zip(cols_p, etiquetas, est)))
            st.plotly_chart(g.histograma_ajuste(x, {dist: b.congelar(dist, params or b.ajustar(x, dist))}),
                            use_container_width=True, key="lab_hfit")
            bloque_bondad(x, dist, alpha, f"«{var}»", params=params, clave="lab", n_mc=nmc,
                          clases=clases or "ancho")

    with t3:
        var = st.selectbox("Variable", nums, key="lab_inf_var")
        x = df[var].dropna()
        c1, c2 = st.columns(2)
        with c1:
            ui.paso("IC", f"Intervalos del {100 * confianza:.0f} %")
            m, v = inf.ic_media(x, confianza), inf.ic_varianza(x, confianza)
            st.latex(rf"{m['li']:.5g}\le\mu\le{m['ls']:.5g}")
            st.latex(rf"{v['li']:.5g}\le\sigma^2\le{v['ls']:.5g}")
            st.latex(rf"{np.sqrt(v['li']):.5g}\le\sigma\le{np.sqrt(v['ls']):.5g}")
            st.caption("El intervalo de σ² supone normalidad: verifíquela en la pestaña de bondad de ajuste.")
        with c2:
            ui.paso("t", "Prueba sobre la media")
            mu0 = st.number_input("μ₀ (valor de referencia)", value=float(round(x.mean(), 3)), key="lab_mu0")
            t = inf.prueba_t_una(x, mu0, alpha)
            st.latex(rf"t=\frac{{{t['media']:.5g}-{mu0:.5g}}}{{{t['s']:.5g}/\sqrt{{{t['n']}}}}}={t['t']:.4f},"
                     rf"\quad p={t['p']:.4f}")
            ui.veredicto(t["rechaza"])
        ui.paso("p", "Proporción de datos que cumplen un límite")
        cc1, cc2, cc3 = st.columns(3)
        sentido = cc1.selectbox("Condición", ["≤", "≥"], key="lab_sent")
        limite = cc2.number_input("Límite", value=float(x.median()), key="lab_lim")
        p0 = cc3.number_input("p₀ (proporción afirmada)", 0.0, 1.0, 0.5, key="lab_p0")
        exitos = int((x <= limite).sum() if sentido == "≤" else (x >= limite).sum())
        icp = inf.ic_proporcion(exitos, len(x), confianza)
        pp = inf.prueba_proporcion(exitos, len(x), p0, alpha)
        st.markdown(f"Cumplen **{exitos} de {len(x)}** (p̂ = {icp['p']:.4f}) · IC: [{icp['li']:.4f}, "
                    f"{icp['ls']:.4f}] · Z = {pp['z']:.4f}, p = {pp['p']:.4f}")
        ui.veredicto(pp["rechaza"], f"H₀: p = {p0}")

    with t4:
        modo = st.segmented_control("¿Cómo están los grupos?", ["Una columna + columna de grupo", "Dos columnas"],
                                    default="Dos columnas" if len(nums) >= 2 else "Una columna + columna de grupo",
                                    key="lab_modo")
        x1 = x2 = None
        n1 = n2 = ""
        if modo == "Dos columnas":
            if len(nums) < 2:
                st.info("Se necesitan al menos dos columnas numéricas.")
            else:
                c1, c2 = st.columns(2)
                n1 = c1.selectbox("Grupo 1", nums, key="lab_g1")
                n2 = c2.selectbox("Grupo 2", [c for c in nums if c != n1], key="lab_g2")
                x1, x2 = df[n1].dropna().to_numpy(), df[n2].dropna().to_numpy()
        else:
            cats = columnas_categoricas(df)
            if not cats:
                st.info("No hay una columna con categorías (2 a 30 niveles) para formar los grupos.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                var = c1.selectbox("Variable", nums, key="lab_cv")
                gc = c2.selectbox("Columna de grupo", cats, key="lab_gc")
                niveles = list(pd.Series(df[gc].dropna().unique()).sort_values())
                n1 = c3.selectbox("Grupo 1", niveles, key="lab_n1")
                n2 = c4.selectbox("Grupo 2", [n for n in niveles if n != n1], key="lab_n2")
                x1 = df.loc[df[gc] == n1, var].dropna().to_numpy()
                x2 = df.loc[df[gc] == n2, var].dropna().to_numpy()
        if x1 is not None and len(x1) > 2 and len(x2) > 2:
            fig = go.Figure()
            for xx, nn, col in ((x1, n1, g.BERMELLON), (x2, n2, g.PIZARRA)):
                fig.add_trace(go.Box(x=xx, name=str(nn), marker_color=col, orientation="h", boxmean=True,
                                     fillcolor=g.alfa(col, .15)))
            fig.update_layout(height=260, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="lab_cmp")
            f, lv = inf.prueba_f(x1, x2, alpha), inf.prueba_levene(x1, x2, alpha)
            t, z, mw = inf.prueba_t_dos(x1, x2, alpha), inf.prueba_z_dos(x1, x2, alpha), inf.mann_whitney(x1, x2, alpha)
            tabla = pd.DataFrame([
                {"Prueba": "F (varianzas, supone normalidad)", "Estadístico": f["F"], "Valor p": f["p"],
                 "Decisión": "Rechaza H₀" if f["rechaza"] else "No rechaza H₀"},
                {"Prueba": "Levene (varianzas, robusta)", "Estadístico": lv["W"], "Valor p": lv["p"],
                 "Decisión": "Rechaza H₀" if lv["rechaza"] else "No rechaza H₀"},
                {"Prueba": f"t-Student ({t['tipo']})", "Estadístico": t["t"], "Valor p": t["p"],
                 "Decisión": "Rechaza H₀" if t["rechaza"] else "No rechaza H₀"},
                {"Prueba": "Z (normal, n grande)", "Estadístico": z["z"], "Valor p": z["p"],
                 "Decisión": "Rechaza H₀" if z["rechaza"] else "No rechaza H₀"},
                {"Prueba": "Mann–Whitney (no paramétrica)", "Estadístico": mw["z"], "Valor p": mw["p"],
                 "Decisión": "Rechaza H₀" if mw["rechaza"] else "No rechaza H₀"},
            ])
            ui.tabla(tabla, dec=4, indice=False)
            st.caption(f"x̄₁ = {t['m1']:.4g}, x̄₂ = {t['m2']:.4g} · medianas {mw['med1']:.4g} y {mw['med2']:.4g} · "
                       f"P̂(X₁ > X₂) = {mw['prob_superioridad']:.3f}")

    with t5:
        if len(nums) < 2:
            st.info("Se necesitan al menos dos columnas numéricas.")
        else:
            met = st.segmented_control("Coeficiente", ["pearson", "spearman"], default="pearson", key="lab_met",
                                       format_func=str.capitalize)
            m = df[nums].corr(method=met or "pearson")
            st.plotly_chart(g.correlaciones(m, alto=max(320, 46 * len(nums))), use_container_width=True,
                            key="lab_corr")
            c1, c2 = st.columns(2)
            vx = c1.selectbox("X", nums, key="lab_x")
            vy = c2.selectbox("Y", [c for c in nums if c != vx], key="lab_y")
            d2 = df[[vx, vy]].dropna()
            r = inf.correlacion(d2[vx], d2[vy])
            reg = inf.regresion(d2[vx], d2[vy])
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=d2[vx], y=d2[vy], mode="markers",
                                       marker=dict(color=g.PIZARRA, size=6, opacity=.5)))
            xs = np.linspace(d2[vx].min(), d2[vx].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=reg["b0"] + reg["b1"] * xs, mode="lines",
                                     line=dict(color=g.BERMELLON, width=3)))
            fig.update_layout(height=380, showlegend=False)
            fig.update_xaxes(title=vx)
            fig.update_yaxes(title=vy)
            st.plotly_chart(fig, use_container_width=True, key="lab_sc")
            st.latex(rf"\hat y={reg['b0']:.5g}{'+' if reg['b1'] >= 0 else '-'}{abs(reg['b1']):.5g}\,x\qquad "
                     rf"r={r['r']:.4f}\ ({inf.interpretar_r(r['r'])})\qquad \rho_{{S}}={r['rho']:.4f}\qquad "
                     rf"R^2={r['r2']:.4f}")
            st.markdown(f"Prueba de H₀: ρ = 0 → t = {r['t']:.3f}, p {ui.pval(r['p_r'])}")
    ui.pie()
