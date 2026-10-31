"""Punto 3B — Ingeniería Industrial con datos reales: control estadístico de calidad."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src import aplicaciones as ap
from src import graficos as g
from src import ui
from src.paginas.comun import bloque_bondad


@st.cache_data
def _datos():
    return ap.cargar_industrial()


@st.cache_data(show_spinner="Construyendo cartas de control…")
def _analisis(alpha):
    a, j = ap.cargar_industrial()
    return ap.analisis_anillos(a, alpha), ap.analisis_jugo(j, alpha)


def pagina(alpha: float):
    a, j = _datos()
    R, J = _analisis(alpha)
    ui.cabecera("04", "Punto 3 · Ingeniería Industrial · datos reales",
                "¿El proceso es <em>capaz</em> y está bajo control?",
                "Control estadístico de procesos con dos bases <b>reales</b> del libro de Montgomery "
                "(<i>Introduction to Statistical Quality Control</i>), publicadas en el paquete <code>qcc</code> de R: "
                "diámetros de anillos de pistón forjados para motores y latas de jugo de naranja no conformes.")
    ui.kpis([("200", "diámetros medidos", "40 muestras de 5"), (f"{R['cap']['Cpk']:.2f}", "Cpk (fase I)"),
             ("2 700", "latas inspeccionadas", "54 muestras de 50"),
             (f"{100 * J['pbar']:.1f}% → {100 * J['p_fase2']:.1f}%", "no conformes antes → después")], acento=3)

    tabs = st.tabs(["a · Problema", "b · Datos", "c.1 Carta X̄–R", "c.2 Normalidad y capacidad",
                    "c.3 ¿Se corrió el proceso?", "c.4 Latas: carta p y binomial", "d · Conclusiones"])
    with tabs[0]:
        st.markdown("### a) Aplicación y problema")
        ui.lectura(
            "**Aplicación de la Estadística en Ingeniería Industrial:** control estadístico de procesos (SPC) y "
            "análisis de capacidad. Es la herramienta con la que una planta decide si un proceso produce dentro de "
            "especificación y si algo cambió.\n\n**Problema 1 (variables):** los anillos de pistón deben medir "
            "**74.000 ± 0.050 mm**. ¿El proceso está bajo control? ¿Es capaz de cumplir la especificación? ¿Cambió "
            "cuando se tomaron las muestras de vigilancia?\n\n**Problema 2 (atributos):** una máquina llena latas "
            "de jugo; cada media hora se inspeccionan 50 latas. ¿El número de defectuosas sigue una binomial? "
            "¿Mejoró el proceso después de ajustar la máquina?\n\n**Datos requeridos:** mediciones o conteos "
            "agrupados en muestras pequeñas tomadas en el tiempo; se obtienen del registro de inspección de la planta.",
            "teoria")
    with tabs[1]:
        st.markdown("### b) Datos")
        st.markdown(
            "* **Anillos (`pistonrings`)**: 40 muestras de 5 anillos. Las 25 primeras son la **fase I** (para "
            "calcular los límites de control) y las 15 últimas, la **fase II** (vigilancia).\n"
            "* **Jugo (`orangejuice`)**: 54 muestras de 50 latas. Las 30 primeras son la fase I; Montgomery "
            "identificó causas asignables en las muestras **15** (nuevo lote de cartón) y **23** (operario sin "
            "experiencia), que se excluyen. Después se ajustó la máquina y se tomaron 24 muestras más (fase II).")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(a, use_container_width=True, height=300, hide_index=True)
        with c2:
            st.dataframe(j, use_container_width=True, height=300, hide_index=True)

    with tabs[2]:
        Cx = R["cartas"]
        lim, tab = Cx["lim"], Cx["tabla"]
        st.markdown("### c.1) Cartas de control X̄ y R")
        st.latex(r"LSC_{\bar x}=\bar{\bar x}+A_2\bar R,\quad LIC_{\bar x}=\bar{\bar x}-A_2\bar R,\quad "
                 r"LSC_R=D_4\bar R,\quad LIC_R=D_3\bar R\qquad (n=5:\ A_2=0.577,\ D_3=0,\ D_4=2.114)")
        st.latex(rf"\bar{{\bar x}}={lim['LC_x']:.5f}\qquad \bar R={lim['LC_r']:.5f}")
        st.plotly_chart(g.carta_control(tab.xbar.values, lim["LC_x"], lim["LSC_x"], lim["LIC_x"], 25,
                                        "Carta X̄ (media de cada muestra)", "Diámetro medio (mm)"),
                        use_container_width=True, key="cx")
        st.plotly_chart(g.carta_control(tab.R.values, lim["LC_r"], lim["LSC_r"], lim["LIC_r"], 25,
                                        "Carta R (rango de cada muestra)", "Rango (mm)", alto=260),
                        use_container_width=True, key="cr")
        fuera = tab.index[tab.fuera_x].tolist()
        ui.lectura(
            f"En la fase I las 25 muestras están dentro de los límites: el proceso estaba **bajo control** y "
            f"sus límites son confiables. En la fase II la carta R sigue estable (la variabilidad no cambió), pero "
            f"en la carta X̄ las muestras **{', '.join(map(str, fuera))}** salen por encima del límite superior "
            f"y desde la muestra 34 todas las medias quedan por encima de la línea central: **la media del proceso se corrió hacia arriba**. "
            f"Es la señal para detener la línea y buscar la causa asignable.")

    with tabs[3]:
        cap = R["cap"]
        st.markdown("### c.2) ¿Los diámetros son normales? ¿Es capaz el proceso?")
        st.plotly_chart(g.histograma_especificacion(a.loc[a.fase_I, "diametro_mm"], R["lie"], R["lse"],
                                                    R["normal"]["congelada"]), use_container_width=True,
                        key="hist_esp")
        bloque_bondad(a.loc[a.fase_I, "diametro_mm"], "Normal", alpha, "el diámetro", clave="anillos")
        st.latex(rf"C_p=\frac{{74.05-73.95}}{{6({cap['s']:.5f})}}={cap['Cp']:.3f}\qquad "
                 rf"C_{{pk}}=\frac{{\min(74.05-{cap['media']:.5f},\ {cap['media']:.5f}-73.95)}}{{3({cap['s']:.5f})}}"
                 rf"={cap['Cpk']:.3f}")
        ic, tn = R["ic_media"], R["t_nominal"]
        ui.lectura(
            f"Las tres pruebas aceptan la normalidad (AD p = {R['normal']['ad']['p']:.3f}), así que los índices "
            f"de capacidad y los límites de la carta son válidos. **Cp = {cap['Cp']:.3f} y Cpk = {cap['Cpk']:.3f}**: "
            f"ambos superan 1.33, el proceso es **capaz** — la tolerancia es {cap['Cp']:.2f} veces más ancha que "
            f"la variación natural (6σ). Se esperan apenas {cap['ppm_fuera_teorico']:.1f} anillos defectuosos por "
            f"millón. La media está entre {ic['li']:.4f} y {ic['ls']:.4f} mm (IC 95 %) y no difiere del nominal "
            f"74.000 (t = {tn['t']:.3f}, p = {tn['p']:.3f}).")

    with tabs[4]:
        tf, ff = R["t_fases"], R["f_fases"]
        st.markdown("### c.3) Fase I vs fase II: confirmación con pruebas de hipótesis")
        c1, c2 = st.columns(2)
        with c1:
            ui.paso("F", "¿Cambió la variabilidad?")
            st.latex(rf"F={ff['F']:.4f},\ \ p={ff['p']:.4f}")
            ui.veredicto(ff["rechaza"])
        with c2:
            ui.paso("t", f"¿Cambió la media? (t {tf['tipo']})")
            st.latex(rf"t=\frac{{{tf['m1']:.5f}-{tf['m2']:.5f}}}{{{tf['ee']:.5f}}}={tf['t']:.4f},\ \ "
                     rf"p={tf['p']:.5f}")
            ui.veredicto(tf["rechaza"])
        st.plotly_chart(g.curva_prueba(stats.t(tf["gl"]), tf["t"], tf["critico"], "dos", "t", alto=260),
                        use_container_width=True, key="t_fases")
        ui.lectura(
            f"La prueba F {'detecta' if ff['rechaza'] else 'no detecta'} cambio en la variabilidad "
            f"(p = {ff['p']:.3f}){', aunque al límite' if ff['rechaza'] and ff['p'] > .01 else ''}, mientras que la "
            f"prueba t confirma con fuerza (p = {tf['p']:.5f}) lo que se vio en la carta: la media pasó de "
            f"{tf['m1']:.4f} a {tf['m2']:.4f} mm, un corrimiento de {1000 * abs(tf['dif']):.1f} micras "
            f"(IC 95 %: {1000 * abs(tf['ic'][1]):.1f} a {1000 * abs(tf['ic'][0]):.1f} micras). La carta de control y "
            f"la prueba de hipótesis son dos caras de la misma idea.")

    with tabs[5]:
        st.markdown("### c.4) Latas no conformes: carta p y prueba de ajuste a la binomial")
        jj = j.copy()
        jj["p"] = jj.no_conformes / jj.tamano
        st.latex(rf"\bar p={J['pbar']:.4f}\qquad LSC,\ LIC=\bar p\pm3\sqrt{{\frac{{\bar p(1-\bar p)}}{{n}}}}"
                 rf"={J['lsc']:.4f},\ {J['lic']:.4f}")
        st.plotly_chart(g.carta_control(jj.p.values, J["pbar"], J["lsc"], J["lic"], 30,
                                        "Carta p (fracción de latas no conformes)", "Fracción no conforme",
                                        marcar=J["excluidas"]), use_container_width=True, key="cp")
        bi = J["binomial"]
        st.markdown(f"**¿El número de latas defectuosas por muestra sigue una Binomial(n = 50, p)?** "
                    f"(fase I sin las muestras 15 y 23, {J['k1']} muestras; p̂ = {bi['params'][1]:.4f})")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            ui.tabla(bi["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 3, "(Oᵢ − Eᵢ)²/Eᵢ": 4}, indice=False)
            st.markdown(f"EP = **{bi['EP']:.4f}**, v = {bi['k']} − 1 − 1 = {bi['v']}, χ² crítico = "
                        f"{bi['critico']:.4f}, p = **{bi['p']:.4f}**")
            ui.veredicto(bi["rechaza"])
        with c2:
            st.plotly_chart(g.barras_chi(bi["tabla"], alto=300), use_container_width=True, key="chi_jugo")
        pr = J["prueba"]
        ui.lectura(
            f"El modelo binomial, base teórica de la carta p, **no se rechaza** (p = {bi['p']:.3f}). Tras ajustar la "
            f"máquina, la fracción no conforme bajó de {100 * pr['p1']:.2f} % a **{100 * pr['p2']:.2f} %**. La prueba de "
            f"dos proporciones (Z = {pr['z']:.3f}, p {ui.pval(pr['p'])}) confirma que la mejora es real: entre "
            f"{100 * pr['ic'][0]:.1f} y {100 * pr['ic'][1]:.1f} puntos porcentuales menos de latas defectuosas. "
            f"En la fase II se deberían recalcular los límites con el nuevo p̄.")

    with tabs[6]:
        st.markdown("### d) Interpretación, conclusiones y recomendaciones")
        ui.lectura(
            f"1. El proceso de anillos es **capaz** (Cpk = {R['cap']['Cpk']:.2f} > 1.33) y estuvo bajo control en la "
            f"fase I; su distribución es normal, lo que valida los límites de control.\n"
            f"2. En la fase II la media **se desplazó** {1000 * abs(R['t_fases']['dif']):.1f} micras hacia arriba "
            f"(muestras 37–39 fuera de control). Aunque todavía no se producen piezas fuera de especificación, "
            f"hay que corregir la causa antes de que el desplazamiento crezca.\n"
            f"3. El llenado de latas pasó de {100 * J['pbar']:.1f} % a {100 * J['p_fase2']:.1f} % de no conformes: "
            f"el ajuste funcionó. Pero {100 * J['p_fase2']:.0f} % sigue siendo alto: se recomienda un estudio de "
            f"causas (diagrama de Pareto) y recalcular la carta p.\n"
            f"4. Las pruebas de bondad de ajuste no son un trámite: las cartas X̄ y p **dependen** de los supuestos "
            f"normal y binomial que aquí se verificaron.")
    ui.pie()
