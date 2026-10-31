"""Portada — presentación del proyecto y mapa de temas del curso."""

from __future__ import annotations

import streamlit as st

from src import ui

# (tema, punto 1, punto 2, punto 3 Sistemas, punto 3 Industrial)  ·  2 = central, 1 = de apoyo
MAPA = [
    ("Estadística descriptiva (centro, dispersión, cuartiles, cajas)", 2, 2, 2, 1),
    ("Probabilidad y variables aleatorias", 1, 2, 2, 2),
    ("Distribuciones discretas: Binomial, Poisson, Binomial negativa", 1, 0, 2, 2),
    ("Distribuciones continuas: Normal, Gamma, Lognormal", 2, 2, 2, 2),
    ("Esperanza y varianza", 1, 2, 2, 1),
    ("Estimación por máxima verosimilitud", 1, 2, 2, 1),
    ("Teorema del Límite Central", 1, 2, 1, 1),
    ("Simulación de Monte Carlo", 0, 2, 0, 0),
    ("Intervalos de confianza (μ, σ², p)", 2, 1, 1, 2),
    ("Pruebas de hipótesis (Z, t, F, proporciones)", 2, 2, 2, 2),
    ("Bondad de ajuste χ² (guía)", 1, 2, 2, 2),
    ("Kolmogorov–Smirnov y Anderson–Darling", 1, 2, 2, 2),
    ("Prueba no paramétrica de Mann–Whitney", 1, 2, 2, 0),
    ("Correlación y regresión", 2, 0, 2, 0),
    ("Control estadístico de calidad y capacidad", 2, 0, 0, 2),
]


def _mapa_html() -> str:
    filas = []
    for tema, *v in MAPA:
        celdas = "".join(f'<td>{"<span class=si></span>" if x == 2 else "<span class=medio></span>" if x == 1 else ""}</td>'
                         for x in v)
        filas.append(f"<tr><td>{tema}</td>{celdas}</tr>")
    return ('<table class="mapa"><thead><tr><th>Tema del curso</th><th>Punto 1<br>Hornos</th>'
            '<th>Punto 2<br>Equipos</th><th>Punto 3<br>Sistemas</th><th>Punto 3<br>Industrial</th></tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table>')


def pagina():
    st.markdown("""
<div class="portada">
  <div class="kicker">Proyecto de aula · Estadística para Ingeniería · 2026-2</div>
  <h1>Decidir con <em>datos</em>:<br>de los hornos al código</h1>
  <p>Una herramienta en Python que aplica todo el curso a tres problemas de ingeniería: un estudio de mercado
  de hornos a gas, la comparación de dos equipos de procesamiento y dos casos con datos <b>reales</b> de
  nuestras carreras —el historial de un proyecto de software y el control de calidad de una planta—.
  Como valor agregado, las pruebas de <b>Kolmogorov–Smirnov</b> y <b>Anderson–Darling</b> explicadas a fondo y un
  laboratorio para analizar cualquier archivo.</p>
  <div class="equipo">
    <div><b>Juan Manuel Moreno Muñoz</b>Ingeniería de Sistemas</div>
    <div><b>David Restrepo</b>Integrante</div>
    <div><b>Juan Fernando</b>Integrante</div>
    <div><b>Prof. Horacio Fernández C.</b>Facultad de Ingenierías · UdM</div>
  </div>
</div>""", unsafe_allow_html=True)
    ui.kpis([("3 665", "hornos analizados", "punto 1"), ("240", "tiempos simulados", "punto 2"),
             ("2 238", "commits reales", "Streamlit 2024–2025"), ("2 900", "mediciones reales", "Montgomery · qcc"),
             ("3", "pruebas de ajuste", "χ² · KS · AD")], acento=2)
    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.markdown("### Cómo se conectan los temas del curso")
        st.markdown(_mapa_html(), unsafe_allow_html=True)
        st.caption("● uso central   ○ uso de apoyo")
    with c2:
        st.markdown("### Recorrido")
        ui.lectura(
            "**01 · Hornos** — cajas y bigotes, capacidad de proceso (Cp, Cpk), intervalos para μ, σ² y p, "
            "prueba F y prueba de proporción.\n\n"
            "**02 · Equipos A y B** — histogramas, χ², KS y AD sobre datos simulados; t-Student vs normal; "
            "Mann–Whitney; y un estudio de Monte Carlo de la potencia de cada prueba.\n\n"
            "**03 · Sistemas** — ¿la llegada de commits es Poisson? ¿qué distribución tiene el tamaño de un "
            "cambio? Correlación y regresión con datos reales.\n\n"
            "**04 · Industrial** — cartas X̄–R y p, normalidad, capacidad y binomial con datos reales.\n\n"
            "**05 · Laboratorio** — suba su propio archivo.\n\n"
            "**06 · Teoría** y **07 · Taller** — las tres pruebas explicadas y la guía del profesor resuelta.",
            "teoria")
        ui.lectura("Todos los resultados usan la semilla **2026** y se reproducen exactamente con los scripts del "
                   "repositorio, en Excel y en R. El nivel de significancia y la confianza se ajustan en la barra "
                   "lateral y todas las páginas se recalculan.", "")
    ui.pie()
