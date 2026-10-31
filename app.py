"""
app.py — Proyecto de Aula · Estadística para Ingeniería · Universidad de Medellín (2026-2)

Profesor: Horacio Fernández C.
Integrantes: Juan Manuel Moreno Muñoz, David Restrepo, Juan Fernando

Ejecutar con:   python -m streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Proyecto de Aula · Estadística UdM", page_icon="📐", layout="wide",
                   initial_sidebar_state="expanded")

from src import ui  # noqa: E402
from src.paginas import (p_equipos, p_hornos, p_industrial, p_laboratorio, p_portada,  # noqa: E402
                         p_sistemas, p_taller, p_teoria)

ui.estilos()

PAGINAS = {
    "00   Portada": "portada",
    "01   Hornos de cocción": "hornos",
    "02   Equipos A y B": "equipos",
    "03   Datos reales · Sistemas": "sistemas",
    "04   Datos reales · Industrial": "industrial",
    "05   Laboratorio: sus datos": "laboratorio",
    "06   Teoría: χ², KS y AD": "teoria",
    "07   Taller de la guía": "taller",
}

with st.sidebar:
    st.markdown('<div class="lateral-meta">Estadística para Ingeniería · UdM</div>'
                '<div class="lateral-marca">Proyecto<br>de <em>aula</em></div>'
                '<div class="lateral-meta">2026-2 · Prof. Horacio Fernández C.</div>', unsafe_allow_html=True)
    st.markdown("---")
    eleccion = st.radio("Navegación", list(PAGINAS), key="nav", label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="lateral-meta">Parámetros globales</div>', unsafe_allow_html=True)
    alpha = st.select_slider("Nivel de significancia α", [0.01, 0.05, 0.10], value=0.05, key="alpha")
    confianza = st.select_slider("Confianza de los intervalos", [0.90, 0.95, 0.99], value=0.95, key="conf")
    semilla = st.number_input("Semilla del punto 2", 0, 999_999, 2026, step=1, key="semilla")
    rep_mc = st.select_slider("Repeticiones de Monte Carlo", [1000, 2000, 4000, 10000], value=4000, key="rep")
    st.markdown("---")
    st.markdown('<div class="lateral-equipo">Juan Manuel Moreno Muñoz<br>David Restrepo<br>Juan Fernando</div>',
                unsafe_allow_html=True)

pagina = PAGINAS[eleccion]
if pagina == "portada":
    p_portada.pagina()
elif pagina == "hornos":
    p_hornos.pagina(alpha, confianza)
elif pagina == "equipos":
    p_equipos.pagina(alpha, int(semilla), rep_mc)
elif pagina == "sistemas":
    p_sistemas.pagina(alpha)
elif pagina == "industrial":
    p_industrial.pagina(alpha)
elif pagina == "laboratorio":
    p_laboratorio.pagina(alpha, confianza)
elif pagina == "teoria":
    p_teoria.pagina(alpha)
else:
    p_taller.pagina()
