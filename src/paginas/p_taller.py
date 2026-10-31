"""Taller — la guía de pruebas de bondad de ajuste resuelta con el motor de la app."""

from __future__ import annotations

import streamlit as st

from src import graficos as g
from src import taller as t
from src import ui


def _mostrar(r: dict, clave: str):
    c1, c2 = st.columns([1.15, 1])
    with c1:
        ui.tabla(r["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 4, "(Oᵢ − Eᵢ)²/Eᵢ": 5}, indice=False)
    with c2:
        st.plotly_chart(g.barras_chi(r["tabla"], alto=280), use_container_width=True, key=clave)
    st.latex(rf"EP={r['EP']:.5f}\qquad v=k-p-1={r['k']}-{r['p_est']}-1={r['v']}\qquad "
             rf"\chi^2_{{({r['alpha']},\,{r['v']})}}={r['critico']:.4f}\qquad \text{{Valor }}P={r['p']:.4f}")


def pagina():
    ui.cabecera("07", "Guía del profesor", "Taller de <em>bondad de ajuste</em> resuelto",
                "El ejemplo de ASTON y los seis ejercicios de la guía «Pruebas de bondad de ajuste», resueltos con "
                "el mismo código que usa el resto de la app. Replicar el ejemplo de la guía al decimal es la "
                "<b>validación</b> de que el motor está bien programado.")
    ui.paso("Ej.", t.EJEMPLO_ASTON["titulo"])
    r = t.resolver_aston()
    st.markdown(t.EJEMPLO_ASTON["enunciado"] + f" λ̂ = 1341/440 = {r['lam_exacto']:.4f} ≈ 3.048.")
    _mostrar(r, "aston")
    ui.kpis([("6.38393", "EP en la guía"), (f"{r['EP']:.5f}", "EP calculado aquí"),
             ("0.4957", "valor P en la guía"), (f"{r['p']:.4f}", "valor P aquí")], acento=1)
    ui.lectura("El resultado coincide exactamente con la guía (EP = 6.38393, v = 7, χ²₀.₀₅,₇ = 14.067, P = 0.4957). "
               "Como 6.38393 < 14.067 y P > 0.05, **no se rechaza H₀**: el número de reclamaciones es consistente "
               "con una Poisson. *Nota:* en la última línea de la guía dice que H₀ «debe ser rechazada»; por el valor "
               "P, la conclusión correcta es que **no** se rechaza, como dice el párrafo anterior.", "teoria")
    for i in range(1, 7):
        r = t.resolver(i)
        with st.expander(f"{r['titulo']}  —  {'se rechaza' if r['rechaza'] else 'no se rechaza'} H₀ "
                         f"(α = {r['alpha']})", expanded=i == 1):
            st.markdown(f"**H₀:** {r['h0']}  ·  **Modelo:** {r['dist']}  ·  {r['params_txt']}")
            _mostrar(r, f"ej{i}")
            if r["k"] < r["k_inicial"]:
                st.caption(f"Se agruparon clases con Eᵢ < 5: {r['k_inicial']} → {r['k']} clases.")
            ui.veredicto(r["rechaza"], f"EP {'>' if r['rechaza'] else '<'} χ² crítico")
            st.markdown(_interpretacion(i, r))
    ui.pie()


def _interpretacion(i: int, r: dict) -> str:
    p = r["p"]
    textos = {
        1: f"Con 5 % de significancia hay evidencia (apenas, P = {p:.4f}) de que la duración **no** sigue una "
           f"Normal(2.5, 0.4): la última clase (x > 2.95) tiene 22 baterías cuando se esperaban 13. Las baterías "
           f"duran más de lo que dice el modelo propuesto. Con α = 0.01 no se rechazaría.",
        2: f"P = {p:.4f} > 0.05: **no se rechaza** que la rentabilidad diaria sea normal. Con 83 días, las "
           f"frecuencias observadas son muy parecidas a las esperadas en todas las clases.",
        3: f"P = {p:.4f} > 0.01: **no se rechaza** que las averías semanales sigan una Poisson con λ = 2.1. Se puede "
           f"usar ese modelo para planear el mantenimiento.",
        4: f"P = {p:.4f} < 0.05: **se rechaza** la Poisson. Hay demasiados minutos con 2 y 3 llegadas y muy pocos "
           f"con 4 o más: las llegadas son más regulares que un proceso de Poisson (varianza menor que la media).",
        5: f"P = {p:.4f} > 0.05: **no se rechaza** que los defectos sigan una Poisson(0.5). Hay algo más de unidades "
           f"con 2 o más defectos de lo esperado, pero no lo suficiente para descartar el modelo.",
        6: f"P = {p:.4f}: **se rechaza** la binomial. Hay demasiados días con 0 o 1 acciones vendidas (17 frente a "
           f"8.4 esperados): las tres acciones de un mismo día no se comportan de forma independiente, porque "
           f"todas dependen de cómo se movió el mercado ese día.",
    }
    return textos[i]
