"""Punto 1 — Estudio de mercado de 5 marcas de hornos de cocción a gas."""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import stats

from src import bondad as b
from src import graficos as g
from src import hornos as h
from src import ui
from src.paginas.comun import bloque_bondad


@st.cache_data
def _datos():
    return h.cargar()


def pagina(alpha: float, confianza: float):
    df = _datos()
    ui.cabecera("01", "Punto 1 · Datos inventados acordes al enunciado",
                "Cinco marcas de <em>hornos</em> a gas",
                f"Base de <b>{len(df):,}</b> hornos con su marca, tiempo de uso, número de reparaciones, "
                "emisiones de partículas (ppm) y potencia del quemador (kW). Especificaciones: potencia "
                "<b>2.7 ± 0.22 kW</b> y emisiones de <b>máximo 400 ppm</b> por seguridad (la norma permite 1000).")
    conteo = df.marca.value_counts().sort_index()
    ui.kpis([(f"{len(df):,}", "hornos"), ("5", "marcas"),
             (f"{df.tiempo_uso_anios.mean():.1f}<small> años</small>", "uso promedio"),
             (f"{100 * (df.ppm <= 400).mean():.1f}<small>%</small>", "cumple ≤ 400 ppm"),
             (f"{100 * df.potencia_kw.between(h.LIE, h.LSE).mean():.1f}<small>%</small>", "potencia en espec.")])

    tabs = st.tabs(["a · Emisiones", "b · Potencia", "c · Desempeño", "d · Intervalos",
                    "e · Variabilidad M3 vs M4", "f · Afirmación del fabricante", "Relaciones", "Datos"])

    # ------------------------------------------------------------------ a)
    with tabs[0]:
        st.markdown("### ¿Qué marca emite menos partículas contaminantes?")
        st.plotly_chart(g.cajas_ppm(df), use_container_width=True, key="cajas_ppm")
        r = h.resumen_ppm(df)
        ui.tabla(r, dec={"n": 0, "Media": 1, "Mediana": 1, "Q1": 1, "Q3": 1, "RIC": 1, "Bigote sup.": 1,
                          "Atípicos": 0, "% ≤ 400 ppm": 1, "% > 1000 ppm": 2})
        c = h.comparar_ppm(df)
        mw = c["mw"]
        orden = r.sort_values("Mediana").index.tolist()
        ui.lectura(
            f"Cada caja va del primer al tercer cuartil (50 % central de los hornos); la línea es la mediana y la "
            f"línea punteada interior, la media. **{orden[0]}** tiene la caja más baja: mediana de "
            f"**{r.loc[orden[0], 'Mediana']:.1f} ppm** y el {r.loc[orden[0], '% ≤ 400 ppm']:.1f} % de sus hornos "
            f"cumple el límite de seguridad; además es la única con **0 hornos sobre la norma de 1000 ppm**. "
            f"En el otro extremo, **{orden[-1]}** tiene la mediana más alta ({r.loc[orden[-1], 'Mediana']:.1f} ppm), "
            f"la caja más ancha y el {r.loc[orden[-1], '% > 1000 ppm']:.2f} % de sus hornos supera la norma.\n\n"
            f"Todas las cajas son asimétricas a la derecha (bigote superior largo y atípicos arriba): pocos hornos "
            f"emiten mucho más que el resto, típico de una variable que crece con el desgaste.\n\n"
            f"**¿La diferencia es real o es azar?** Kruskal–Wallis compara las 5 marcas a la vez: "
            f"H = {c['H']:.1f}, p {ui.pval(c['p_kw'])} → las marcas sí difieren. Entre las dos mejores "
            f"({c['mejor']} vs {c['segunda']}), Mann–Whitney da Z = {mw['z']:.3f}, p {ui.pval(mw['p'])}: "
            f"**{c['mejor']} emite significativamente menos**.")

    # ------------------------------------------------------------------ b)
    with tabs[1]:
        st.markdown("### ¿Qué marca cumple mejor la especificación de potencia?")
        st.plotly_chart(g.potencia_marcas(df), use_container_width=True, key="pot")
        rp = h.resumen_potencia(df)
        ui.tabla(rp, dec={"n": 0, "Media": 4, "Desv. (s)": 4, "|x̄ − 2.70|": 4, "% dentro de espec.": 2,
                           "Cp": 3, "Cpk": 3, "ppm fuera (teórico)": 0})
        st.latex(r"C_p=\frac{LSE-LIE}{6s}\qquad C_{pk}=\min\left(\frac{LSE-\bar x}{3s},\ \frac{\bar x-LIE}{3s}\right)"
                 r"\qquad LIE = 2.48,\ LSE = 2.92")
        mejor = rp["Cpk"].idxmax()
        centrada = rp["|x̄ − 2.70|"].idxmin()
        ui.lectura(
            f"No basta con que la media esté cerca de 2.70 kW: también importa **qué tan dispersa** es la "
            f"potencia. El índice **Cpk** combina las dos cosas: mide cuántas desviaciones estándar caben entre "
            f"la media y el límite más cercano.\n\n"
            f"**{mejor}** es la que mejor se ajusta: Cpk = {rp.loc[mejor, 'Cpk']:.3f}, la menor desviación "
            f"(s = {rp.loc[mejor, 'Desv. (s)']:.4f} kW) y el {rp.loc[mejor, '% dentro de espec.']:.1f} % de sus "
            f"hornos dentro de 2.48–2.92 kW. **{centrada}** está mejor centrada (x̄ = {rp.loc[centrada, 'Media']:.4f}) "
            f"pero es más dispersa. Ninguna llega a Cpk ≥ 1.33, el mínimo usual para un proceso \"capaz\", así "
            f"que todas tienen margen de mejora; {rp['Cpk'].idxmin()} es la peor "
            f"(Cpk = {rp['Cpk'].min():.3f}, {rp.loc[rp['Cpk'].idxmin(), '% dentro de espec.']:.1f} % en especificación).")

    # ------------------------------------------------------------------ c)
    with tabs[2]:
        st.markdown("### ¿Qué marca tiene el mejor desempeño global?")
        t = h.tabla_desempeno(df)
        st.plotly_chart(g.mapa_puestos(t), use_container_width=True, key="mapa")
        ui.tabla(t, dec={"Mediana ppm": 1, "% ≤ 400 ppm": 1, "Cpk potencia": 3, "% potencia en espec.": 2,
                          "Reparaciones / año de uso": 4, "Reparaciones promedio": 3, "Suma de puestos": 0})
        m1, m2 = t.index[0], t.index[1]
        ui.lectura(
            f"Cada marca recibe un puesto (1 = mejor) en cinco criterios, todos con el mismo peso; la tasa de "
            f"reparaciones se divide entre los años de uso para no castigar a las marcas con hornos más viejos.\n\n"
            f"**{m1} es la de mejor desempeño** (suma de puestos {t.loc[m1, 'Suma de puestos']:.0f}): es la que "
            f"menos contamina y la que menos se daña ({t.loc[m1, 'Reparaciones / año de uso']:.3f} reparaciones "
            f"por año de uso), y es segunda en potencia. **{m2}** queda muy cerca "
            f"({t.loc[m2, 'Suma de puestos']:.0f}) gracias a su potencia más estable. La decisión depende de qué "
            f"pese más: si la prioridad es la seguridad ambiental, {m1}; si es la estabilidad del quemador, {m2}. "
            f"{t.index[-1]} es la peor en todos los criterios.")

    mejor = h.mejor_marca(df)
    g_m = df[df.marca == mejor]
    # ------------------------------------------------------------------ d)
    with tabs[3]:
        st.markdown(f"### Intervalos de confianza del {100 * confianza:.0f} % para **{mejor}**")
        iv = h.intervalos(df, mejor, confianza)
        a = 1 - confianza
        mu, var, pr = iv["media"], iv["varianza"], iv["proporcion"]
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.paso("d.1", "Media de la potencia")
            st.latex(rf"\bar x \pm t_{{\alpha/2,\,n-1}}\frac{{s}}{{\sqrt n}}")
            st.latex(rf"{mu['media']:.4f}\pm {mu['t']:.4f}\cdot\frac{{{mu['s']:.4f}}}{{\sqrt{{{mu['n']}}}}}")
            st.latex(rf"{mu['li']:.4f}\ \le\ \mu\ \le\ {mu['ls']:.4f}")
        with c2:
            ui.paso("d.2", "Varianza de la potencia")
            st.latex(r"\frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\le\sigma^2\le\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}}")
            st.latex(rf"\frac{{{mu['n'] - 1}({var['s2']:.6f})}}{{{var['chi_der']:.2f}}}\le\sigma^2\le"
                     rf"\frac{{{mu['n'] - 1}({var['s2']:.6f})}}{{{var['chi_izq']:.2f}}}")
            st.latex(rf"{var['li']:.6f}\le\sigma^2\le{var['ls']:.6f}")
        with c3:
            ui.paso("d.3", "Proporción que cumple ≤ 400 ppm")
            st.latex(r"\hat p\pm z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}")
            st.latex(rf"{pr['p']:.4f}\pm {pr['z']:.4f}\sqrt{{\frac{{{pr['p']:.4f}({1 - pr['p']:.4f})}}{{{pr['n']}}}}}")
            st.latex(rf"{pr['li']:.4f}\le p\le{pr['ls']:.4f}")
        st.plotly_chart(g.intervalo(mu["media"], mu["li"], mu["ls"], "Media de la potencia μ", 2.70,
                                    "nominal 2.70", unidad="kW"), use_container_width=True, key="ic_mu")
        ui.lectura(
            f"Con {100 * confianza:.0f} % de confianza, la potencia media de los hornos de {mejor} está entre "
            f"**{mu['li']:.4f} y {mu['ls']:.4f} kW**: el valor nominal 2.70 queda **dentro**, así que el proceso "
            f"está centrado. La varianza está entre {var['li']:.6f} y {var['ls']:.6f} kW², es decir, la "
            f"desviación estándar entre **{np.sqrt(var['li']):.4f} y {np.sqrt(var['ls']):.4f} kW**. "
            f"Entre el **{100 * pr['li']:.2f} % y el {100 * pr['ls']:.2f} %** de los hornos cumple el límite de "
            f"400 ppm (Wilson: {100 * pr['wilson_li']:.2f} %–{100 * pr['wilson_ls']:.2f} %).")
        with st.expander("Antes de confiar en el intervalo de la varianza: ¿la potencia es normal?"):
            st.markdown("El intervalo con χ² **exige normalidad** (a diferencia del de la media, que se apoya en "
                        "el TLC). Se verifica con las tres pruebas:")
            bloque_bondad(g_m["potencia_kw"], "Normal", alpha, "la potencia", clave="pot_mejor",
                          color=g.MARCAS.get(mejor, g.PIZARRA))

    # ------------------------------------------------------------------ e)
    with tabs[4]:
        st.markdown("### ¿Hay diferencia en la variabilidad de la potencia entre las marcas 3 y 4?")
        v = h.variabilidad(df, alpha=alpha)
        F = v["F"]
        st.latex(r"H_0:\ \sigma_3^2=\sigma_4^2\qquad H_a:\ \sigma_3^2\ne\sigma_4^2\qquad "
                 r"F=\frac{s_{\text{mayor}}^2}{s_{\text{menor}}^2}\sim F_{(n_1-1,\,n_2-1)}")
        num_m, den_m = (v["m2"], v["m1"]) if F["invertido"] else (v["m1"], v["m2"])
        st.latex(rf"F=\frac{{s^2_{{\text{{{num_m}}}}}}}{{s^2_{{\text{{{den_m}}}}}}}=\frac{{{F['s2_num']:.6f}}}"
                 rf"{{{F['s2_den']:.6f}}}={F['F']:.4f}\qquad F_{{({alpha / 2},\,{F['gl1']},\,{F['gl2']})}}"
                 rf"={F['critico']:.4f}")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.plotly_chart(g.curva_prueba(stats.f(F["gl1"], F["gl2"]), F["F"], F["critico"], "der", "F"),
                            use_container_width=True, key="curva_f")
        with c2:
            ui.kpis([(f"{v['s1']:.4f}", f"s {v['m1']}"), (f"{v['s2']:.4f}", f"s {v['m2']}")])
            ui.veredicto(F["rechaza"], f"p {ui.pval(F['p'])}")
            st.markdown(f"Levene (no exige normalidad): W = {v['levene']['W']:.2f}, p {ui.pval(v['levene']['p'])}")
        ui.lectura(
            f"La desviación estándar de {v['m2']} ({v['s2']:.4f} kW) es {v['s2'] / v['s1']:.2f} veces la de "
            f"{v['m1']} ({v['s1']:.4f} kW). El estadístico F = {F['F']:.3f} supera ampliamente el valor crítico "
            f"{F['critico']:.3f} (p {ui.pval(F['p'])}), y el intervalo del {100 * (1 - alpha):.0f} % para la razón "
            f"de varianzas ({F['ic_razon'][0]:.3f} a {F['ic_razon'][1]:.3f}) no contiene el 1. **Sí existe "
            f"diferencia: la tecnología de {v['m1']} produce una potencia más uniforme.** La prueba F supone "
            f"normalidad; se verificó en d) y Levene (robusta) llega a la misma conclusión.")
        with st.expander("Normalidad de la potencia de cada marca (supuesto de la prueba F)"):
            for m in (v["m1"], v["m2"]):
                st.markdown(f"**{m}**")
                bloque_bondad(df.loc[df.marca == m, "potencia_kw"], "Normal", alpha, f"la potencia de {m}",
                              clave=f"norm_{m}")

    # ------------------------------------------------------------------ f)
    with tabs[5]:
        st.markdown(f"### \"De cada 100 hornos, 80 cumplen\" — ¿es cierto para {mejor}?")
        af = h.afirmacion(df, mejor, 0.80, alpha)
        dos, men = af["dos"], af["menor"]
        st.latex(r"Z=\frac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}}"
                 rf"=\frac{{{dos['p_hat']:.4f}-0.80}}{{\sqrt{{0.80(0.20)/{dos['n']}}}}}={dos['z']:.4f}")
        c1, c2 = st.columns(2)
        with c1:
            ui.paso("f.1", "Bilateral: H₀: p = 0.80  vs  Hₐ: p ≠ 0.80")
            st.plotly_chart(g.curva_prueba(stats.norm(), dos["z"], dos["critico"], "dos", "Z", alto=260),
                            use_container_width=True, key="z_dos")
            ui.veredicto(dos["rechaza"], f"p = {ui.pval(dos['p'])}")
        with c2:
            ui.paso("f.2", "Unilateral: H₀: p ≥ 0.80  vs  Hₐ: p < 0.80")
            st.plotly_chart(g.curva_prueba(stats.norm(), men["z"], men["critico"], "izq", "Z", alto=260),
                            use_container_width=True, key="z_men")
            ui.veredicto(men["rechaza"], f"p = {ui.pval(men['p'])}")
        ui.lectura(
            f"En la muestra cumplen {dos['x']} de {dos['n']} hornos: p̂ = **{100 * dos['p_hat']:.2f} %**. "
            f"La diferencia con el 80 % equivale a Z = {dos['z']:.3f} errores estándar, muy por debajo del "
            f"crítico {dos['critico']:.2f} (p = {dos['p']:.4f}). Tampoco hay evidencia de que la proporción "
            f"sea **menor** al 80 % (p = {men['p']:.4f}).\n\n" + (
                "**Sí, estamos de acuerdo con la afirmación del fabricante:** los datos son consistentes con que 80 "
                "de cada 100 hornos cumplen, y el intervalo de d) también contiene 0.80." if not dos["rechaza"]
                else "**No estamos de acuerdo:** la proporción observada difiere significativamente del 80 %."))

    # ------------------------------------------------------------------ Relaciones
    with tabs[6]:
        st.markdown("### ¿Cómo se relacionan las variables?")
        c1, c2 = st.columns([1, 1.25])
        with c1:
            met = st.segmented_control("Coeficiente", ["Pearson", "Spearman"], default="Pearson", key="metcor")
            st.plotly_chart(g.correlaciones(h.matriz_correlacion(df, (met or "Pearson").lower())),
                            use_container_width=True, key="corr")
        with c2:
            reg = h.regresion_ppm_uso(df)
            st.plotly_chart(g.dispersion_uso_ppm(df, reg), use_container_width=True, key="disp")
        st.latex(rf"\ln(\widehat{{ppm}})={reg['b0']:.4f}+{reg['b1']:.4f}\cdot\text{{uso}}\quad"
                 rf"\Rightarrow\quad e^{{{reg['b1']:.4f}}}={np.exp(reg['b1']):.4f}\quad(R^2={reg['r2']:.3f})")
        poi = b.chi_cuadrado_datos_discretos(g_m["reparaciones"], "Poisson", alpha)
        ui.lectura(
            f"Las emisiones crecen con el **tiempo de uso** (r = {h.matriz_correlacion(df).iloc[0, 2]:.3f}) y con "
            f"las **reparaciones** (r = {h.matriz_correlacion(df).iloc[1, 2]:.3f}); la potencia baja levemente con "
            f"los años. Son correlaciones débiles a moderadas: la marca explica más que la edad. El modelo "
            f"log-lineal dice que **cada año de uso aumenta las emisiones un {reg['pct_por_anio']:.2f} %** en "
            f"promedio (p {ui.pval(reg['p'])}).\n\nLas reparaciones de {mejor} se ajustan a una **Poisson** "
            f"(λ = {poi['params'][0]:.4f}; χ² = {poi['EP']:.4f}, v = {poi['v']}, p = {poi['p']:.4f}), el "
            f"modelo natural para conteos de eventos raros.")
        with st.expander(f"Tabla χ² de la Poisson para las reparaciones de {mejor}"):
            ui.tabla(poi["tabla"], dec={"Pᵢ": 5, "Eᵢ = nPᵢ": 3, "(Oᵢ − Eᵢ)²/Eᵢ": 4}, indice=False)

    # ------------------------------------------------------------------ Datos
    with tabs[7]:
        st.markdown("Los datos son **inventados** pero coherentes: se generaron con `scripts/generar_datos.py` "
                    "(semilla 2026). Las emisiones dependen de la marca, el uso y las reparaciones; la potencia, "
                    "de la marca y del uso.")
        marca_f = st.multiselect("Filtrar marcas", list(conteo.index), default=list(conteo.index), key="fm")
        st.dataframe(df[df.marca.isin(marca_f)], use_container_width=True, height=360, hide_index=True)
        st.download_button("Descargar hornos.csv", df.to_csv(index=False).encode("utf-8"), "hornos.csv",
                           "text/csv")
    ui.pie()
