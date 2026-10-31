"""
generar_informe.py — Calcula TODOS los números del informe y exporta sus figuras.

Escribe:
  informe/valores.tex   → un comando LaTeX por cada número que aparece en el texto
  informe/tablas/*.tex  → tablas en formato booktabs
  figuras/*.pdf         → figuras vectoriales (las mismas funciones de la app)

Así ningún número del informe se copia a mano: si cambian los datos, se vuelve a correr.
Ejecutar desde la raíz:   python scripts/generar_informe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import aplicaciones as ap  # noqa: E402
from src import bondad as b  # noqa: E402
from src import graficos as g  # noqa: E402
from src import hornos as h  # noqa: E402
from src import inferencia as inf  # noqa: E402
from src import montecarlo as mc  # noqa: E402
from src import taller as tl  # noqa: E402
from src.descriptiva import tabla_resumen  # noqa: E402
from src.generacion import generar_simulacion  # noqa: E402

FIG = RAIZ / "figuras"
INF = RAIZ / "informe"
TAB = INF / "tablas"
for d in (FIG, INF, TAB):
    d.mkdir(exist_ok=True, parents=True)

ALPHA = 0.05
V: dict[str, str] = {}


# -----------------------------------------------------------------------------
# utilidades
# -----------------------------------------------------------------------------
def val(nombre: str, x, d: int = 4):
    """Registra un número como comando LaTeX (\\nombre). Solo letras en el nombre."""
    assert nombre.isalpha(), nombre
    if isinstance(x, str):
        V[nombre] = x
    elif isinstance(x, (int, np.integer)):
        V[nombre] = f"{int(x):,}".replace(",", "\\,")
    else:
        V[nombre] = f"{x:,.{d}f}".replace(",", "\\,")


def pv(p: float) -> str:
    return r"<\,0.0001" if p < 1e-4 else f"{p:.4f}"


def fig(f, nombre, ancho=900, alto=420):
    f = g.con_fondo(f)
    r_actual = f.layout.margin.r if f.layout.margin and f.layout.margin.r else 20
    t_actual = 60 if (f.layout.legend and f.layout.showlegend is not False) else 30
    f.update_layout(font=dict(family="Manrope Proj", size=13), title_font=dict(family="Fraunces Proj"),
                    margin=dict(l=78, r=max(25, r_actual), t=max(t_actual, 45), b=62))
    f.update_xaxes(automargin=True)
    f.update_yaxes(automargin=True)
    f.write_image(FIG / f"{nombre}.pdf", width=ancho, height=alto)
    f.write_image(FIG / f"{nombre}.png", width=ancho, height=alto, scale=2)


def tabla_tex(nombre: str, df: pd.DataFrame, formatos: dict, alineacion: str | None = None,
              indice: str | None = None):
    """Escribe una tabla booktabs. formatos = {columna: '{:.3f}'} o función."""
    cols = list(df.columns)
    alin = alineacion or ("l" + "r" * len(cols) if indice is not None else "l" + "r" * (len(cols) - 1))
    lineas = [r"\begin{tabular}{" + alin + "}", r"\toprule"]
    cab = ([indice] if indice is not None else []) + [str(c) for c in cols]
    lineas.append(" & ".join(r"\textbf{" + _esc(c) + "}" for c in cab) + r" \\ \midrule")
    for idx, fila in df.iterrows():
        celdas = [_esc(str(idx))] if indice is not None else []
        for c in cols:
            v = fila[c]
            f = formatos.get(c)
            if f is None:
                celdas.append(_esc(str(v)))
            elif callable(f):
                celdas.append(f(v))
            else:
                celdas.append("—" if (isinstance(v, float) and np.isnan(v)) else f.format(v))
        lineas.append(" & ".join(celdas) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{nombre}.tex").write_text("\n".join(lineas), encoding="utf-8")


def _esc(t: str) -> str:
    t = t.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    for a, b_ in [("(Oᵢ − Eᵢ)²/Eᵢ", r"$(O_i-E_i)^2/E_i$"), ("Eᵢ = nPᵢ", r"$E_i=nP_i$"), ("Oᵢ", r"$O_i$"),
                  ("Pᵢ", r"$P_i$"), ("|x̄ − 2.70|", r"$|\bar x-2.70|$"), ("x̄", r"$\bar x$"), ("χ²", r"$\chi^2$"),
                  ("A²", r"$A^2$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("²", r"$^2$"), ("→", r"$\to$"), ("−", "-")]:
        t = t.replace(a, b_)
    return t


def dec(d):
    return lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}".replace(",", r"\,")


# =============================================================================
# PUNTO 1
# =============================================================================
def punto1():
    df = h.cargar()
    val("nHornos", len(df))
    for i, (m, n) in enumerate(df.marca.value_counts().sort_index().items(), 1):
        val("nMarca" + "ABCDE"[i - 1], n)
    val("usoProm", df.tiempo_uso_anios.mean(), 2)
    val("pctCumpleTotal", 100 * (df.ppm <= 400).mean(), 1)

    rp = h.resumen_ppm(df)
    tabla_tex("ppm", rp[["n", "Media", "Mediana", "Q1", "Q3", "RIC", "Atípicos", "% ≤ 400 ppm", "% > 1000 ppm"]],
              {"n": dec(0), "Media": dec(1), "Mediana": dec(1), "Q1": dec(1), "Q3": dec(1), "RIC": dec(1),
               "Atípicos": dec(0), "% ≤ 400 ppm": dec(1), "% > 1000 ppm": dec(2)}, indice="Marca")
    for m, k in zip(rp.index, "ABCDE"):
        val("medPpm" + k, rp.loc[m, "Mediana"], 1)
        val("cumple" + k, rp.loc[m, "% ≤ 400 ppm"], 1)
        val("sobreNorma" + k, rp.loc[m, "% > 1000 ppm"], 2)
    c = h.comparar_ppm(df)
    val("kwH", c["H"], 1)
    val("kwP", pv(c["p_kw"]))
    val("mwPpmZ", c["mw"]["z"], 3)
    val("mwPpmP", pv(c["mw"]["p"]))
    fig(g.cajas_ppm(df, alto=430), "f_cajas_ppm", 900, 430)

    rw = h.resumen_potencia(df)
    tabla_tex("potencia", rw[["n", "Media", "Desv. (s)", "|x̄ − 2.70|", "% dentro de espec.", "Cp", "Cpk"]],
              {"n": dec(0), "Media": dec(4), "Desv. (s)": dec(4), "|x̄ − 2.70|": dec(4),
               "% dentro de espec.": dec(2), "Cp": dec(3), "Cpk": dec(3)}, indice="Marca")
    for m, k in zip(rw.index, "ABCDE"):
        val("cpk" + k, rw.loc[m, "Cpk"], 3)
        val("potMedia" + k, rw.loc[m, "Media"], 4)
        val("potS" + k, rw.loc[m, "Desv. (s)"], 4)
        val("potDentro" + k, rw.loc[m, "% dentro de espec."], 2)
    fig(g.potencia_marcas(df, alto=400), "f_potencia", 900, 400)

    t = h.tabla_desempeno(df)
    tabla_tex("desempeno", t[["Mediana ppm", "% ≤ 400 ppm", "Cpk potencia", "% potencia en espec.",
                              "Reparaciones / año de uso", "Suma de puestos", "Puesto final"]]
              .rename(columns={"Reparaciones / año de uso": "Rep./año", "% potencia en espec.": "% pot. en espec.",
                               "Suma de puestos": "Suma", "Puesto final": "Puesto"}),
              {"Mediana ppm": dec(1), "% ≤ 400 ppm": dec(1), "Cpk potencia": dec(3), "% pot. en espec.": dec(2),
               "Rep./año": dec(4), "Suma": dec(0), "Puesto": dec(0)}, indice="Marca")
    mejor = t.index[0]
    val("mejor", mejor)
    val("segunda", t.index[1])
    val("sumaMejor", t.iloc[0]["Suma de puestos"], 0)
    val("sumaSegunda", t.iloc[1]["Suma de puestos"], 0)
    val("repAnioMejor", t.iloc[0]["Reparaciones / año de uso"], 3)
    val("repAnioPeor", t.iloc[-1]["Reparaciones / año de uso"], 3)
    fig(g.mapa_puestos(t, alto=300), "f_puestos", 900, 300)

    iv = h.intervalos(df, mejor, 0.95)
    mu, va, pr = iv["media"], iv["varianza"], iv["proporcion"]
    val("nMejor", mu["n"])
    val("muMedia", mu["media"], 4)
    val("muS", mu["s"], 4)
    val("muT", mu["t"], 4)
    val("muE", mu["error"], 4)
    val("muLi", mu["li"], 4)
    val("muLs", mu["ls"], 4)
    val("vaSdos", va["s2"], 6)
    val("vaChiDer", va["chi_der"], 2)
    val("vaChiIzq", va["chi_izq"], 2)
    val("vaLi", va["li"], 6)
    val("vaLs", va["ls"], 6)
    val("sLi", np.sqrt(va["li"]), 4)
    val("sLs", np.sqrt(va["ls"]), 4)
    val("prX", pr["x"])
    val("prP", pr["p"], 4)
    val("prE", pr["error"], 4)
    val("prLi", pr["li"], 4)
    val("prLs", pr["ls"], 4)
    val("prWLi", pr["wilson_li"], 4)
    val("prWLs", pr["wilson_ls"], 4)
    gb = b.pruebas_bondad(df.loc[df.marca == mejor, "potencia_kw"], "Normal")
    val("normMejorChi", gb["chi"]["EP"], 3)
    val("normMejorV", gb["chi"]["v"])
    val("normMejorChiP", gb["chi"]["p"], 3)
    val("normMejorKsD", gb["ks"]["D"], 4)
    val("normMejorKsP", gb["ks"]["p"], 3)
    val("normMejorAd", gb["ad"]["A2"], 3)
    val("normMejorAdP", gb["ad"]["p"], 3)

    v = h.variabilidad(df)
    F = v["F"]
    val("fF", F["F"], 4)
    val("fNum", F["s2_num"], 6)
    val("fDen", F["s2_den"], 6)
    val("fGlUno", F["gl1"])
    val("fGlDos", F["gl2"])
    val("fCrit", F["critico"], 4)
    val("fP", pv(F["p"]))
    val("fIcLi", F["ic_razon"][0], 3)
    val("fIcLs", F["ic_razon"][1], 3)
    val("sTres", v["s1"], 4)
    val("sCuatro", v["s2"], 4)
    val("razonS", v["s2"] / v["s1"], 2)
    val("levW", v["levene"]["W"], 2)
    val("levP", pv(v["levene"]["p"]))
    for m, k in (("Marca 3", "Tres"), ("Marca 4", "Cuatro")):
        r = b.pruebas_bondad(df.loc[df.marca == m, "potencia_kw"], "Normal")
        val("normAd" + k, r["ad"]["p"], 3)
        val("normKs" + k, r["ks"]["p"], 3)
    fig(g.curva_prueba(stats.f(F["gl1"], F["gl2"]), F["F"], F["critico"], "der", "F", alto=260), "f_curva_f",
        700, 260)

    af = h.afirmacion(df, mejor)
    val("afZ", af["dos"]["z"], 4)
    val("afP", af["dos"]["p"], 4)
    val("afPmen", af["menor"]["p"], 4)
    val("afCrit", af["dos"]["critico"], 4)
    fig(g.curva_prueba(stats.norm(), af["dos"]["z"], af["dos"]["critico"], "dos", "Z", alto=250), "f_curva_prop",
        700, 250)

    m = h.matriz_correlacion(df)
    val("rUsoPpm", m.iloc[0, 2], 3)
    val("rRepPpm", m.iloc[1, 2], 3)
    val("rUsoRep", m.iloc[0, 1], 3)
    val("rUsoPot", m.iloc[0, 3], 3)
    reg = h.regresion_ppm_uso(df)
    val("regBcero", reg["b0"], 4)
    val("regBuno", reg["b1"], 4)
    val("regPct", reg["pct_por_anio"], 2)
    val("regRdos", reg["r2"], 3)
    fig(g.correlaciones(m, alto=330), "f_corr", 560, 330)
    fig(g.dispersion_uso_ppm(df, reg, alto=330), "f_disp_uso", 640, 330)
    poi = b.chi_cuadrado_datos_discretos(df.loc[df.marca == mejor, "reparaciones"], "Poisson")
    val("poiLam", poi["params"][0], 4)
    val("poiEP", poi["EP"], 4)
    val("poiV", poi["v"])
    val("poiP", poi["p"], 4)


# =============================================================================
# PUNTO 2
# =============================================================================
GR = {"E1_equipo_A": "E1 · A", "E1_equipo_B": "E1 · B", "E2_equipo_A": "E2 · A", "E2_equipo_B": "E2 · B"}


def punto2():
    s = generar_simulacion(2026)
    s.to_csv(RAIZ / "data" / "simulacion.csv", index=False)
    teo = {"E1_equipo_A": stats.norm(175, 30), "E1_equipo_B": stats.norm(170, 50),
           "E2_equipo_A": stats.gamma(1.2, scale=150), "E2_equipo_B": stats.gamma(0.95, scale=190)}
    titulos = ["Escenario 1 · Equipo A  N(175, 30)", "Escenario 1 · Equipo B  N(170, 50)",
               "Escenario 2 · Equipo A  Γ(1.2, 150)", "Escenario 2 · Equipo B  Γ(0.95, 190)"]
    f = make_subplots(rows=2, cols=2, subplot_titles=titulos, horizontal_spacing=.08, vertical_spacing=.17)
    for i, col in enumerate(GR):
        r, c = i // 2 + 1, i % 2 + 1
        x = s[col]
        color = g.BERMELLON if col.endswith("A") else g.PIZARRA
        f.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=6, showlegend=False,
                                 marker=dict(color=g.alfa(color, .35), line=dict(color=color, width=1.2))), r, c)
        xs = np.linspace(max(0.5, x.min() * .9) if "E2" in col else x.min() - 10, x.max() * 1.05, 300)
        f.add_trace(go.Scatter(x=xs, y=teo[col].pdf(xs), mode="lines", showlegend=i == 0, name="Densidad teórica",
                               line=dict(color=g.TINTA, width=2, dash="dot")), r, c)
    for a in f.layout.annotations:
        a.font = dict(family="Fraunces Proj", size=14)
    f.update_layout(legend=dict(y=1.1))
    fig(f, "f_hist4", 1000, 560)

    tr = tabla_resumen({GR[k]: s[k] for k in GR})
    tr = tr.loc[["Media", "Mediana", "Desv. estándar (s)", "Coef. de variación (%)", "Mínimo", "Q1", "Q3",
                 "Máximo", "Asimetría", "Curtosis (exceso)"]]
    tabla_tex("descr2", tr, {c: dec(2) for c in tr.columns}, indice="Medida")
    for k, col in zip("ABCD", GR):
        val("media" + k, s[col].mean(), 2)
        val("desv" + k, s[col].std(ddof=1), 2)
        val("asim" + k, stats.skew(s[col], bias=False), 2)

    filas = []
    res_all = {}
    for col in GR:
        for dist in ("Normal", "Gamma"):
            r = b.pruebas_bondad(s[col], dist)
            res_all[(col, dist)] = r
            filas.append({"Grupo": GR[col], "Modelo": dist, "χ² guía": r["chi"]["p"], "v guía": r["chi"]["v"],
                          "χ² equiprob.": r["chi_eq"]["p"], "KS": r["ks"]["p"], "AD": r["ad"]["p"]})
    d = pd.DataFrame(filas)

    def pcol(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r"\textit{v<1}"
        txt = pv(v) if v >= 1e-4 else r"$<$0.0001"
        return r"\textbf{\color{bermellon}" + txt + "}" if v < ALPHA else txt
    tabla_tex("gof4", d, {"Grupo": None, "Modelo": None, "χ² guía": pcol, "v guía": dec(0), "χ² equiprob.": pcol,
                          "KS": pcol, "AD": pcol}, alineacion="llrrrrr")
    # detalle χ² del grupo E1-A normal (tabla del cuerpo)
    r = res_all[("E1_equipo_A", "Normal")]
    tabla_tex("chi_e1a", r["chi"]["tabla"], {"Clase": None, "Oᵢ": dec(0), "Pᵢ": dec(5), "Eᵢ = nPᵢ": dec(3),
                                              "(Oᵢ − Eᵢ)²/Eᵢ": dec(4)}, alineacion="lrrrr")
    val("eaMu", r["params"][0], 3)
    val("eaS", r["params"][1], 3)
    val("eaEP", r["chi"]["EP"], 4)
    val("eaV", r["chi"]["v"])
    val("eaK", r["chi"]["k"])
    val("eaCrit", r["chi"]["critico"], 4)
    val("eaP", r["chi"]["p"], 4)
    val("eaD", r["ks"]["D"], 4)
    val("eaKsP", r["ks"]["p"], 3)
    val("eaKsPc", r["ks"]["p_clasico"], 3)
    val("eaAd", r["ad"]["A2"], 4)
    val("eaAdMod", r["ad"]["A_mod"], 4)
    val("eaAdP", r["ad"]["p"], 3)
    r = res_all[("E2_equipo_B", "Normal")]
    val("ebD", r["ks"]["D"], 4)
    val("ebKsP", pv(r["ks"]["p"]))
    val("ebAd", r["ad"]["A2"], 3)
    val("ebAdP", pv(r["ad"]["p"]))
    fig(g.ks_grafico(s["E2_equipo_B"], r["congelada"], r["ks"], alto=330), "f_ks_e2b", 620, 330)
    fig(g.qq(s["E2_equipo_B"], r["congelada"], alto=330), "f_qq_e2b", 480, 330)
    rg = res_all[("E2_equipo_B", "Gamma")]
    val("ebGalfa", rg["params"][0], 3)
    val("ebGbeta", rg["params"][1], 2)
    val("ebGAdP", rg["ad"]["p"], 3)
    ra = res_all[("E2_equipo_A", "Gamma")]
    val("eaGalfa", ra["params"][0], 3)
    val("eaGbeta", ra["params"][1], 2)

    # b) medias
    filas = []
    for e in ("E1", "E2"):
        a, bb = s[f"{e}_equipo_A"], s[f"{e}_equipo_B"]
        f_ = inf.prueba_f(a, bb)
        t = inf.prueba_t_dos(a, bb)
        z = inf.prueba_z_dos(a, bb)
        m = inf.mann_whitney(a, bb)
        k = "Uno" if e == "E1" else "Dos"
        val("fe" + k, f_["F"], 3)
        val("fep" + k, pv(f_["p"]))
        val("tt" + k, t["t"], 4)
        val("tgl" + k, t["gl"], 2)
        val("tp" + k, t["p"], 4)
        val("tcrit" + k, t["critico"], 4)
        val("ttipo" + k, t["tipo"])
        val("ee" + k, t["ee"], 4)
        val("zz" + k, z["z"], 4)
        val("zp" + k, z["p"], 4)
        val("dif" + k, t["dif"], 3)
        val("mwR" + k, m["R1"], 1)
        val("mwU" + k, m["U1"], 1)
        val("mwMu" + k, m["mu_U"], 0)
        val("mwSig" + k, m["sigma_U"], 3)
        val("mwZ" + k, m["z"], 4)
        val("mwP" + k, m["p"], 4)
        val("mwPs" + k, m["p_scipy"], 4)
        val("mwSup" + k, m["prob_superioridad"], 3)
        val("medA" + k, m["med1"], 2)
        val("medB" + k, m["med2"], 2)
        filas.append({"Escenario": "1 (normales)" if e == "E1" else "2 (gamma)", "x̄_A − x̄_B": t["dif"],
                      "F (p)": f_["p"], "t": t["t"], "gl": t["gl"], "p (t)": t["p"], "Z": z["z"], "p (Z)": z["p"],
                      "U_A": m["U1"], "Z (MW)": m["z"], "p (MW)": m["p"]})
        fig(g.rangos_mw(a.to_numpy(), bb.to_numpy()), f"f_rangos_{e}", 900, 220)
    d = pd.DataFrame(filas).set_index("Escenario")
    d.columns = [r"$\bar x_A-\bar x_B$", r"$p$ (F)", r"$t$", "gl", r"$p$ ($t$)", r"$Z$", r"$p$ ($Z$)", r"$U_A$",
                 r"$Z_{MW}$", r"$p$ (MW)"]
    tabla_tex("medias", d, {c: (dec(2) if c in ("gl",) else dec(0) if "U" in c else dec(4)) for c in d.columns},
              indice="Escenario")
    (TAB / "medias.tex").write_text((TAB / "medias.tex").read_text().replace(r"\$", "$").replace(r"\_", "_"))

    res = mc.resumen_escenarios(4000, 2026)
    fig(g.barras_potencia(res, alto=330), "f_potencia_mc", 900, 330)
    e1, e2, c0 = res.values()
    for k, r in (("Uno", e1), ("Dos", e2), ("Ctrl", c0)):
        val("mcT" + k, 100 * r["t de Welch"], 1)
        val("mcZ" + k, 100 * r["Z (normal)"], 1)
        val("mcMw" + k, 100 * r["Mann–Whitney"], 1)
        val("mcSup" + k, r["P(A > B)"], 3)


# =============================================================================
# PUNTO 3
# =============================================================================
def punto3():
    c, d = ap.cargar_sistemas()
    val("nCommits", len(c))
    val("nAutores", c.autor.nunique())
    L = ap.llegadas_diarias(d)
    val("nDias", L["n"])
    val("llMedia", L["media"], 4)
    val("llVar", L["var"], 4)
    val("llDisp", L["indice_dispersion"], 3)
    val("poiCEP", L["poisson"]["EP"], 2)
    val("poiCV", L["poisson"]["v"])
    val("poiCCrit", L["poisson"]["critico"], 3)
    val("poiCP", pv(L["poisson"]["p"]))
    val("poiCeros", int(L["poisson"]["tabla"].iloc[0]["Oᵢ"]))
    val("poiCerosE", L["poisson"]["tabla"].iloc[0]["Eᵢ = nPᵢ"], 1)
    val("bnR", L["binneg"]["params"][0], 3)
    val("bnPp", L["binneg"]["params"][1], 3)
    val("bnEP", L["binneg"]["EP"], 2)
    val("bnV", L["binneg"]["v"])
    val("bnCrit", L["binneg"]["critico"], 3)
    val("bnP", L["binneg"]["p"], 4)
    val("pnPoi", L["p95_poisson"])
    val("pnBn", L["p95_binneg"])
    val("pnReal", L["p95_real"], 0)
    fig(g.conteos_diarios(L["binneg"]["tabla"], L["media"], alto=320), "f_llegadas", 900, 320)
    comp = pd.DataFrame({
        "Modelo": ["Poisson", "Binomial negativa"],
        "Parámetros": [f"λ = {L['poisson']['params'][0]:.4f}",
                       f"r = {L['binneg']['params'][0]:.3f}, p = {L['binneg']['params'][1]:.3f}"],
        "EP": [L["poisson"]["EP"], L["binneg"]["EP"]], "v": [L["poisson"]["v"], L["binneg"]["v"]],
        "Crítico": [L["poisson"]["critico"], L["binneg"]["critico"]],
        "Valor P": [L["poisson"]["p"], L["binneg"]["p"]],
        "P95": [L["p95_poisson"], L["p95_binneg"]]})
    tabla_tex("llegadas", comp, {"Modelo": None, "Parámetros": lambda t: t.replace("λ", r"$\lambda$"),
                                 "EP": dec(2), "v": dec(0), "Crítico": dec(3), "Valor P": lambda p: pv(p),
                                 "P95": dec(0)}, alineacion="llrrrrr")

    T = ap.tamano_cambios(c)
    lg = T["lognormal"]
    val("tamMed", T["mediana"], 0)
    val("tamMedia", T["media"], 0)
    val("tamMax", T["x"].max(), 0)
    val("tamPnov", T["p90"], 0)
    val("lgMu", lg["params"][0], 3)
    val("lgSig", lg["params"][1], 3)
    val("lgChi", lg["chi"]["EP"], 2)
    val("lgChiV", lg["chi"]["v"])
    val("lgChiP", pv(lg["chi"]["p"]))
    val("lgD", lg["ks"]["D"], 4)
    val("lgKsP", pv(lg["ks"]["p"]))
    val("lgAd", lg["ad"]["A2"], 3)
    val("lgAdP", pv(lg["ad"]["p"]))
    val("lgPeq", int(lg["chi"]["tabla"].iloc[0]["Oᵢ"]))
    val("lgPeqE", lg["chi"]["tabla"].iloc[0]["Eᵢ = nPᵢ"], 1)
    rk = T["ranking"][["Distribución", "AIC", "D (KS)", "A² (AD)"]]
    tabla_tex("ranking", rk.set_index("Distribución"), {"AIC": dec(1), "D (KS)": dec(4), "A² (AD)": dec(2)},
              indice="Distribución")
    fig(g.histograma_ajuste(T["x"], {"Lognormal ajustada": lg["congelada"]}, nbins=40, log_x=True, alto=320),
        "f_lognormal", 620, 320)

    C = ap.correccion_vs_funcionalidad(c)
    val("cfNf", C["n_f"])
    val("cfNc", C["n_c"])
    val("cfMf", C["med_f"], 0)
    val("cfMc", C["med_c"], 0)
    val("cfZ", C["mw"]["z"], 2)
    val("cfP", pv(C["mw"]["p"]))
    val("cfSup", 100 * C["mw"]["prob_superioridad"], 1)
    val("cfRazon", C["med_f"] / C["med_c"], 1)
    fig(g.cajas_tipo(c, alto=330), "f_cajas_tipo", 620, 330)

    A = ap.archivos_vs_lineas(c)
    val("arR", A["cor"]["r"], 3)
    val("arRho", A["cor"]["rho"], 3)
    val("arRlog", A["cor_log"]["r"], 3)
    val("arBcero", A["reg"]["b0"], 4)
    val("arBuno", A["reg"]["b1"], 4)
    val("arRdos", A["reg"]["r2"], 3)
    val("arDoble", 2 ** A["reg"]["b1"], 2)
    fig(g.dispersion_log(c, A["reg"], alto=330), "f_disp_log", 560, 330)

    P = ap.proporcion_correcciones(c)
    pr = P["prueba"]
    val("pcTot", 100 * P["ic"]["p"], 1)
    val("pcLi", 100 * P["ic"]["li"], 1)
    val("pcLs", 100 * P["ic"]["ls"], 1)
    val("pcUno", 100 * pr["p1"], 1)
    val("pcDos", 100 * pr["p2"], 1)
    val("pcZ", pr["z"], 3)
    val("pcP", pr["p"], 4)
    val("pcNuno", P["n1"])
    val("pcNdos", P["n2"])

    a, j = ap.cargar_industrial()
    R = ap.analisis_anillos(a)
    lim = R["cartas"]["lim"]
    val("xbb", lim["LC_x"], 5)
    val("rb", lim["LC_r"], 5)
    val("lscx", lim["LSC_x"], 5)
    val("licx", lim["LIC_x"], 5)
    val("lscr", lim["LSC_r"], 5)
    tab = R["cartas"]["tabla"]
    fig(g.carta_control(tab.xbar.values, lim["LC_x"], lim["LSC_x"], lim["LIC_x"], 25, None,
                        "Media de la muestra (mm)", alto=280), "f_carta_x", 900, 280)
    fig(g.carta_control(tab.R.values, lim["LC_r"], lim["LSC_r"], lim["LIC_r"], 25, None, "Rango (mm)", alto=220),
        "f_carta_r", 900, 220)
    nr = R["normal"]
    val("anChi", nr["chi"]["EP"], 3)
    val("anChiV", nr["chi"]["v"])
    val("anChiP", nr["chi"]["p"], 3)
    val("anKsP", nr["ks"]["p"], 3)
    val("anAdP", nr["ad"]["p"], 3)
    val("anMu", nr["params"][0], 5)
    val("anS", nr["params"][1], 5)
    cap = R["cap"]
    val("anCp", cap["Cp"], 3)
    val("anCpk", cap["Cpk"], 3)
    val("anPpm", cap["ppm_fuera_teorico"], 1)
    val("anIcLi", R["ic_media"]["li"], 4)
    val("anIcLs", R["ic_media"]["ls"], 4)
    val("anTn", R["t_nominal"]["t"], 3)
    val("anTnP", R["t_nominal"]["p"], 3)
    tf, ff = R["t_fases"], R["f_fases"]
    val("faT", tf["t"], 3)
    val("faP", pv(tf["p"]))
    val("faMuno", tf["m1"], 4)
    val("faMdos", tf["m2"], 4)
    val("faMicras", 1000 * abs(tf["dif"]), 1)
    val("faIcLi", 1000 * abs(tf["ic"][1]), 1)
    val("faIcLs", 1000 * abs(tf["ic"][0]), 1)
    val("faF", ff["F"], 3)
    val("faFP", ff["p"], 3)
    fig(g.histograma_especificacion(a.loc[a.fase_I, "diametro_mm"], R["lie"], R["lse"], nr["congelada"], alto=300),
        "f_anillos_hist", 560, 300)

    J = ap.analisis_jugo(j)
    val("juPbar", J["pbar"], 4)
    val("juLsc", J["lsc"], 4)
    val("juLic", J["lic"], 4)
    val("juPdos", J["p_fase2"], 4)
    bi = J["binomial"]
    val("juBp", bi["params"][1], 4)
    val("juEP", bi["EP"], 4)
    val("juV", bi["v"])
    val("juK", bi["k"])
    val("juCrit", bi["critico"], 4)
    val("juP", bi["p"], 4)
    val("juZ", J["prueba"]["z"], 3)
    val("juPz", pv(J["prueba"]["p"]))
    val("juIcLi", 100 * J["prueba"]["ic"][0], 1)
    val("juIcLs", 100 * J["prueba"]["ic"][1], 1)
    tabla_tex("binomial_jugo", bi["tabla"], {"Clase": None, "Oᵢ": dec(0), "Pᵢ": dec(5), "Eᵢ = nPᵢ": dec(3),
                                             "(Oᵢ − Eᵢ)²/Eᵢ": dec(4)}, alineacion="lrrrr")
    jj = j.copy()
    jj["p"] = jj.no_conformes / jj.tamano
    fig(g.carta_control(jj.p.values, J["pbar"], J["lsc"], J["lic"], 30, None, "Fracción no conforme",
                        alto=280, marcar=J["excluidas"]), "f_carta_p", 900, 280)


# =============================================================================
# Anexos: taller y figuras teóricas
# =============================================================================
def anexos():
    r = tl.resolver_aston()
    val("astonEP", r["EP"], 5)
    val("astonP", r["p"], 4)
    filas = [{"Problema": "Ejemplo ASTON", "Modelo": "Poisson(3.048)", "k": r["k"], "p": r["p_est"], "v": r["v"],
              "EP": r["EP"], "Crítico": r["critico"], "Valor P": r["p"], "α": 0.05,
              "Decisión": "Rechaza" if r["rechaza"] else "No rechaza"}]
    for i in range(1, 7):
        r = tl.resolver(i)
        filas.append({"Problema": f"Ejercicio {i}", "Modelo": r["dist"].replace("μ", "$\\mu$")
                      .replace("σ", "$\\sigma$").replace("λ", "$\\lambda$").replace("x̄", "$\\bar x$")
                      .replace("p̂", "$\\hat p$"), "k": r["k"],
                      "p": r["p_est"], "v": r["v"], "EP": r["EP"], "Crítico": r["critico"], "Valor P": r["p"],
                      "α": r["alpha"], "Decisión": "Rechaza" if r["rechaza"] else "No rechaza"})
        tabla_tex(f"taller{i}", r["tabla"], {"Clase": None, "Oᵢ": dec(0), "Pᵢ": dec(5), "Eᵢ = nPᵢ": dec(4),
                                              "(Oᵢ − Eᵢ)²/Eᵢ": dec(4)}, alineacion="lrrrr")
    d = pd.DataFrame(filas)
    tabla_tex("taller", d, {"Problema": None, "Modelo": lambda t: t, "k": dec(0), "p": dec(0), "v": dec(0),
                            "EP": dec(4), "Crítico": dec(3), "Valor P": dec(4), "α": dec(2), "Decisión": None},
              alineacion="llrrrrrrrl")
    L = mc.lilliefors()
    val("liConocido", 100 * L["rech_conocido"], 1)
    val("liClasico", 100 * L["rech_clasico"], 1)
    val("liDtabla", L["d95_clasico"], 4)
    val("liDreal", L["d95_est"], 4)
    P = mc.potencias_bondad()
    P["Datos generados desde"] = ["$t$ de Student (4 gl): colas pesadas", "Gamma($\\alpha=4$): asimétrica",
                                  "Uniforme: sin colas"]
    tabla_tex("potencia_gof", P.set_index("Datos generados desde"),
              {c: (lambda v: f"{100 * v:.1f}\\,\\%") for c in P.columns if c != "Datos generados desde"},
              indice="Datos generados desde")
    # Figura conceptual de KS con una muestra pequeña
    rng = np.random.default_rng(3)
    x = rng.gamma(2, 10, 12)
    dd = stats.norm(x.mean(), x.std(ddof=1))
    fig(g.ks_grafico(x, dd, b.estadistico_ks(x, dd), alto=300), "f_ks_concepto", 560, 300)
    fig(g.peso_ad(alto=300), "f_peso_ad", 560, 300)


def escribir_valores():
    lineas = ["% Archivo generado por scripts/generar_informe.py — NO editar a mano"]
    for k, v in sorted(V.items()):
        lineas.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (INF / "valores.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{len(V)} valores escritos en informe/valores.tex")


if __name__ == "__main__":
    punto1()
    punto2()
    punto3()
    anexos()
    escribir_valores()
