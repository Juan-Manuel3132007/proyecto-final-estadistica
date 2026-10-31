"""
graficos.py — Gráficas interactivas (Plotly) con la identidad visual del proyecto.

Las mismas funciones se usan en la app y para exportar las figuras del informe PDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats

# -----------------------------------------------------------------------------
# Identidad visual: papel, tinta y bermellón
# -----------------------------------------------------------------------------
PAPEL = "#F4EFE6"
TARJETA = "#FFFDF8"
TINTA = "#1D1A16"
GRIS = "#6E665B"
LINEA = "#DCD2C1"
BERMELLON = "#D9480F"
OCRE = "#D9982B"
OLIVA = "#5E7D3A"
CIRUELA = "#7A3B5E"
PIZARRA = "#3E5C76"
ARENA = "#C9B99A"

MARCAS = {"Marca 1": PIZARRA, "Marca 2": BERMELLON, "Marca 3": OLIVA,
          "Marca 4": CIRUELA, "Marca 5": OCRE}
SERIE = [BERMELLON, PIZARRA, OLIVA, CIRUELA, OCRE, "#8C6D46", "#2E7F7A"]

FUENTE = "Manrope, 'Manrope Proj', 'Helvetica Neue', Arial, sans-serif"
FUENTE_TITULO = "Fraunces, 'Fraunces Proj', Georgia, serif"
FUENTE_MONO = "'JetBrains Mono', monospace"

pio.templates["papel"] = go.layout.Template(
    layout=dict(
        font=dict(family=FUENTE, size=13, color=TINTA),
        title=dict(font=dict(family=FUENTE_TITULO, size=18, color=TINTA), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIE,
        xaxis=dict(showgrid=False, linecolor=TINTA, linewidth=1.2, ticks="outside",
                   tickcolor=TINTA, ticklen=4, zeroline=False, title=dict(standoff=8)),
        yaxis=dict(gridcolor=LINEA, griddash="dot", linecolor="rgba(0,0,0,0)", zeroline=False,
                   ticks="", title=dict(standoff=8)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=TINTA, font=dict(color=PAPEL, family=FUENTE)),
        bargap=0.06,
    )
)
pio.templates.default = "papel"


def _base(fig, titulo=None, alto=420, **kw):
    fig.update_layout(title=titulo, height=alto, **kw)
    return fig


def con_fondo(fig):
    """Para exportar al informe: fondo de color tarjeta en vez de transparente."""
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return fig


# =============================================================================
# Punto 1 — Hornos
# =============================================================================
def cajas_ppm(df, alto=460):
    fig = go.Figure()
    for m in sorted(df.marca.unique()):
        fig.add_trace(go.Box(y=df.loc[df.marca == m, "ppm"], name=m, marker_color=MARCAS[m],
                             line=dict(width=1.6), boxmean=True, marker=dict(size=3, opacity=.55),
                             fillcolor=_alfa(MARCAS[m], .18)))
    fig.add_hline(y=400, line=dict(color=TINTA, width=1.4, dash="dash"),
                  annotation_text="Límite de seguridad · 400 ppm", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA), annotation_bgcolor="rgba(255,253,248,.9)")
    fig.add_hline(y=1000, line=dict(color=BERMELLON, width=1.4, dash="dot"),
                  annotation_text="Norma · 1000 ppm", annotation_position="top left",
                  annotation_font=dict(size=11, color=BERMELLON))
    tope = 1250
    fuera = int((df["ppm"] > tope).sum())
    fig.update_yaxes(title="Emisiones (ppm)", range=[0, tope])
    if fuera:
        fig.add_annotation(x=1, xref="paper", y=tope, yanchor="top", xanchor="right", showarrow=False,
                           text=f"{fuera} valores > {tope} ppm fuera del recuadro", font=dict(size=10, color=GRIS))
    return _base(fig, None, alto, showlegend=False)


def potencia_marcas(df, lie=2.48, lse=2.92, alto=440):
    fig = go.Figure()
    fig.add_hrect(y0=lie, y1=lse, fillcolor=_alfa(OLIVA, .08), line_width=0)
    for y, t in [(lie, "LIE 2.48"), (lse, "LSE 2.92")]:
        fig.add_hline(y=y, line=dict(color=OLIVA, width=1.4, dash="dash"),
                      annotation_text=t, annotation_position="right",
                      annotation_font=dict(size=11, color=OLIVA))
    fig.add_hline(y=2.70, line=dict(color=GRIS, width=1, dash="dot"))
    for m in sorted(df.marca.unique()):
        fig.add_trace(go.Violin(y=df.loc[df.marca == m, "potencia_kw"], name=m, box_visible=True,
                                meanline_visible=True, line_color=MARCAS[m],
                                fillcolor=_alfa(MARCAS[m], .22), points=False, spanmode="hard"))
    fig.update_yaxes(title="Potencia (kW)")
    return _base(fig, None, alto, showlegend=False)


def mapa_puestos(tabla, alto=330):
    criterios = ["Mediana ppm", "% ≤ 400 ppm", "Cpk potencia", "% potencia en espec.",
                 "Reparaciones / año de uso"]
    asc = {"Mediana ppm": True, "Reparaciones / año de uso": True}
    pu = pd.DataFrame({c: tabla[c].rank(ascending=asc.get(c, False)) for c in criterios})
    texto = [[f"{tabla.loc[m, c]:.3g}<br><b>#{int(pu.loc[m, c])}</b>" for c in criterios]
             for m in tabla.index]
    fig = go.Figure(go.Heatmap(z=pu.values, x=criterios, y=list(tabla.index), text=texto,
                               texttemplate="%{text}", textfont=dict(size=12),
                               colorscale=[[0, OLIVA], [0.5, "#EFE6D4"], [1, BERMELLON]],
                               showscale=False, xgap=3, ygap=3))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", linecolor="rgba(0,0,0,0)", ticks="")
    return _base(fig, None, alto, margin=dict(l=10, r=10, t=40, b=10))


def correlaciones(matriz, alto=380):
    fig = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=matriz.index, zmin=-1, zmax=1,
                               colorscale=[[0, PIZARRA], [0.5, "#F7F1E6"], [1, BERMELLON]],
                               text=np.round(matriz.values, 3), texttemplate="%{text}",
                               xgap=3, ygap=3, colorbar=dict(thickness=10, len=.8)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(linecolor="rgba(0,0,0,0)", ticks="")
    return _base(fig, None, alto)


def dispersion_uso_ppm(df, reg, alto=420):
    fig = go.Figure()
    for m in sorted(df.marca.unique()):
        g = df[df.marca == m]
        fig.add_trace(go.Scattergl(x=g.tiempo_uso_anios, y=g.ppm, mode="markers", name=m,
                                   marker=dict(color=MARCAS[m], size=4, opacity=.45)))
    xs = np.linspace(df.tiempo_uso_anios.min(), df.tiempo_uso_anios.max(), 100)
    fig.add_trace(go.Scatter(x=xs, y=np.exp(reg["b0"] + reg["b1"] * xs), mode="lines",
                             name="Modelo ln(ppm) = b₀ + b₁·uso", line=dict(color=TINTA, width=2.5)))
    fig.update_yaxes(title="Emisiones (ppm)", type="log")
    fig.update_xaxes(title="Tiempo de uso (años)")
    return _base(fig, None, alto)


def intervalo(estimado, li, ls, etiqueta, referencia=None, ref_txt=None, alto=170, unidad=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[li, ls], y=[0, 0], mode="lines", line=dict(color=BERMELLON, width=8),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[li, estimado, ls], y=[0, 0, 0], mode="markers+text",
                             marker=dict(color=[BERMELLON, TINTA, BERMELLON], size=[12, 16, 12],
                                         symbol=["line-ns", "diamond", "line-ns"],
                                         line=dict(width=2, color=[BERMELLON, TINTA, BERMELLON])),
                             text=[f"{li:.5g}", f"{estimado:.5g}", f"{ls:.5g}"],
                             textposition=["bottom center", "top center", "bottom center"],
                             showlegend=False, hoverinfo="skip"))
    if referencia is not None:
        fig.add_vline(x=referencia, line=dict(color=OLIVA, dash="dash", width=1.5),
                      annotation_text=ref_txt, annotation_font=dict(color=OLIVA, size=11))
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_xaxes(title=f"{etiqueta}{(' (' + unidad + ')') if unidad else ''}")
    return _base(fig, None, alto, margin=dict(l=10, r=10, t=20, b=10))


# =============================================================================
# Pruebas: curva de la distribución del estadístico con la región de rechazo
# =============================================================================
def curva_prueba(dist, estadistico, critico, cola="dos", nombre="Z", alto=300, p=None):
    """Dibuja la distribución del estadístico bajo H₀, la región de rechazo y el valor observado."""
    lo, hi = dist.ppf(0.0005), dist.ppf(0.9995)
    lo, hi = min(lo, estadistico * 1.1, -abs(estadistico) * 1.1 if cola == "dos" else lo), \
        max(hi, estadistico * 1.1)
    xs = np.linspace(lo, hi, 600)
    ys = dist.pdf(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TINTA, width=2),
                             fill="tozeroy", fillcolor=_alfa(ARENA, .25), showlegend=False,
                             hoverinfo="skip"))
    regiones = []
    if cola == "dos":
        regiones = [xs <= -abs(critico), xs >= abs(critico)]
    elif cola == "der":
        regiones = [xs >= critico]
    else:
        regiones = [xs <= critico]
    for r in regiones:
        fig.add_trace(go.Scatter(x=xs[r], y=ys[r], mode="lines", line=dict(color=BERMELLON, width=0),
                                 fill="tozeroy", fillcolor=_alfa(BERMELLON, .55), showlegend=False,
                                 hoverinfo="skip"))
    fig.add_vline(x=estadistico, line=dict(color=BERMELLON if _en_region(estadistico, critico, cola)
                                           else OLIVA, width=3),
                  annotation_text=f"{nombre} obs. = {estadistico:.4g}",
                  annotation_position="top", annotation_font=dict(size=12))
    for c in ([-abs(critico), abs(critico)] if cola == "dos" else [critico]):
        fig.add_vline(x=c, line=dict(color=GRIS, width=1, dash="dot"),
                      annotation_text=f"crítico {c:.4g}", annotation_position="bottom",
                      annotation_font=dict(size=10, color=GRIS))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title=nombre)
    return _base(fig, None, alto, margin=dict(l=10, r=10, t=36, b=10))


def _en_region(e, c, cola):
    if cola == "dos":
        return abs(e) > abs(c)
    return e > c if cola == "der" else e < c


# =============================================================================
# Bondad de ajuste
# =============================================================================
def histograma_ajuste(x, dists: dict, nbins=None, alto=380, titulo=None, color=PIZARRA,
                      log_x=False):
    """Histograma en escala de densidad con una o varias densidades teóricas encima."""
    x = np.asarray(x, dtype=float)
    fig = go.Figure()
    datos = np.log10(x) if log_x else x
    fig.add_trace(go.Histogram(x=datos, histnorm="probability density", nbinsx=nbins,
                               marker=dict(color=_alfa(color, .35), line=dict(color=color, width=1)),
                               name="Datos"))
    xs = np.linspace(datos.min(), datos.max(), 400)
    for (nombre, d), col in zip(dists.items(), [BERMELLON, OLIVA, CIRUELA, OCRE]):
        if log_x:  # densidad de log10(X): f(10^u)·10^u·ln10
            ys = d.pdf(10 ** xs) * 10 ** xs * np.log(10)
        else:
            ys = d.pdf(xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nombre,
                                 line=dict(color=col, width=2.6)))
    fig.update_yaxes(title="Densidad")
    if log_x:
        fig.update_xaxes(title="log₁₀(x)")
    return _base(fig, titulo, alto)


def ks_grafico(x, dist, ks: dict, alto=380, titulo=None):
    """F_n(x) (escalera) vs F(x) (curva) y la máxima distancia D marcada."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    xs = np.linspace(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x), 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.repeat(x, 2)[1:], y=np.repeat(np.arange(1, n + 1) / n, 2)[:-1],
                             mode="lines", name="Fₙ(x) empírica", line=dict(color=PIZARRA, width=2,
                                                                         shape="hv")))
    fig.add_trace(go.Scatter(x=xs, y=dist.cdf(xs), mode="lines", name="F(x) teórica",
                             line=dict(color=BERMELLON, width=2.4)))
    xm = ks["x_max"]
    y1 = ks["Fn_despues"] if ks["lado"] == "+" else ks["Fn_antes"]
    y2 = ks["F_en_x"]
    fig.add_trace(go.Scatter(x=[xm, xm], y=[y1, y2], mode="lines+markers", name=f"D = {ks['D']:.4f}",
                             line=dict(color=TINTA, width=3), marker=dict(size=8, color=TINTA)))
    fig.update_yaxes(title="Probabilidad acumulada", range=[-0.02, 1.02])
    return _base(fig, titulo, alto)


def qq(x, dist, alto=380, titulo=None):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    pp = (np.arange(1, n + 1) - 0.5) / n
    teo = dist.ppf(pp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=teo, y=x, mode="markers", marker=dict(color=PIZARRA, size=6,
                                                                    opacity=.75), name="Datos"))
    lo, hi = min(teo.min(), x.min()), max(teo.max(), x.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color=BERMELLON, dash="dash", width=2)))
    fig.update_xaxes(title="Cuantiles teóricos")
    fig.update_yaxes(title="Cuantiles observados")
    return _base(fig, titulo, alto, showlegend=False)


def barras_chi(tabla, alto=330, titulo=None):
    """Frecuencia observada Oᵢ vs esperada Eᵢ por clase."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tabla["Clase"], y=tabla["Oᵢ"], name="Observada Oᵢ",
                         marker=dict(color=_alfa(PIZARRA, .8))))
    fig.add_trace(go.Scatter(x=tabla["Clase"], y=tabla["Eᵢ = nPᵢ"], name="Esperada Eᵢ",
                             mode="lines+markers", line=dict(color=BERMELLON, width=2.5),
                             marker=dict(size=9, symbol="diamond")))
    fig.update_yaxes(title="Frecuencia")
    fig.update_xaxes(type="category")
    return _base(fig, titulo, alto)


def peso_ad(alto=300):
    """Función de peso de Anderson–Darling ψ(F) = 1 / [F(1 − F)] frente al peso plano de KS."""
    u = np.linspace(0.005, 0.995, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u, y=1 / (u * (1 - u)), mode="lines", name="Peso de AD: 1/[F(1−F)]",
                             line=dict(color=BERMELLON, width=3), fill="tozeroy",
                             fillcolor=_alfa(BERMELLON, .12)))
    fig.add_trace(go.Scatter(x=u, y=np.full_like(u, 4), mode="lines", name="Peso plano (Cramér–von Mises)",
                             line=dict(color=PIZARRA, width=2, dash="dash")))
    fig.update_yaxes(title="Peso", type="log")
    fig.update_xaxes(title="F(x)  — posición dentro de la distribución")
    return _base(fig, None, alto)


# =============================================================================
# Punto 2
# =============================================================================
def rangos_mw(x1, x2, n1="A", n2="B", alto=260):
    todos = np.concatenate([x1, x2])
    r = stats.rankdata(todos)
    fig = go.Figure()
    for nombre, sl, col, y in [(n1, slice(0, len(x1)), BERMELLON, 1), (n2, slice(len(x1), None), PIZARRA, 0)]:
        fig.add_trace(go.Scatter(x=r[sl], y=np.full(len(r[sl]), y) + np.random.default_rng(1)
                                 .uniform(-.18, .18, len(r[sl])), mode="markers", name=f"Equipo {nombre}",
                                 marker=dict(color=col, size=8, opacity=.8, line=dict(width=0))))
    fig.update_yaxes(tickvals=[0, 1], ticktext=[f"Equipo {n2}", f"Equipo {n1}"], showgrid=False,
                     range=[-.6, 1.6])
    fig.update_xaxes(title="Rango en la muestra combinada (1 = el más rápido)")
    return _base(fig, None, alto, showlegend=False)


def barras_potencia(res: dict, alto=360):
    nombres = list(res.keys())
    pruebas = ["t de Welch", "Z (normal)", "Mann–Whitney"]
    fig = go.Figure()
    for p, col in zip(pruebas, [PIZARRA, OCRE, BERMELLON]):
        fig.add_trace(go.Bar(x=[n.replace(" (", "<br>(") for n in nombres],
                             y=[100 * res[n][p] for n in nombres], name=p, marker_color=col,
                             text=[f"{100 * res[n][p]:.1f}%" for n in nombres], textposition="outside"))
    fig.add_hline(y=5, line=dict(color=TINTA, dash="dash", width=1.2),
                  annotation_text="α = 5 %", annotation_position="top right")
    fig.update_yaxes(title="% de veces que se rechaza H₀", rangemode="tozero")
    return _base(fig, None, alto, barmode="group")


# =============================================================================
# Punto 3
# =============================================================================
def _prob_clase(dist, etiqueta: str) -> float:
    """Probabilidad de una clase discreta escrita como '3', '13 o más', '0 o menos', '8 y 9' o '0–8'."""
    e = etiqueta.strip()
    if e.endswith("o más"):
        return float(dist.sf(int(e.split()[0]) - 1))
    if e.endswith("o menos"):
        return float(dist.cdf(int(e.split()[0])))
    for sep in (" y ", "–"):
        if sep in e:
            a, b = (int(v) for v in e.split(sep))
            return float(dist.cdf(b) - dist.cdf(a - 1))
    return float(dist.pmf(int(e)))


def conteos_diarios(tabla_bn, lam: float, alto=360):
    """Días observados vs. esperados por la binomial negativa y por la Poisson, en las mismas clases."""
    n = tabla_bn["Oᵢ"].sum()
    poi = stats.poisson(lam)
    e_poi = [n * _prob_clase(poi, c) for c in tabla_bn["Clase"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tabla_bn["Clase"], y=tabla_bn["Oᵢ"], name="Días observados",
                         marker_color=_alfa(PIZARRA, .8)))
    fig.add_trace(go.Scatter(x=tabla_bn["Clase"], y=tabla_bn["Eᵢ = nPᵢ"], name="Esperados · binomial negativa",
                             mode="lines+markers", line=dict(color=OLIVA, width=2.5)))
    fig.add_trace(go.Scatter(x=tabla_bn["Clase"], y=e_poi, name="Esperados · Poisson",
                             mode="lines+markers", line=dict(color=BERMELLON, width=2.5, dash="dash")))
    fig.update_xaxes(type="category", title="Commits en un día hábil")
    fig.update_yaxes(title="Número de días")
    return _base(fig, None, alto)


def cajas_tipo(commits, alto=400):
    orden = commits.groupby("tipo")["lineas_cambiadas"].median().sort_values().index
    fig = go.Figure()
    for t, col in zip(orden, SERIE):
        fig.add_trace(go.Box(x=commits.loc[commits.tipo == t, "lineas_cambiadas"], name=t,
                             marker_color=col, orientation="h", boxpoints=False,
                             fillcolor=_alfa(col, .2), line=dict(width=1.6)))
    fig.update_xaxes(type="log", title="Líneas modificadas por commit (escala log)")
    return _base(fig, None, alto, showlegend=False)


def dispersion_log(commits, reg, alto=400):
    x, y = np.log10(commits.archivos), np.log10(commits.lineas_cambiadas)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", marker=dict(color=PIZARRA, size=5, opacity=.35),
                               name="Commits"))
    xs = np.linspace(x.min(), x.max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=reg["b0"] + reg["b1"] * xs, mode="lines", name="Recta de regresión",
                             line=dict(color=BERMELLON, width=3)))
    fig.update_xaxes(title="log₁₀(archivos modificados)")
    fig.update_yaxes(title="log₁₀(líneas modificadas)")
    return _base(fig, None, alto)


def carta_control(valores, lc, lsc, lic, fase_I, titulo, ytitulo, alto=300, marcar=None):
    k = np.arange(1, len(valores) + 1)
    fig = go.Figure()
    fig.add_vrect(x0=fase_I + .5, x1=len(valores) + .5, fillcolor=_alfa(OCRE, .10), line_width=0,
                  annotation_text="Fase II (vigilancia)", annotation_position="top left",
                  annotation_font=dict(size=11, color=GRIS))
    for y, t, c in [(lsc, "LSC", BERMELLON), (lc, "LC", TINTA), (lic, "LIC", BERMELLON)]:
        fig.add_hline(y=y, line=dict(color=c, dash="dash" if t != "LC" else "solid", width=1.3),
                      annotation_text=f"{t} = {y:.4g}", annotation_position="right",
                      annotation_font=dict(size=10, color=c))
    fuera = (np.asarray(valores) > lsc) | (np.asarray(valores) < lic)
    if marcar is not None:
        fuera = fuera | np.isin(k, marcar)
    fig.add_trace(go.Scatter(x=k, y=valores, mode="lines+markers", line=dict(color=PIZARRA, width=1.6),
                             marker=dict(size=8, color=np.where(fuera, BERMELLON, PIZARRA),
                                         line=dict(color=TARJETA, width=1)), showlegend=False))
    fig.update_xaxes(title="Muestra")
    fig.update_yaxes(title=ytitulo)
    return _base(fig, titulo, alto, margin=dict(l=10, r=70, t=40, b=10))


def histograma_especificacion(x, lie, lse, dist, alto=340):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=14,
                               marker=dict(color=_alfa(PIZARRA, .35), line=dict(color=PIZARRA, width=1)),
                               name="Diámetros"))
    xs = np.linspace(lie - .01, lse + .01, 400)
    fig.add_trace(go.Scatter(x=xs, y=dist.pdf(xs), mode="lines", line=dict(color=BERMELLON, width=2.6),
                             name="Normal ajustada"))
    for v, t in [(lie, "LIE"), (lse, "LSE")]:
        fig.add_vline(x=v, line=dict(color=OLIVA, dash="dash", width=2), annotation_text=t,
                      annotation_font=dict(color=OLIVA))
    fig.update_xaxes(title="Diámetro (mm)")
    fig.update_yaxes(title="Densidad")
    return _base(fig, None, alto)


# -----------------------------------------------------------------------------
def _alfa(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


alfa = _alfa
