"""
ui.py — Estilos y componentes visuales de la app (identidad "papel y tinta").
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,800;0,9..144,900;1,9..144,400;1,9..144,600&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --papel:#F4EFE6; --tarjeta:#FFFDF8; --tinta:#1D1A16; --gris:#6E665B; --linea:#DCD2C1;
  --bermellon:#D9480F; --ocre:#D9982B; --oliva:#5E7D3A; --ciruela:#7A3B5E; --pizarra:#3E5C76;
}
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea, select, button {
  font-family: 'Manrope', sans-serif;
}
.stApp{
  background-color: var(--papel);
  background-image: radial-gradient(rgba(29,26,22,.07) 0.7px, transparent 0.7px);
  background-size: 16px 16px;
}
.block-container{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }
h1, h2, h3, h4 { font-family: 'Fraunces', Georgia, serif !important; color: var(--tinta) !important;
  letter-spacing: -0.01em; }
h2 { font-weight: 800 !important; }
h3 { font-weight: 600 !important; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ---------- Barra lateral: índice de tinta ---------- */
section[data-testid="stSidebar"]{ background: var(--tinta); border-right: 0; }
section[data-testid="stSidebar"] *{ color: #EFE7D8; }
section[data-testid="stSidebar"] .stRadio > label{ display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  padding: 9px 12px; border-radius: 0; border-left: 3px solid transparent; width:100%;
  transition: all .15s ease; margin:0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{ background: rgba(255,255,255,.05);
  border-left-color: rgba(217,72,15,.5); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] label[data-selected="true"]{
  background: rgba(217,72,15,.16); border-left-color: var(--bermellon); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]){ display:none; }
section[data-testid="stSidebar"] label[data-selected="true"] p{ color:#fff !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p{ font-size: .93rem; font-weight:600; }
section[data-testid="stSidebar"] input{ color: var(--tinta) !important; font-family:'JetBrains Mono',monospace; }
section[data-testid="stSidebar"] hr{ border-color: rgba(255,255,255,.12); }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"]{ background: var(--bermellon); }
.lateral-marca{ font-family:'Fraunces',serif; font-size:1.55rem; line-height:1.05; font-weight:800; margin:.2rem 0 .3rem 0;}
.lateral-marca em{ color: var(--bermellon); font-style: italic; font-weight:600;}
.lateral-meta{ font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.08em;
  text-transform:uppercase; color:#A79C8A !important; }
.lateral-equipo{ font-size:.8rem; line-height:1.5; color:#CFC5B3 !important; }

/* ---------- Encabezado de sección ---------- */
.cabecera{ display:flex; align-items:flex-end; gap:22px; padding-top: 14px; border-bottom: 2.5px solid var(--tinta);
  padding-bottom: 14px; margin-bottom: 18px; }
.cabecera .num{ font-family:'Fraunces',serif; font-weight:900; font-size: 5.2rem; line-height:.8;
  color: transparent; -webkit-text-stroke: 2px var(--bermellon); }
.cabecera .kicker{ font-family:'JetBrains Mono',monospace; font-size:.74rem; letter-spacing:.14em;
  text-transform:uppercase; color: var(--bermellon); margin-bottom:4px; }
.cabecera h1{ font-size: 2.45rem !important; margin:0 !important; padding:0 !important; line-height:1.05; font-weight:800 !important;}
.cabecera h1 em{ font-style: italic; font-weight: 400; color: var(--bermellon); }
.bajada{ font-size:1.05rem; color: var(--gris); max-width: 900px; margin: -4px 0 16px 0; line-height:1.55; }

/* ---------- Portada ---------- */
.portada{ border: 2.5px solid var(--tinta); background: var(--tarjeta); padding: 38px 42px 30px 42px;
  box-shadow: 9px 9px 0 var(--tinta); position:relative; overflow:hidden; margin: 6px 8px 34px 0;}
.portada::after{ content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px;
  border-radius:50%; background: radial-gradient(circle, rgba(217,72,15,.18), rgba(217,72,15,0) 70%);}
.portada .kicker{ font-family:'JetBrains Mono',monospace; letter-spacing:.16em; font-size:.78rem;
  text-transform:uppercase; color: var(--bermellon); }
.portada h1{ font-size: 3.6rem !important; line-height: .98 !important; margin: 14px 0 12px 0 !important; font-weight:900 !important;}
.portada h1 em{ font-weight:400; font-style:italic; color: var(--bermellon); }
.portada p{ font-size: 1.08rem; max-width: 760px; color:#3B362F; line-height:1.6; }
.portada .equipo{ display:flex; gap:28px; flex-wrap:wrap; margin-top: 18px; border-top:1px solid var(--linea); padding-top:14px;}
.portada .equipo div{ font-family:'JetBrains Mono',monospace; font-size:.78rem; color: var(--gris); }
.portada .equipo b{ display:block; font-family:'Manrope',sans-serif; font-size:.98rem; color: var(--tinta); }

/* ---------- Tarjetas y KPIs ---------- */
.tarjeta{ background: var(--tarjeta); border: 1.5px solid var(--tinta); box-shadow: 5px 5px 0 var(--tinta);
  padding: 18px 20px; margin: 4px 6px 18px 0; }
.kpis{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0;
  border: 1.5px solid var(--tinta); background: var(--tarjeta); margin: 6px 0 22px 0; box-shadow: 5px 5px 0 var(--tinta);}
.kpi{ padding: 16px 18px; border-right: 1px solid var(--linea); }
.kpi:last-child{ border-right: 0; }
.kpi .v{ font-family:'Fraunces',serif; font-weight:800; font-size: 2.05rem; line-height:1.05; color: var(--tinta);}
.kpi .v small{ font-size: .95rem; font-weight:600; color: var(--gris); margin-left:2px;}
.kpi .e{ font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase;
  color: var(--gris); margin-top: 5px;}
.kpi .n{ font-size:.82rem; color: var(--gris); margin-top:2px;}
.kpi.acento .v{ color: var(--bermellon); }

/* ---------- Lectura (interpretación) ---------- */
.lectura{ background: #FBF4E9; border-left: 4px solid var(--bermellon); padding: 14px 18px 12px 18px;
  margin: 10px 0 20px 0; line-height: 1.62; font-size: .99rem; }
.lectura::before{ content:"LECTURA"; display:block; font-family:'JetBrains Mono',monospace; font-size:.66rem;
  letter-spacing:.18em; color: var(--bermellon); margin-bottom:5px; font-weight:600;}
.lectura.teoria{ border-left-color: var(--pizarra); background:#F0EFEA;}
.lectura.teoria::before{ content:"IDEA CLAVE"; color: var(--pizarra);}
.lectura.alerta{ border-left-color: var(--ocre); background:#FBF1DF;}
.lectura.alerta::before{ content:"OJO"; color:#9C6A12;}

/* ---------- Veredictos ---------- */
.veredicto{ display:inline-flex; align-items:center; gap:10px; padding: 8px 14px; font-weight:700;
  font-size:.92rem; border:1.5px solid var(--tinta); margin: 4px 0 12px 0; }
.veredicto .sello{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.12em;
  padding: 2px 7px; border:1.5px solid currentColor; }
.veredicto.rechaza{ background: var(--bermellon); color: #fff; }
.veredicto.no{ background: var(--oliva); color: #fff; }

/* ---------- Paso a paso ---------- */
.paso{ display:flex; gap:12px; align-items:baseline; margin: 16px 0 6px 0; }
.paso .k{ font-family:'JetBrains Mono',monospace; font-size:.74rem; background: var(--tinta); color: var(--papel);
  padding: 2px 8px; }
.paso .t{ font-family:'Fraunces',serif; font-weight:600; font-size:1.18rem; }

/* ---------- Tabla "mapa de temas" ---------- */
.mapa{ width:100%; border-collapse: collapse; font-size:.9rem; background: var(--tarjeta); border:1.5px solid var(--tinta);}
.mapa th{ font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  text-align:center; padding:9px 6px; border-bottom:1.5px solid var(--tinta); background:#EFE7D8;}
.mapa th:first-child, .mapa td:first-child{ text-align:left; padding-left:12px;}
.mapa td{ padding:7px 6px; border-bottom:1px solid var(--linea); text-align:center;}
.mapa .si{ display:inline-block; width:12px; height:12px; background: var(--bermellon); border-radius:50%;}
.mapa .medio{ display:inline-block; width:12px; height:12px; border:2px solid var(--bermellon); border-radius:50%;}

/* ---------- Pestañas ---------- */
.stTabs [data-baseweb="tab-list"]{ gap: 0; border-bottom: 1.5px solid var(--tinta); }
.stTabs [data-baseweb="tab"]{ font-family:'JetBrains Mono',monospace; font-size:.8rem; letter-spacing:.04em;
  padding: 10px 16px; background: transparent; border-radius:0; }
.stTabs [aria-selected="true"]{ background: var(--tinta) !important; color: var(--papel) !important; }
.stTabs [aria-selected="true"] p{ color: var(--papel) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------- Varios ---------- */
div[data-testid="stDataFrame"]{ border: 1.5px solid var(--tinta); }
div[data-testid="stExpander"]{ border: 1.5px solid var(--tinta); border-radius:0; background: var(--tarjeta);}
.stButton > button, .stDownloadButton > button{ border-radius:0; border:1.5px solid var(--tinta);
  box-shadow: 3px 3px 0 var(--tinta); font-weight:700; background: var(--tarjeta); }
.stButton > button:hover, .stDownloadButton > button:hover{ border-color: var(--bermellon); color: var(--bermellon);}
div[data-testid="stFileUploader"] section{ border: 2px dashed var(--tinta); border-radius:0; background: var(--tarjeta);}
.etiqueta{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:var(--gris);}
.pie{ margin-top: 40px; border-top: 1.5px solid var(--tinta); padding-top: 10px; font-family:'JetBrains Mono',monospace;
  font-size:.72rem; color: var(--gris); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;}
.katex{ font-size: 1.12em !important; }
</style>
"""


def estilos():
    st.markdown(CSS, unsafe_allow_html=True)


def cabecera(numero: str, kicker: str, titulo_html: str, bajada: str | None = None):
    st.markdown(
        f"""<div class="cabecera"><div class="num">{numero}</div>
        <div><div class="kicker">{kicker}</div><h1>{titulo_html}</h1></div></div>""",
        unsafe_allow_html=True)
    if bajada:
        st.markdown(f'<div class="bajada">{bajada}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple], acento: int | None = None):
    """items = [(valor, etiqueta, nota opcional), …]"""
    celdas = []
    for i, it in enumerate(items):
        v, e = it[0], it[1]
        n = it[2] if len(it) > 2 else ""
        clase = "kpi acento" if acento == i else "kpi"
        celdas.append(f'<div class="{clase}"><div class="v">{v}</div><div class="e">{e}</div>'
                      f'{f"<div class=n>{n}</div>" if n else ""}</div>')
    st.markdown(f'<div class="kpis">{"".join(celdas)}</div>', unsafe_allow_html=True)


def lectura(texto_md: str, tipo: str = ""):
    """Caja de interpretación. Admite **negritas** y saltos de línea simples."""
    st.markdown(f'<div class="lectura {tipo}">{_md(texto_md)}</div>', unsafe_allow_html=True)


def veredicto(rechaza: bool, texto: str = ""):
    if rechaza is None:
        return
    clase = "rechaza" if rechaza else "no"
    sello = "SE RECHAZA H₀" if rechaza else "NO SE RECHAZA H₀"
    st.markdown(f'<div class="veredicto {clase}"><span class="sello">{sello}</span>{html.escape(texto)}</div>',
                unsafe_allow_html=True)


def paso(k: str, titulo: str):
    st.markdown(f'<div class="paso"><span class="k">{k}</span><span class="t">{titulo}</span></div>',
                unsafe_allow_html=True)


def tabla(df: pd.DataFrame, dec: int | dict = 4, alto: int | None = None, indice: bool = True):
    """Muestra una tabla con un número fijo de decimales por columna (NaN → «—»)."""
    d = df.copy()
    if isinstance(dec, int):
        dec = {c: dec for c in d.columns if pd.api.types.is_float_dtype(d[c])}
    for c, k in dec.items():
        if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v, k=k: "—" if pd.isna(v) else f"{v:,.{k}f}")
    for c in d.columns:   # columnas mixtas (p. ej. con None)
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    kw = {"use_container_width": True, "hide_index": not indice}
    if alto:
        kw["height"] = alto
    st.dataframe(d, **kw)


def pie():
    st.markdown('<div class="pie"><span>Estadística para Ingeniería · Universidad de Medellín · 2026-2</span>'
                '<span>Prof. Horacio Fernández C.</span></div>', unsafe_allow_html=True)


def num(x, d=4) -> str:
    """Número con d cifras decimales y separador de miles."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}"


def pval(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _md(t: str) -> str:
    import re
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\n\n", "<br><br>").replace("\n", "<br>")
