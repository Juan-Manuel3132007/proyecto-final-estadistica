"""
generar_excel.py — Libro de Excel con los cálculos del proyecto hechos con FÓRMULAS de Excel.

Usa los mismos datos de la carpeta data/ (generados con semilla 2026), así que los resultados
coinciden con la app y con el informe. Ejecutar desde la raíz:  python scripts/generar_excel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import aplicaciones as ap  # noqa: E402

SALIDA = RAIZ / "excel" / "Proyecto_Aula_Estadistica.xlsx"
SALIDA.parent.mkdir(exist_ok=True)

F = "Arial"
TINTA, BERM, CREMA, ARENA = "1D1A16", "D9480F", "FBF4E9", "EFE7D8"
f_titulo = Font(name=F, size=15, bold=True, color=TINTA)
f_sub = Font(name=F, size=10, italic=True, color="6E665B")
f_cab = Font(name=F, size=10, bold=True, color="FFFFFF")
f_norm = Font(name=F, size=10, color=TINTA)
f_dato = Font(name=F, size=10, color="0000FF")          # datos de entrada (azul)
f_res = Font(name=F, size=10, bold=True, color=BERM)    # resultados clave
fill_cab = PatternFill("solid", fgColor=TINTA)
fill_res = PatternFill("solid", fgColor=CREMA)
fill_sec = PatternFill("solid", fgColor=ARENA)
borde = Border(bottom=Side(style="thin", color="DCD2C1"))

wb = Workbook()


def hoja(nombre, titulo, sub):
    ws = wb.create_sheet(nombre)
    ws["A1"] = titulo
    ws["A1"].font = f_titulo
    ws["A2"] = sub
    ws["A2"].font = f_sub
    ws.sheet_view.showGridLines = False
    return ws


def cab(ws, fila, col, textos):
    for j, t in enumerate(textos):
        c = ws.cell(fila, col + j, t)
        c.font, c.fill = f_cab, fill_cab
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


NUEVAS = ["T.INV.2T", "CHISQ.INV.RT", "CHISQ.DIST.RT", "NORM.S.DIST", "NORM.S.INV", "NORM.DIST",
          "F.INV.RT", "F.DIST.RT", "F.TEST", "T.TEST", "POISSON.DIST", "BINOM.DIST", "RANK.AVG"]


def xlfn(f: str) -> str:
    """Excel guarda las funciones de 2010+ con el prefijo _xlfn. (su interfaz lo oculta)."""
    import re
    for nombre in NUEVAS:
        f = re.sub(r"(?<![\w.])" + re.escape(nombre) + r"\(", "_xlfn." + nombre + "(", f)
    return f


def put(ws, ref, valor, fmt=None, font=f_norm, fill=None):
    if isinstance(valor, str) and valor.startswith("="):
        valor = xlfn(valor)
    c = ws[ref]
    c.value = valor
    c.font = font
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    return c


def etiqueta(ws, ref, texto, negrita=False):
    put(ws, ref, texto, font=Font(name=F, size=10, bold=negrita, color=TINTA))


def seccion(ws, ref, texto):
    c = put(ws, ref, texto, font=Font(name=F, size=11, bold=True, color=BERM))
    return c


def anchos(ws, d):
    for col, w in d.items():
        ws.column_dimensions[col].width = w


# =============================================================================
# Portada
# =============================================================================
ws = wb.active
ws.title = "Portada"
ws.sheet_view.showGridLines = False
ws["B2"] = "Proyecto de Aula · Estadística para Ingeniería · UdM 2026-2"
ws["B2"].font = Font(name=F, size=18, bold=True, color=TINTA)
ws["B3"] = "Integrantes: Juan Manuel Moreno Muñoz, David Restrepo, Juan Fernando · Prof. Horacio Fernández C."
ws["B3"].font = f_sub
filas = [
    ("Hornos_datos", "Base inventada de 3665 hornos (punto 1)."),
    ("P1_Por_marca", "Emisiones y potencia separadas por marca, para usar MEDIAN, QUARTILE y STDEV."),
    ("P1_Calculos", "a) emisiones, b) capacidad Cp y Cpk, d) intervalos, e) prueba F, f) prueba de proporción."),
    ("P2_Datos", "60 tiempos por equipo y escenario (semilla 2026), con descriptivas."),
    ("P2_Chi_E1A", "Chi-cuadrado paso a paso para el equipo A del escenario 1 contra la normal."),
    ("P2_KS_AD_E1A", "Kolmogorov–Smirnov y Anderson–Darling fila por fila."),
    ("P2_Medias_MW", "Pruebas F, t, Z y Mann–Whitney (rangos con RANK.AVG)."),
    ("P3_Llegadas", "Commits por día hábil: χ² para Poisson y binomial negativa (datos reales)."),
    ("P3_Anillos", "Cartas X̄–R y capacidad de los anillos de pistón (datos reales)."),
    ("P3_Jugo", "Carta p y χ² para la binomial en las latas de jugo (datos reales)."),
    ("Taller_ASTON", "Ejemplo de la guía de bondad de ajuste replicado con fórmulas."),
]
cab(ws, 5, 2, ["Hoja", "Contenido"])
for i, (h, t) in enumerate(filas, 6):
    ws.cell(i, 2, h).font = Font(name=F, size=10, bold=True, color=BERM)
    ws.cell(i, 3, t).font = f_norm
ws.cell(18, 2, "Convención de colores: azul = dato de entrada; negro = fórmula; rojo en negrita = resultado clave.").font = f_sub
ws.cell(19, 2, "Todas las cifras se calculan con fórmulas; si cambia un dato, el libro se recalcula.").font = f_sub
anchos(ws, {"A": 3, "B": 18, "C": 95})

# =============================================================================
# Punto 1
# =============================================================================
h = pd.read_csv(RAIZ / "data" / "hornos.csv")
ws = hoja("Hornos_datos", "Base de hornos (datos inventados, semilla 2026)", "Fuente: scripts/generar_datos.py")
cab(ws, 4, 1, ["id_horno", "marca", "tiempo_uso_anios", "reparaciones", "ppm", "potencia_kw"])
for i, fila in enumerate(h.itertuples(index=False), 5):
    for j, v in enumerate(fila, 1):
        c = ws.cell(i, j, v)
        c.font = f_dato
anchos(ws, {"A": 11, "B": 11, "C": 16, "D": 13, "E": 10, "F": 12})
N_H = len(h) + 4

marcas = sorted(h.marca.unique())
ws = hoja("P1_Por_marca", "Emisiones y potencia por marca", "Mismos datos de Hornos_datos, en una columna por marca")
cab(ws, 4, 1, [f"ppm · {m}" for m in marcas] + [""] + [f"potencia · {m}" for m in marcas])
for j, m in enumerate(marcas):
    sub = h[h.marca == m]
    for i, (p, w) in enumerate(zip(sub.ppm, sub.potencia_kw), 5):
        ws.cell(i, 1 + j, p).font = f_dato
        ws.cell(i, 7 + j, w).font = f_dato
for j in range(11):
    ws.column_dimensions[get_column_letter(j + 1)].width = 15
FIN = 5 + h.marca.value_counts().max()


def rng_ppm(j):
    col = get_column_letter(1 + j)
    return f"P1_Por_marca!${col}$5:${col}${FIN}"


def rng_pot(j):
    col = get_column_letter(7 + j)
    return f"P1_Por_marca!${col}$5:${col}${FIN}"


ws = hoja("P1_Calculos", "Punto 1 · Cálculos con fórmulas", "Especificaciones: potencia 2.7 ± 0.22 kW · emisiones ≤ 400 ppm")
etiqueta(ws, "A4", "LIE (kW)")
put(ws, "B4", 2.48, "0.00", f_dato)
etiqueta(ws, "A5", "LSE (kW)")
put(ws, "B5", 2.92, "0.00", f_dato)
etiqueta(ws, "A6", "Límite de seguridad (ppm)")
put(ws, "B6", 400, "0", f_dato)
etiqueta(ws, "A7", "Norma (ppm)")
put(ws, "B7", 1000, "0", f_dato)
etiqueta(ws, "A8", "Nivel de significancia α")
put(ws, "B8", 0.05, "0.00", f_dato)

seccion(ws, "A10", "a) Emisiones y b) potencia por marca")
cols = ["Marca", "n", "Media ppm", "Mediana ppm", "Q1", "Q3", "% ≤ 400", "% > 1000", "Media kW", "s kW",
        "% en espec.", "Cp", "Cpk"]
cab(ws, 11, 1, cols)
for j, m in enumerate(marcas):
    r = 12 + j
    P, W = rng_ppm(j), rng_pot(j)
    put(ws, f"A{r}", m, font=Font(name=F, size=10, bold=True))
    put(ws, f"B{r}", f"=COUNT({P})", "0")
    put(ws, f"C{r}", f"=AVERAGE({P})", "0.0")
    put(ws, f"D{r}", f"=MEDIAN({P})", "0.0")
    put(ws, f"E{r}", f"=QUARTILE({P},1)", "0.0")
    put(ws, f"F{r}", f"=QUARTILE({P},3)", "0.0")
    put(ws, f"G{r}", f'=COUNTIF({P},"<="&$B$6)/B{r}', "0.0%")
    put(ws, f"H{r}", f'=COUNTIF({P},">"&$B$7)/B{r}', "0.00%")
    put(ws, f"I{r}", f"=AVERAGE({W})", "0.0000")
    put(ws, f"J{r}", f"=STDEV({W})", "0.0000")
    put(ws, f"K{r}", f'=COUNTIFS({W},">="&$B$4,{W},"<="&$B$5)/B{r}', "0.00%")
    put(ws, f"L{r}", f"=($B$5-$B$4)/(6*J{r})", "0.000")
    put(ws, f"M{r}", f"=MIN($B$5-I{r},I{r}-$B$4)/(3*J{r})", "0.000", f_res)
ws.cell(17, 1, "Nota: Excel calcula los cuartiles con QUARTILE (método inclusivo), igual que Python.").font = f_sub

seccion(ws, "A19", "d) Intervalos de confianza para la Marca 2 (mejor desempeño)")
etiqueta(ws, "A20", "Confianza")
put(ws, "B20", 0.95, "0%", f_dato)
etiqueta(ws, "A21", "n")
put(ws, "B21", "=B13", "0")
etiqueta(ws, "A22", "Media x̄ (kW)")
put(ws, "B22", "=I13", "0.0000")
etiqueta(ws, "A23", "s (kW)")
put(ws, "B23", "=J13", "0.0000")
etiqueta(ws, "A24", "t (α/2, n−1)")
put(ws, "B24", "=T.INV.2T(1-B20,B21-1)", "0.0000")
etiqueta(ws, "A25", "IC media: límite inferior")
put(ws, "B25", "=B22-B24*B23/SQRT(B21)", "0.0000", f_res, fill_res)
etiqueta(ws, "A26", "IC media: límite superior")
put(ws, "B26", "=B22+B24*B23/SQRT(B21)", "0.0000", f_res, fill_res)
etiqueta(ws, "A27", "χ² que deja α/2 a la derecha")
put(ws, "B27", "=CHISQ.INV.RT((1-B20)/2,B21-1)", "0.00")
etiqueta(ws, "A28", "χ² que deja 1−α/2 a la derecha")
put(ws, "B28", "=CHISQ.INV.RT(1-(1-B20)/2,B21-1)", "0.00")
etiqueta(ws, "A29", "IC varianza: límite inferior")
put(ws, "B29", "=(B21-1)*B23^2/B27", "0.000000", f_res, fill_res)
etiqueta(ws, "A30", "IC varianza: límite superior")
put(ws, "B30", "=(B21-1)*B23^2/B28", "0.000000", f_res, fill_res)
etiqueta(ws, "A31", "p̂ (cumple ≤ 400 ppm)")
put(ws, "B31", "=G13", "0.0000")
etiqueta(ws, "A32", "z (α/2)")
put(ws, "B32", "=NORM.S.INV(1-(1-B20)/2)", "0.0000")
etiqueta(ws, "A33", "IC proporción: límite inferior")
put(ws, "B33", "=B31-B32*SQRT(B31*(1-B31)/B21)", "0.0000", f_res, fill_res)
etiqueta(ws, "A34", "IC proporción: límite superior")
put(ws, "B34", "=B31+B32*SQRT(B31*(1-B31)/B21)", "0.0000", f_res, fill_res)

seccion(ws, "D19", "e) Prueba F: varianza de la potencia, Marca 4 vs Marca 3")
etiqueta(ws, "D20", "s² Marca 4 (numerador)")
put(ws, "E20", "=J15^2", "0.000000")
etiqueta(ws, "D21", "s² Marca 3 (denominador)")
put(ws, "E21", "=J14^2", "0.000000")
etiqueta(ws, "D22", "F = s²₄ / s²₃")
put(ws, "E22", "=E20/E21", "0.0000", f_res, fill_res)
etiqueta(ws, "D23", "F crítico (α/2, cola derecha)")
put(ws, "E23", "=F.INV.RT($B$8/2,B15-1,B14-1)", "0.0000")
etiqueta(ws, "D24", "Valor p (bilateral)")
put(ws, "E24", "=MIN(1,2*F.DIST.RT(E22,B15-1,B14-1))", "0.0000E+00", f_res, fill_res)
etiqueta(ws, "D25", "Decisión")
put(ws, "E25", '=IF(E24<$B$8,"Se rechaza H0: varianzas distintas","No se rechaza H0")', font=f_res)

seccion(ws, "D27", "f) Prueba de proporción: H0: p = 0.80")
etiqueta(ws, "D28", "p₀ afirmado por el fabricante")
put(ws, "E28", 0.8, "0.00", f_dato)
etiqueta(ws, "D29", "Z")
put(ws, "E29", "=(B31-E28)/SQRT(E28*(1-E28)/B21)", "0.0000", f_res, fill_res)
etiqueta(ws, "D30", "Valor p bilateral")
put(ws, "E30", "=2*(1-NORM.S.DIST(ABS(E29),TRUE))", "0.0000", f_res, fill_res)
etiqueta(ws, "D31", "Valor p unilateral (Ha: p < 0.80)")
put(ws, "E31", "=NORM.S.DIST(E29,TRUE)", "0.0000")
etiqueta(ws, "D32", "Decisión")
put(ws, "E32", '=IF(E30<$B$8,"Se rechaza H0","No se rechaza H0: se acepta la afirmación")', font=f_res)

seccion(ws, "D34", "Correlaciones (Pearson) con toda la base")
etiqueta(ws, "D35", "r (uso, ppm)")
put(ws, "E35", f"=CORREL(Hornos_datos!C5:C{N_H},Hornos_datos!E5:E{N_H})", "0.000")
etiqueta(ws, "D36", "r (reparaciones, ppm)")
put(ws, "E36", f"=CORREL(Hornos_datos!D5:D{N_H},Hornos_datos!E5:E{N_H})", "0.000")
etiqueta(ws, "D37", "r (uso, potencia)")
put(ws, "E37", f"=CORREL(Hornos_datos!C5:C{N_H},Hornos_datos!F5:F{N_H})", "0.000")
anchos(ws, {"A": 30, "B": 12, "C": 11, "D": 32, "E": 14, "F": 10, "G": 10, "H": 10, "I": 10, "J": 10, "K": 11,
            "L": 9, "M": 9})

# =============================================================================
# Punto 2
# =============================================================================
s = pd.read_csv(RAIZ / "data" / "simulacion.csv")
ws = hoja("P2_Datos", "Punto 2 · Tiempos de procesamiento (ms)", "Generados en Python con semilla 2026")
cab(ws, 4, 1, list(s.columns))
for i, fila in enumerate(s.itertuples(index=False), 5):
    for j, v in enumerate(fila, 1):
        ws.cell(i, j, v).font = f_dato
cab(ws, 4, 6, ["Medida"] + list(s.columns))
medidas = [("n", "COUNT", "0"), ("Media", "AVERAGE", "0.00"), ("Mediana", "MEDIAN", "0.00"),
           ("Desv. estándar", "STDEV", "0.00"), ("Varianza", "VAR", "0.00"), ("Mínimo", "MIN", "0.00"),
           ("Q1", "QUARTILE1", "0.00"), ("Q3", "QUARTILE3", "0.00"), ("Máximo", "MAX", "0.00"),
           ("Asimetría", "SKEW", "0.000"), ("Curtosis", "KURT", "0.000")]
for i, (nom, fn, fmt) in enumerate(medidas, 5):
    put(ws, f"F{i}", nom, font=Font(name=F, size=10, bold=True))
    for j in range(4):
        col = get_column_letter(1 + j)
        r = f"{col}5:{col}64"
        if fn.startswith("QUARTILE"):
            frm = f"=QUARTILE({r},{fn[-1]})"
        else:
            frm = f"={fn}({r})"
        put(ws, f"{get_column_letter(7 + j)}{i}", frm, fmt)
put(ws, "F16", "CV (%)", font=Font(name=F, size=10, bold=True))
for j in range(4):
    c = get_column_letter(7 + j)
    put(ws, f"{c}16", f"=100*{c}8/{c}6", "0.00")
anchos(ws, {"A": 13, "B": 13, "C": 13, "D": 13, "E": 3, "F": 16, "G": 13, "H": 13, "I": 13, "J": 13})

# ---------- χ² E1-A -----------------------------------------------------------
ws = hoja("P2_Chi_E1A", "Chi-cuadrado: ¿el equipo A del escenario 1 es normal?",
          "Método de la guía: k con 2^k > n, clases de igual ancho, Eᵢ ≥ 5, v = k − p − 1")
items = [("n", "=COUNT(P2_Datos!A5:A64)", "0"), ("Media x̄", "=AVERAGE(P2_Datos!A5:A64)", "0.000"),
         ("Desv. s", "=STDEV(P2_Datos!A5:A64)", "0.000"), ("Mínimo", "=MIN(P2_Datos!A5:A64)", "0.00"),
         ("Máximo", "=MAX(P2_Datos!A5:A64)", "0.00"), ("k (2^k > n)", "=FLOOR(LOG(B4,2),1)+1", "0"),
         ("Ancho de clase", "=(B8-B7)/B9", "0.000"), ("Parámetros estimados p", 2, "0"), ("α", 0.05, "0.00")]
for i, (e, v, fmt) in enumerate(items, 4):
    etiqueta(ws, f"A{i}", e)
    put(ws, f"B{i}", v, fmt, f_dato if not str(v).startswith("=") else f_norm)
ws["A14"] = "Clases de igual ancho (6) y agrupación: la última clase (E < 5) se une con la anterior → 5 clases."
ws["A14"].font = f_sub
cab(ws, 15, 1, ["Clase", "Límite inf.", "Límite sup.", "Oᵢ", "Pᵢ", "Eᵢ = nPᵢ", "(Oᵢ−Eᵢ)²/Eᵢ"])
# bordes interiores usados tras agrupar: min + 1..4 anchos
for i in range(5):
    r = 16 + i
    put(ws, f"A{r}", i + 1, "0")
    li = "-1E+99" if i == 0 else f"=$B$7+{i}*$B$10"
    ls = "1E+99" if i == 4 else f"=$B$7+{i + 1}*$B$10"
    put(ws, f"B{r}", li if i else -1e99, "0.00")
    put(ws, f"C{r}", ls if i < 4 else 1e99, "0.00")
    put(ws, f"D{r}", f'=COUNTIFS(P2_Datos!$A$5:$A$64,">"&B{r},P2_Datos!$A$5:$A$64,"<="&C{r})', "0")
    put(ws, f"E{r}", f"=NORM.DIST(C{r},$B$5,$B$6,TRUE)-NORM.DIST(B{r},$B$5,$B$6,TRUE)", "0.00000")
    put(ws, f"F{r}", f"=$B$4*E{r}", "0.000")
    put(ws, f"G{r}", f"=(D{r}-F{r})^2/F{r}", "0.0000")
ws["B16"].comment = Comment("−∞: la primera clase es abierta (x ≤ límite superior).", "Proyecto")
ws["C20"].comment = Comment("+∞: la última clase es abierta.", "Proyecto")
put(ws, "C21", "Suma", font=Font(name=F, size=10, bold=True))
put(ws, "D21", "=SUM(D16:D20)", "0")
put(ws, "E21", "=SUM(E16:E20)", "0.00000")
put(ws, "F21", "=SUM(F16:F20)", "0.000")
etiqueta(ws, "A23", "EP")
put(ws, "B23", "=SUM(G16:G20)", "0.0000", f_res, fill_res)
etiqueta(ws, "A24", "v = k − p − 1")
put(ws, "B24", "=COUNT(D16:D20)-B11-1", "0")
etiqueta(ws, "A25", "χ² crítico")
put(ws, "B25", "=CHISQ.INV.RT(B12,B24)", "0.0000")
etiqueta(ws, "A26", "Valor P")
put(ws, "B26", "=CHISQ.DIST.RT(B23,B24)", "0.0000", f_res, fill_res)
etiqueta(ws, "A27", "Decisión")
put(ws, "B27", '=IF(B23>B25,"Se rechaza H0","No se rechaza H0: es normal")', font=f_res)
anchos(ws, {"A": 26, "B": 14, "C": 14, "D": 9, "E": 11, "F": 11, "G": 13})

# ---------- KS y AD ------------------------------------------------------------
ws = hoja("P2_KS_AD_E1A", "Kolmogorov–Smirnov y Anderson–Darling para E1·A contra la normal",
          "Los datos se ordenan de menor a mayor; F(x) es la normal con x̄ y s estimados")
x = np.sort(s["E1_equipo_A"].to_numpy())
etiqueta(ws, "A4", "n")
put(ws, "B4", "=COUNT(B9:B68)", "0")
etiqueta(ws, "A5", "x̄")
put(ws, "B5", "=AVERAGE(B9:B68)", "0.000")
etiqueta(ws, "A6", "s")
put(ws, "B6", "=STDEV(B9:B68)", "0.000")
cab(ws, 8, 1, ["i", "x₍ᵢ₎ (ordenado)", "F(x₍ᵢ₎)", "i/n − F", "F − (i−1)/n", "F(x₍ₙ₊₁₋ᵢ₎)",
               "(2i−1)[ln F + ln(1−F(x₍ₙ₊₁₋ᵢ₎))]"])
for i, v in enumerate(x, 1):
    r = 8 + i
    put(ws, f"A{r}", i, "0")
    put(ws, f"B{r}", float(v), "0.00", f_dato)
    put(ws, f"C{r}", f"=NORM.DIST(B{r},$B$5,$B$6,TRUE)", "0.00000")
    put(ws, f"D{r}", f"=A{r}/$B$4-C{r}", "0.00000")
    put(ws, f"E{r}", f"=C{r}-(A{r}-1)/$B$4", "0.00000")
    put(ws, f"F{r}", f"=INDEX($C$9:$C$68,$B$4+1-A{r})", "0.00000")
    put(ws, f"G{r}", f"=(2*A{r}-1)*(LN(C{r})+LN(1-F{r}))", "0.0000")
etiqueta(ws, "I4", "D⁺ = máx(i/n − F)")
put(ws, "J4", "=MAX(D9:D68)", "0.0000")
etiqueta(ws, "I5", "D⁻ = máx(F − (i−1)/n)")
put(ws, "J5", "=MAX(E9:E68)", "0.0000")
etiqueta(ws, "I6", "D (Kolmogorov–Smirnov)")
put(ws, "J6", "=MAX(J4,J5)", "0.0000", f_res, fill_res)
etiqueta(ws, "I7", "Crítico aproximado 1.36/√n (parámetros conocidos)")
put(ws, "J7", "=1.36/SQRT(B4)", "0.0000")
etiqueta(ws, "I8", "Crítico de Lilliefors aprox. 0.886/√n (μ, σ estimados)")
put(ws, "J8", "=0.886/SQRT(B4)", "0.0000")
etiqueta(ws, "I10", "A² (Anderson–Darling)")
put(ws, "J10", "=-B4-SUM(G9:G68)/B4", "0.0000", f_res, fill_res)
etiqueta(ws, "I11", "A* = A²(1 + 0.75/n + 2.25/n²)")
put(ws, "J11", "=J10*(1+0.75/B4+2.25/B4^2)", "0.0000", f_res, fill_res)
etiqueta(ws, "I12", "Crítico de Stephens (normal, α = 0.05)")
put(ws, "J12", 0.752, "0.000", f_dato)
etiqueta(ws, "I13", "Decisión KS")
put(ws, "J13", '=IF(J6>J8,"Se rechaza H0","No se rechaza H0")', font=f_res)
etiqueta(ws, "I14", "Decisión AD")
put(ws, "J14", '=IF(J11>J12,"Se rechaza H0","No se rechaza H0")', font=f_res)
ws["I16"] = "En la app el valor p se obtiene por Monte Carlo (bootstrap paramétrico); aquí se usan los críticos de tabla."
ws["I16"].font = f_sub
anchos(ws, {"A": 6, "B": 16, "C": 12, "D": 12, "E": 13, "F": 15, "G": 30, "H": 3, "I": 48, "J": 16})

# ---------- Medias y Mann–Whitney ------------------------------------------------
ws = hoja("P2_Medias_MW", "Comparación de los equipos A y B", "F, t-Student, Z (normal) y Mann–Whitney")
cab(ws, 4, 1, ["", "Escenario 1", "Escenario 2"])
filas2 = [("x̄_A", "=AVERAGE(P2_Datos!{a}5:{a}64)", "0.000"), ("x̄_B", "=AVERAGE(P2_Datos!{b}5:{b}64)", "0.000"),
          ("s²_A", "=VAR(P2_Datos!{a}5:{a}64)", "0.00"), ("s²_B", "=VAR(P2_Datos!{b}5:{b}64)", "0.00"),
          ("n_A", "=COUNT(P2_Datos!{a}5:{a}64)", "0"), ("n_B", "=COUNT(P2_Datos!{b}5:{b}64)", "0"),
          ("Valor p de F (F.TEST)", "=F.TEST(P2_Datos!{a}5:{a}64,P2_Datos!{b}5:{b}64)", "0.0000"),
          ("¿Varianzas iguales?", '=IF({c}11<0.05,"No → t de Welch","Sí → t combinada")', None),
          ("Error estándar √(s²_A/n_A + s²_B/n_B)", "=SQRT({c}7/{c}9+{c}8/{c}10)", "0.0000"),
          ("t = Z = (x̄_A − x̄_B)/EE", "=({c}5-{c}6)/{c}13", "0.0000"),
          ("Valor p t (T.TEST)", '=T.TEST(P2_Datos!{a}5:{a}64,P2_Datos!{b}5:{b}64,2,IF({c}11<0.05,3,2))', "0.0000"),
          ("Valor p Z (normal)", "=2*(1-NORM.S.DIST(ABS({c}14),TRUE))", "0.0000"),
          ("R_A (suma de rangos de A)", "=SUMIF(P2_Medias_MW!$F$5:$F$124,\"A\",P2_Medias_MW!{r}5:{r}124)", "0.0"),
          ("U_A = R_A − n_A(n_A+1)/2", "={c}17-{c}9*({c}9+1)/2", "0.0"),
          ("E[U] = n_A n_B / 2", "={c}9*{c}10/2", "0.0"),
          ("σ_U = √(n_A n_B (N+1)/12)", "=SQRT({c}9*{c}10*({c}9+{c}10+1)/12)", "0.000"),
          ("Z Mann–Whitney", "=({c}18-{c}19)/{c}20", "0.0000"),
          ("Valor p Mann–Whitney", "=2*(1-NORM.S.DIST(ABS({c}21),TRUE))", "0.0000")]
for i, (e, frm, fmt) in enumerate(filas2, 5):
    etiqueta(ws, f"A{i}", e)
    for col, a, b_, rcol in (("B", "A", "B", "H"), ("C", "C", "D", "I")):
        put(ws, f"{col}{i}", frm.format(a=a, b=b_, c=col, r=rcol), fmt,
            f_res if i in (14, 15, 16, 21, 22) else f_norm, fill_res if i in (14, 15, 16, 21, 22) else None)
cab(ws, 4, 5, ["Valor E1", "Equipo", "Valor E2", "Rango E1", "Rango E2"])
for i in range(60):
    r = 5 + i
    put(ws, f"E{r}", f"=P2_Datos!A{5 + i}", "0.00")
    put(ws, f"F{r}", "A")
    put(ws, f"G{r}", f"=P2_Datos!C{5 + i}", "0.00")
    r2 = 65 + i
    put(ws, f"E{r2}", f"=P2_Datos!B{5 + i}", "0.00")
    put(ws, f"F{r2}", "B")
    put(ws, f"G{r2}", f"=P2_Datos!D{5 + i}", "0.00")
for r in range(5, 125):
    put(ws, f"H{r}", f"=RANK.AVG(E{r},$E$5:$E$124,1)", "0.0")
    put(ws, f"I{r}", f"=RANK.AVG(G{r},$G$5:$G$124,1)", "0.0")
ws["A24"] = "Sin corrección por empates (no hay empates en estos datos)."
ws["A24"].font = f_sub
anchos(ws, {"A": 38, "B": 18, "C": 18, "D": 3, "E": 10, "F": 8, "G": 10, "H": 10, "I": 10})

# =============================================================================
# Punto 3
# =============================================================================
c, d = ap.cargar_sistemas()
L = ap.llegadas_diarias(d)
x = L["x"]
ws = hoja("P3_Llegadas", "Commits por día hábil en Streamlit (2024–2025, datos reales)",
          "Fuente: git log de github.com/streamlit/streamlit · días hábiles con 0 o más commits")
cab(ws, 4, 1, ["commits_dia"])
for i, v in enumerate(x, 5):
    ws.cell(i, 1, int(v)).font = f_dato
FX = 4 + len(x)
R = f"$A$5:$A${FX}"
etiqueta(ws, "C4", "n (días)")
put(ws, "D4", f"=COUNT({R})", "0")
etiqueta(ws, "C5", "Media x̄ = λ̂")
put(ws, "D5", f"=AVERAGE({R})", "0.0000")
etiqueta(ws, "C6", "Varianza s²")
put(ws, "D6", f"=VAR({R})", "0.0000")
etiqueta(ws, "C7", "Índice de dispersión s²/x̄")
put(ws, "D7", "=D6/D5", "0.000", f_res, fill_res)
etiqueta(ws, "C8", "r (binomial negativa, momentos)")
put(ws, "D8", "=D5^2/(D6-D5)", "0.0000")
etiqueta(ws, "C9", "p (binomial negativa)")
put(ws, "D9", "=D8/(D8+D5)", "0.0000")
cab(ws, 11, 3, ["Commits", "Oᵢ", "P Poisson", "E Poisson", "P bin. neg.", "E bin. neg.", "(O−E)²/E bin. neg."])
for k in range(14):
    r = 12 + k
    put(ws, f"C{r}", k if k < 13 else "13 o más", "0")
    if k < 13:
        put(ws, f"D{r}", f"=COUNTIF({R},{k})", "0")
        put(ws, f"E{r}", f"=POISSON.DIST({k},$D$5,FALSE)", "0.00000")
        put(ws, f"G{r}", f"=EXP(GAMMALN({k}+$D$8)-GAMMALN($D$8)-GAMMALN({k}+1)+$D$8*LN($D$9)+{k}*LN(1-$D$9))",
            "0.00000")
    else:
        put(ws, f"D{r}", f'=COUNTIF({R},">=13")', "0")
        put(ws, f"E{r}", "=1-SUM(E12:E24)", "0.00000")
        put(ws, f"G{r}", "=1-SUM(G12:G24)", "0.00000")
    put(ws, f"F{r}", f"=$D$4*E{r}", "0.00")
    put(ws, f"H{r}", f"=$D$4*G{r}", "0.00")
    put(ws, f"I{r}", f"=(D{r}-H{r})^2/H{r}", "0.0000")
etiqueta(ws, "C27", "Bin. neg.: EP")
put(ws, "D27", "=SUM(I12:I25)", "0.0000", f_res, fill_res)
etiqueta(ws, "C28", "v = 14 − 2 − 1")
put(ws, "D28", 11, "0", f_dato)
etiqueta(ws, "C29", "Valor P")
put(ws, "D29", "=CHISQ.DIST.RT(D27,D28)", "0.0000", f_res, fill_res)
ws["C31"] = "Poisson: como Eᵢ < 5 desde 9 commits, se agrupan las clases 9 a 13+ en «9 o más» (10 clases)."
ws["C31"].font = f_sub
cab(ws, 32, 3, ["Clase", "Oᵢ", "Eᵢ Poisson", "(O−E)²/E"])
for k in range(10):
    r = 33 + k
    put(ws, f"C{r}", k if k < 9 else "9 o más", "0")
    if k < 9:
        put(ws, f"D{r}", f"=D{12 + k}", "0")
        put(ws, f"E{r}", f"=F{12 + k}", "0.00")
    else:
        put(ws, f"D{r}", "=SUM(D21:D25)", "0")
        put(ws, f"E{r}", "=SUM(F21:F25)", "0.00")
    put(ws, f"F{r}", f"=(D{r}-E{r})^2/E{r}", "0.0000")
etiqueta(ws, "C44", "Poisson: EP")
put(ws, "D44", "=SUM(F33:F42)", "0.00", f_res, fill_res)
etiqueta(ws, "C45", "v = 10 − 1 − 1")
put(ws, "D45", 8, "0", f_dato)
etiqueta(ws, "C46", "Valor P")
put(ws, "D46", "=CHISQ.DIST.RT(D44,D45)", "0.00E+00", f_res, fill_res)
anchos(ws, {"A": 13, "B": 3, "C": 30, "D": 12, "E": 12, "F": 12, "G": 12, "H": 12, "I": 18})

a, j = ap.cargar_industrial()
ws = hoja("P3_Anillos", "Anillos de pistón: cartas X̄–R y capacidad (datos reales, Montgomery)",
          "Especificación 74.000 ± 0.050 mm · muestras de 5 · fase I = muestras 1 a 25")
cab(ws, 4, 1, ["Muestra", "Fase", "x1", "x2", "x3", "x4", "x5", "X̄", "R"])
for m in range(1, 41):
    r = 4 + m
    sub = a[a.muestra == m]
    put(ws, f"A{r}", m, "0")
    put(ws, f"B{r}", "I" if bool(sub.fase_I.iloc[0]) else "II")
    for k, v in enumerate(sub.diametro_mm, 3):
        ws.cell(r, k, float(v)).font = f_dato
        ws.cell(r, k).number_format = "0.000"
    put(ws, f"H{r}", f"=AVERAGE(C{r}:G{r})", "0.0000")
    put(ws, f"I{r}", f"=MAX(C{r}:G{r})-MIN(C{r}:G{r})", "0.000")
items = [("X̄̄ (fase I)", "=AVERAGE(H5:H29)", "0.00000"), ("R̄ (fase I)", "=AVERAGE(I5:I29)", "0.00000"),
         ("A2 (n = 5)", 0.577, "0.000"), ("D4 (n = 5)", 2.114, "0.000"),
         ("LSC X̄", "=L5+L7*L6", "0.00000"), ("LIC X̄", "=L5-L7*L6", "0.00000"), ("LSC R", "=L8*L6", "0.00000"),
         ("Muestras fuera (X̄)", '=COUNTIF(H5:H44,">"&L9)+COUNTIF(H5:H44,"<"&L10)', "0"),
         ("s fase I", "=STDEV(C5:G29)", "0.00000"), ("LIE", 73.95, "0.000"), ("LSE", 74.05, "0.000"),
         ("Cp", "=(L15-L14)/(6*L13)", "0.000"), ("Cpk", "=MIN(L15-L5,L5-L14)/(3*L13)", "0.000")]
for i, (e, v, fmt) in enumerate(items, 5):
    etiqueta(ws, f"K{i}", e)
    put(ws, f"L{i}", v, fmt, f_dato if not str(v).startswith("=") else (f_res if e.startswith(("Cp", "LSC", "LIC"))
                                                                         else f_norm))
ws["H46"] = "Las fórmulas de la columna X̄ alimentan la carta; las muestras 37–39 superan el LSC."
ws["H46"].font = f_sub
anchos(ws, {"A": 9, "B": 6, "C": 9, "D": 9, "E": 9, "F": 9, "G": 9, "H": 10, "I": 9, "J": 3, "K": 22, "L": 12})

ws = hoja("P3_Jugo", "Latas de jugo no conformes: carta p y ajuste binomial (datos reales, Montgomery)",
          "Muestras de 50 latas · las muestras 15 y 23 tienen causas asignables y se excluyen de la fase I")
cab(ws, 4, 1, ["Muestra", "Fase", "No conformes", "Tamaño", "p", "Usar en fase I"])
for i, fila in enumerate(j.itertuples(index=False), 5):
    put(ws, f"A{i}", int(fila.muestra), "0")
    put(ws, f"B{i}", "I" if bool(fila.fase_I) else "II")
    put(ws, f"C{i}", int(fila.no_conformes), "0", f_dato)
    put(ws, f"D{i}", int(fila.tamano), "0", f_dato)
    put(ws, f"E{i}", f"=C{i}/D{i}", "0.000")
    put(ws, f"F{i}", f'=IF(AND(B{i}="I",A{i}<>15,A{i}<>23),1,0)', "0")
items = [("Muestras fase I usadas", "=SUM(F5:F58)", "0"), ("p̄", "=SUMPRODUCT(C5:C58,F5:F58)/(50*H5)", "0.0000"),
         ("LSC", "=H6+3*SQRT(H6*(1-H6)/50)", "0.0000"), ("LIC", "=MAX(0,H6-3*SQRT(H6*(1-H6)/50))", "0.0000"),
         ("p fase II", '=SUMIF(B5:B58,"II",C5:C58)/(50*COUNTIF(B5:B58,"II"))', "0.0000"),
         ("Z (dos proporciones)", "=(H6-H9)/SQRT(((H6*50*H5+H9*50*24)/(50*H5+50*24))*(1-(H6*50*H5+H9*50*24)/"
                                  "(50*H5+50*24))*(1/(50*H5)+1/(50*24)))", "0.000")]
for i, (e, v, fmt) in enumerate(items, 5):
    etiqueta(ws, f"G{i}", e)
    put(ws, f"H{i}", v, fmt, f_res if e in ("p̄", "LSC", "LIC", "Z (dos proporciones)") else f_norm)
ws["G12"] = "χ² para la binomial (fase I): clases agrupadas para que Eᵢ ≥ 5"
ws["G12"].font = Font(name=F, size=10, bold=True, color=BERM)
cab(ws, 13, 7, ["Clase", "Desde", "Hasta", "Oᵢ", "Pᵢ", "Eᵢ", "(O−E)²/E"])
clases = [("0–8", 0, 8), ("9 y 10", 9, 10), ("11 y 12", 11, 12), ("13–50", 13, 50)]
for i, (nom, lo, hi) in enumerate(clases, 14):
    put(ws, f"G{i}", nom)
    put(ws, f"H{i}", lo, "0", f_dato)
    put(ws, f"I{i}", hi, "0", f_dato)
    put(ws, f"J{i}", f'=COUNTIFS(C$5:C$58,">="&H{i},C$5:C$58,"<="&I{i},F$5:F$58,1)', "0")
    put(ws, f"K{i}", f"=BINOM.DIST(I{i},50,$H$6,TRUE)-IF(H{i}=0,0,BINOM.DIST(H{i}-1,50,$H$6,TRUE))", "0.00000")
    put(ws, f"L{i}", f"=$H$5*K{i}", "0.000")
    put(ws, f"M{i}", f"=(J{i}-L{i})^2/L{i}", "0.0000")
etiqueta(ws, "G19", "EP")
put(ws, "H19", "=SUM(M14:M17)", "0.0000", f_res, fill_res)
etiqueta(ws, "G20", "v = 4 − 1 − 1")
put(ws, "H20", 2, "0", f_dato)
etiqueta(ws, "G21", "Valor P")
put(ws, "H21", "=CHISQ.DIST.RT(H19,H20)", "0.0000", f_res, fill_res)
anchos(ws, {"A": 9, "B": 6, "C": 13, "D": 9, "E": 8, "F": 13, "G": 22, "H": 10, "I": 8, "J": 8, "K": 10, "L": 9,
            "M": 11})

# =============================================================================
# Taller: ejemplo ASTON
# =============================================================================
ws = hoja("Taller_ASTON", "Ejemplo de la guía: reclamaciones de Construcciones ASTON S.A.",
          "H0: el número de reclamaciones sigue una Poisson · α = 0.05")
cab(ws, 4, 1, ["Reclamaciones", "Oᵢ", "x·Oᵢ", "Pᵢ (Poisson)", "Eᵢ = nPᵢ", "(Oᵢ−Eᵢ)²/Eᵢ"])
O = [18, 53, 103, 107, 82, 46, 18, 10, 2, 1]
for k, o in enumerate(O):
    r = 5 + k
    put(ws, f"A{r}", k, "0")
    put(ws, f"B{r}", o, "0", f_dato)
    put(ws, f"C{r}", f"=A{r}*B{r}", "0")
    if k < 8:
        put(ws, f"D{r}", f"=ROUND(POISSON.DIST(A{r},$I$5,FALSE),5)", "0.00000")
        put(ws, f"E{r}", f"=$I$4*D{r}", "0.0000")
        put(ws, f"F{r}", f"=(B{r}-E{r})^2/E{r}", "0.00000")
ws["A15"] = "8 y 9"
ws["A15"].font = Font(name=F, size=10, bold=True)
put(ws, "B15", "=B13+B14", "0")
put(ws, "D15", "=1-SUM(D5:D12)", "0.00000")
put(ws, "E15", "=$I$4*D15", "0.0000")
put(ws, "F15", "=(B15-E15)^2/E15", "0.00000")
ws["G13"] = "Las filas 8 y 9 tienen Eᵢ < 5: se agrupan en la fila 15 (como en la guía)."
ws["G13"].font = f_sub
etiqueta(ws, "H4", "n")
put(ws, "I4", "=SUM(B5:B14)", "0")
etiqueta(ws, "H5", "λ̂ = Σx·O / n (redondeado como en la guía)")
put(ws, "I5", "=ROUND(SUM(C5:C14)/I4,3)", "0.000")
etiqueta(ws, "H6", "EP")
put(ws, "I6", "=SUM(F5:F12)+F15", "0.00000", f_res, fill_res)
etiqueta(ws, "H7", "v = k − p − 1 = 9 − 1 − 1")
put(ws, "I7", 7, "0", f_dato)
etiqueta(ws, "H8", "χ² crítico (0.05, 7)")
put(ws, "I8", "=CHISQ.INV.RT(0.05,I7)", "0.000")
etiqueta(ws, "H9", "Valor P")
put(ws, "I9", "=CHISQ.DIST.RT(I6,I7)", "0.0000", f_res, fill_res)
etiqueta(ws, "H10", "Decisión")
put(ws, "I10", '=IF(I6>I8,"Se rechaza H0","No se rechaza H0: sigue una Poisson")', font=f_res)
anchos(ws, {"A": 24, "B": 8, "C": 8, "D": 13, "E": 12, "F": 14, "G": 3, "H": 40, "I": 14})

for w in wb.worksheets:
    w.sheet_properties.tabColor = BERM if w.title.startswith(("P1", "P2", "P3")) else TINTA
wb.save(SALIDA)
print("Guardado:", SALIDA)
