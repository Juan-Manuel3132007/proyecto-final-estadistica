"""
datos.py — Lectura de CUALQUIER archivo de datos que suba el usuario.

Formatos: CSV / TXT / TSV (con , ; tabulador o espacios; decimales con punto o coma),
Excel (.xlsx, .xls, .xlsm, .ods), JSON y Parquet.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXTENSIONES = ["csv", "txt", "tsv", "dat", "xlsx", "xls", "xlsm", "ods", "json", "parquet"]


def _a_numero(serie: pd.Series) -> pd.Series:
    """Convierte texto como '1.234,56' o '2,7' en número cuando casi toda la columna lo permite."""
    if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie):
        return serie
    s = serie.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    validos = s.dropna()
    if validos.empty:
        return serie
    intento = pd.to_numeric(validos, errors="coerce")
    if intento.notna().mean() < 0.9:
        # formato latino: miles con punto y decimales con coma
        latino = validos.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True).str.replace(",", ".", regex=False)
        intento = pd.to_numeric(latino, errors="coerce")
        if intento.notna().mean() >= 0.9:
            s2 = s.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True).str.replace(",", ".", regex=False)
            return pd.to_numeric(s2, errors="coerce")
        return serie
    return pd.to_numeric(s, errors="coerce")


def leer_texto(contenido: bytes) -> pd.DataFrame:
    texto = None
    for cod in ("utf-8-sig", "latin-1"):
        try:
            texto = contenido.decode(cod)
            break
        except UnicodeDecodeError:
            continue
    primeras = texto.splitlines()[:20]
    muestra = "\n".join(primeras)
    if muestra.count(";") > muestra.count(",") and ";" in muestra:
        sep = ";"
    elif "\t" in muestra:
        sep = "\t"
    elif "," in muestra and not all(_es_numero_coma(l) for l in primeras if l.strip()):
        sep = ","
    else:
        sep = r"\s+"
    df = pd.read_csv(io.StringIO(texto), sep=sep, engine="python")
    # Si la primera fila parecía un dato (no un encabezado), se vuelve a leer sin encabezado
    if all(_es_numero(str(c)) for c in df.columns):
        df = pd.read_csv(io.StringIO(texto), sep=sep, engine="python", header=None)
        df.columns = [f"columna_{i + 1}" for i in range(df.shape[1])]
    return df


def _es_numero(t: str) -> bool:
    try:
        float(t.replace(",", "."))
        return True
    except ValueError:
        return False


def _es_numero_coma(linea: str) -> bool:
    """'2,75' sola en la línea = número con coma decimal, no dos columnas."""
    partes = linea.strip().split(",")
    return len(partes) == 2 and all(p.strip().lstrip("-").isdigit() for p in partes)


def leer_archivo(nombre: str, contenido: bytes, hoja: str | int | None = 0) -> pd.DataFrame:
    ext = Path(nombre).suffix.lower().lstrip(".")
    if ext in ("xlsx", "xlsm", "xls", "ods"):
        motor = {"xls": "xlrd", "ods": "odf"}.get(ext)
        df = pd.read_excel(io.BytesIO(contenido), sheet_name=hoja, engine=motor)
    elif ext == "json":
        datos = json.loads(contenido.decode("utf-8-sig"))
        df = pd.json_normalize(datos) if isinstance(datos, list) else pd.DataFrame(datos)
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(contenido))
    else:
        df = leer_texto(contenido)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for c in df.columns:
        df[c] = _a_numero(df[c])
    return df.reset_index(drop=True)


def hojas_excel(nombre: str, contenido: bytes) -> list[str]:
    ext = Path(nombre).suffix.lower().lstrip(".")
    if ext not in ("xlsx", "xlsm", "xls", "ods"):
        return []
    motor = {"xls": "xlrd", "ods": "odf"}.get(ext)
    return list(pd.ExcelFile(io.BytesIO(contenido), engine=motor).sheet_names)


def columnas_numericas(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].notna().sum() >= 3]


def columnas_categoricas(df: pd.DataFrame, max_niveles: int = 30) -> list[str]:
    return [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])
                                      or pd.api.types.is_bool_dtype(df[c]))
            and 2 <= df[c].nunique() <= max_niveles] + \
           [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and 2 <= df[c].nunique() <= 8]


def es_discreta(x) -> bool:
    x = pd.Series(x).dropna()
    return bool(len(x) and (x >= 0).all() and np.allclose(x, np.round(x)) and x.nunique() <= 60)
