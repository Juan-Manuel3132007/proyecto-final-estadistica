"""
generar_datos.py — Genera TODOS los datos del proyecto (reproducibles con semilla fija).

1. data/hornos.csv          → Punto 1: base inventada de 3665 hornos de 5 marcas.
2. data/simulacion.csv      → Punto 2: 60 datos por equipo en los dos escenarios.
3. data/sistemas_commits.csv, data/sistemas_diario.csv
                            → Punto 3 (Ing. de Sistemas): historial REAL de commits de
                              Streamlit 2024–2025, ya limpio y clasificado.
4. data/industrial_anillos.csv, data/industrial_jugo.csv
                            → Punto 3 (Ing. Industrial): datos REALES de control de
                              calidad (Montgomery), tomados del paquete qcc de R.

Ejecutar desde la raíz del repositorio:   python scripts/generar_datos.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.generacion import generar_hornos, generar_simulacion  # noqa: E402

DATA = RAIZ / "data"

# =============================================================================
# 3. Datos reales — Ingeniería de Sistemas (commits de Streamlit)
# =============================================================================
REGLAS_TIPO = [
    ("Corrección", r"\bfix|bug|hotfix|regression|broken|issue"),
    ("Pruebas", r"\btest|e2e|flaky|playwright|pytest"),
    ("Refactorización", r"refactor|migrate|clean ?up|modernize|rename|move |simplif"),
    ("Documentación", r"\bdocs?\b|docstring|readme|typo|comment"),
    ("Funcionalidad", r"\badd|feat|support|allow|introduc|implement|new |enable|expose"),
]


def clasificar(asunto: str) -> str:
    s = asunto.lower()
    for tipo, patron in REGLAS_TIPO:
        if re.search(patron, s):
            return tipo
    return "Mantenimiento"


def preparar_sistemas() -> tuple[pd.DataFrame, pd.DataFrame]:
    crudo = pd.read_csv(DATA / "raw" / "streamlit_commits_2024_2025_raw.csv")
    df = crudo.copy()
    # Limpieza: se quitan bots, publicaciones de versión y commits sin cambios de texto
    es_bot = df["author"].str.contains(r"\[bot\]|dependabot|snyk", case=False, regex=True)
    es_release = df["subject"].str.contains(r"^release|version bump|^bump version|^Up version",
                                            case=False, regex=True)
    df = df[~es_bot & ~es_release].copy()
    df["lineas_cambiadas"] = df["adds"] + df["dels"]
    df = df[df["lineas_cambiadas"] > 0].copy()

    fecha = pd.to_datetime(df["commit_date"], utc=True)
    df["fecha"] = fecha.dt.strftime("%Y-%m-%d")
    df["anio"] = fecha.dt.year
    df["dia_semana"] = fecha.dt.dayofweek               # 0 = lunes
    df["hora_utc"] = fecha.dt.hour
    df["tipo"] = df["subject"].map(clasificar)
    df = df.rename(columns={"files": "archivos", "adds": "lineas_agregadas",
                            "dels": "lineas_eliminadas", "author": "autor",
                            "subject": "asunto", "hash": "commit"})
    df = df[["commit", "fecha", "anio", "dia_semana", "hora_utc", "autor", "tipo", "archivos",
             "lineas_agregadas", "lineas_eliminadas", "lineas_cambiadas", "asunto"]]
    df = df.sort_values("fecha").reset_index(drop=True)

    # Llegadas por día hábil (lunes a viernes) — incluye los días con 0 commits
    dias = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    conteo = df.groupby("fecha").size()
    diario = pd.DataFrame({"fecha": dias.strftime("%Y-%m-%d"), "dia_semana": dias.dayofweek})
    diario["commits"] = diario["fecha"].map(conteo).fillna(0).astype(int)
    diario["habil"] = diario["dia_semana"] < 5
    return df, diario


# =============================================================================
# 4. Datos reales — Ingeniería Industrial (control de calidad, Montgomery)
# =============================================================================
def preparar_industrial() -> tuple[pd.DataFrame, pd.DataFrame]:
    anillos = pd.read_csv(DATA / "raw" / "pistonrings_qcc.csv")
    anillos = anillos.rename(columns={"diameter": "diametro_mm", "sample": "muestra",
                                      "trial": "fase_I"})
    jugo = pd.read_csv(DATA / "raw" / "orangejuice_qcc.csv")
    jugo = jugo.rename(columns={"sample": "muestra", "D": "no_conformes", "size": "tamano",
                                "trial": "fase_I"})
    return anillos, jugo


def main() -> None:
    DATA.mkdir(exist_ok=True)
    hornos = generar_hornos()
    hornos.to_csv(DATA / "hornos.csv", index=False)
    sim = generar_simulacion()
    sim.to_csv(DATA / "simulacion.csv", index=False)
    commits, diario = preparar_sistemas()
    commits.to_csv(DATA / "sistemas_commits.csv", index=False)
    diario.to_csv(DATA / "sistemas_diario.csv", index=False)
    anillos, jugo = preparar_industrial()
    anillos.to_csv(DATA / "industrial_anillos.csv", index=False)
    jugo.to_csv(DATA / "industrial_jugo.csv", index=False)
    print(f"hornos: {hornos.shape}  simulación: {sim.shape}  commits: {commits.shape}  "
          f"días: {diario.shape}  anillos: {anillos.shape}  jugo: {jugo.shape}")


if __name__ == "__main__":
    main()
