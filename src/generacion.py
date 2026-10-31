"""
generacion.py — Generación de los datos simulados (reproducible con semilla fija).

  • generar_hornos()      → Punto 1: base inventada de 3665 hornos de 5 marcas.
  • generar_simulacion()  → Punto 2: 60 tiempos de procesamiento por equipo y escenario.
"""

import numpy as np
import pandas as pd

SEMILLA = 2026

# =============================================================================
# 1. Hornos (datos inventados, pero con una estructura coherente)
# =============================================================================
# Cada marca tiene su "perfil". Las emisiones crecen con el tiempo de uso y con
# el número de reparaciones; la potencia se degrada levemente con los años.
PERFILES = {
    #            n,   uso_forma, uso_escala, tasa_rep, mu_ln_ppm, sigma_ln, pot_media, pot_desv
    "Marca 1": (812, 2.6, 1.95, 0.140, 5.565, 0.420, 2.680, 0.100),
    "Marca 2": (694, 2.4, 1.90, 0.075, 5.420, 0.450, 2.715, 0.082),
    "Marca 3": (735, 2.5, 1.85, 0.100, 5.530, 0.360, 2.745, 0.064),
    "Marca 4": (768, 2.7, 1.90, 0.120, 5.600, 0.390, 2.715, 0.096),
    "Marca 5": (656, 2.8, 2.05, 0.205, 5.760, 0.470, 2.640, 0.120),
}
B_USO_PPM = 0.030      # +3 % de emisiones por cada año de uso
B_REP_PPM = 0.045      # +4.5 % por cada reparación
B_USO_POT = -0.003     # la potencia cae 0.003 kW por año


def generar_hornos(semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    bloques = []
    for marca, (n, k, th, tasa, mu, sg, pm, ps) in PERFILES.items():
        uso = np.clip(rng.gamma(k, th, n), 0.2, 18.0).round(1)
        rep = rng.poisson(tasa * uso)
        ppm = np.exp(mu + B_USO_PPM * uso + B_REP_PPM * rep + rng.normal(0, sg, n)).round(1)
        pot = (pm + B_USO_POT * uso + rng.normal(0, ps, n)).round(3)
        bloques.append(pd.DataFrame({
            "marca": marca, "tiempo_uso_anios": uso, "reparaciones": rep,
            "ppm": ppm, "potencia_kw": pot,
        }))
    df = pd.concat(bloques, ignore_index=True)
    # Se mezclan las filas para que parezca una base real, no ordenada por marca
    df = df.sample(frac=1, random_state=semilla).reset_index(drop=True)
    df.insert(0, "id_horno", [f"H-{i:04d}" for i in range(1, len(df) + 1)])
    return df


# =============================================================================
# 2. Simulación de los equipos A y B
# =============================================================================
def generar_simulacion(semilla: int = SEMILLA, n: int = 60) -> pd.DataFrame:
    """Escenario 1: normales.  Escenario 2: gamma (β = parámetro de ESCALA)."""
    rng = np.random.default_rng(semilla)
    return pd.DataFrame({
        "E1_equipo_A": rng.normal(175, 30, n).round(2),
        "E1_equipo_B": rng.normal(170, 50, n).round(2),
        "E2_equipo_A": rng.gamma(1.2, 150, n).round(2),
        "E2_equipo_B": rng.gamma(0.95, 190, n).round(2),
    })


