"""
taller.py — El ejemplo y los 6 ejercicios de la guía "Pruebas de bondad de ajuste"
(Prof. Horacio Fernández C.), resueltos con el mismo motor de la app.
"""

from __future__ import annotations

from scipy import stats

from . import bondad as b

EJEMPLO_ASTON = {
    "titulo": "Ejemplo · Reclamaciones de Construcciones ASTON S.A.",
    "enunciado": "440 apartamentos y su número de reclamaciones por errores constructivos. ¿Sigue una Poisson?",
    "valores": list(range(10)), "O": [18, 53, 103, 107, 82, 46, 18, 10, 2, 1],
}


def resolver_aston(alpha=0.05):
    v, O = EJEMPLO_ASTON["valores"], EJEMPLO_ASTON["O"]
    lam = sum(a * o for a, o in zip(v, O)) / sum(O)
    # La guía redondea λ a 3,048 y las probabilidades a 5 decimales: se replica exactamente
    res = b.chi_cuadrado_discreta(v, O, stats.poisson(3.048), 1, alpha, decimales_p=5)
    res["lam_exacto"] = lam
    return res


def resolver(i: int, alpha: float | None = None) -> dict:
    """Devuelve el resultado de la χ² del ejercicio i (1 a 6) y su planteamiento."""
    if i == 1:
        a = alpha or 0.05
        bordes = [1.45, 1.75, 2.05, 2.35, 2.65, 2.95, 3.25, 3.55]
        O = [0, 6, 10, 15, 27, 20, 17, 5, 0]
        r = b.chi_cuadrado_continua(None, stats.norm(2.5, 0.4), 0, a, bordes=bordes, observados=O)
        r.update(titulo="Ejercicio 1 · Duración de 100 baterías", dist="Normal(μ = 2.5, σ = 0.4)",
                 params_txt="μ = 2.5 y σ = 0.4 son DADOS → p = 0",
                 h0="la duración sigue una Normal(2.5, 0.4)")
    elif i == 2:
        a = alpha or 0.05
        r = b.chi_cuadrado_continua(None, stats.norm(0.173, 0.066), 2, a, bordes=[0.10, 0.15, 0.20, 0.25],
                                    observados=[12, 20, 23, 15, 13])
        r.update(titulo="Ejercicio 2 · Rentabilidad diaria de un activo", dist="Normal(x̄ = 0.173, s = 0.066)",
                 params_txt="μ y σ se ESTIMAN con la muestra → p = 2",
                 h0="la rentabilidad diaria sigue una distribución normal")
    elif i == 3:
        a = alpha or 0.01
        r = b.chi_cuadrado_discreta(range(6), [10, 24, 32, 23, 6, 5], stats.poisson(2.1), 1, a)
        r.update(titulo="Ejercicio 3 · Averías semanales de una máquina", dist="Poisson(λ = 2.1)",
                 params_txt="λ = 2.1 es la media OBSERVADA (se estima) → p = 1",
                 h0="el número semanal de averías sigue una Poisson")
    elif i == 4:
        a = alpha or 0.05
        r = b.chi_cuadrado_discreta(range(5), [10, 26, 35, 24, 5], stats.poisson(1.9), 1, a)
        r.update(titulo="Ejercicio 4 · Llegadas por minuto a un peaje", dist="Poisson(λ = 190/100 = 1.9)",
                 params_txt="λ = 190 vehículos / 100 minutos = 1.9 (se estima) → p = 1",
                 h0="las llegadas por minuto siguen una Poisson")
    elif i == 5:
        a = alpha or 0.05
        r = b.chi_cuadrado_discreta(range(4), [63, 23, 13, 1], stats.poisson(0.5), 0, a)
        r.update(titulo="Ejercicio 5 · Defectos del componente AH233", dist="Poisson(λ = 0.5)",
                 params_txt="λ = 0.5 es DADO por calidad → p = 0",
                 h0="los defectos por unidad siguen una Poisson(0.5)")
    elif i == 6:
        a = alpha or 0.05
        O = [1, 16, 55, 228]
        p = (0 * 1 + 1 * 16 + 2 * 55 + 3 * 228) / (3 * sum(O))
        r = b.chi_cuadrado_discreta(range(4), O, stats.binom(3, p), 1, a, ultima_abierta=False)
        r.update(titulo="Ejercicio 6 · Acciones vendidas por día", dist=f"Binomial(n = 3, p̂ = {p:.4f})",
                 params_txt=f"p̂ = Σx·O / (3·300) = 810/900 = {p:.4f} (se estima) → p = 1",
                 h0="el número de acciones vendidas sigue una Binomial(3, p)")
    else:
        raise ValueError(i)
    r["alpha"] = a
    return r
