# Decidir con datos: de los hornos al código

**Proyecto de Aula · Estadística para Ingeniería · Universidad de Medellín · 2026-2**
Profesor: Horacio Fernández C.
Integrantes: Juan Manuel Moreno Muñoz · David Restrepo · Juan Fernando

Aplicación en Python (Streamlit) que desarrolla los tres puntos del proyecto de aula y, como valor agregado,
explica y aplica las pruebas de **Kolmogorov–Smirnov** y **Anderson–Darling** junto a la χ² de la guía, con un
**laboratorio para subir cualquier archivo de datos**.

| Punto | Qué se hace | Datos |
|---|---|---|
| 1 · Hornos | Cajas y bigotes, capacidad (Cp, Cpk), mejor marca, IC para μ, σ² y p, prueba F, prueba de proporción, correlación y regresión | Inventados (3665 hornos, coherentes con el enunciado) |
| 2 · Equipos A y B | Histogramas, descriptivas, χ² + KS + AD, t-Student vs normal, Mann–Whitney y estudio de potencia por Monte Carlo | Simulados (semilla 2026) |
| 3 · Sistemas | ¿Los commits por día son Poisson? Tamaño lognormal de los cambios, Mann–Whitney, Spearman y regresión log-log | **Reales**: historial de Git de Streamlit 2024–2025 |
| 3 · Industrial | Cartas X̄–R y p, normalidad, capacidad, prueba binomial, fase I vs fase II | **Reales**: Montgomery, paquete `qcc` de R |

---

## Cómo ejecutarlo

Requiere **Python 3.10 o superior**.

```bash
git clone <url-del-repo>
cd proyecto-aula-estadistica

python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py          # se abre en http://localhost:8501
```

## La aplicación

| Página | Contenido |
|---|---|
| 00 Portada | Presentación y **mapa de temas del curso** (dónde aparece cada tema) |
| 01 Hornos | Literales a) a f) del punto 1 + relaciones entre variables |
| 02 Equipos A y B | Literales a) a c) del punto 2 + estudio de Monte Carlo |
| 03 Sistemas | Punto 3 con datos reales de software (a, b, c, d) |
| 04 Industrial | Punto 3 con datos reales de control de calidad (a, b, c, d) |
| 05 Laboratorio | **Suba cualquier archivo** (CSV, TXT, TSV, Excel, ODS, JSON, Parquet; decimales con punto o coma) o pegue datos de Excel, y aplíquele todo |
| 06 Teoría | χ², KS y AD explicadas a fondo, corrección de Lilliefors, potencia comparada y simulador interactivo |
| 07 Taller | Ejemplo ASTON y los 6 ejercicios de la guía de bondad de ajuste resueltos |

En la barra lateral se cambian α, la confianza, la semilla del punto 2 y las repeticiones de Monte Carlo.

## Estructura

```
proyecto-aula-estadistica/
├── app.py                     # Interfaz (navegación y estilos)
├── requirements.txt
├── .streamlit/config.toml     # Tema "papel y tinta"
├── src/                       # Motor estadístico (independiente de la interfaz)
│   ├── bondad.py              # χ² (método de la guía), KS, AD, Monte Carlo para el valor p
│   ├── inferencia.py          # IC, pruebas Z, t, F, proporciones, Mann–Whitney, Cp/Cpk, correlación
│   ├── descriptiva.py         # Medidas descriptivas y regla 2^k > n
│   ├── montecarlo.py          # Potencia de las pruebas y simulación de Lilliefors
│   ├── generacion.py          # Datos inventados (punto 1) y simulados (punto 2)
│   ├── hornos.py              # Cálculos del punto 1
│   ├── aplicaciones.py        # Cálculos del punto 3 (Sistemas e Industrial)
│   ├── taller.py              # Ejemplo y ejercicios de la guía
│   ├── datos.py               # Lectura de cualquier archivo subido
│   ├── graficos.py            # Gráficas Plotly (las mismas de la app y del informe)
│   ├── ui.py                  # Componentes visuales
│   └── paginas/               # Una página de la app por archivo
├── data/                      # Bases (y data/raw con los datos reales originales)
├── scripts/
│   ├── generar_datos.py       # Regenera data/
│   ├── generar_informe.py     # Calcula TODOS los números y figuras del informe
│   └── generar_excel.py       # Construye el libro de Excel con fórmulas
├── excel/Proyecto_Aula_Estadistica.xlsx
├── r/analisis_proyecto.R      # Los mismos cálculos en R / RStudio
├── figuras/                   # Figuras del informe (PDF y PNG)
└── informe/                   # informe.tex, valores.tex, tablas/ e informe.pdf
```

## Reproducibilidad

Ningún número del informe se copió a mano: todos salen de `scripts/generar_informe.py` (que escribe
`informe/valores.tex`). Para regenerar todo:

```bash
python scripts/generar_datos.py
python scripts/generar_informe.py
python scripts/generar_excel.py
cd informe && xelatex informe.tex && xelatex informe.tex
```

El Excel y el script de R leen los mismos archivos de `data/`, así que los tres entornos dan los mismos resultados.
Los datos reales de Sistemas se obtuvieron con:

```bash
git clone --bare https://github.com/streamlit/streamlit.git
git --git-dir=streamlit.git log --first-parent --since=2024-01-01 --until=2026-01-01 \
    --numstat --format='@@%H|%cI|%aI|%an|%s'
```

## Tecnologías

Python · NumPy · SciPy · pandas · Plotly · Streamlit · openpyxl · R · LaTeX (XeLaTeX)

## Referencias principales

- Fernández C., H. *Pruebas de bondad de ajuste*. Guía del curso, Universidad de Medellín.
- Montgomery, D. C. *Introduction to Statistical Quality Control*. Wiley.
- Stephens, M. A. (1974). EDF statistics for goodness of fit and some comparisons. *JASA*, 69(347).
- Lilliefors, H. W. (1967). On the Kolmogorov–Smirnov test for normality with mean and variance unknown. *JASA*, 62(318).
- Mann, H. B. y Whitney, D. R. (1947). *Annals of Mathematical Statistics*, 18(1).
