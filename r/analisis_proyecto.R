# =============================================================================
# analisis_proyecto.R — Proyecto de Aula · Estadística para Ingeniería · UdM 2026-2
# Integrantes: Juan Manuel Moreno Muñoz, David Restrepo, Juan Fernando
# Profesor: Horacio Fernández C.
#
# Reproduce en R (RStudio) los cálculos principales del proyecto, con los MISMOS
# datos que usa la aplicación en Python (carpeta data/).
#
# Uso en RStudio: abrir el proyecto, fijar el directorio de trabajo en la raíz del
# repositorio (Session > Set Working Directory) y ejecutar con Ctrl + Shift + Enter.
# Paquete opcional: nortest (ad.test y lillie.test). Si no está, se usan las
# funciones propias de este archivo.
# =============================================================================

options(digits = 6, scipen = 6)
invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "C.UTF-8")))   # acentos en los datos
alpha <- 0.05
tiene_nortest <- requireNamespace("nortest", quietly = TRUE)

titulo <- function(t) cat("\n", strrep("=", 78), "\n ", t, "\n", strrep("=", 78), "\n", sep = "")

# -----------------------------------------------------------------------------
# Funciones de bondad de ajuste (igual que en src/bondad.py)
# -----------------------------------------------------------------------------
numero_clases <- function(n) { k <- 1; while (2^k <= n) k <- k + 1; k }   # regla 2^k > n

# Une clases adyacentes hasta que todas tengan E_i >= 5 (nota de la guía)
agrupar <- function(O, P, n) {
  O <- as.numeric(O); P <- as.numeric(P); i <- 1
  while (i < length(O)) {
    if (n * P[i] < 5) { O[i] <- O[i] + O[i + 1]; P[i] <- P[i] + P[i + 1]; O <- O[-(i + 1)]; P <- P[-(i + 1)] }
    else i <- i + 1
  }
  while (length(O) > 1 && n * P[length(P)] < 5) {
    k <- length(O); O[k - 1] <- O[k - 1] + O[k]; P[k - 1] <- P[k - 1] + P[k]; O <- O[-k]; P <- P[-k]
  }
  list(O = O, P = P)
}

chi_cuadrado <- function(O, P, n, p_est, alpha = 0.05) {
  g <- agrupar(O, P, n)
  E <- n * g$P
  EP <- sum((g$O - E)^2 / E)
  k <- length(g$O); v <- k - p_est - 1
  data.frame(EP = EP, k = k, v = v, critico = if (v >= 1) qchisq(1 - alpha, v) else NA,
             valor_p = if (v >= 1) pchisq(EP, v, lower.tail = FALSE) else NA)
}

# Chi-cuadrado continua con k clases de igual ancho (guía); cdf = función de distribución
chi_continua <- function(x, cdf, p_est, alpha = 0.05) {
  n <- length(x); k <- numero_clases(n)
  b <- seq(min(x), max(x), length.out = k + 1)[2:k]
  O <- tabulate(findInterval(x, b, left.open = TRUE) + 1, nbins = k)
  P <- diff(c(0, cdf(b), 1))
  chi_cuadrado(O, P, n, p_est, alpha)
}

# Estadísticos KS (D) y Anderson–Darling (A^2), con las fórmulas del marco teórico
estadistico_ks <- function(x, cdf) {
  x <- sort(x); n <- length(x); F <- cdf(x); i <- seq_len(n)
  max(max(i / n - F), max(F - (i - 1) / n))
}
estadistico_ad <- function(x, cdf) {
  x <- sort(x); n <- length(x); F <- pmin(pmax(cdf(x), 1e-300), 1 - 1e-16); i <- seq_len(n)
  -n - sum((2 * i - 1) * (log(F) + log(1 - rev(F)))) / n
}

# Valor p por bootstrap paramétrico (corrección de Lilliefors) para normal o gamma
p_bootstrap <- function(x, modelo = c("normal", "gamma"), B = 999, semilla = 2026) {
  modelo <- match.arg(modelo); set.seed(semilla); n <- length(x)
  ajusta <- function(y) {
    if (modelo == "normal") { m <- mean(y); s <- sd(y); function(q) pnorm(q, m, s) }
    else { f <- MASS_gamma(y); function(q) pgamma(q, shape = f[1], scale = f[2]) }
  }
  genera <- function(cdf_par) {
    if (modelo == "normal") rnorm(n, mean(x), sd(x))
    else { f <- MASS_gamma(x); rgamma(n, shape = f[1], scale = f[2]) }
  }
  F0 <- ajusta(x); d0 <- estadistico_ks(x, F0); a0 <- estadistico_ad(x, F0)
  sim <- replicate(B, { y <- genera(); Fy <- ajusta(y); c(estadistico_ks(y, Fy), estadistico_ad(y, Fy)) })
  c(D = d0, p_KS = (sum(sim[1, ] >= d0) + 1) / (B + 1), A2 = a0, p_AD = (sum(sim[2, ] >= a0) + 1) / (B + 1))
}

# Máxima verosimilitud de la gamma (forma, escala) sin paquetes externos
MASS_gamma <- function(y) {
  nll <- function(par) -sum(dgamma(y, shape = exp(par[1]), scale = exp(par[2]), log = TRUE))
  m <- mean(y); v <- var(y)
  op <- optim(c(log(m^2 / v), log(v / m)), nll)
  exp(op$par)
}

# Mann–Whitney paso a paso (aproximación normal con corrección por empates)
mann_whitney <- function(x1, x2) {
  n1 <- length(x1); n2 <- length(x2); N <- n1 + n2
  r <- rank(c(x1, x2)); R1 <- sum(r[1:n1]); U1 <- R1 - n1 * (n1 + 1) / 2
  t <- table(c(x1, x2)); var_u <- n1 * n2 / 12 * ((N + 1) - sum(t^3 - t) / (N * (N - 1)))
  z <- (U1 - n1 * n2 / 2) / sqrt(var_u)
  c(R1 = R1, U1 = U1, Z = z, valor_p = 2 * pnorm(-abs(z)), P_superioridad = U1 / (n1 * n2))
}

# =============================================================================
titulo("PUNTO 1 · Hornos de cocción a gas")
# =============================================================================
hornos <- read.csv("data/hornos.csv", fileEncoding = "UTF-8")
LIE <- 2.48; LSE <- 2.92

cat("\n(a) Emisiones por marca\n")
print(aggregate(ppm ~ marca, hornos, function(x) round(c(n = length(x), media = mean(x), mediana = median(x),
      Q1 = unname(quantile(x, .25)), Q3 = unname(quantile(x, .75)), cumple400 = 100 * mean(x <= 400),
      sobre1000 = 100 * mean(x > 1000)), 2)))
boxplot(ppm ~ marca, hornos, col = c("#3E5C76", "#D9480F", "#5E7D3A", "#7A3B5E", "#D9982B"),
        main = "Emisiones por marca", ylab = "ppm", outline = TRUE)
abline(h = c(400, 1000), lty = c(2, 3), col = c("black", "#D9480F"))
print(kruskal.test(ppm ~ marca, hornos))

cat("\n(b) Potencia y capacidad (Cp, Cpk)\n")
cap <- do.call(rbind, lapply(split(hornos$potencia_kw, hornos$marca), function(x) {
  m <- mean(x); s <- sd(x)
  round(c(media = m, s = s, dentro = 100 * mean(x >= LIE & x <= LSE), Cp = (LSE - LIE) / (6 * s),
          Cpk = min(LSE - m, m - LIE) / (3 * s)), 4)
}))
print(cap)

mejor <- "Marca 2"
cat("\n(d) Intervalos del 95 % para", mejor, "\n")
x <- hornos$potencia_kw[hornos$marca == mejor]; n <- length(x)
print(t.test(x)$conf.int)                                        # media
ic_var <- c((n - 1) * var(x) / qchisq(0.975, n - 1), (n - 1) * var(x) / qchisq(0.025, n - 1))
cat("Varianza:", round(ic_var, 6), "\n")
cumple <- sum(hornos$ppm[hornos$marca == mejor] <= 400)
p_hat <- cumple / n
cat("Proporción:", round(p_hat + c(-1, 1) * qnorm(0.975) * sqrt(p_hat * (1 - p_hat) / n), 4), "\n")
cat("Normalidad de la potencia (supuesto del IC de la varianza):\n")
print(chi_continua(x, function(q) pnorm(q, mean(x), sd(x)), p_est = 2))
print(p_bootstrap(x, "normal"))
if (tiene_nortest) print(nortest::ad.test(x))

cat("\n(e) Variabilidad Marca 3 vs Marca 4\n")
print(var.test(hornos$potencia_kw[hornos$marca == "Marca 4"], hornos$potencia_kw[hornos$marca == "Marca 3"]))

cat("\n(f) Afirmación: 80 de cada 100 cumplen\n")
z <- (p_hat - 0.8) / sqrt(0.8 * 0.2 / n)
cat("Z =", round(z, 4), " valor p bilateral =", round(2 * pnorm(-abs(z)), 4),
    " unilateral (p < 0.8) =", round(pnorm(z), 4), "\n")
print(cor(hornos[, c("tiempo_uso_anios", "reparaciones", "ppm", "potencia_kw")]))
print(summary(lm(log(ppm) ~ tiempo_uso_anios, hornos))$coefficients)

# =============================================================================
titulo("PUNTO 2 · Equipos A y B (datos generados con semilla 2026 en Python)")
# =============================================================================
sim <- read.csv("data/simulacion.csv")
print(round(sapply(sim, function(x) c(media = mean(x), mediana = median(x), s = sd(x), min = min(x),
                                     max = max(x))), 3))
par(mfrow = c(2, 2))
for (v in names(sim)) hist(sim[[v]], breaks = seq(min(sim[[v]]), max(sim[[v]]), length.out = 7),
                           main = v, xlab = "ms", col = "#C9B99A", border = "white")
par(mfrow = c(1, 1))

cat("\n(a) Bondad de ajuste: normal y gamma\n")
for (v in names(sim)) {
  x <- sim[[v]]; g <- MASS_gamma(x)
  cat("\n--", v, "--\n")
  cat("Normal: "); print(chi_continua(x, function(q) pnorm(q, mean(x), sd(x)), 2))
  print(round(p_bootstrap(x, "normal"), 4))
  cat("Gamma (forma =", round(g[1], 3), ", escala =", round(g[2], 2), "): ")
  print(chi_continua(x, function(q) pgamma(q, shape = g[1], scale = g[2]), 2))
  print(round(p_bootstrap(x, "gamma", B = 499), 4))
}

cat("\n(b) t-Student y normal\n")
for (e in c("E1", "E2")) {
  a <- sim[[paste0(e, "_equipo_A")]]; b <- sim[[paste0(e, "_equipo_B")]]
  iguales <- var.test(a, b)$p.value >= alpha
  tt <- t.test(a, b, var.equal = iguales)
  z <- (mean(a) - mean(b)) / sqrt(var(a) / length(a) + var(b) / length(b))
  cat(e, ": t =", round(tt$statistic, 4), " gl =", round(tt$parameter, 2), " p(t) =", round(tt$p.value, 4),
      " | Z =", round(z, 4), " p(Z) =", round(2 * pnorm(-abs(z)), 4), "\n")
  cat("   Mann–Whitney:", round(mann_whitney(a, b), 4), "\n")
  print(wilcox.test(a, b, exact = FALSE))
}

# =============================================================================
titulo("PUNTO 3 · Datos reales")
# =============================================================================
diario <- read.csv("data/sistemas_diario.csv")
x <- diario$commits[diario$habil == "True" | diario$habil == TRUE]
cat("Commits por día hábil: media =", round(mean(x), 4), " varianza =", round(var(x), 4), "\n")
tope <- unname(floor(quantile(x, 0.99, type = 1)))
O <- c(sapply(0:(tope - 1), function(k) sum(x == k)), sum(x >= tope))
lam <- mean(x)
P_poi <- c(dpois(0:(tope - 1), lam), ppois(tope - 1, lam, lower.tail = FALSE))
cat("Poisson: "); print(chi_cuadrado(O, P_poi, length(x), 1))
r <- mean(x)^2 / (var(x) - mean(x)); pr <- r / (r + mean(x))
P_bn <- c(dnbinom(0:(tope - 1), r, pr), pnbinom(tope - 1, r, pr, lower.tail = FALSE))
cat("Binomial negativa (r =", round(r, 3), ", p =", round(pr, 3), "): "); print(chi_cuadrado(O, P_bn, length(x), 2))

commits <- read.csv("data/sistemas_commits.csv", fileEncoding = "UTF-8")
print(wilcox.test(commits$lineas_cambiadas[commits$tipo == "Funcionalidad"],
                  commits$lineas_cambiadas[commits$tipo == "Corrección"], exact = FALSE))
cat("Pearson:", round(cor(commits$archivos, commits$lineas_cambiadas), 4),
    " Spearman:", round(cor(commits$archivos, commits$lineas_cambiadas, method = "spearman"), 4), "\n")

anillos <- read.csv("data/industrial_anillos.csv")
f1 <- anillos[anillos$fase_I == "True" | anillos$fase_I == TRUE, ]
xbar <- tapply(f1$diametro_mm, f1$muestra, mean); R <- tapply(f1$diametro_mm, f1$muestra, function(v) diff(range(v)))
cat("Carta X-R: LC =", round(mean(xbar), 5), " LSC =", round(mean(xbar) + 0.577 * mean(R), 5),
    " LIC =", round(mean(xbar) - 0.577 * mean(R), 5), " R barra =", round(mean(R), 5), "\n")
s <- sd(f1$diametro_mm); m <- mean(f1$diametro_mm)
cat("Cp =", round(0.1 / (6 * s), 3), " Cpk =", round(min(74.05 - m, m - 73.95) / (3 * s), 3), "\n")
if (tiene_nortest) print(nortest::ad.test(f1$diametro_mm))
print(t.test(f1$diametro_mm, anillos$diametro_mm[!(anillos$fase_I == "True" | anillos$fase_I == TRUE)]))

jugo <- read.csv("data/industrial_jugo.csv")
j1 <- jugo[(jugo$fase_I == "True" | jugo$fase_I == TRUE) & !(jugo$muestra %in% c(15, 23)), ]
j2 <- jugo[!(jugo$fase_I == "True" | jugo$fase_I == TRUE), ]
pbar <- sum(j1$no_conformes) / (50 * nrow(j1))
cat("Carta p: p barra =", round(pbar, 4), " LSC =", round(pbar + 3 * sqrt(pbar * (1 - pbar) / 50), 4), "\n")
print(prop.test(c(sum(j1$no_conformes), sum(j2$no_conformes)), c(50 * nrow(j1), 50 * nrow(j2)), correct = FALSE))
