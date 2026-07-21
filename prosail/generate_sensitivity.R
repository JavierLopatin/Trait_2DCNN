#!/usr/bin/env Rscript
# Theoretical spectral sensitivity via PROSAIL forward simulation.
#
# For each of the 8 GreenHyperSpectra traits, sweep that trait across its range
# while holding all other parameters at their mean (baseline), and record the
# resulting directional reflectance (rsot, 400:2500 nm). The spread of these
# curves per band gives a continuous, physically-grounded "theoretical
# sensitivity" that replaces the hand-drawn binary absorption regions in the
# spectral-importance analysis.
#
# Fixed params and baseline means match prosail/generate_lut.R (Cherif 2025).
# Output: data/GreenHyperSpectra/prosail_sensitivity_raw.csv (long format).

.libPaths("/mnt/rapidita_4T/R/libs")
suppressMessages(library(prosail))

args <- commandArgs(trailingOnly = TRUE)
n_steps <- ifelse(length(args) >= 1, as.integer(args[1]), 25L)

# --- Baseline (all traits at their GHS mean) --------------------------------
base <- list(
  n_struct = 1.5, chl = 40.0, car = 8.4, ant = 1.27, brown = 0.25,
  ewt = 0.017, lma = 0, prot = 0.001, cbc = 0.011, lai = 3.4,
  lidf_a = 57, hotspot = 0.01, tts = 30, tto = 0, psi = 0, rsoil = 0.8
)

# --- Per-trait sweep ranges (min, max) from generate_lut.R clamps -----------
# Each entry: the PROSAIL argument(s) to vary and the [min, max] range.
# GHS trait -> PROSPECT-PRO parameter mapping.
sweeps <- list(
  Cab  = list(param = "chl",  lo = 5,      hi = 90),
  Car  = list(param = "car",  lo = 0.5,    hi = 25),
  Anth = list(param = "ant",  lo = 0,      hi = 5),
  Cw   = list(param = "ewt",  lo = 0.001,  hi = 0.05),
  Cm   = list(param = "cbc",  lo = 0.001,  hi = 0.039),  # total dry matter (cbc dominates cm)
  LAI  = list(param = "lai",  lo = 0.1,    hi = 10),
  Cp   = list(param = "prot", lo = 0.0001, hi = 0.005),
  Cbc  = list(param = "cbc",  lo = 0.0001, hi = 0.039)
)

wl <- 400:2500
run_one <- function(p) {
  do.call(prosail, p)$rsot
}

rows <- list()
k <- 0
t0 <- proc.time()
for (trait in names(sweeps)) {
  sw <- sweeps[[trait]]
  values <- seq(sw$lo, sw$hi, length.out = n_steps)
  cat(sprintf("Sweeping %-4s via %-4s: %d steps in [%g, %g]\n",
              trait, sw$param, n_steps, sw$lo, sw$hi))
  for (v in values) {
    p <- base
    p[[sw$param]] <- v
    refl <- run_one(p)
    k <- k + 1
    rows[[k]] <- c(trait = trait, param = sw$param, param_value = v, refl)
  }
}

# --- Assemble long data frame -----------------------------------------------
refl_mat <- do.call(rbind, lapply(rows, function(r) as.numeric(r[-(1:3)])))
colnames(refl_mat) <- as.character(wl)
meta <- data.frame(
  trait       = sapply(rows, function(r) r[["trait"]]),
  param       = sapply(rows, function(r) r[["param"]]),
  param_value = sapply(rows, function(r) as.numeric(r[["param_value"]])),
  stringsAsFactors = FALSE
)
out <- cbind(meta, as.data.frame(refl_mat))

outpath <- "data/GreenHyperSpectra/prosail_sensitivity_raw.csv"
write.csv(out, outpath, row.names = FALSE)
elapsed <- (proc.time() - t0)[3]
cat(sprintf("\nSaved: %s (%d rows x %d cols) in %.0f sec\n",
            outpath, nrow(out), ncol(out), elapsed))
cat("Next: python prosail/postprocess_sensitivity.py\n")
