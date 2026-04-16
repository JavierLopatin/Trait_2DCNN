#!/usr/bin/env Rscript
# Generate PROSAIL-PRO LUT for Case Study 2
# Uses prosail() directly with PROSPECT-PRO parameters
# Distributions from GreenHyperSpectra real data

.libPaths("/mnt/rapidita_4T/R/libs")
library(prosail)

args <- commandArgs(trailingOnly = TRUE)
n <- ifelse(length(args) >= 1, as.integer(args[1]), 50000)

set.seed(42)
cat(sprintf("=== PROSAIL-PRO LUT (%d samples) ===\n", n))

# Parameter distributions from GreenHyperSpectra
chl  <- pmax(5,  pmin(90,  rnorm(n, 40.0, 14.4)))
car  <- pmax(0.5, pmin(25, rnorm(n, 8.4, 2.9)))
ant  <- pmax(0,   pmin(5,  rnorm(n, 1.27, 0.41)))
ewt  <- pmax(0.001, pmin(0.05, rnorm(n, 0.017, 0.015)))
lma  <- pmax(0.001, pmin(0.04, rnorm(n, 0.012, 0.009)))
lai  <- pmax(0.1, pmin(10, rnorm(n, 3.4, 1.7)))
prot <- pmax(0.0001, pmin(0.005, rnorm(n, 0.001, 0.0005)))
cbc  <- pmax(0.0001, lma - prot)
ns   <- pmax(1.2, pmin(1.8, rnorm(n, 1.5, 0.3)))

cat(sprintf("  chl:  %.1f - %.1f (mean %.1f)\n", min(chl), max(chl), mean(chl)))
cat(sprintf("  car:  %.2f - %.2f (mean %.2f)\n", min(car), max(car), mean(car)))
cat(sprintf("  ant:  %.2f - %.2f (mean %.2f)\n", min(ant), max(ant), mean(ant)))
cat(sprintf("  lai:  %.2f - %.2f (mean %.2f)\n", min(lai), max(lai), mean(lai)))

# Run simulations
cat("\nRunning PROSAIL-PRO simulations...\n")
results <- list()
t0 <- proc.time()

for (i in 1:n) {
  # In PROSPECT-PRO: lma = prot + cbc internally, so set lma=0
  r <- prosail(
    n_struct = ns[i], chl = chl[i], car = car[i], ant = ant[i],
    brown = 0.25, ewt = ewt[i], lma = 0,
    prot = prot[i], cbc = cbc[i],
    lai = lai[i], lidf_a = 57, hotspot = 0.01,
    tts = 30, tto = 0, psi = 0, rsoil = 0.8
  )
  results[[i]] <- r$rsot  # directional reflectance factor
  if (i %% 10000 == 0) {
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  %d/%d (%.0f sec)\n", i, n, elapsed))
  }
}

cat("Simulations complete.\n")

# Build reflectance matrix
refl_mat <- matrix(unlist(results), nrow = n, byrow = TRUE)
wl <- 400:2500
colnames(refl_mat) <- as.character(wl)
cat(sprintf("Reflectance: %d x %d (%.0f - %.0f nm)\n",
    nrow(refl_mat), ncol(refl_mat), min(wl), max(wl)))

# Build output
# In PROSPECT-PRO: cm = prot + cbc (lma was set to 0 in the simulation)
traits <- data.frame(
  cab = chl, car = car, anth = ant, cw = ewt,
  cm = prot + cbc, LAI = lai, cp = prot, cbc = cbc
)
output <- cbind(traits, as.data.frame(refl_mat))

outpath <- "data/GreenHyperSpectra/prosail_lut_raw_PRO.csv"
write.csv(output, outpath, row.names = FALSE)
elapsed <- (proc.time() - t0)[3]
cat(sprintf("\nSaved: %s (%d x %d) in %.0f sec\n",
    outpath, nrow(output), ncol(output), elapsed))
cat("Next: python prosail/postprocess_lut.py --input", outpath, "\n")
