#!/usr/bin/env Rscript
# Generate PROSAIL-PRO Look-Up Table for Case Study 2
#
# Uses jbferet/prosail R package with PROSPECT-PRO + 4SAIL
# Parameters based on Cherif et al. (2025) Table 13 + ATBD distributions
#
# Usage:
#   Rscript prosail/generate_lut.R [n_samples] [output_file]
#   Rscript prosail/generate_lut.R 50000 data/GreenHyperSpectra/prosail_lut.csv

library(prosail)

args <- commandArgs(trailingOnly = TRUE)
n_samples <- ifelse(length(args) >= 1, as.integer(args[1]), 50000)
output_file <- ifelse(length(args) >= 2, args[2],
                      "data/GreenHyperSpectra/prosail_lut_raw.csv")

cat("=== PROSAIL-PRO LUT Generation ===\n")
cat(sprintf("  Samples: %d\n", n_samples))
cat(sprintf("  Output: %s\n", output_file))

# ---------------------------------------------------------------
# 1. Define parameter distributions
# ---------------------------------------------------------------
# Based on ATBD (Weiss et al. 2020) + PROSPECT-PRO additions
# Fixed params from Cherif 2025 Table 13:
#   Brown=0.25, Ns=1.5, LIDF=5, psoil=0.8, hspot=0.01
#   tto=0, tts=30, psi=0

# Use ATBD as base and customize for PROSPECT-PRO
# ATBD ranges (Table 5 from S2 Toolbox ATBD V2.1):
#   Cab: 20-90, mode=45, sd=30, Gaussian
#   Cdm: 0.003-0.011, mode=0.005, sd=0.005, Gaussian
#   Cw_rel: 0.60-0.85, mode=0.75, sd=0.08, Gaussian
#   N: 1.2-1.8, mode=1.5, sd=0.3, Gaussian
#   LAI: 0-15, mode=2, sd=3, Gaussian (log-normal modified)
#   ALA: 30-80, mode=60, sd=30, Gaussian

# Generate ATBD-based input parameters
cat("Generating input parameters (ATBD-based)...\n")

# We'll use get_input_prosail with ATBD distributions as starting point
# then add PROSPECT-PRO specific params
input_prosail <- get_input_prosail(
  nbSamples = n_samples,
  atbd = TRUE
)

# Override fixed params to match Cherif 2025 Table 13
input_prosail$brown <- rep(0.25, n_samples)
input_prosail$N <- runif(n_samples, min = 1.2, max = 1.8)

# Add PROSPECT-PRO specific parameters based on real data statistics
# Car: correlated with Cab (r=0.77), range 1.2-40.4, mean=8.4, sd=2.9
# Using relationship: Car ~ Cab/5 + noise (from literature)
input_prosail$car <- pmax(0.5, input_prosail$CHL / 5 +
                          rnorm(n_samples, 0, 2))

# Anth: range 0.56-2.98, mean=1.27, sd=0.41
input_prosail$ant <- pmax(0, rnorm(n_samples, mean = 1.27, sd = 0.5))

# Protein content (Cp): range 0-0.005, mean=0.001, sd=0.0005
input_prosail$prot <- pmax(0, rnorm(n_samples, mean = 0.001, sd = 0.0005))

# CBC = Cm - Cp (carbon-based constituents)
# Cm is already in input_prosail as LMA (but in different units)
# In PROSPECT-PRO: Cm = Cp + Cbc
# We need to ensure Cbc = Cm - Cp >= 0
input_prosail$cbc <- pmax(0, input_prosail$LMA - input_prosail$prot)

# Fixed geometry (Cherif 2025)
input_prosail$tts <- rep(30, n_samples)
input_prosail$tto <- rep(0, n_samples)
input_prosail$psi <- rep(0, n_samples)

# Hotspot
input_prosail$hotspot <- rep(0.01, n_samples)

cat("Input parameters generated.\n")
cat(sprintf("  Cab range: %.1f - %.1f\n", min(input_prosail$CHL), max(input_prosail$CHL)))
cat(sprintf("  Car range: %.2f - %.2f\n", min(input_prosail$car), max(input_prosail$car)))
cat(sprintf("  Anth range: %.2f - %.2f\n", min(input_prosail$ant), max(input_prosail$ant)))
cat(sprintf("  LAI range: %.2f - %.2f\n", min(input_prosail$LAI), max(input_prosail$LAI)))
cat(sprintf("  Cw range: %.4f - %.4f\n", min(input_prosail$EWT), max(input_prosail$EWT)))

# ---------------------------------------------------------------
# 2. Run PROSAIL simulations
# ---------------------------------------------------------------
cat("\nRunning PROSAIL-PRO simulations...\n")

lut <- generate_lut_prosail(input_prosail = input_prosail)

cat("Simulations complete.\n")
cat(sprintf("  Reflectance matrix: %d x %d\n",
            nrow(lut$Reflectance), ncol(lut$Reflectance)))

# ---------------------------------------------------------------
# 3. Build output dataframe
# ---------------------------------------------------------------
# Extract reflectance (wavelengths as columns)
refl <- as.data.frame(lut$Reflectance)

# Get wavelength names
wl <- as.numeric(colnames(refl))
cat(sprintf("  Wavelength range: %.0f - %.0f nm (%d bands)\n",
            min(wl), max(wl), length(wl)))

# Build trait columns matching GreenHyperSpectra format
traits <- data.frame(
  cab = input_prosail$CHL,
  car = input_prosail$car,
  anth = input_prosail$ant,
  cw = input_prosail$EWT,
  cm = input_prosail$LMA,
  LAI = input_prosail$LAI,
  cp = input_prosail$prot,
  cbc = input_prosail$cbc
)

# Combine traits + reflectance
output <- cbind(traits, refl)

# ---------------------------------------------------------------
# 4. Save raw LUT (post-processing done in Python)
# ---------------------------------------------------------------
cat(sprintf("\nSaving to %s...\n", output_file))
write.csv(output, output_file, row.names = FALSE)
cat(sprintf("Saved: %d samples x %d columns\n", nrow(output), ncol(output)))
cat("Done! Run prosail/postprocess_lut.py next.\n")
