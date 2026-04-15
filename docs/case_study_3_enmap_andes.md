# Case Study 3: EnMAP Satellite Validation in the Mediterranean Andes

## Status: DRAFT — Review after completing Case Study 2 experiments

## Motivation

Case Studies 1-2 demonstrate the advantage of 2D spectral representations using
field-spectrometer data (GreenHyperSpectra). Case Study 3 would demonstrate
**scalability from field to satellite** in a real, ecologically meaningful context:
the Mediterranean Andes of central Chile, a biodiversity hotspot.

This is the missing piece for high-impact publication: showing the method works
not just with lab/field spectrometers but with actual satellite imagery over
complex terrain.

## Study Area

- **Location**: Mapocho River basin, Mediterranean Andes, central Chile
- **Coordinates**: -33.42 to -33.39 lat, -70.38 to -70.33 lon
- **Elevation**: 1200–2400 m a.s.l.
- **Climate**: Semi-arid Mediterranean, 14°C mean annual, ~360 mm precipitation
- **Vegetation**: Andean sclerophyllous forest and scrubland
  (Lithraea caustica, Quillaja saponaria, Kageneckia oblonga/angustifolia)
- **41 species, 32 families**

## Field Data (Cerda-Paredes et al., BDJ paper)

- **133 plots**, fractal hierarchical design (100–900 m scales)
- **Plot size**: circular, radius 6 m (113.1 m²)
- **5 fractals** along the elevational gradient

### Measured Traits

| Trait | PROSAIL equivalent | Available |
|-------|-------------------|-----------|
| Leaf mass per area (LMA) | Cm | Yes |
| Water content | Cw | Yes |
| Chlorophyll content (SPAD) | Cab (needs calibration) | Yes |
| Leaf area | — | Yes |
| Leaf thickness | — | Yes |
| Leaf dry matter content (LDMC) | — | Yes |
| Wood density | — | Yes |

**3 traits overlap directly with PROSAIL**: LMA (Cm), water content (Cw),
chlorophyll (Cab). These are also the most reliably predicted traits in
Case Studies 1-2.

**Note**: SPAD values need calibration to Cab (µg/cm²). A SPAD-Cab calibration
model specific to the species or a general relationship would be needed.

## Satellite Data

- **Sensor**: EnMAP (Environmental Mapping and Analysis Program)
- **Level**: L2A (atmospherically corrected surface reflectance)
- **Topographic correction**: Already applied (type TBD — need to verify:
  SCS+C, Minnaert, or similar)
- **Spectral range**: 420–2450 nm, 224 bands (VNIR + SWIR)
- **Spatial resolution**: 30 m
- **Coverage**: Covers the study area

### Band Alignment with PROSAIL LUT

EnMAP has 224 bands (not 1721 like field spectrometers). Two options:

1. **Resample PROSAIL LUT to EnMAP bands**: Apply EnMAP spectral response
   functions to the 1nm PROSAIL output → 224 bands. This is more physically
   correct.

2. **Interpolate EnMAP to 1nm**: Less ideal, introduces interpolation artifacts
   in water absorption regions.

**Recommendation**: Option 1 — generate PROSAIL at 1nm, then convolve with
EnMAP spectral response functions (SRFs) before the reshape transform.

## Experimental Design

### Approach A: PROSAIL standard (if EnMAP has topographic correction)

If EnMAP L2A already normalizes reflectance to flat terrain:
- PROSAIL with fixed geometry (tts=30°, tto=0°, psi=0°) is compatible
- Same LUT generation pipeline as Case Study 2
- Just need to resample to EnMAP bands

### Approach B: PROSAIL with variable geometry (if correction is partial)

If residual topographic effects remain:
- Calculate local solar/view angles per pixel from DEM + EnMAP metadata
- Generate LUT varying tts, tto, psi according to terrain geometry
- This captures ~80% of topographic effect without implementing 4SAILT

### Approach C: 4SAILT (full topographic RTM)

Shi & Xiao (2021) 4SAILT model:
- Adds slope, aspect, and sky view factor (F_sky) to 4SAIL
- Includes gravitropism (leaves grow vertical regardless of slope)
- Validated against DART: R² > 0.99
- Errors from ignoring topography: up to 60% for slopes >40°
- **No public implementation** — would need to code from the paper
- **Recommended only if**: Approaches A/B fail, or if targeting a journal
  where RTM novelty is a key contribution

### Proposed Experiments

| # | Train data | Test data | Method |
|---|-----------|-----------|--------|
| A1 | PROSAIL LUT (EnMAP bands) | EnMAP pixels with field validation | Supervised 2D (Reshape) |
| A2 | PROSAIL LUT (EnMAP bands) | EnMAP pixels | MAE-2D pretrained + PROSAIL fine-tune |
| A3 | GHS real + PROSAIL | EnMAP pixels | Transfer: field-spec pretrain → satellite |

### Validation Strategy

- Extract EnMAP pixels corresponding to the 133 field plots
- Average spectra within each plot footprint (considering plot size vs pixel size)
- Compare predicted traits (LMA, Cw, Cab) with field measurements
- Metrics: R², RMSE, nRMSE per trait
- **Challenge**: 30m pixels vs 6m radius plots — mixed pixels likely.
  May need to select "pure" vegetation pixels using NDVI threshold.

## Key Questions to Resolve Before Implementation

1. **What topographic correction was applied to EnMAP L2?**
   - If SCS+C or similar normalization → Approach A (simple)
   - If only illumination angle correction → Approach B or C

2. **SPAD to Cab calibration?**
   - Need species-specific or general SPAD-Cab relationship
   - Literature values exist for sclerophyllous species?

3. **Plot-pixel correspondence?**
   - 6m radius plot ≈ 113 m² vs 30m pixel ≈ 900 m²
   - Pixel is ~8x larger than plot → mixed signal
   - Need vegetation fraction map or NDVI filtering

4. **EnMAP acquisition date vs field campaign date?**
   - Temporal mismatch affects trait values (phenology)
   - Mediterranean climate: strong seasonal variation

5. **Number of validation samples?**
   - 133 plots, but after filtering for pure pixels and trait availability,
     could be <100 — may need to aggregate at fractal level

## Impact Assessment

### Without Case Study 3 (current paper)
- Target: RSE, MEE, or ISPRS — strong methods paper
- Message: "2D transforms improve spectral trait prediction"

### With Case Study 3
- Target: RSE, New Phytologist, or potentially higher
- Message: "2D transforms + pretraining enable satellite-scale trait mapping
  in complex terrain without field-labeled training data"
- The "zero real labels" angle (PROSAIL → satellite) is novel and impactful

### Effort Estimate
- Band alignment + LUT resampling: ~1 day
- EnMAP data preparation: ~1 day
- Pixel extraction + validation setup: ~1 day
- Training + evaluation: ~1 day
- **Total: ~4-5 days additional work** (assuming EnMAP data is ready)

## References

- Shi & Xiao (2021) — 4SAILT: improved 4SAIL for sloping terrain. IEEE TGRS.
- Cerda-Paredes et al. — Mediterranean plant diversity dataset (BDJ, in prep)
- Guanter et al. (2015) — EnMAP mission description
- Storch et al. (2023) — EnMAP L2A processing chain
