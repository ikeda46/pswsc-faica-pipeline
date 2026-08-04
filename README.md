# pswsc-faica-pipeline

A pipeline for extracting calibrated, error-barred astronomical spectra from DESHIMA/ASTE position-switching (pswsc) observations, using factor-analysis + ICA (FA+ICA / "faica") to blindly extract the ON/OFF-chop-correlated signal component, rather than naive ON−OFF differencing.

## Status

Early development — plan agreed, implementation not yet started.

## Planned pipeline

1. Calibrate beam A's ON+OFF time series (calibration method still open — mini-skydip calibration is the default/simplest option, but may need to be replaced with an OFF-point-based recalibration; to be decided during implementation), apply LPF, then run FA+ICA (`faica`).
2. Identify the independent component (IC) whose time series correlates with the known ON/OFF chop pattern (a square wave).
3. Rescale that IC's time series so OFF → 0 and ON → 1, using the known ON/OFF state labels as a calibration reference (resolves ICA's inherent scale/sign ambiguity).
4. Take that IC's per-channel mixing coefficients — after the step-3 rescaling, these should correspond to η_atm(freq, elevation, pwv) × T\* per KID channel, where T\* is the source's intrinsic brightness temperature.
5. Estimate PWV at the relevant time (e.g. from OFF-point data), excluding strong spectral features, matching against η_atm.
6. Estimate the background error from OFF-state scatter, and the ON-state error similarly.
7. Divide the step-4 amplitude by η_atm(freq, elevation, pwv) (evaluated at the step-5 PWV) to get the source spectrum T\*(freq), with error bars from step 6.

Both the casatools/ATM-model-based η_atm and the empirical η_atm (see [empirical-eta-atm](https://github.com/ikeda46/empirical-eta-atm)) will be tried and compared.

## Target dataset

Development/testing target: NGC1068, obsid `20240716104712` (beam A+B, ~137s / 21,732 samples) — a short pswsc observation of the same source used in earlier (much longer, ~44 min) investigations, chosen for fast iteration.

## Dependencies

- [empirical-eta-atm](https://github.com/ikeda46/empirical-eta-atm)
- `d24_tools` (Arend Moerman's DESHIMA analysis utilities)
- `decode`, `demerge`, `dems`, `numpy`, `scipy`, `pandas`, `xarray`
