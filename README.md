# ML for Option Pricing

Neural networks that learn the **Black–Scholes pricing function** of a European call,
with **uncertainty quantification** (deep ensembles and Monte-Carlo dropout).

> Project from the *Machine Learning & Econometrics* course (NTHU, 2025), re-implemented as a
> clean, tested and reproducible codebase.

## Why synthetic data?

The original plan was to learn directly from **TSMC option chains**. Free sources turned out to
be incomplete (missing quotes, inconsistent strikes and maturities, unreliable implied
volatilities), far too noisy for deep learning. The project therefore follows the
"learning the pricing formula" approach: generate clean options in a realistic parameter
domain, price them with Black–Scholes, and train networks to approximate the mapping.
Because the true price is known, model error can be measured exactly.

## Problem

Inputs: moneyness `m = S0/K`, time to maturity `τ`, risk-free rate `r`, volatility `σ`.
Target: scaled call price `π = C/K`.

| Parameter | Range |
|---|---|
| `m` | 0.4 – 1.6 |
| `τ` (years) | 0.2 – 1.1 |
| `r` | 0.02 – 0.10 |
| `σ` | 0.01 – 1.00 |

## Models

| Model | Description |
|---|---|
| MLP | 4 → 50 → 50 → 1, dropout 0.1 |
| Residual | MLP with a skip-connection |
| Highway | MLP with a learnable gate |
| Deep ensemble | N networks with different seeds → mean and spread |
| MC dropout | Dropout kept active at inference → predictive distribution |

## Results

All numbers are measured against the **exact Black–Scholes price** (`π = C/K`), seed 42,
5000/1000/1000 train/val/test split, scaler fit on train only. Each table states which split it
uses; the test split is used only for the final MC-dropout evaluation, never to choose hyperparameters.

### 1. The original optimiser under-fits (validation split, MLP, 100 epochs)

| config | val RMSE | val MAE | val median rel. err | best epoch | final train MSE |
|---|---|---|---|---|---|
| SGD lr=1e-4 (original report) | 1.462e-01 | 1.186e-01 | 4.417e-01 | 99 | 2.491e-02 |
| **Adam lr=1e-3** | **8.593e-03** | **6.588e-03** | **3.113e-02** | 16 | 3.559e-04 |
| SGD lr=1e-3 (crossed) | 4.296e-02 | 3.394e-02 | 1.381e-01 | 99 | 6.151e-03 |
| Adam lr=1e-4 (crossed) | 1.234e-02 | 9.358e-03 | 4.363e-02 | 57 | 8.792e-04 |
| Baseline (train mean) | 2.180e-01 | 1.884e-01 | 6.133e-01 | - | - |

With the report's SGD configuration the network under-fits: it ends ~33 % below the constant
baseline and its best epoch is the last one (still improving). Adam lr=1e-3 reduces the RMSE by
~17x. The crossed runs show both factors matter: SGD with a 10x larger lr is still 5x worse than
Adam lr=1e-3, and Adam with the small lr is 1.4x worse. All following experiments use Adam lr=1e-3.

### 2. Deep ensemble vs single network (validation split, N=5, MLP)

| model | RMSE | MAE | median rel. err |
|---|---|---|---|
| single model (mean over the 5 members) | 8.106e-03 | 6.276e-03 | 2.501e-02 |
| **ensemble (N=5)** | **5.769e-03** | **4.245e-03** | **1.554e-02** |

Averaging 5 networks cuts the RMSE by ~29 % and the median relative error from 2.5 % to 1.6 %.

### 3. Monte-Carlo dropout (test split, T=100 passes)

| RMSE (deterministic) | RMSE (MC mean) | MAE (MC) | median rel. err (MC) | mean std | coverage 95 % |
|---|---|---|---|---|---|
| 8.347e-03 | 1.267e-02 | 9.618e-03 | 4.879e-02 | 2.663e-02 | 1.000 |

The nominal 95 % interval covers **100 %** of test points: MC dropout (p = 0.1) is
**over-conservative** here, and its mean is less accurate than the deterministic forward pass.

### 4. Is uncertainty larger at the edges of the domain? (grid over (m, τ), σ=0.3, r=0.06)

Reported as edge/interior ratios (1.00 = no edge effect) per border (`m_low`, `m_high`,
`tau_low`, `tau_high`) for the std, the absolute error and the relative error, as
mean ± std over 3 seeds (42, 142, 242), plus Spearman(std, err). Generate the exact table with
`python -m option_pricing.uncertainty_maps`; it is written to `outputs/` (git-ignored).

| method | metric | m_low | m_high | tau_low | tau_high | Spearman(std, err) |
|---|---|---|---|---|---|---|
| deep ensemble (N=5) | std ratio | 0.83 ± 0.16 | 1.54 ± 0.22 | 1.25 ± 0.09 | 1.09 ± 0.11 | 0.063 ± 0.217 |
| deep ensemble (N=5) | err ratio | 0.70 ± 0.10 | 2.73 ± 0.57 | 1.46 ± 0.21 | 0.97 ± 0.26 |  |
| deep ensemble (N=5) | relerr ratio | n/a | 0.36 ± 0.08 | 0.99 ± 0.26 | 1.13 ± 0.16 |  |
| MC dropout (T=100) | std ratio | 0.82 ± 0.44 | 2.34 ± 0.12 | 1.07 ± 0.01 | 1.16 ± 0.08 | 0.413 ± 0.153 |
| MC dropout (T=100) | err ratio | 0.44 ± 0.24 | 1.65 ± 0.30 | 0.96 ± 0.17 | 1.36 ± 0.37 |  |
| MC dropout (T=100) | relerr ratio | n/a | 0.30 ± 0.07 | 1.30 ± 0.44 | 1.22 ± 0.45 |  |

`n/a` for `relerr` at `m_low`: with σ=0.3, r=0.06 the true price is below 0.01 (K = 1) for
m ≤ 0.46 (≈ 0.0002 at m=0.4, ≈ 0.0008 at m=0.46, both at the most favourable τ=1.1; this holds
for σ=0.3, r=0.06 only), so no point of that band has a defined relative error
(relative error is only computed where price ≥ 0.01, to avoid dividing by ~0). Absolute-error
and std ratios are unaffected.

Note: the MC dropout map uses **member 0 of the ensemble** (same network), so the
ensemble vs MC dropout contrast should be read with that caveat.

![Ensemble uncertainty map](docs/figures/uncertainty_map_ensemble_sigma0.3_r0.06.png)
![MC dropout uncertainty map](docs/figures/uncertainty_map_mc_dropout_sigma0.3_r0.06.png)

### 5. Towards market data

TSMC daily prices can be downloaded reproducibly (`scripts/download_tsmc.py`, extra `market`) and
`option_pricing.historical_vol` compares rolling realised volatility (21/63 days) with the `σ`
range of the synthetic domain.

### Limitations

- Tables 1–3 use a single seed (42) and a single 5000/1000/1000 split: no confidence intervals. Section 4 is averaged over 3 seeds.
- Tables 1–2 are on the validation split, table 3 on the test split; they are not directly comparable.
- Residual and Highway architectures are implemented (`models.py`) but only the MLP is reported above.

## Reproduce the figures

```bash
pip install -e ".[dev]"          # 1. install
pytest -q                        # 2. run the tests (< 30 s)
python -m option_pricing.uncertainty_maps   # 3. section 4 only: train ensemble + MC dropout, write figures to docs/figures/
```

Other entry points: `python -m option_pricing.optimizer_study` (table 1),
`python -m option_pricing.uncertainty` (table 2), `python -m option_pricing.mc_dropout --split test` (table 3),
and `notebooks/demo.ipynb` for a guided tour.

## References

- F. Black, M. Scholes (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81, 637–654.
- L. van Mieghem, A. Papapantoleon, J. Papazoglou-Hennig (2023). *Machine learning for option pricing: an empirical investigation of network architectures*. [arXiv:2307.07657](https://arxiv.org/abs/2307.07657)
- J. Hull (2020). *Machine Learning in Business: An Introduction to the World of Data Science*.
