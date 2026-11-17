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

_Coming soon_: RMSE / relative error per architecture, error by moneyness bucket,
uncertainty heatmaps and interval coverage.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

## References

- F. Black, M. Scholes (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81, 637–654.
- L. van Mieghem, A. Papapantoleon, J. Papazoglou-Hennig (2023). *Machine learning for option pricing: an empirical investigation of network architectures*. [arXiv:2307.07657](https://arxiv.org/abs/2307.07657)
- J. Hull (2020). *Machine Learning in Business: An Introduction to the World of Data Science*.
