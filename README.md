# Replication – Decomposing US Economic Fluctuations: A Trend–Cycle Approach (Fosso, 2025)

This repository replicates and critically assesses the methodology and main results of Luca Fosso (2025), *“Decomposing US economic fluctuations: a trend-cycle approach”*, ECB Working Paper No. 3138.

It accompanies the project report *Macroeconometrics: Advanced Time-Series Analysis* (2026), which extends the original analysis to recent data and evaluates the robustness of the Trend–Cycle VAR (TC‑VAR) framework.

> Original paper: [ECB Working Paper 3138](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3138~13c472834e.en.pdf)

---

## Overview

The goal of this project is to:

- Replicate Fosso’s decomposition of US macroeconomic time series into **trend** (permanent) and **cycle** (transitory) components using a **Trend–Cycle VAR (TC‑VAR)**.
- Recover key unobserved states:
  - Trend output growth \( \Delta g_t^* \)
  - Trend inflation \( \pi_t^* \)
  - The natural rate of interest \( r_t^* \)
  - Gaps for output, inflation, inflation expectations, nominal interest rate, and commodity prices.
- Compare the replicated estimates to the figures reported in Fosso (2025).
- Discuss identification, prior sensitivity, and conceptual limitations of the framework.

---

## Data

The dataset closely follows Fosso (2025) and covers US quarterly data:

- Real GDP growth (annualized QoQ)
- PCE inflation (annualized QoQ)
- 3‑month Treasury yield (level)
- 1‑year‑ahead inflation expectations
- Commodity price index (CRB‑type index, proxied by US crude oil price pre‑1994)

**Sample:**

- Pre‑sample: **1954Q1–1959Q4** (used to calibrate priors)
- Estimation sample: **1960Q1–2025Q4**

Main data sources:

- FRED (GDP, PCE, Treasury yield)
- Cleveland Fed (inflation expectations)
- Investing.com (commodity index / crude oil proxy)

---

## Model and Methodology

The empirical framework is a **Trend–Cycle VAR (TC‑VAR)**:

- Each observed variable \( y_t \) is decomposed into a **trend** \( y_t^* \) and a **cycle** \( \tilde{y}_t \):
  \[
  y_t = \Lambda^* y_t^* + \tilde{y}_t
  \]
- Trends follow a multivariate **random walk**, capturing low‑frequency movements and structural changes.
- Cyclical components follow a stationary **VAR(1)** process.
- **Orthogonality assumption**: trend and cycle shocks are restricted to be uncorrelated, allowing a sharp separation between permanent and transitory components.
- Long‑run co‑movements are pinned down by New‑Keynesian theory via the **Euler equation**, imposing structure on the loading matrix \( \Lambda^* \) and linking trend growth, trend inflation, preferences, and the natural nominal rate.

### State-space representation and estimation

- The model is written in **state-space form** and estimated with:
  - **Kalman filter / smoother**
  - **Durbin & Koopman (2002) simulation smoother** for latent states
- Estimation is **Bayesian**, using a **Gibbs sampler**:
  - Conservative priors on trend innovation variances (Inverse‑Wishart), adjusted in the replication to match the volatility of published trends.
  - Minnesota‑type priors for the cycle VAR, adapted to enforce stationarity (mean‑reverting cycles).
  - Long MCMC run with burn‑in; posterior summaries reported for trends, gaps, and \( r_t^* \).

---

## Repository Structure

Layout for this project is:

```text
.
├── data/
│   ├── raw/           # Original / downloaded series
│   └── processed/     # Transformed data (growth rates, aligned sample)
├── src/
│   ├── tcvar_model.py # State-space, priors, Gibbs sampler
│   ├── data_prep.py   # Data loading and transformations
│   ├── run_tcvar.py   # Main script to estimate the model 
│   └── plots.py       # Scripts to reproduce figures
├── models/
│   |──tcvar_ar1_model.plk #Model obtained from run_tc_var.py
    └──tcvar_ar1_results.plk #Results obtained from run_tc_var.py
    
├── outputs/
│   └── figures/       # Replicated figures (trends, gaps, r*)
│ 
├── requirements.txt
└── README.md
