import numpy as np
import pandas as pd
import pickle
import warnings
from tc_var_model import TrendCycleVAR_AR1

warnings.filterwarnings("ignore")

# Data loading and preparation
data = pd.read_csv(
    "data\\processed\\ecb_estimation_prepared.csv", index_col=0, parse_dates=True
)
prior_raw = pd.read_csv(
    "data\\processed\\ecb_prior_prepared.csv", index_col=0, parse_dates=True
)

zlb_mask = data.iloc[:, 2] < 0.25
data_copy = data.copy()
data_copy.loc[zlb_mask, "3-Month Treasury Bill (%)"] = np.nan
data_values = data_copy.values
dates = data_copy.index

prior = prior_raw.copy()
prior.iloc[:, 0] = prior.iloc[:, 0].fillna(method="ffill").fillna(3.0)


# RUN AR(1) MODEL

# Beaware that this might take a while to run (several hours)
model = TrendCycleVAR_AR1(data_values, dates, prior, n_trends=4, p_lags=1, sigma=1.0)
model.fit(n_iter=150_000, n_burn=25_000)

# Save the model and data
with open("tcvar_ar1_model.pkl", "wb") as f:
    pickle.dump(model, f)

trends = np.array(model.store_trends)
cycles = np.array(model.store_cycles)

results = {
    "trends": trends,
    "cycles": cycles,
    "Sigma_star": model.Sigma_star,
    "Sigma_tilde": model.Sigma_tilde,
    "Phi": model.Phi,
    "dates": dates,
    "data_values": data_values,
}

with open("models\\tcvar_ar1_results.pkl", "wb") as f:
    pickle.dump(results, f)
