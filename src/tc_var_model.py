import numpy as np
import pandas as pd
from scipy.stats import invwishart
from numpy.linalg import inv, LinAlgError
import warnings

warnings.filterwarnings("ignore")

# TCVAR


class TrendCycleVAR_AR1:
    def __init__(self, data, dates, prior_df, n_trends=4, p_lags=1, sigma=1.0):
        self.data = data
        self.dates = dates
        self.T, self.n = data.shape
        self.r = n_trends
        self.p = p_lags
        self.sigma = sigma
        # The definition of lambda depends on the order of the columns in data
        self.Lambda_star = np.zeros((self.n, self.r))
        self.Lambda_star[0, 0] = 1.0
        self.Lambda_star[1, 1] = 1.0
        self.Lambda_star[2, 0] = sigma
        self.Lambda_star[2, 1] = 1.0
        self.Lambda_star[2, 2] = 1.0
        self.Lambda_star[3, 1] = 1.0
        self.Lambda_star[4, 3] = 1.0

        self.y0_mean = prior_df.iloc[-1].values
        self.setup_priors(prior_df)
        self.initialize_states()

    def setup_priors(self, prior_df):
        self.kappa_star = 100
        target_variance = 0.1  # Here is a change from the article : we had to do so to have something reasonable, otherwise it was too flat.
        Sigma_star_mode = np.eye(self.r) * target_variance
        self.scale_prior_star = (self.kappa_star + self.r + 1) * Sigma_star_mode
        self.Sigma_star = Sigma_star_mode.copy()

        self.kappa_tilde = self.n + 2

        # Variance calculation
        prior_vals = prior_df.fillna(method="ffill").fillna(method="bfill").values
        prior_diff = np.diff(prior_vals, axis=0)
        cycle_vars_prior = np.nanvar(prior_diff, axis=0)

        data_temp = pd.DataFrame(self.data)
        data_filled = data_temp.fillna(method="ffill").fillna(method="bfill").values
        data_diff = np.diff(data_filled, axis=0)
        cycle_vars_data = np.nanvar(data_diff, axis=0)

        cycle_vars = np.where(
            cycle_vars_prior > 1e-3, cycle_vars_prior, cycle_vars_data
        )
        self.Sigma_tilde = np.diag(cycle_vars)
        self.scale_prior_tilde = (self.kappa_tilde + self.n + 1) * self.Sigma_tilde

        # Minnesota Prior
        self.Phi_bar = np.zeros((self.n, self.n * self.p))
        overall_tightness = 0.2
        self.Omega_phi = np.zeros((self.n, self.n * self.p, self.n * self.p))

        for i in range(self.n):
            sigma_sq_i = cycle_vars[i]
            for k in range(self.p):
                lag = k + 1
                for j in range(self.n):
                    sigma_sq_j = cycle_vars[j]
                    idx = k * self.n + j
                    if i == j:
                        var = (overall_tightness / (lag**2)) ** 2
                    else:
                        var = (
                            overall_tightness * sigma_sq_i / sigma_sq_j / (lag**2)
                        ) ** 2
                    self.Omega_phi[i, idx, idx] = var + 1e-8

    def initialize_states(self):
        self.y_star = np.zeros((self.T, self.r))
        self.y_tilde = np.zeros((self.T, self.n))
        for i in range(self.n):
            if i < self.r:
                series = (
                    pd.Series(self.data[:, i])
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )
                self.y_star[:, i] = (
                    series.rolling(window=40, min_periods=1, center=True).mean().values
                )
        self.Phi = self.Phi_bar.copy()

    def simulation_smoother(self):
        n_state_lags = max(self.p, 2)
        d_state = self.r + self.n * n_state_lags

        # F (Transition)
        F = np.zeros((d_state, d_state))
        F[: self.r, : self.r] = np.eye(self.r)

        # Bloc 1: VAR Equation
        F[self.r : self.r + self.n, self.r : self.r + self.n * self.p] = self.Phi

        if n_state_lags > 1:
            shift_rows = self.n * (n_state_lags - 1)
            F[
                self.r + self.n : self.r + self.n + shift_rows,
                self.r : self.r + shift_rows,
            ] = np.eye(shift_rows)

        # Q (Covariance)
        Q = np.zeros((d_state, d_state))
        Q[: self.r, : self.r] = self.Sigma_star
        Q[self.r : self.r + self.n, self.r : self.r + self.n] = self.Sigma_tilde

        # H (Mesure) ---
        H = np.zeros((self.n, d_state))
        H[:, : self.r] = self.Lambda_star

        # Mapping Cycles
        # GDP (0) & Commo (4) : Diff = 4 * (Cycle_t - Cycle_t-1)
        # Indices : Cycle_t starts at r, Cycle_t-1 starts at r+n
        H[0, self.r] = 4.0
        H[0, self.r + self.n] = -4.0
        H[4, self.r + 4] = 4.0
        H[4, self.r + self.n + 4] = -4.0

        # Others : Level = Cycle_t
        H[1, self.r + 1] = 1.0
        H[2, self.r + 2] = 1.0
        H[3, self.r + 3] = 1.0

        R = np.eye(self.n) * 1e-6

        # Kalman Filter
        x_filt = np.zeros((self.T, d_state))
        P_filt = np.zeros((self.T, d_state, d_state))
        x_pred = np.zeros(d_state)
        x_pred[: self.r] = self.y_star[0]
        P_pred = np.eye(d_state) * 1.0

        for t in range(self.T):
            if t > 0:
                x_pred = F @ x_filt[t - 1]
                P_pred = F @ P_filt[t - 1] @ F.T + Q

            y_t = self.data[t]
            mask = ~np.isnan(y_t)

            if np.any(mask):
                H_t = H[mask]
                R_t = R[np.ix_(mask, mask)]
                y_obs = y_t[mask]
                innov = y_obs - H_t @ x_pred
                S = H_t @ P_pred @ H_t.T + R_t
                try:
                    K = P_pred @ H_t.T @ inv(S)
                except LinAlgError:
                    K = P_pred @ H_t.T @ inv(S + np.eye(len(S)) * 1e-6)
                x_filt[t] = x_pred + K @ innov
                P_filt[t] = P_pred - K @ H_t @ P_pred
            else:
                x_filt[t] = x_pred
                P_filt[t] = P_pred

        # Smoothing (Simulation)
        x_draw = np.zeros((self.T, d_state))
        try:
            x_draw[-1] = np.random.multivariate_normal(x_filt[-1], P_filt[-1])
        except LinAlgError:
            x_draw[-1] = x_filt[-1]

        for t in range(self.T - 2, -1, -1):
            x_pred_next = F @ x_filt[t]
            P_pred_next = F @ P_filt[t] @ F.T + Q
            try:
                J_t = P_filt[t] @ F.T @ inv(P_pred_next + np.eye(d_state) * 1e-8)
            except LinAlgError:
                J_t = np.zeros_like(F)
            mu = x_filt[t] + J_t @ (x_draw[t + 1] - x_pred_next)
            cov = P_filt[t] - J_t @ P_pred_next @ J_t.T
            cov = (cov + cov.T) / 2
            try:
                x_draw[t] = np.random.multivariate_normal(
                    mu, cov + np.eye(d_state) * 1e-10
                )
            except LinAlgError:
                x_draw[t] = mu

        self.y_star = x_draw[:, : self.r]
        self.y_tilde = x_draw[:, self.r : self.r + self.n]

    def draw_sigma_star(self):
        residuals = np.diff(self.y_star, axis=0)
        SS = residuals.T @ residuals
        kappa_post = self.kappa_star + self.T - 1
        scale_post = self.scale_prior_star + SS
        try:
            self.Sigma_star = invwishart.rvs(df=kappa_post, scale=scale_post)
        except LinAlgError:
            pass

    def draw_sigma_tilde_phi(self):
        Y = self.y_tilde[self.p :]
        X = np.zeros((len(Y), self.n * self.p))
        for i in range(self.p):
            X[:, i * self.n : (i + 1) * self.n] = self.y_tilde[
                self.p - (i + 1) : -(i + 1)
            ]

        eps = Y - X @ self.Phi.T
        SS = eps.T @ eps
        kappa_post = self.kappa_tilde + len(Y)
        scale_post = self.scale_prior_tilde + SS
        try:
            self.Sigma_tilde = invwishart.rvs(df=kappa_post, scale=scale_post)
        except LinAlgError:
            pass

        Phi_new = np.zeros_like(self.Phi)
        for i in range(self.n):
            try:
                Omega_inv = inv(self.Omega_phi[i])
            except LinAlgError:
                Omega_inv = inv(self.Omega_phi[i] + np.eye(self.n * self.p) * 1e-6)

            Sigma_ii = self.Sigma_tilde[i, i]
            V_post_inv = (X.T @ X) / Sigma_ii + Omega_inv
            try:
                V_post = inv(V_post_inv)
            except LinAlgError:
                V_post = inv(V_post_inv + np.eye(len(V_post_inv)) * 1e-6)

            mu_post = V_post @ (
                (X.T @ Y[:, i]) / Sigma_ii + Omega_inv @ self.Phi_bar[i]
            )
            V_post = (V_post + V_post.T) / 2
            try:
                Phi_new[i] = np.random.multivariate_normal(mu_post, V_post)
            except LinAlgError:
                Phi_new[i] = mu_post

        # Stationarity Check
        if np.all(np.abs(np.linalg.eigvals(Phi_new)) < 0.999):
            self.Phi = Phi_new

    def fit(self, n_iter=150_000, n_burn=25_000):
        print(f"Starting Gibbs ({n_iter} iters)...")
        self.store_trends = []
        self.store_cycles = []
        for i in range(n_iter):
            self.simulation_smoother()
            self.draw_sigma_star()
            self.draw_sigma_tilde_phi()
            if i >= n_burn:
                self.store_trends.append(self.y_star.copy())
                self.store_cycles.append(self.y_tilde.copy())
            if (i + 1) % (n_iter // 10) == 0:
                print(f" Iter {i + 1}")
        print("Done.")
