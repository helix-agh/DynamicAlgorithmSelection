"""Concrete GP-BO variants with EI and UCB acquisition functions."""

import numpy as np
from scipy.stats import norm

from das.optimizers.BO.bo import BO


class GPBO_EI(BO):
    """GP Bayesian Optimization with Expected Improvement.

    EI(x) = (f* - mu - xi) * Phi(Z) + sigma * phi(Z),  Z = (f* - mu - xi) / sigma
    where f* is the current best observed value and xi > 0 controls exploration.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.xi: float = options.get("xi", 0.01)

    def _acquisition(self, x_norm: np.ndarray, obs_y: np.ndarray) -> float:
        mu, sigma = self._gp.predict(x_norm.reshape(1, -1), return_std=True)
        mu, sigma = float(mu), float(sigma)
        if sigma < 1e-10:
            return 0.0
        f_best = float(np.min(obs_y))
        Z = (f_best - mu - self.xi) / sigma
        ei = (f_best - mu - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -max(ei, 0.0)


class GPBO_UCB(BO):
    """GP Bayesian Optimization with Lower Confidence Bound.

    LCB(x) = mu(x) - beta * sigma(x)
    Minimising LCB favours points with low predicted mean (exploitation) or
    high predicted variance (exploration); beta controls the trade-off.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.beta: float = options.get("beta", 2.0)

    def _acquisition(self, x_norm: np.ndarray, obs_y: np.ndarray) -> float:
        mu, sigma = self._gp.predict(x_norm.reshape(1, -1), return_std=True)
        return float(mu) - self.beta * float(sigma)
