"""TDE: Trigonometric-mutation Differential Evolution."""

import numpy as np

from .base import DE


class TDE(DE):
    """DE/rand/1/bin with probabilistic fitness-weighted trigonometric mutation.

    With probability tm the standard DE/rand/1 donor vector is replaced by a
    centroid-based mutation biased toward the lowest-fitness of three randomly
    selected individuals.
    """

    def __init__(self, problem: dict, options: dict):
        options = dict(options)
        options.setdefault("n_individuals", 30)
        super().__init__(problem, options)
        self.F: float = options.get("F", 0.99)
        self.CR: float = options.get("CR", 0.85)
        self.tm: float = options.get("tm", 0.05)

    def iterate(self, x, y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            perm = self.rng_optimization.permutation(self.n_individuals)
            r = perm[perm != i][:3]
            r0, r1, r2 = int(r[0]), int(r[1]), int(r[2])

            if self.rng_optimization.random() < self.tm:
                p_sum = np.abs(y[r0]) + np.abs(y[r1]) + np.abs(y[r2])
                if p_sum < 1e-12:
                    p_sum = 1e-12
                p0 = np.abs(y[r0]) / p_sum
                p1 = np.abs(y[r1]) / p_sum
                p2 = np.abs(y[r2]) / p_sum
                donor = (
                    (x[r0] + x[r1] + x[r2]) / 3.0
                    + (p1 - p0) * (x[r0] - x[r1])
                    + (p2 - p1) * (x[r1] - x[r2])
                    + (p0 - p2) * (x[r2] - x[r0])
                )
            else:
                donor = x[r0] + self.F * (x[r1] - x[r2])

            trial = self._crossover(
                x[i], np.clip(donor, self.lower_boundary, self.upper_boundary)
            )
            trial_y = self._evaluate_fitness(trial)
            if trial_y <= y[i]:
                x[i], y[i] = trial, trial_y

        self._n_generations += 1
        self._warm_start = {"x": x, "y": y}
        return x, y
