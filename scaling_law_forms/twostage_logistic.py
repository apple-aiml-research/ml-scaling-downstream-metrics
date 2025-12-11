#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np
from scipy.optimize import curve_fit


class TwoStageLogistic:
    """Two-stage model with logistic dependence between accuracy and validation loss."""

    def __init__(self):
        self.alpha_N: float | None = None
        self.C_N: float | None = None
        self.a: float | None = None
        self.b: float | None = None
        self.k: float | None = None
        self.l0: float | None = None

    @staticmethod
    def _logistic(z: np.ndarray, a: float, b: float, k: float, l0: float) -> np.ndarray:
        """Logistic function mapping loss to accuracy."""
        return a / (1.0 + np.exp(-k * (z - l0))) + b

    def fit(self, flops: np.ndarray, nll: np.ndarray, acc: np.ndarray) -> "TwoStageLogistic":
        """Fit two-stage logistic model to FLOPs, NLL, and accuracy data."""
        flops = np.asarray(flops, dtype=float).ravel()
        nll = np.asarray(nll, dtype=float).ravel()
        acc = np.asarray(acc, dtype=float).ravel()

        eps = 1e-12
        x = np.clip(flops, eps, None)
        y = np.clip(nll,   eps, None)

        A = np.vstack([np.log(x), np.ones_like(x)]).T
        alpha_N, log_C_coeff = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
        C_coeff = np.exp(log_C_coeff)
        C_N = (1.0 / C_coeff) ** (1.0 / alpha_N) if alpha_N != 0 else 1.0

        p0 = [float(np.max(acc)), float(np.min(acc)), 1.0, float(np.median(y))]
        a, b, k, l0 = curve_fit(self._logistic, y, acc, p0=p0, maxfev=10000)[0]

        self.alpha_N, self.C_N = float(alpha_N), float(C_N)
        self.a, self.b, self.k, self.l0 = float(a), float(b), float(k), float(l0)
        return self

    def predict(self, flops: np.ndarray) -> np.ndarray:
        """Predict accuracy for given FLOPs values."""
        x = np.asarray(flops, dtype=float).ravel()
        nll_hat = (np.clip(x, 1e-12, None) / self.C_N) ** self.alpha_N
        return self._logistic(nll_hat, self.a, self.b, self.k, self.l0)