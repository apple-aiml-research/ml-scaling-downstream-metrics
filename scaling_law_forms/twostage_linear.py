#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np


class TwoStageLinear:
    """Two-stage model with linear dependence between accuracy and validation loss."""

    def __init__(self):
        self.alpha_N: float | None = None
        self.C_N: float | None = None
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, flops: np.ndarray, nll: np.ndarray, acc: np.ndarray) -> "TwoStageLinear":
        """Fit two-stage linear model to FLOPs, NLL, and accuracy data."""
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

        B = np.vstack([np.ones_like(y), y]).T
        a, b = np.linalg.lstsq(B, acc, rcond=None)[0]

        self.alpha_N, self.C_N, self.a, self.b = float(alpha_N), float(C_N), float(a), float(b)
        return self

    def predict(self, flops: np.ndarray) -> np.ndarray:
        """Predict accuracy for given FLOPs values."""
        x = np.asarray(flops, dtype=float).ravel()
        nll_hat = (np.clip(x, 1e-12, None) / self.C_N) ** self.alpha_N
        return self.a + self.b * nll_hat