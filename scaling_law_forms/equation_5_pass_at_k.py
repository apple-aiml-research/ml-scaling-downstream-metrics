#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np


class PasskLaw:
    """Fit and predict using pass@k scaling law (Equation 5)."""

    def __init__(self) -> None:
        self.coeffs: np.ndarray | None = None
        self.min_acc: float | None = None
        self.name: str = 'LogBilinear'

    def fit(
        self,
        flops: np.ndarray,
        acc: np.ndarray,
        ks: np.ndarray,
        max_flops: float | None = None
    ) -> None:
        """Fit pass@k model to FLOPs, accuracy, and k values."""
        x_fit = np.stack([np.log(np.array(flops)), np.log(np.array(ks)), np.log(np.array(flops)) * np.log(np.array(ks)), np.ones(flops.shape[0])], axis=1)
        y_fit = np.log(-np.log(np.array(acc)[:, None]))
        self.coeffs, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    def predict(self, flops: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Predict pass@k accuracy for given FLOPs and k values."""
        assert self.coeffs is not None
        x_predict = np.stack([np.log(np.array(flops)), np.log(np.array(ks)), np.log(np.array(flops)) * np.log(np.array(ks)), np.ones(flops.shape[0])], axis=1)
        y_predict = x_predict @ self.coeffs
        accuracy = np.exp(-np.exp(y_predict))

        return accuracy[:, 0]
