#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import numpy as np


class LogAccPowerLaw:
    """Fit and predict using log-accuracy power law (Equation 2)."""

    def __init__(self) -> None:
        self.coeffs: np.ndarray | None = None
        self.min_acc: float | None = None

    def fit(
        self,
        flops: np.ndarray,
        acc: np.ndarray,
        max_flops: float | None = None,
        min_acc: float | None = None,
        min_acc_to_fit: float = 0.
    ) -> None:
        """Fit power law model to FLOPs and accuracy data with optional constraints."""
        if max_flops is not None:
            mask = flops <= max_flops
            flops = flops[mask]
            acc = acc[mask]

        mask_min_acc_to_fit = acc > min_acc_to_fit
        flops = flops[mask_min_acc_to_fit]
        acc = acc[mask_min_acc_to_fit]
        x_fit = np.stack([np.log(np.array(flops)), np.ones(flops.shape[0])], axis=1)

        if min_acc is not None:
            acc = (acc - min_acc) / (1 - min_acc)
            self.min_acc = min_acc

        y_fit = np.log(-np.log(np.array(acc)[:, None]))
        self.coeffs, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    def predict(self, flops: np.ndarray) -> np.ndarray:
        """Predict accuracy for given FLOPs values."""
        assert self.coeffs is not None
        x_predict = np.stack([np.log(np.array(flops)), np.ones(flops.shape[0])], axis=1)
        y_predict = x_predict @ self.coeffs
        accuracy = np.exp(-np.exp(y_predict))

        if self.min_acc is not None:
            accuracy = accuracy * (1 - self.min_acc) + self.min_acc

        return accuracy[:, 0]