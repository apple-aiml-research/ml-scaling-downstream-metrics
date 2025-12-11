#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import LBFGS
import pandas as pd


def huber_loss(
    N: torch.Tensor,
    a: torch.Tensor,
    alpha: torch.Tensor,
    D: torch.Tensor,
    b: torch.Tensor,
    beta: torch.Tensor,
    NegLogAcc: torch.Tensor,
    delta: float
) -> torch.Tensor:
    """Compute Huber loss for multi-token-to-param ratio model."""
    inp = torch.logsumexp(
        torch.stack([a - alpha * torch.log(N), b - beta * torch.log(D)]), 0
    )
    target = torch.log(NegLogAcc)

    loss = F.huber_loss(inp, target, delta=delta, reduction="sum")

    return loss


def fit_coeffs_for_a_given_init(
    N: torch.Tensor,
    D: torch.Tensor,
    NegLogAcc: torch.Tensor,
    A_init: float,
    alpha_init: float,
    B_init: float,
    beta_init: float,
    huber_delta: float = 0.001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit coefficients for given initialization using L-BFGS optimizer."""
    alpha = torch.Tensor([alpha_init]).requires_grad_(True)
    a = torch.Tensor([A_init]).requires_grad_(True)
    beta = torch.Tensor([beta_init]).requires_grad_(True)
    b = torch.Tensor([B_init]).requires_grad_(True)

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = huber_loss(N, a, alpha, D, b, beta, NegLogAcc, huber_delta)
        loss.backward()
        return loss

    lbfgs = LBFGS(
        [a, alpha, b, beta],
        history_size=10000,
        lr=0.1,
        max_iter=100000,
        line_search_fn="strong_wolfe",
    )

    lbfgs.step(closure)

    A = torch.exp(a)
    B = torch.exp(b)

    final_loss = huber_loss(N, a, alpha, D, b, beta, NegLogAcc, 0.001)

    return A, alpha, B, beta, final_loss


class MultiTPRHuberPowerLaw:
    """Fit and predict using multi-token-to-param ratio power law (Equation 4)."""

    def __init__(self, min_acc: float = 0.0):
        self.coeffs: dict[str, float] = dict()
        self.min_acc = min_acc

    def fit(self, N: np.ndarray, D: np.ndarray, Acc: np.ndarray, min_acc_to_fit: float = 0.0) -> None:
        """Fit model with varying token-to-param ratios and parameter counts."""
        min_acc_mask = Acc > min_acc_to_fit
        Acc = (Acc - self.min_acc) / (1 - self.min_acc)

        N = torch.Tensor(np.array(N)[min_acc_mask])
        D = torch.Tensor(np.array(D)[min_acc_mask])
        NegLogAcc = torch.Tensor(np.array(-np.log(Acc[min_acc_mask])))

        alpha_inits = [0.5, 1.0, 1.5]
        beta_inits = [0.5, 1.0, 1.5]
        a_inits = [5.0, 10.0, 15.0]
        b_inits = [5.0, 10.0, 15.0]

        df = pd.DataFrame(columns=["A", "alpha", "B", "beta", "huber_loss"])
        for alpha_init, beta_init, A_init, B_init in product(
            alpha_inits, beta_inits, a_inits, b_inits
        ):
            df.loc[len(df)] = fit_coeffs_for_a_given_init(
                N, D, NegLogAcc, A_init, alpha_init, B_init, beta_init
            )

        self.coeffs = df.loc[df["huber_loss"].idxmin()].iloc[:-1].to_dict()

    def predict(self, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Predict accuracy for given parameter counts and token ratios."""
        assert len(self.coeffs) > 0
        N = torch.Tensor(np.array(N))
        D = torch.Tensor(np.array(D))

        res = torch.exp(
            -torch.exp(
                torch.logsumexp(
                    torch.stack(
                        [
                            torch.log(self.coeffs["A"])
                            - self.coeffs["alpha"] * torch.log(N),
                            torch.log(self.coeffs["B"])
                            - self.coeffs["beta"] * torch.log(D),
                        ]
                    ),
                    dim=0,
                )
            )
        ).detach().numpy()
        res = res * (1 - self.min_acc) + self.min_acc

        return res
