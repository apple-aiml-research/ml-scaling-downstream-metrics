#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import LBFGS
from tqdm import tqdm


def huber_loss(N, a, alpha, e, NegLogAcc, delta):
    """Compute Huber loss for power law model with irreducible error."""
    inp = torch.logsumexp(
        torch.stack([a - alpha * torch.log(N), e.repeat(N.shape[0])]), 0
    )
    target = torch.log(NegLogAcc)

    loss = F.huber_loss(inp, target, delta=delta, reduction="sum")

    return loss


def fit_coeffs_for_a_given_init(
    N, NegLogAcc, A_init, alpha_init, e_init, huber_delta=0.001
):
    """Fit coefficients for given initialization using L-BFGS optimizer."""
    alpha = torch.Tensor([alpha_init]).requires_grad_(True)
    a = torch.Tensor([A_init]).requires_grad_(True)
    e = torch.Tensor([e_init]).requires_grad_(True)

    def closure():
        lbfgs.zero_grad()
        loss = huber_loss(N, a, alpha, e, NegLogAcc, huber_delta)
        loss.backward()
        return loss

    lbfgs = LBFGS(
        [a, alpha, e],
        history_size=10000,
        lr=0.1,
        max_iter=100000,
        line_search_fn="strong_wolfe",
    )

    lbfgs.step(closure)

    A = torch.exp(a)
    E = torch.exp(e)

    final_loss = huber_loss(N, a, alpha, e, NegLogAcc, 0.001)

    return A, alpha, E, final_loss


class LogAccPowerLawWithIrreducibleError:
    """Fit and predict using log-accuracy power law with irreducible error (Equation 6)."""

    def __init__(self, min_acc=0.0):
        self.coeffs = dict()
        self.min_acc = min_acc

    def fit(self, flops, acc, min_acc_to_fit=0.0):
        """Fit power law model with irreducible error to FLOPs and accuracy data."""
        min_acc_mask = acc > min_acc_to_fit
        acc = (acc - self.min_acc) / (1 - self.min_acc)

        flops = torch.Tensor(np.array(flops)[min_acc_mask])
        NegLogAcc = torch.Tensor(np.array(-np.log(acc[min_acc_mask])))

        alpha_inits = np.linspace(0, 2, 5)
        a_inits = np.linspace(0, 25, 6)
        e_inits = np.linspace(-1., 1., 5)

        df = pd.DataFrame(columns=["A", "alpha", "E", "huber_loss"])
        for alpha_init, A_init, e_init in tqdm(product(
            alpha_inits, a_inits, e_inits
        )):
            df.loc[len(df)] = fit_coeffs_for_a_given_init(
                flops, NegLogAcc, A_init, alpha_init, e_init
            )

        self.coeffs = df.loc[df["huber_loss"].idxmin()].iloc[:-1].to_dict()

        max_acc_norm = torch.exp(-self.coeffs["E"])              
        max_acc = max_acc_norm * (1 - self.min_acc) + self.min_acc

        print(f"Max accuracy: {max_acc}")

    def predict(self, flops):
        """Predict accuracy for given FLOPs values."""
        assert len(self.coeffs) > 0
        flops = torch.Tensor(np.array(flops))

        res = torch.exp(
            -torch.exp(
                torch.logsumexp(
                    torch.stack(
                        [
                            torch.log(self.coeffs["A"])
                            - self.coeffs["alpha"] * torch.log(flops),
                            torch.log(self.coeffs["E"].repeat(flops.shape[0]))
                        ]
                    ),
                    dim=0,
                )
            )
        ).detach().numpy()
        res = res * (1 - self.min_acc) + self.min_acc

        return res
