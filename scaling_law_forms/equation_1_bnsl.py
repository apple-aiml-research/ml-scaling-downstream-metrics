#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from copy import deepcopy
from typing import Any, Callable

import jax.numpy as jnp
import jax
from scipy.optimize import  basinhopping
from jax.flatten_util import ravel_pytree
import numpy as np


def uniform_sampling(bounds: dict[str, tuple[float, float, int]], seed: int) -> dict[str, jnp.ndarray]:
    """Sample parameters uniformly within specified bounds using JAX random key."""
    key = jax.random.PRNGKey(seed)
    params = {}
    for k, (low, high, size) in bounds.items():
        params[k] = jax.random.uniform(key, shape=(size,), minval=low, maxval=high)
        key, _ = jax.random.split(key)
    return params


def huber_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    delta: float = 1.0e-3,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Huber loss between predictions and targets with optional weights."""
    residual = y_true - y_pred
    cond = jnp.abs(residual) <= delta
    loss = jnp.where(cond, 0.5 * residual**2, delta * (jnp.abs(residual) - 0.5 * delta))
    return jnp.sum(loss * weights) if weights is not None else jnp.mean(loss)


def get_loss_fn(model_fn: Callable, results: tuple[jnp.ndarray, jnp.ndarray], weights: str = "uniform") -> Callable:
    """Build loss function for model fitting with uniform or per-size weighting."""
    inputs = results[0]
    n, _ = inputs.shape
    if weights == "uniform":
        weights = jnp.ones(n) / n
    elif weights == "per_size":
        unique_model_sizes = jnp.unique(inputs[:, -2])
        weights = jnp.zeros(n)
        for model_size in unique_model_sizes:
            mask = inputs[:, -2] == model_size
            weights = weights.at[mask].set(1.0 / jnp.sum(mask))
        weights = weights / jnp.sum(weights)

    def partial_pred_fn(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        preds = model_fn(params, results[0])
        return huber_loss(preds, results[1], weights=weights)

    return partial_pred_fn


def minimize_over_grid(
    model_fn: Callable,
    bounds: dict[str, tuple[float, float, int]],
    results: tuple[jnp.ndarray, jnp.ndarray],
    method_kwargs: dict[str, Any]
) -> tuple[dict[str, jnp.ndarray], float, Callable, list[float]]:
    """Optimize model parameters via grid search with basin hopping."""
    kwargs = deepcopy(method_kwargs)
    partial_pred_fn = get_loss_fn(model_fn, results)
    params = uniform_sampling(bounds, 0)
    _, unravel_fn = ravel_pytree(params)

    def flat_fn(flat_params: jnp.ndarray) -> float:
        params = unravel_fn(flat_params)
        return partial_pred_fn(params)

    grid_size = kwargs["grid_size"]
    del kwargs["grid_size"]

    flat_fn = jax.jit(flat_fn)
    grad_fn = jax.jit(jax.grad(flat_fn))

    def minimizer_fn(flat_params: jnp.ndarray, seed: int) -> tuple[dict[str, jnp.ndarray], float, Any]:
        # Filter out kwargs that basinhopping doesn't accept
        basinhopping_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["n_jobs"]
        }  # Remove n_jobs and any other unsupported params
        res = basinhopping(
            flat_fn,
            flat_params,
            minimizer_kwargs=dict(
                method="L-BFGS-B", jac=grad_fn, options=dict(gtol=1e-15, ftol=1e-15)
            ),
            seed=seed,
            **basinhopping_kwargs,
        )
        return unravel_fn(res.x), res.fun, res


    def get_results(seed: int) -> tuple[dict[str, jnp.ndarray], float, Any]:
        params = uniform_sampling(bounds, seed)
        flat_params, _ = ravel_pytree(params)
        res = minimizer_fn(flat_params, seed)
        return res

    res = [get_results(seed) for seed in range(grid_size)]

    funvals = [r[1] for r in res]
    best_idx = jnp.nanargmin(jnp.array(funvals))
    best_res = res[best_idx]
    best_params, best_loss, _ = best_res

    def pred_fn(hists: jnp.ndarray) -> jnp.ndarray:
        return model_fn(best_params, hists)

    return best_params, best_loss, pred_fn, funvals


def bnsl_model_fn(params: dict[str, jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
    """Compute Broken Neural Scaling Law predictions given parameters and inputs."""
    x = X[:, 0]
    # Add small epsilon to prevent numerical issues
    epsilon = 1e-12
    power_term = params["b"] * (x ** -params["c0"])
    break_term = (1 + (x / params["d1"]) ** (1 / params["f1"])) ** (
        -params["c1"] * params["f1"]
    )
    result = params["a"] + power_term * break_term
    # Ensure numerical stability
    return jnp.clip(
        result, -50, 50
    )  # Prevent extreme values that could cause exp() overflow


class BrokenNeuralScalingLaw:
    """Fit and predict using Broken Neural Scaling Law (Equation 1)."""

    def __init__(self, min_acc: float = 0.):
        self.min_acc = min_acc
        self.params: dict[str, float] | None = None

    def fit(self, flops: np.ndarray, acc: np.ndarray) -> None:
        """Fit BNSL model to FLOPs and accuracy data."""
        flops = np.asarray(flops, dtype=float)[:, None]
        acc = np.asarray(acc, dtype=float)

        bounds = {
                "a": (-2, 5, 1),  # Allow negative bias (was 0, 5, 1)
                "b": (0, 50, 1),  # Increase upper bound (was 0, 15, 1)
                "c0": (0.1, 3, 1),  # Ensure c0 > 0 and reasonable range (was 0, 5, 1)
                "d1": (1e18, 1e24, 1),  # Expand range (was 1e18, 1e23, 1)
                "f1": (0.1, 3, 1),  # Ensure f1 > 0 (was 0, 5, 1)
                "c1": (0.1, 3, 1),  # Ensure c1 > 0 (was 0, 5, 1)
            }

        best_params, _, pred_fn, _ = minimize_over_grid(
                bnsl_model_fn,
                bounds,
                (flops, acc),
                {'grid_size': 100, 'niter': 100},
            )

        self.params = jax.tree_util.tree_map(lambda x: x[0].item(), best_params)

    def predict(self, flops: np.ndarray) -> np.ndarray:
        """Predict accuracy for given FLOPs values."""
        flops = np.asarray(flops, dtype=float)[:, None]
        return bnsl_model_fn(self.params, flops)