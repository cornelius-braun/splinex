"""Utils for spline evaluation."""

import chex
import jax.numpy as jnp


def compute_knots(n: int, degree: int = 3) -> chex.Array:
    """Get the knot vector for a given number of points.

    Args:
        n: Number of evaluation points.
        degree: Degree of the B-spline.

    Returns:
        chex.Array: Knot vector.

    """
    u = jnp.linspace(0.0, 1.0, n, True)
    if degree % 2 == 0:
        middle_knots = (u[:-1] + u[1:]) / 2.0
        middle_knots = jnp.append(middle_knots, u[-1])
    else:
        middle_knots = u
    kv = jnp.r_[(u[0],) * degree, middle_knots, (u[-1],) * degree]
    return kv


def clamp_ctrls(x: chex.Array, degree: int = 2) -> chex.Array:
    """Clamp a vector of controls for inference.

    Clamping is guarantees that the spline passes through start and endpoints.

    Args:
        x (chex.Array): Control vector.
        degree (int): Degree of the B-spline.

    Returns:
        chex.Array: Clamped control vector with multiplicities at start and end.

    """
    if x.ndim == 2:
        x = x[jnp.newaxis, :, :]

    nrep = degree // 2
    first_points = jnp.tile(x[:, 0, :][:, jnp.newaxis, :], (1, nrep, 1))
    last_points = jnp.tile(x[:, -1, :][:, jnp.newaxis, :], (1, nrep, 1))
    ctrl_points = jnp.concatenate([first_points, x, last_points], axis=1)

    return jnp.hstack(ctrl_points)
