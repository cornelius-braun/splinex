"""B-Spline interpolation in jax."""

from functools import partial

import chex
import jax
import jax.numpy as jnp

from splinex.utils import clamp_ctrls, compute_knots


class BSpline:
    """B-Spline interpolation in jax."""

    def __init__(self, n_in: int, n_out: int, degree: int = 3, order: int = 2):
        """Initialize the B-spline.

        Args:
            n_in: Number of input points.
            n_out: Number of output points.
            degree: Degree of the B-spline.
            order: Order of derivatives to compute.

        """
        self.knots = compute_knots(n_in, degree)
        self.degree = degree
        self.t_values = jnp.linspace(
            self.knots[degree], self.knots[-degree - 1], n_out, True
        )
        self.n = len(self.knots) - degree - 1
        self.order = order
        self.Phi = self.compute_basis_matrix()
        self.dPhi = self.compute_derivative_matrix(1) if order > 0 else None
        self.ddPhi = self.compute_derivative_matrix(2) if order > 1 else None

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Evaluate the B-spline and its derivatives at the given control points.

        Supports evaluation in batches, i.e., for a (m, n, l) matrix the input
         will be treated as m batches of n x l inputs, where n is the number
         of control points and l the target dimension.

        Args:
            x: Control points for the B-spline.

        Returns:
            tuple[chex.Array, chex.Array, chex.Array]:
            Evaluated spline, first derivative, and second derivative.

        """
        xc = clamp_ctrls(x, self.degree)

        y = self.Phi @ xc
        y_dot = self.dPhi @ xc if self.order > 0 else None
        y_ddot = self.ddPhi @ xc if self.order > 1 else None

        if x.ndim > 2:
            y = y.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)
            if self.order > 0:
                y_dot = y_dot.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)
            if self.order > 1:
                y_ddot = y_ddot.reshape(-1, x.shape[0], x.shape[-1]).transpose(1, 0, 2)

        return y, y_dot, y_ddot

    def compute_basis_matrix(self) -> chex.Array:
        """Compute the basis matrix for the B-spline.

        Returns:
            chex.Array: Basis matrix.

        """
        basis_matrix = jax.vmap(
            lambda i: jax.vmap(lambda t: self.basis_function(t, self.degree, i))(
                self.t_values
            )
        )(jnp.arange(self.n))

        return basis_matrix.T

    def compute_derivative_matrix(self, order: int = 1) -> chex.Array:
        """Compute the derivative matrix for the B-spline.

        Args:
            order: Order of the derivative.

        Returns:
            chex.Array: Derivative matrix.

        """
        derivative_matrix = jax.vmap(
            lambda i: jax.vmap(
                lambda t: self.derivative_basis_function(t, self.degree, i, order)
            )(self.t_values)
        )(jnp.arange(self.n))

        return derivative_matrix.T

    def basis_function(self, t: float, k: int, i: int) -> float:
        """Cox-de Boor recursion formula for B-spline basis function.

        Args:
            t: Parameter value.
            k: Degree of the basis function.
            i: Knot span index.

        Returns:
            float: Basis function value at t.

        """
        if k == 0:
            return jnp.where((self.knots[i] <= t) & (t < self.knots[i + 1]), 1.0, 0.0)
        else:
            left_term = jnp.where(
                self.knots[i + k] != self.knots[i],
                (t - self.knots[i])
                / (self.knots[i + k] - self.knots[i])
                * self.basis_function(t, k - 1, i),
                0.0,
            )
            right_term = jnp.where(
                self.knots[i + k + 1] != self.knots[i + 1],
                (self.knots[i + k + 1] - t)
                / (self.knots[i + k + 1] - self.knots[i + 1])
                * self.basis_function(t, k - 1, i + 1),
                0.0,
            )
            special_case = jnp.where(
                ((i == 0) & (t == self.knots[0]))
                | ((i == self.n - 1) & (t == self.knots[-1])),
                0.0,
                1.0,
            )
            return (left_term + right_term) * special_case + 1.0 - special_case

    def derivative_basis_function(
        self, t: float, k: int, i: int, order: int = 1
    ) -> float:
        """Compute the derivative of the B-spline basis function.

        Args:
            t: Parameter value.
            k: Degree of the basis function.
            i: Knot span index.
            order: Order of the derivative.

        Returns:
            float: Derivative of the basis function at t.

        """
        if order == 0:
            return self.basis_function(t, k, i)
        elif k == 0:
            return 0.0
        else:
            left_term = jnp.where(
                self.knots[i + k] != self.knots[i],
                k
                / (self.knots[i + k] - self.knots[i])
                * self.derivative_basis_function(t, k - 1, i, order - 1),
                0.0,
            )
            right_term = jnp.where(
                self.knots[i + k + 1] != self.knots[i + 1],
                k
                / (self.knots[i + k + 1] - self.knots[i + 1])
                * self.derivative_basis_function(t, k - 1, i + 1, order - 1),
                0.0,
            )
            return left_term - right_term
