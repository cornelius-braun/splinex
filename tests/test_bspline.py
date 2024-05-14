import jax
import jax.numpy as jnp
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

from splinex import BSpline
from splinex.utils import clamp_ctrls, compute_knots


@pytest.mark.parametrize("dim", [2, 3, 8])
def test_bspline(dim):
    rng = jax.random.PRNGKey(0)
    tau = 0.1
    degree = 3
    points = jax.random.uniform(rng, (4, dim), minval=0, maxval=8)
    n_pts = points.shape[0]
    sim_steps = int(n_pts // tau)

    # scipy
    knots = compute_knots(n_pts, degree)
    ctrls = clamp_ctrls(points)
    times = jnp.linspace(0., 1., sim_steps, True)
    bspl = ScipyBSpline(knots, ctrls, degree)
    scipy_curve = bspl(times)
    scipy_derivative = bspl.derivative()(times)
    scipy_dderivative = bspl.derivative(nu=2)(times)

    # splinex
    splinex_bspl = BSpline(n_pts, sim_steps)
    splinex_curve, splinex_derivative, splinex_dderivative = splinex_bspl(points)

    # Note: we set the threshold relatively high bec. scipy itself comes with numerical imprecisions
    assert jnp.allclose(scipy_curve, splinex_curve, atol=1e-4)
    assert jnp.allclose(scipy_derivative, splinex_derivative, atol=1e-4)
    assert jnp.allclose(scipy_dderivative, splinex_dderivative, atol=1e-4)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_bspline_batched(dim):
    rng = jax.random.PRNGKey(123)
    tau = 0.1
    degree = 3
    points = jax.random.uniform(rng, (dim, 5, 4), minval=0, maxval=8)
    n_pts = points.shape[1]
    sim_steps = int(n_pts // tau)

    # scipy
    knots = compute_knots(n_pts, degree)
    ctrls = clamp_ctrls(points)
    times = jnp.linspace(0., 1., sim_steps, True)
    bspl = ScipyBSpline(knots, ctrls, degree)
    scipy_curve = bspl(times)
    scipy_curve = scipy_curve.reshape(sim_steps, points.shape[0], points.shape[-1]).transpose(1, 0, 2)
    scipy_derivative = bspl.derivative()(times)
    scipy_derivative = scipy_derivative.reshape(sim_steps, points.shape[0], points.shape[-1]).transpose(1, 0, 2)
    scipy_dderivative = bspl.derivative(nu=2)(times)
    scipy_dderivative = scipy_dderivative.reshape(sim_steps, points.shape[0], points.shape[-1]).transpose(1, 0, 2)

    # splinex
    splinex_bspl = BSpline(n_pts, sim_steps)
    splinex_curve, splinex_derivative, splinex_dderivative = splinex_bspl(points)

    assert jnp.allclose(scipy_curve, splinex_curve, atol=1e-4)
    assert jnp.allclose(scipy_derivative, splinex_derivative, atol=1e-4)
    assert jnp.allclose(scipy_dderivative, splinex_dderivative, atol=1e-4)
