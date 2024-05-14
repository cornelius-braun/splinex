import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline as ScipyBSpline

from splinex import BSpline
from splinex.utils import clamp_ctrls, compute_knots

if __name__ == "__main__":
    # Dummy points for interpolation
    tau = .1
    degree = 3
    points = jax.random.uniform(jax.random.PRNGKey(123), (7, 2), minval=0, maxval=8)
    n_pts = points.shape[0]
    sim_steps = int(n_pts // tau)

    # scipy -- only included for comparison
    sim_steps = int(points.shape[-2] // tau)
    knots = compute_knots(n_pts, degree)
    ctrls = clamp_ctrls(points)
    times = jnp.linspace(0., 1., sim_steps, True)
    bspl = ScipyBSpline(knots, ctrls, degree)
    scipy_curve = bspl(times)

    # splinex
    splinex_bspl = BSpline(n_pts, sim_steps)
    splinex_curve, _, _ = splinex_bspl(points)

    # Plotting
    plt.plot(*scipy_curve.T, "bd-", label="B-spline scipy")
    plt.plot(*splinex_curve.T, "yo-", label="B-spline jax", alpha=.5)
    plt.plot(*points.T, "ro:", label="Ctrl points")
    plt.legend()
    plt.show()
