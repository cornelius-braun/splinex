# splinex

jax implementation of spline interpolation in jax beyond 2D, as I did not find this anywhere else.

Currently, B-Spline basis functions via the [Cox - de Boor recursion formula](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) are implemented.
Other methods such as Ordinal Basis Functions (OBF) may follow eventually. Contributions are always welcome.

### Example usage
```
import jax
import matplotlib.pyplot as plt

from splinex import BSpline


# Dummy points for interpolation
tau = .1
degree = 3
points = jax.random.uniform(jax.random.PRNGKey(123), (7, 2), minval=0, maxval=8)
n_pts = points.shape[0]
sim_steps = int(n_pts // tau)

# splinex
splinex_bspl = BSpline(n_pts, sim_steps)
splinex_curve, _, _ = splinex_bspl(points)

# Plotting
plt.plot(*splinex_curve.T, "yo-", label="B-spline jax")
plt.plot(*points.T, "ro:", label="Ctrl points")
plt.legend()
plt.show()
```

For more examples, please look in the [examples](examples) folder.

### Dependencies
The code is tested on:
```
Python >= 3.10
jax == 0.4.28
```
I believe that it should generally be compatible with most `jax` versions up to this point as it only uses basic functionalities

### License
[MIT](LICENSE)
