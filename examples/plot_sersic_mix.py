import gauss2d.fit as g2f
from lsst.multiprofit.plots import plot_sersicmix_interp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

interps = {
    "lin": (g2f.LinearSersicMixInterpolator(), "-"),
    "gsl-csp": (g2f.GSLSersicMixInterpolator(interp_type=g2f.InterpType.cspline), (0, (8, 8))),
    "scipy-csp": ((CubicSpline, {}), (0, (4, 4))),
}

for n_low, n_hi in ((0.5, 0.7), (0.8, 1.2), (2.2, 4.4)):
    n_ser = 10 ** np.linspace(np.log10(n_low), np.log10(n_hi), 100)
    plot_sersicmix_interp(interps=interps, n_ser=n_ser, figsize=(10, 8))
    plt.tight_layout()
    plt.show()
