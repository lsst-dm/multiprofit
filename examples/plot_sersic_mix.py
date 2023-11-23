import gauss2d.fit as g2f
from lsst.multiprofit.plots import plot_sersicmix_interp
import matplotlib.pyplot as plt
import numpy as np

interps = {
    'lin': (g2f.LinearSersicMixInterpolator(), (0, (5, 1))),
    'g-csp': (g2f.GSLSersicMixInterpolator(interp_type=g2f.GSLInterpType.cspline), '-'),
}

for n_low, n_hi in ((0.5, 0.7), (2.2, 4.4)):
    n_ser = 10**np.linspace(np.log10(n_low), np.log10(n_hi), 100)
    plot_sersicmix_interp(interps=interps, n_ser=n_ser, figsize=(10, 8))
    plt.tight_layout()
    plt.show()
