import multiprofit.tests as mpftest
import numpy as np
from timeit import default_timer as timer

start = timer()
reffs = [2.0, 5.0]
angs = np.linspace(0, 90, 7)
axrats = [1, 0.5, 0.2, 0.1, 0.01]
nsers = [0.5, 1.0, 2.0, 4.0, 6.0]

mpftest.mgsersic_test(reff=reffs[-1], nser=2, axrat=axrats[-1],
                      angle=angs[-1], plot=True)

for nser in nsers:
    for reff in reffs:
        for axrat in axrats:
            for ang in angs:
                diffs = mpftest.mgsersic_test(
                    reff=reff, nser=nser, axrat=axrat, angle=ang, plot=False
                )['mgs']
                diff_abs = np.sum(np.abs(diffs))
                print(f'Test: nser={nser} reff={reff} axrat={axrat} ang={ang}'
                      f' diff={diff_abs:.3f}')
