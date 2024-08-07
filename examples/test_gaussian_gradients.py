import numpy as np
from test_utils import gradient_test

np.random.seed(1)
for reff, axrat, ang in [(4.457911011776755, 0.6437167462922668, 44.55485360075)]:
    for reff_psf, axrat_psf, ang_psf in [
        (0, 0, 0),
        (1.5121054822774742, 0.9135936343054303, 50.30562156585181),
        (3.7442185156735914, 0.8695066738347554, -39.40729158864958),
    ]:
        grads, dlls, diffabs = gradient_test(
            xdim=23,
            ydim=19,
            reff=5,
            angle=20,
            reff_psf=reff_psf,
            axrat_psf=axrat_psf,
            angle_psf=ang_psf,
            printout=False,
            plot=False,
        )
        print("Gradient     ", grads)
        print("Finite Diff. ", dlls)
        print("FinD. - Grad.", dlls - grads)
        print("FinD. / Grad.", dlls / grads)
        print("Jacobian sum abs. diff.", diffabs)
