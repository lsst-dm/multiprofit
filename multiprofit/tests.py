# This file is part of multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from importlib.util import find_spec
import numpy as np
import matplotlib.pyplot as plt
from multiprofit.fitutils import get_model
import gauss2d
import gauss2d.utils
from multiprofit.objects import get_gsparams, names_params_gauss
from multiprofit.utils import estimate_ellipse
import timeit


# Example usage:
# test = gaussian_test(nbenchmark=1000)
# for x in test:
#   print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))
def gaussian_test(xdim=49, ydim=51, reffs=None, angs=None, axrats=None, nbenchmark=0, nsub=1,
                  do_like=True, do_grad=False, do_jac=False, do_meas_modelfit=False):
    """
    Test and/or benchmark different gaussian evaluation methods.

    Example usage:

    test = gaussian_test(nbenchmark=1000)
    for x in test:
        print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))

    :param xdim: int; x dimension of image
    :param ydim: int; y dimension of image
    :param reffs: float[]; iterable of effective radii
    :param angs: float[]; iterable of position angles in degrees CCW from +x
    :param axrats: float[]; iterable of major-to-minor axis ratios
    :param nbenchmark: int; number of times to repeat function evaluation for benchmarking
    :param nsub: int; number of (identical) Gaussians to evaluate (as for a GMM)
    :param do_like: bool; whether to evaluate the likelihood
    :param do_grad: bool; whether to evaluate the likelihood gradient
    :param do_jac: bool; whether to evaluate the model Jacobian
    :param do_meas_modelfit: bool; whether to test meas_modelfit's code
    :return: results: list of dicts with results for each combination of parameters
    """
    if reffs is None:
        reffs = [2.0, 5.0]
    if angs is None:
        angs = np.linspace(0, 90, 7)
    if axrats is None:
        axrats = [0.01, 0.1, 0.2, 0.5, 1]
    results = []
    hasgs = find_spec('galsim') is not None
    num_params = nsub*6
    if hasgs:
        import galsim as gs
    for reff in reffs:
        for ang in angs:
            for axrat in axrats:
                gaussmpfold = gauss2d.make_gaussian_pixel_sersic(
                    xdim/2, ydim/2, 1, reff, axrat, ang, 0, xdim, 0, ydim, xdim, ydim)
                gaussmpf = gauss2d.make_gaussian_pixel(
                    xdim/2, ydim/2, 1, reff, axrat, ang, 0, xdim, 0, ydim, xdim, ydim)
                oldtonew = np.sum(np.abs(gaussmpf-gaussmpfold))
                result = 'Old/new residual=({:.3e})'.format(oldtonew)
                if hasgs:
                    gaussgs = gs.Gaussian(flux=1, half_light_radius=reff*np.sqrt(axrat)).shear(
                        q=axrat, beta=ang*gs.degrees).drawImage(
                        nx=xdim, ny=ydim, scale=1, method='no_pixel').array
                    gstonew = np.sum(np.abs(gaussmpf-gaussgs))
                    result += '; GalSim/new residual=({:.3e})'.format(gstonew)
                if nbenchmark > 0:
                    argsmpf = ('(' + ','.join(np.repeat('{}', 12)) + ')').format(
                        xdim/2, ydim/2, 1, reff, axrat, ang, 0, xdim, 0, ydim, xdim, ydim)
                    functions = {
                        'old': 'gauss2d.make_gaussian_pixel_sersic' + argsmpf,
                        'new': 'gauss2d.make_gaussian_pixel' + argsmpf,
                    }
                    if do_like or do_grad or do_jac:
                        ell = gauss2d.Ellipse(gauss2d.EllipseMajor(
                            gauss2d.M_SIGMA_HWHM*reff, axrat, ang, degrees=True,
                        ))
                        gaussparams = (f'({xdim/2}, {ydim/2}, {1/nsub}, {ell.sigma_x}, {ell.sigma_y},'
                                       f' {ell.rho}, 0, 0, 0)')
                        gaussarrays = f'np.array([{",".join(np.repeat(gaussparams, nsub))}])'
                    if do_like:
                        functions['like'] = f'g2dutils.loglike_gaussians_pixel(data, data, {gaussarrays})'
                    if do_grad:
                        functions['grad'] = (f'g2dutils.loglike_gaussians_pixel(data, data, {gaussarrays},'
                                             f' grad=grads)')
                    if do_jac:
                        functions['jac'] = (f'g2dutils.loglike_gaussians_pixel(data, data, {gaussarrays},'
                                            f' grad=grads)')
                    times = {
                        key: np.min(timeit.repeat(
                            callstr,
                            setup='import gauss2d; import gauss2d.utils as g2dutils' + (
                                  f'; import numpy as np; data=np.zeros(({ydim}, {xdim}))'
                                  f'; zeros = np.zeros(({ydim}, {xdim}))'
                                  f'; zeros_s = np.zeros(0, dtype=np.uint64)'
                                if key in ['grad', 'like', 'jac'] else ''
                            ) + (
                                f'; grads = np.zeros(({num_params}))' if key == 'grad' else (
                                    f'; grads = np.zeros(({ydim},{xdim},{num_params}))'
                                    if key == 'jac' else ''
                                )
                            ),
                            repeat=nbenchmark, number=1
                        ))
                        for key, callstr in functions.items()
                    }
                    if do_meas_modelfit:
                        ang_rad = np.deg2rad(ang)
                        for key in ("dev", "exp"):
                            times[f"mmf-{key}"] = np.min(timeit.repeat(
                                f"msf.evaluate().addToImage(img)",
                                setup=(
                                    f"import numpy as np;"
                                    f"from lsst.shapelet import RadialProfile;"
                                    f"from lsst.afw.geom.ellipses import Ellipse, Axes;"
                                    f"from lsst.geom import Point2D;"
                                    f"is_exp = '{key}' == 'exp';"
                                    f"profile = RadialProfile.get('lux' if is_exp else 'luv');"
                                    f"basis = profile.getBasis(6 if is_exp else 8, 4 if is_exp else 8);"
                                    f"xc, yc = {xdim/2} - 0.5, {ydim}/2. - 0.5;"
                                    f"ellipse = Ellipse("
                                    f"Axes(a={reff}, b={reff*axrat}, theta={ang_rad}), Point2D(xc, yc));"
                                    f"msf = basis.makeFunction(ellipse, np.array([1.]));"
                                    f"img=np.zeros(({ydim}, {xdim}))"
                                ),
                                repeat=nbenchmark, number=1,
                            ))
                    if hasgs:
                        times['GalSim'] = np.min(timeit.repeat(
                            f'x=gs.Gaussian(flux=1, half_light_radius={reff*np.sqrt(axrat)}).shear('
                            f'q={axrat}, beta={ang}*gs.degrees)'
                            f'.drawImage(nx={xdim}, ny={ydim}, scale=1, method="no_pixel").array',
                            setup='import galsim as gs', repeat=nbenchmark, number=1
                        ))
                    result += f";{'/'.join(times.keys())} times=(" \
                        f"{','.join(['{:.3e}'.format(x) for x in times.values()])})"
                results.append({
                    'string': result,
                    'xdim': xdim,
                    'ydim': ydim,
                    'reff': reff,
                    'axrat': axrat,
                    'ang': ang,
                })
    return results


def gradient_test(dimx=5, dimy=4, flux=1e4, reff=2, axrat=0.5, ang=0, bg=1e3,
                  reff_psf=0, axrat_psf=0.95, ang_psf=0, printout=False, plot=False):
    cenx, ceny = dimx/2., dimy/2.
    # Keep this in units of sigma, not re==FWHM/2
    source = gauss2d.EllipseMajor(reff*gauss2d.M_SIGMA_HWHM, axrat, ang, degrees=True)
    source_g = gauss2d.Ellipse(source)
    has_psf = reff_psf > 0
    psf = gauss2d.EllipseMajor(reff_psf*gauss2d.M_SIGMA_HWHM, axrat_psf, ang_psf, degrees=True)
    psf_g = gauss2d.Ellipse(psf)
    conv = gauss2d.EllipseMajor(source_g.make_convolution(psf_g), degrees=True)
    model = gauss2d.make_gaussian_pixel(
        cenx, ceny, flux, conv.r_major*gauss2d.M_HWHM_SIGMA, conv.axrat, conv.angle,
        0, dimx, 0, dimy, dimx, dimy
    )

    if printout:
        psf_c = gauss2d.Covariance(psf_g)
        deconv_params = [psf_c.sigma_x_sq, psf_c.sigma_y_sq, psf_c.cov_xy]
        deconvs = (False, True) if has_psf else (False,)
        covar_ests = [
            gauss2d.Covariance(
                *estimate_ellipse(model, cenx=cenx, ceny=ceny, denoise=False, return_as_params=True,
                                  deconvolution_params=deconv_params if deconv else None)
            ) for deconv in deconvs
        ]
        sum_img = np.sum(model)
        print(f"Modelsum = {sum_img:5e} vs flux {flux:.5e} ({(flux-sum_img)/sum_img*100.:.2f}% missing)")
        print(f"Source: {source}")
        print(f"PSF: {psf}")
        print(f"Convolved: {conv}")
        print("reff, axrat, ang:", (conv.r_major, conv.axrat, conv.angle))
        print("Estimated ellipse (covar):", covar_ests[0])
        print("Estimated ellipse (ellipse):", gauss2d.EllipseMajor(covar_ests[0]))
        if has_psf:
            print("Estimated deconvolved ellipse (covar):", covar_ests[1])
            print("Estimated deconvolved ellipse (ellipse):", gauss2d.EllipseMajor(covar_ests[1]))

    data = np.random.poisson(model + bg) - bg
    sigma_inv = np.array([[1/bg]])
    output = np.zeros_like(data)
    num_params = 6
    grads = np.zeros(num_params)
    params_init = np.array([[
        cenx, ceny, flux,
        source_g.sigma_x, source_g.sigma_y, source_g.rho,
        psf_g.sigma_x, psf_g.sigma_y, psf_g.rho
    ]])
    print(conv, source_g.make_convolution(gauss2d.Ellipse(psf)))
    # Compute the log likelihood and gradients
    gauss2d.utils.loglike_gaussians_pixel(data, sigma_inv, params_init, output=output)
    if plot:
        fig, axes = plt.subplots(ncols=2)
        fig.suptitle(f'gauss2d evaluation comparison')
        axes[0].imshow(model)
        axes[0].set_title("Model (make_gaussian_pixel)")
        axes[1].imshow(output)
        axes[1].set_title("Model (loglike_gaussians_pixel)")
        plt.show()
    ll = gauss2d.utils.loglike_gaussians_pixel(data, sigma_inv, params_init, grad=grads)
    jacobian = np.zeros([dimy, dimx, num_params])
    gauss2d.utils.loglike_gaussians_pixel(data, sigma_inv, params_init, grad=jacobian)
    dxs = [1e-6, 1e-6, flux*1e-6, 1e-8, 1e-8, 1e-8]
    dlls = np.zeros(6)
    diffabs = np.zeros(6)
    format_param_name = f'<{max(len(param_name) for param_name in names_params_gauss)}'

    for i, dxi in enumerate(dxs):
        dx = np.zeros(6)
        dx[i] = dxi
        # Note that mpf computes dll/drho where the diagonal term is rho*sigma_x*sigma_y
        params = np.array([[
            cenx + dx[0], ceny + dx[1], flux + dx[2],
            source_g.sigma_x + dx[3], source_g.sigma_y + dx[4], source_g.rho + dx[5],
            psf_g.sigma_x, psf_g.sigma_y, psf_g.rho
        ]])
        # Note that there's no option to return the log likelihood and the Jacobian - the latter skips
        # computing the former for efficiency, assuming that you won't need it
        llnew = gauss2d.utils.loglike_gaussians_pixel(data, sigma_inv, params, output=output)
        # It's actually the jacobian of the residual
        findiff = -(output-model)/dxi*sigma_inv
        jacparam = jacobian[:, :, i]
        diffabs[i] = np.sum(np.abs(findiff - jacparam))
        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=3)
            fig.suptitle(f'{names_params_gauss[i]} gradients')
            axes[0][0].imshow(model)
            axes[0][0].set_title("Model")
            axes[0][1].imshow(output)
            axes[0][1].set_title("Model (modified)")
            axes[1][0].imshow(findiff)
            axes[1][0].set_title(f"Finite difference (dx={dxi:.2e})")
            axes[1][1].imshow(jacparam)
            axes[1][1].set_title("Exact Jacobian")
            pcdiff = 100*(1-jacparam/findiff)
            pcdiff[np.abs(jacparam - findiff) < 10*dxi] = 0
            axes[1][2].imshow(pcdiff)
            axes[1][2].set_title("Percent difference")
            plt.show()
        if printout:
            print((f'{names_params_gauss[i]:{format_param_name}} LLnew={llnew:.3f} '
                   f'LLdiff={llnew-ll:.3e} with dx={dxi:.2e}'))
        dlls[i] = (llnew - ll)/dxi
    return grads, dlls, diffabs


def mgsersic_test(reff=3, nser=1, dimx=15, dimy=None, plot=False, use_fast_gauss=True, mgsersic_order=8,
                  do_meas_modelfit=False, flux=1.):
    """
    Test multi-Gaussian Sersic approximations compared to the 'true' Sersic profile in 2D.
    :param reff: float; the circular effective radius in pixels.
    :param nser: float; the Sersic index. Must be >=0.5 and should be <= 6.3.
    :param dimx: int; image x dimensions in pixels.
    :param dimy: int; image y dimensions in pixels.
    :param plot: bool; whether to plot images and residuals.
    :param use_fast_gauss: bool; whether to use the built-in fast Gaussian evaluation. If False, the default
        rendering engine will be used (likely GalSim).
    :param mgsersic_order: int; the order of the Gaussian approximation.
        See MultiGaussianApproximationComponent for supported values.
    :param do_meas_modelfit: bool; whether to test meas_modelfit's Tractor-based (Hogg & Lang '13) profiles.
        Only n=1 and n=4 are supported.
    :param flux: float; the total flux of the source.
    :return: No return; diagnostics are printed.
    """
    if dimy is None:
        dimy = dimx
    engineopts = {"use_fast_gauss": True} if use_fast_gauss else None
    band = 'i'
    is_gauss = nser == 0.5
    keys = ("gaussian:1" if is_gauss else "sersic:1", f"mgsersic{mgsersic_order}:1")
    models = {
        key[0:3]: get_model({band: flux}, key, (dimx, dimy), sigma_xs=[reff], sigma_ys=[reff], slopes=[nser])
        for key in keys
    }
    keys = list(models.keys())
    models[keys[0]].engineopts = {"gsparams": get_gsparams(None), "drawmethod": "no_pixel"}
    imgs = {}
    for key, model in models.items():
        model.evaluate(get_likelihood=False, keep_images=True, do_draw_image=True, engineopts=engineopts)
        imgs[key] = model.data.exposures[band][0].meta['img_model']
    img_ref = imgs[keys[0]]
    if do_meas_modelfit:
        is_exp = nser == 1
        model = "lux" if is_exp else ("luv" if nser == 4 else None)
        if model is None:
            do_meas_modelfit = False
        else:
            from lsst.shapelet import RadialProfile
            from lsst.afw.geom.ellipses import Ellipse, Axes
            from lsst.geom import Point2D
            basis = RadialProfile.get(model).getBasis(6 if is_exp else 8, 4 if is_exp else 8)
            xc, yc, = dimx/2 - 0.5, dimy/2. - 0.5
            ellipse = Ellipse(Axes(a=reff, b=reff), Point2D(xc, yc))
            msf = basis.makeFunction(ellipse, np.array([flux]))
            img_mmf = np.zeros((dimy, dimx))
            msf.evaluate().addToImage(img_mmf)
            imgs["mmf"] = img_mmf
            keys.append("mmf")
    diffs = {key: imgs[key] - img_ref for key in keys[1:]}
    if plot:
        nrows = 1+do_meas_modelfit
        fig, axes = plt.subplots(nrows=nrows, ncols=4)
        for idx in range(1, nrows+1):
            axes_row = axes[idx-1] if nrows > 1 else axes
            for col, idx_key in enumerate([0, idx]):
                key = keys[idx_key]
                axes_row[col].imshow(np.log10(imgs[key]))
                axes_row[col].set_title(f"log10({key})")
            diff = diffs[key]
            axes_row[2].imshow(diff)
            axes_row[2].set_title(f"{keys[idx]}-{keys[0]}")
            axes_row[3].imshow(diff/img_ref)
            axes_row[3].set_title(f"({keys[idx]}-{keys[0]})/{keys[0]}")
        plt.show()
    return diffs
