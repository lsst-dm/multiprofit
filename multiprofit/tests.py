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
import multiprofit as mpf
from multiprofit.ellipse import Ellipse
from multiprofit.fitutils import get_model
import multiprofit.gaussutils as mpfgauss
from multiprofit.objects import names_params_gauss
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
                gaussmpfold = mpf.make_gaussian_pixel_sersic(xdim/2, ydim/2, 1, reff, axrat, ang,
                                                             0, xdim, 0, ydim, xdim, ydim)
                gaussmpf = mpf.make_gaussian_pixel(xdim/2, ydim/2, 1, reff, axrat, ang,
                                                   0, xdim, 0, ydim, xdim, ydim)
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
                        'old': 'mpf.make_gaussian_pixel_sersic' + argsmpf,
                        'new': 'mpf.make_gaussian_pixel' + argsmpf,
                    }
                    if do_like or do_grad or do_jac:
                        sigma_x, sigma_y, rho = mpfgauss.ellipse_to_covar(
                            mpfgauss.reff_to_sigma(reff), axrat, ang, return_as_matrix=False,
                            return_as_params=True)
                        gaussarrays = ','.join(np.repeat('[' + ','.join(np.repeat('{}', 9)).format(
                            xdim/2, ydim/2, 1/nsub, sigma_x, sigma_y, rho, 0, 0, 0) + ']', nsub))
                    if do_like:
                        functions['like'] = (
                            'mpfg.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']))')
                    if do_grad:
                        functions['grad'] = (
                            'mpfg.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']), grad=grads)')
                    if do_jac:
                        functions['jac'] = 'mpfg.loglike_gaussians_pixel(data, data, ' \
                                           'np.array([{}]), grad=grads)'.format(gaussarrays)
                    times = {
                        key: np.min(timeit.repeat(
                            callstr,
                            setup='import multiprofit as mpf; import multiprofit.gaussutils as mpfg;' + (
                                  'import numpy as np; data=np.zeros(({}, {})); zeros = np.zeros((0, 0));'
                                  'zeros_s = np.zeros(0, dtype=np.uint64);'.format(ydim, xdim)
                                if key in ['grad', 'like', 'jac'] else '') + (
                                'grads=np.zeros(({}))'.format(num_params) if key == 'grad' else (
                                    'grads=np.zeros(({},{},{}))'.format(ydim, xdim, num_params) if
                                    key == 'jac' else ''
                                )
                            ),
                            repeat=nbenchmark, number=1))
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
                            'x=gs.Gaussian(flux=1, half_light_radius={}).shear(q={}, beta={}*gs.degrees)'
                            '.drawImage(nx={}, ny={}, scale=1, method="no_pixel").array'.format(
                                reff*np.sqrt(axrat), axrat, ang, xdim, ydim
                            ),
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
    source = Ellipse().set_from_covariance(
        mpfgauss.ellipse_to_covar(mpfgauss.reff_to_sigma(reff), axrat, ang))
    psf = Ellipse()
    if reff_psf > 0:
        psf.set_from_covariance(mpfgauss.ellipse_to_covar(
            mpfgauss.reff_to_sigma(reff_psf), axrat_psf, ang_psf))
        conv = source.convolve(psf, new=True)
        reff_conv, axrat_conv, ang_conv = mpfgauss.covar_to_ellipse(conv)
        reff_conv = mpfgauss.sigma_to_reff(reff_conv)
    else:
        conv = source
        reff_conv, axrat_conv, ang_conv = reff, axrat, ang

    sigma_x, sigma_y, rho = source.get()
    sigma_x_psf, sigma_y_psf, rho_psf = psf.get()

    model = mpf.make_gaussian_pixel(cenx, ceny, flux, reff_conv, axrat_conv, ang_conv,
                                    0, dimx, 0, dimy, dimx, dimy)
    if printout:
        sum_img = np.sum(model)
        print("Modelsum = {:5e} vs flux {:.5e} ({:.2f}% missing)".format(
            sum_img, flux, (flux-sum_img)/sum_img/100))
        print("Source: {}".format(source))
        print("PSF: {}".format(psf))
        print("Convolved: {}".format(conv))
        print("reff, axrat, ang: ", (reff_conv, axrat_conv, ang_conv))
        print("Estimated ellipse:", Ellipse.covar_matrix_as(
            estimate_ellipse(model, cenx=cenx, ceny=ceny, denoise=False), params=True))
        print("Estimated deconvolved ellipse:", Ellipse.covar_matrix_as(estimate_ellipse(
            model, cenx=cenx, ceny=ceny, denoise=False, deconvolution_matrix=psf.get_covariance()),
            params=True))
    data = np.random.poisson(model + bg) - bg
    output = np.zeros_like(data)
    num_params = 6
    grads = np.zeros(num_params)
    params_init = np.array([[cenx, ceny, flux, sigma_x, sigma_y, rho, sigma_x_psf, sigma_y_psf, rho_psf]])
    # Compute the log likelihood and gradients
    mpfgauss.loglike_gaussians_pixel(data, np.array([[1/bg]]), params_init)
    ll = mpfgauss.loglike_gaussians_pixel(data, np.array([[1/bg]]), params_init, grad=grads)
    jacobian = np.zeros([dimy, dimx, num_params])
    mpfgauss.loglike_gaussians_pixel(data, np.array([[1/bg]]), params_init, grad=jacobian)
    dxs = [1e-6, 1e-6, flux*1e-6, 1e-8, 1e-8, 1e-8]
    dlls = np.zeros(6)
    diffabs = np.zeros(6)
    format_param_name = '{:<' + str(max(len(param_name) for param_name in names_params_gauss)) + '}'
    for i, dxi in enumerate(dxs):
        dx = np.zeros(6)
        dx[i] = dxi
        # Note that mpf computes dll/drho where the diagonal term is rho*sigma_x*sigma_y
        params = np.array([[cenx + dx[0], ceny + dx[1], flux+dx[2], sigma_x+dx[3], sigma_y+dx[4], rho+dx[5],
                            sigma_x_psf, sigma_y_psf, rho_psf]])
        # Note that there's no option to return the log likelihood and the Jacobian - the latter skips
        # computing the former for efficiency, assuming that you won't need it
        llnew = mpfgauss.loglike_gaussians_pixel(data, np.array([[1/bg]]), params, output=output)
        findiff = (output-model)/dxi
        jacparam = jacobian[:, :, i]
        diffabs[i] = np.sum(np.abs(findiff - jacparam))
        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=3)
            fig.suptitle(names_params_gauss[i] + ' gradients sx={:.2e} sy={:.2e} ')
            axes[0][0].imshow(model)
            axes[0][0].set_title("Model")
            axes[0][1].imshow(output)
            axes[0][1].set_title("Model (modified)")
            axes[1][0].imshow(findiff)
            axes[1][0].set_title("Finite difference (dx={:.2e})".format(dxi))
            axes[1][1].imshow(jacparam)
            axes[1][1].set_title("Exact Jacobian")
            pcdiff = 100*(1-jacparam/findiff)
            pcdiff[np.abs(jacparam - findiff) < 10*dxi] = 0
            axes[1][2].imshow(pcdiff)
            axes[1][2].set_title("Percent difference")
            plt.show()
        if printout:
            print((format_param_name + ' LLnew={:.3f} LLdiff={:.3e} with dx={:.2e}').format(
                names_params_gauss[i], llnew, llnew-ll, dxi))
        dlls[i] = (llnew - ll)/dxi
    return grads, dlls, diffabs


def mgsersic_test(reff=3, nser=1, dimx=15, dimy=None, plot=False, use_fast_gauss=True, mgsersic_order=8,
                  do_meas_modelfit=False, flux=1.):
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
    imgs = {}
    for key, model in models.items():
        model.evaluate(get_likelihood=False, keep_images=True, do_draw_image=True, engineopts=engineopts)
        imgs[key] = model.data.exposures[band][0].meta['img_model']
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
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=3+2*do_meas_modelfit)
        for idx, (key, img) in enumerate(imgs.items()):
            axes[idx].imshow(img)
            axes[idx].set_title(key)
        idx += 1
        axes[idx].imshow(imgs[keys[1]] - imgs[keys[0]])
        axes[idx].set_title(f"{keys[1]}-{keys[0]}")
        if do_meas_modelfit:
            idx += 1
            axes[idx].imshow(imgs[keys[2]] - imgs[keys[0]])
            axes[idx].set_title(f"{keys[2]}-{keys[0]}")
        plt.show()