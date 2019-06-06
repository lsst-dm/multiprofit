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
import multiprofit as mpf
from multiprofit.ellipse import Ellipse
import multiprofit.gaussutils as mpfgauss
from multiprofit.objects import names_params_gauss
from multiprofit.utils import estimateellipse
import timeit


# Example usage:
# test = testgaussian(nbenchmark=1000)
# for x in test:
#   print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))
def gaussian_test(xdim=49, ydim=51, reffs=None, angs=None, axrats=None, nbenchmark=0, nsub=1,
                  do_like=True, do_grad=False, do_jac=False):
    if reffs is None:
        reffs = [2.0, 5.0]
    if angs is None:
        angs = np.linspace(0, 90, 7)
    if axrats is None:
        axrats = [0.01, 0.1, 0.2, 0.5, 1]
    results = []
    hasgs = find_spec('galsim') is not None
    nparams = nsub*6
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
                        sigma_x, sigma_y, rho = mpfgauss.ellipsetocovar(
                            mpfgauss.reff2sigma(reff), axrat, ang, returnmatrix=False, returnparams=True)
                        gaussarrays = ','.join(np.repeat('[' + ','.join(np.repeat('{}', 9)).format(
                            xdim/2, ydim/2, 1/nsub, sigma_x, sigma_y, rho, 0, 0, 0) + ']', nsub))
                    if do_like:
                        functions['like'] = (
                            'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']), 0, {}, 0, {}, False, zeros, zeros, zeros_s, zeros)').format(
                            xdim, ydim, ydim, xdim)
                    if do_grad:
                        functions['grad'] = (
                            'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']), 0, {}, 0, {}, False, zeros, grads, zeros_s, zeros)').format(
                            xdim, ydim, ydim, xdim)
                    if do_jac:
                        functions['jac'] = (
                                'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                                ']), 0, {}, 0, {}, False, zeros, grads, zeros_s, zeros)').format(
                            xdim, ydim, ydim, xdim)
                    timesmpf = {
                        key: np.min(timeit.repeat(
                            callstr,
                            setup='import multiprofit as mpf;' + (
                                  'import numpy as np; data=np.zeros(({}, {})); zeros = np.zeros((0, 0));'
                                  'zeros_s = np.zeros(0, dtype=np.uint64);'.format(ydim, xdim)
                                if key in ['grad', 'like', 'jac'] else '') + (
                                'grads=np.zeros(({}))'.format(nparams) if key == 'grad' else (
                                    'grads=np.zeros(({},{},{}))'.format(ydim, xdim, nparams) if key == 'jac'
                                    else ''
                                )
                            ),
                            repeat=nbenchmark, number=1))
                        for key, callstr in functions.items()
                    }
                    if hasgs:
                        timegs = np.min(timeit.repeat(
                            'x=gs.Gaussian(flux=1, half_light_radius={}).shear(q={}, beta={}*gs.degrees)'
                            '.drawImage(nx={}, ny={}, scale=1, method="no_pixel").array'.format(
                                reff*np.sqrt(axrat), axrat, ang, xdim, ydim
                            ),
                            setup='import galsim as gs', repeat=nbenchmark, number=1
                        ))
                    mpffuncs = list(timesmpf.keys())
                    times = [timesmpf[x] for x in mpffuncs] + [timegs] if hasgs else []
                    result += ';' + '/'.join(mpffuncs) + ('/GalSim' if hasgs else '') + ' times=(' + \
                              ','.join(['{:.3e}'.format(x) for x in times]) + ')'
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
    source = Ellipse().set_from_covariance(mpfgauss.ellipsetocovar(mpfgauss.reff2sigma(reff), axrat, ang))
    psf = Ellipse()
    if reff_psf > 0:
        psf.set_from_covariance(mpfgauss.ellipsetocovar(mpfgauss.reff2sigma(reff_psf), axrat_psf, ang_psf))
        conv = source.convolve(psf, new=True)
        reff_conv, axrat_conv, ang_conv = mpfgauss.covartoellipse(conv)
        reff_conv = mpfgauss.sigma2reff(reff_conv)
    else:
        conv = source
        reff_conv, axrat_conv, ang_conv = reff, axrat, ang

    sigma_x, sigma_y, rho = source.get()
    sigma_x_psf, sigma_y_psf, rho_psf = psf.get()

    model = mpf.make_gaussian_pixel(cenx, ceny, flux, reff_conv, axrat_conv, ang_conv,
                                    0, dimx, 0, dimy, dimx, dimy)
    if printout:
        sumimg = np.sum(model)
        print("Modelsum = {:5e} vs flux {:.5e} ({:.2f}% missing)".format(
            sumimg, flux, (flux-sumimg)/sumimg/100))
        print("Source: {}".format(source))
        print("PSF: {}".format(psf))
        print("Convolved: {}".format(conv))
        print("reff, axrat, ang: ", (reff_conv, axrat_conv, ang_conv))
        print("Estimated ellipse:", Ellipse.covar_matrix_as(
            estimateellipse(model, cenx=cenx, ceny=ceny, denoise=False), params=True))
        print("Estimated deconvolved ellipse:", Ellipse.covar_matrix_as(estimateellipse(
            model, cenx=cenx, ceny=ceny, denoise=False, deconvolution_matrix=psf.getcovariance()),
            params=True))
    data = np.random.poisson(model + bg) - bg
    output = np.zeros_like(data)
    nparams = 6
    grads = np.zeros(nparams)
    jacobian = np.zeros([dimy, dimx, nparams])
    zeros = np.zeros((0, 0))
    zeros_s = np.zeros(0, dtype=np.uint64)
    paramsinit = np.array([[cenx, ceny, flux, sigma_x, sigma_y, rho, sigma_x_psf, sigma_y_psf, rho_psf]])
    # Compute the log likelihood and gradients
    ll = mpf.loglike_gaussians_pixel(
        data, np.array([[1/bg]]), paramsinit, 0, dimx, 0, dimy, False, zeros, grads, zeros_s, zeros)
    mpf.loglike_gaussians_pixel(
        data, np.array([[1/bg]]), paramsinit, 0, dimx, 0, dimy, False, zeros, jacobian, zeros_s, zeros)
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
        llnew = mpf.loglike_gaussians_pixel(
            data, np.array([[1/bg]]), params, 0, dimx, 0, dimy, False, zeros, zeros, zeros_s, zeros
        )
        mpf.loglike_gaussians_pixel(
            data, np.array([[1/bg]]), params, 0, dimx, 0, dimy, False, output, zeros, zeros_s, zeros
        )
        findiff = (output-model)/dxi
        jacparam = jacobian[:, :, i]
        diffabs[i] = np.sum(np.abs(findiff - jacparam))
        if plot:
            import matplotlib.pyplot as plt
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
