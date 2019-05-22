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
import multiprofit.gaussutils as mpfgauss
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
                gaussmpfold = mpf.make_gaussian_pixel_sersic(xdim/2, ydim/2, 1, reff, ang, axrat,
                                                             0, xdim, 0, ydim, xdim, ydim)
                gaussmpf = mpf.make_gaussian_pixel(xdim/2, ydim/2, 1, reff, ang, axrat,
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
                        xdim/2, ydim/2, 1, reff, ang, axrat, 0, xdim, 0, ydim, xdim, ydim)
                    functions = {
                        'old': 'mpf.make_gaussian_pixel_sersic' + argsmpf,
                        'new': 'mpf.make_gaussian_pixel' + argsmpf,
                    }
                    gaussarrays = ','.join(np.repeat('[' + ','.join(np.repeat('{}', 6)).format(
                        xdim/2, ydim/2, 1/nsub, reff, ang, axrat) + ']', nsub))
                    if do_like:
                        functions['like'] = (
                            'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']), False, 0, {}, 0, {}, False, zeros, zeros, zeros_s, zeros)').format(
                            xdim, ydim, ydim, xdim)
                    if do_grad:
                        functions['grad'] = (
                            'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                            ']), False, 0, {}, 0, {}, False, zeros, grads, zeros_s, zeros)').format(
                            xdim, ydim, ydim, xdim)
                    if do_jac:
                        functions['jac'] = (
                                'mpf.loglike_gaussians_pixel(data, data, np.array([' + gaussarrays +
                                ']), False, 0, {}, 0, {}, False, zeros, grads, zeros_s, zeros)').format(
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


def gradient_test(dimx=5, dimy=4, reff=2, ang=0, axrat=0.5, flux=1e4, bg=1e3, printout=False):
    cenx, ceny = dimx/2., dimy/2.
    model = mpf.make_gaussian_pixel(cenx, ceny, flux, reff, ang, axrat, 0, dimx, 0, dimy, dimx, dimy)
    data = np.random.poisson(model + bg) - bg
    output = np.zeros_like(data)
    nparams = 6
    grads = np.zeros(nparams)
    jacobian = np.zeros([dimy, dimx, nparams])
    zeros = np.zeros((0, 0))
    zeros_s = np.zeros(0, dtype=np.uint64)
    paramsinit = np.array([[cenx, ceny, flux, reff, ang, axrat]])
    # Compute the log likelihood and gradients
    ll = mpf.loglike_gaussians_pixel(
        data, np.array([[1/bg]]), paramsinit, False, 0, dimx, 0, dimy, False, zeros, grads, zeros_s, zeros)
    mpf.loglike_gaussians_pixel(
        data, np.array([[1/bg]]), paramsinit, False, 0, dimx, 0, dimy, False, zeros, jacobian, zeros_s, zeros)
    # Keep this in units of sigma, not re==FWHM/2
    covar = mpfgauss.ellipsetocovar(mpfgauss.reff2sigma(reff), axrat, ang)
    dxs = [1e-8, 1e-8, 1e-3, 1e-10, 1e-10, 1e-8]
    dlls = np.zeros(6)
    diffabs = np.zeros(6)
    sigmatore = 1/mpfgauss.reff2sigma(1)
    for i, dxi in enumerate(dxs):
        dx = np.zeros(6)
        dx[i] = dxi
        # Note that mpf computes dll/drho where the diagonal term is rho*sigma_x*sigma_y
        covarx = np.sqrt(covar[0, 0]) + dx[3]
        covary = np.sqrt(covar[1, 1]) + dx[4]
        covardiag = covar[0, 1] + dx[5]*covarx*covary
        # Convert covariance back to ellipse coordinates used by mpf (for now)
        # The eigen version is more accurate and doesn't break down at 45 degrees
        reff2, axrat2, ang2 = np.array(mpfgauss.covartoellipseeig(
            np.array([[np.square(covarx), covardiag], [covardiag, np.square(covary)]])
        ))
        reff2 *= sigmatore
        params = np.array([[cenx + dx[0], ceny + dx[1], flux+dx[2], reff2, ang2, axrat2]])
        # Note that there's no option to return the log likelihood and the Jacobian - the latter skips
        # computing the former for efficiency, assuming that you won't need it
        llnew = mpf.loglike_gaussians_pixel(
            data, np.array([[1/bg]]), params, False, 0, dimx, 0, dimy, False, zeros, zeros, zeros_s, zeros
        )
        mpf.loglike_gaussians_pixel(
            data, np.array([[1/bg]]), params, False, 0, dimx, 0, dimy, False, output, zeros, zeros_s, zeros
        )
        diffabs[i] = np.sum(np.abs((output - model) - jacobian[:, :, i]*dxi))
        if printout:
            print(reff2, axrat2, ang2, llnew, llnew-ll, dxi)
        dlls[i] = (llnew - ll)/dxi
    return grads, dlls, diffabs
