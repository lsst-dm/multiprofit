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
# TODO: Replace with e.g. fit_source.get_model?
from lsst.multiprofit.fitutils import get_model
import gauss2d as g2
import timeit

names_params_ellipse = ['sigma_x', 'sigma_y', 'rho']
names_params_ellipse_psf = ['psf_' + x for x in names_params_ellipse]
names_params_gauss = ["cen_x", "cen_y", "flux"] + names_params_ellipse
num_params_gauss = len(names_params_gauss)
order_params_gauss = {name: idx for idx, name in enumerate(names_params_gauss)}


# Fairly standard moment of inertia estimate of ellipse orientation and size
# TODO: compare with galsim's convenient calculateHLR/FWHM
# TODO: replace with the stack's method (in meas_?)
def estimate_ellipse(
        img, cen_x=None, cen_y=None, denoise=True, deconvolution_params=None, sigma_sq_min=1e-8,
        do_recenter=True, return_cens=False, validate=True, contiguous=False, pixel_min=None,
        pixel_sn_min=None, sigma_inverse=None, num_pix_min=2, raise_on_fail=False, return_as_params=False):
    """

    :param img: ndarray; an image of a source to estimate the moments of.
    :param cen_x, cen_y: float; initial estimate of the centroid of the source.
    :param denoise: bool; whether to attempt to naively de-noise the image by zeroing all negative pixels and
        the faintest positive pixels while conserving the total flux.
    :param deconvolution_params: array-like; xx, yy and xy moments to subtract from measurements for
        deconvolution
    :param sigma_sq_min: float; the minimum variance to return, especially if deconvolving.
    :param do_recenter: bool; whether to iteratively re-estimate the centroid.
    :param return_cens: bool; whether to return the centroid.
    :param validate: bool; whether to check if the image has positive net flux before processing.
    :param contiguous: bool; whether to keep only the contiguous above-threshold pixels around the centroid.
    :param pixel_min: float; the minimum threshold pixel value. Default 0. Clipped to positive.
    :param pixel_sn_min: float; the minimum threshold pixel S/N. Default 0. Clipped to positive. Ignored if
        sigma_inverse is not supplied.
    :param sigma_inverse: ndarray; an inverse error map for applying S/N cuts. Must be same size/units as img.
    :param num_pix_min: int; the minimum number of positive pixels required for processing.
    :param raise_on_fail: bool; whether to raise an Exception on any failure instead of attempting to
        continue by not strictly obeying the inputs.
    :param return_as_params: bool; whether to return a tuple of sigma_x^2, sigma_y^2, covar instead of the
        full matrix (which is symmetric)
    :return: inertia: ndarray; the moment of inertia/covariance matrix or parameters.
        cen_x, cen_y: The centroids, if return_cens is True.
    """
    if img is None:
        raise ValueError('img cannot be None')
    if not (sigma_sq_min >= 0):
        raise ValueError(f'sigma_sq_min={sigma_sq_min} !>= 0')
    if validate or denoise:
        sum_img = np.sum(img)
    if validate and raise_on_fail:
        if not sum_img > 0:
            raise RuntimeError(f"Tried to estimate ellipse for img(shape={img.shape}, sum={sum_img})")
    if cen_x is None:
        cen_x = img.shape[1]/2.
    if cen_y is None:
        cen_y = img.shape[0]/2.
    pixel_min = 0 if pixel_min is None else np.max([0, pixel_min])
    pixel_sn_min = 0 if pixel_sn_min is None else np.max([0, pixel_sn_min])
    if denoise and sum_img > 0:
        img = absconservetotal(np.copy(img))
    mask = img > pixel_min
    if sigma_inverse is not None:
        mask &= (img > pixel_sn_min)
    if contiguous:
        pix_cen_x, pix_cen_y = round(cen_x), round(cen_y)
        if img[pix_cen_y, pix_cen_x] > pixel_min:
            labels, _ = ndimage.label(mask)
            mask = labels == labels[pix_cen_y, pix_cen_x]
    y_0, x_0 = np.nonzero(mask)
    num_pix = len(y_0)
    flux = img[y_0, x_0]
    flux_sum = np.sum(flux)
    if not ((num_pix >= num_pix_min) and (flux_sum > 0)):
        if raise_on_fail:
            raise RuntimeError(f'estimate_ellipse failed with n_pix={num_pix} !>= min={num_pix_min}'
                               f' and/or flux_sum={flux_sum} !>0')
        finished = True
        i_xx, i_yy, i_xy = sigma_sq_min, sigma_sq_min, 0.
    else:
        finished = False
        i_xx, i_yy, i_xy = 0., 0., 0.

    while not finished:
        y = y_0 + 0.5 - cen_y
        x = x_0 + 0.5 - cen_x
        x_sq = x**2
        y_sq = y**2
        xy = x*y
        i_xx = np.sum(flux*x_sq)/flux_sum
        i_yy = np.sum(flux*y_sq)/flux_sum
        i_xy = np.sum(flux*xy)/flux_sum

        if do_recenter:
            x_shift = np.sum(flux*x)/flux_sum
            y_shift = np.sum(flux*y)/flux_sum
            finished = np.abs(x_shift) < 0.1 and np.abs(y_shift) < 0.1
            if not finished:
                cen_x_new = cen_x + x_shift
                cen_y_new = cen_y + y_shift
                if not ((0 < cen_x_new < img.shape[1]) and (0 < cen_y_new < img.shape[0])):
                    if raise_on_fail:
                        raise RuntimeError(f'cen_y,cen_x={cen_y},{cen_x} outside img.shape={img.shape}')
                    finished = True
                cen_x = cen_x_new
                cen_y = cen_y_new
        else:
            finished = True

    deconvolve = deconvolution_params is not None
    if deconvolve:
        d_xx, d_yy, d_xy = deconvolution_params
        if not ((i_xx > d_xx) and (i_yy > d_yy)):
            if raise_on_fail:
                raise RuntimeError(f'Moments {i_xx},{i_yy} not > deconvolution {d_xx},{d_yy}')
            cor = i_xy/np.sqrt(i_xx*i_yy) if (i_xx > 0 and i_yy > 0) else 0
            i_xx /= 2
            i_yy /= 2
            i_xy = 0 if (cor == 0) else i_xy/2
        else:
            cor = i_xy/np.sqrt(i_xx*i_yy)
            i_xx -= d_xx
            i_yy -= d_yy
            i_xy -= d_xy

    # Need to clip here before finishing deconvolving or else could get !(-1 < cor_new < 1)
    i_xx = np.clip(i_xx, sigma_sq_min, np.Inf)
    i_yy = np.clip(i_yy, sigma_sq_min, np.Inf)

    if deconvolve:
        cor_new = i_xy/np.sqrt(i_xx*i_yy) if (i_xx > 0 and i_yy > 0) else np.nan
        if not (-1 < cor_new < 1):
            if raise_on_fail:
                raise RuntimeError(f'Deconvolved moments {i_xx},{i_yy},{i_xy} give !(-1<rho={cor_new}<1)')
            if cor is not None and -1 < cor < 1:
                i_xy = cor*np.sqrt(i_xx*i_yy)
            else:
                i_xy = 0

    inertia = np.array(((i_xx, i_xy), (i_xy, i_yy))) if not return_as_params else (i_xx, i_yy, i_xy)
    if not np.all(np.isfinite(inertia)):
        raise RuntimeError(f'Inertia {inertia} not all finite')
    if return_cens:
        return inertia, cen_x, cen_y
    return inertia


# TODO: Make this a class?
def get_gsparams(engineopts=None):
    return engineopts.get("gsparams", gs.GSParams()) if engineopts is not None else gs.GSParams()


def get_setup(xdim=15, ydim=15, r_major=1, axrat=1, angle=0, nsub=1,
              do_output=False, do_like=False, do_residual=False, do_grad=False, do_jac=False, noise=1e-2):
    cmds = [
        'import gauss2d as g2',
        'import numpy as np',
        f'xdim={xdim}',
        f'ydim={ydim}',
        f'centroid = g2.Centroid({xdim}/2, {ydim}/2)',
        'kernel = g2.Gaussian(centroid=g2.Centroid(0, 0),'
        'ellipse=g2.Ellipse(sigma_x=0., sigma_y=0))',
        'ellipse = g2.Ellipse(g2.EllipseMajor('
        f'r_major={r_major}*g2.M_SIGMA_HWHM, axrat={axrat}, angle={angle}, degrees=True))',
        f'source = g2.Gaussian(centroid=centroid, ellipse=ellipse,'
        f' integral=g2.GaussianIntegralValue(1/{nsub}))',
        f'gaussians = g2.ConvolvedGaussians([g2.ConvolvedGaussian(source, kernel) for _ in range({nsub})])',
    ]
    img = "g2.ImageD"
    arr = "g2.ImageArrayD"
    cmds.extend([
            f'data = {img}(data=g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim).data'
            f' + np.random.normal(scale={noise}, size=[{ydim}, {xdim}]))',
            f'sigma_inv = {img}(data=1./np.array([[{noise}]]))',
        ] if (do_like or do_residual or do_grad or do_jac) else ['data, sigma_inv = None, None'])
    cmds.append(
        f'output = {f"{img}(n_rows={ydim}, n_cols={xdim})" if do_output else "None"}'
    )
    cmds.append(
        f'residual = {f"{img}(n_rows={ydim}, n_cols={xdim})" if do_residual else "None"}'
    )
    # TODO: Find a better way to set n_params=6
    args_grad = f"{arr}([{img}(n_rows={ydim}, n_cols={xdim}) for _ in range(6*{nsub})])" if do_jac else (
        f"{arr}([{img}(n_rows={nsub}, n_cols=6)])" if do_grad else None
    )
    cmds.append(f'grads = {args_grad}')
    if do_output or do_like or do_residual or do_grad or do_jac:
        args = ", ".join([f"{x}={x}"
                          for x in ("gaussians", "data", "sigma_inv", "output", "residual", "grads")
                          ])
        cmds.append(f'evaluator = g2.GaussianEvaluatorD({args})')
    return '; '.join(cmds)


# Example usage:
# test = gaussian_test(nbenchmark=1000)
# for x in test:
#   print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))
def gaussian_test(xdim=49, ydim=51, reffs=None, angs=None, axrats=None, nbenchmark=0, nsub=1,
                  do_like=True, do_residual=False, do_grad=False, do_jac=False, do_meas_modelfit=False,
                  plot=False):
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
    :param do_residual: bool; whether to evaluate the residual
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
        axrats = [1, 0.5, 0.2, 0.1, 0.01]
    results = []
    hasgs = find_spec('galsim') is not None
    num_params = nsub*6
    if hasgs:
        import galsim as gs

    centroid = g2.Centroid(xdim/2, ydim/2)
    kernel = g2.Gaussian(centroid=g2.Centroid(0, 0), ellipse=g2.Ellipse(sigma_x=0., sigma_y=0))

    cmd_func = 'g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim,)'
    cmd_obj = 'evaluator.loglike_pixel()'

    functions = {
        'eval': (cmd_func, {'nsub': nsub}),
        'output': (cmd_obj, {'nsub': nsub, 'do_output': True}),
    }
    if do_like:
        functions['like'] = (cmd_obj, {'nsub': nsub, 'do_like': True})
    if do_residual:
        functions['residual'] = (cmd_obj, {'nsub': nsub, 'do_residual': True})
    if do_grad:
        functions['grad'] = (cmd_obj, {'nsub': nsub, 'do_grad': True})
    if do_jac:
        functions['jac'] = (cmd_obj, {'nsub': nsub, 'do_jac': True})

    for key, (cmd, kwargs_setup) in functions.items():
        setup = get_setup(**kwargs_setup)
        print(f'Evaluating with {key}: "{setup}; {cmd}"')
        exec('; '.join((setup, cmd)), locals(), locals())

    for reff in reffs:
        for ang in angs:
            for axrat in axrats:
                ellipse = g2.Ellipse(
                    g2.EllipseMajor(r_major=reff*g2.M_SIGMA_HWHM, axrat=axrat, angle=ang, degrees=True)
                )
                source = g2.Gaussian(centroid=centroid, ellipse=ellipse,
                                     integral=g2.GaussianIntegralValue(1/nsub))
                gaussians = g2.ConvolvedGaussians([g2.ConvolvedGaussian(source, kernel) for _ in range(nsub)])
                gaussmpf = g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim).data
                result = 'Ran make'
                if hasgs:
                    gaussgs = gs.Gaussian(flux=1, half_light_radius=reff*np.sqrt(axrat)).shear(
                        q=axrat, beta=ang*gs.degrees).drawImage(
                        nx=xdim, ny=ydim, scale=1, method='no_pixel').array
                    if plot:
                        fig, ax = plt.subplots(ncols=3)
                        ax[0].imshow(np.log10(gaussmpf))
                        ax[1].imshow(np.log10(gaussgs))
                        ax[2].imshow(gaussgs - gaussmpf)
                        plt.show()
                    gstonew = np.sum(np.abs(gaussmpf-gaussgs))
                    result += f'; GalSim/new residual=({gstonew:.3e})'
                if nbenchmark > 0:
                    times = {
                        key: np.min(timeit.repeat(
                            cmd,
                            setup=get_setup(xdim, ydim, reff, axrat, ang, **kwargs_setup),
                            repeat=nbenchmark,
                            number=1,
                        ))
                        for key, (cmd, kwargs_setup) in functions.items()
                    }
                    if do_meas_modelfit:
                        ang_rad = np.deg2rad(ang)
                        for key in ("dev", "exp"):
                            times[f"mmf-{key}"] = np.min(timeit.repeat(
                                "msf.evaluate().addToImage(img)",
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
                            f'x=profile.drawImage(nx={xdim}, ny={ydim}, scale=1, method="no_pixel").array',
                            setup=f'import galsim as gs; profile = gs.Gaussian(flux=1,'
                                  f' half_light_radius={reff*np.sqrt(axrat)}).shear('
                                  f'q={axrat}, beta={ang}*gs.degrees)',
                            repeat=nbenchmark, number=1
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
                  reff_psf=0, axrat_psf=0.95, ang_psf=0, n_psfs=1, printout=False, plot=False):

    if n_psfs > 1:
        raise ValueError("n_psfs>1 not yet supported")

    cen_x, cen_y = dimx/2., dimy/2.
    # Keep this in units of sigma, not re==FWHM/2
    source = g2.EllipseMajor(reff*g2.M_SIGMA_HWHM, axrat, ang, degrees=True)
    source_g = g2.Ellipse(source)
    has_psf = reff_psf > 0
    psf = g2.EllipseMajor(reff_psf*g2.M_SIGMA_HWHM, axrat_psf, ang_psf, degrees=True)
    psf_g = g2.Ellipse(psf)
    conv = g2.EllipseMajor(source_g.make_convolution(psf_g), degrees=True)

    source = g2.Gaussian(centroid=g2.Centroid(x=cen_x, y=cen_y), ellipse=source_g,
                         integral=g2.GaussianIntegralValue(flux))
    values = (source.centroid.x, source.centroid.y, source.integral.value,
              source.ellipse.sigma_x, source.ellipse.sigma_y, source.ellipse.rho,)

    def set_param(gauss, idx, value):
        if idx == 0:
            gauss.centroid.x = value
        elif idx == 1:
            gauss.centroid.y = value
        elif idx == 2:
            gauss.integral.value = value
        elif idx == 3:
            gauss.ellipse.sigma_x = value
        elif idx == 4:
            gauss.ellipse.sigma_y = value
        else:
            gauss.ellipse.rho = value

    gaussians = g2.ConvolvedGaussians([
        g2.ConvolvedGaussian(
            source=source,
            kernel=g2.Gaussian(centroid=g2.Centroid(x=dimx/2, y=dimy/2), ellipse=psf_g,
                               integral=g2.GaussianIntegralValue(1./n_psfs)),
        )
        for _ in range(n_psfs)
    ])
    model = g2.make_gaussians_pixel_D(gaussians, n_rows=dimy, n_cols=dimx).data

    if printout:
        psf_c = g2.Covariance(psf_g)
        deconv_params = [psf_c.sigma_x_sq, psf_c.sigma_y_sq, psf_c.cov_xy]
        deconvs = (False, True) if has_psf else (False,)
        covar_ests = [
            g2.Covariance(
                *estimate_ellipse(model, cen_x=cen_x, cen_y=cen_y, denoise=False, return_as_params=True,
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
        print("Estimated ellipse (ellipse):", g2.EllipseMajor(covar_ests[0]))
        if has_psf:
            print("Estimated deconvolved ellipse (covar):", covar_ests[1])
            print("Estimated deconvolved ellipse (ellipse):", g2.EllipseMajor(covar_ests[1]))

    data = g2.ImageD(np.random.poisson(model + bg) - bg)
    sigma_inv = g2.ImageD(np.array([[1/bg]]))
    output = g2.ImageD(np.zeros_like(data.data))
    n_params = 6
    grads = g2.ImageArrayD([g2.ImageD(np.zeros((n_psfs, n_params)))])

    # Compute the log likelihood and gradients
    evaluator_i = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, output=output)
    ll_i = evaluator_i.loglike_pixel()

    evaluator_g = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, grads=grads)
    ll_g = evaluator_g.loglike_pixel()

    jacobian = np.zeros([dimy, dimx, n_params*n_psfs])
    jacobian_arr = g2.ImageArrayD([g2.ImageD(jacobian[:, :, idx].view())
                                   for idx in range(n_params*n_psfs)])
    evaluator_j = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, grads=jacobian_arr)
    evaluator_j.loglike_pixel()

    dxs = [1e-6, 1e-6, flux*1e-6, 1e-8, 1e-8, 1e-8]
    dlls = np.zeros(n_params)
    diffabs = np.zeros(n_params)
    format_param_name = f'<{max(len(param_name) for param_name in names_params_gauss)}'

    for i, dxi in enumerate(dxs):
        dx = np.zeros(n_params)
        dx[i] = dxi
        value = values[i]
        set_param(source, i, value + dxi)
        # Note that there's no option to return the log likelihood and the Jacobian - the latter skips
        # computing the former for efficiency, assuming that you won't need it
        llnewg, llnewi = (evaltor.loglike_pixel() for evaltor in (evaluator_g, evaluator_i))

        # It's actually the jacobian of the residual
        findiff = -(output.data - model)/dxi*sigma_inv.data
        jacparam = jacobian[:, :, i]
        diffabs[i] = np.sum(np.abs(findiff - jacparam))
        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=3)
            fig.suptitle(f'{names_params_gauss[i]} gradients')
            axes[0][0].imshow(model)
            axes[0][0].set_title("Model")
            axes[0][1].imshow(output.data)
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
            print(
                f'{names_params_gauss[i]:{format_param_name}} LL, LLnew1, llnew12 = {ll_g:.4e}, {llnewg:.4e}'
                f', {llnewi:.4e}; LLdiff={llnewg-ll_g:.4e} with dx={dxi:.3e}'
            )
        dlls[i] = (llnewi - ll_i)/dxi
        set_param(source, i, value)
    return grads.at(0).data, dlls, diffabs


# TODO: port/fix this
def mgsersic_test(reff=3, nser=1, axrat=1, angle=0, dimx=15, dimy=None, plot=False, use_fast_gauss=True,
                  mgsersic_order=8, do_meas_modelfit=False, flux=1.):
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
    engineopts = {"use_fast_gauss": True, "drawmethod": "no_pixel"} if use_fast_gauss else None
    band = 'i'
    is_gauss = nser == 0.5
    keys = ("gaussian:1" if is_gauss else "sersic:1", f"mgsersic{mgsersic_order}:1")
    ell = g2.Ellipse(g2.EllipseMajor(r_major=reff, axrat=axrat, angle=angle, degrees=True))
    models = {
        key[0:3]: get_model({band: flux}, key, (dimx, dimy),
                            sigma_xs=[ell.sigma_x], sigma_ys=[ell.sigma_y], slopes=[nser])
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
