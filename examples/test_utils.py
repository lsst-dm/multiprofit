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
import math
import timeit
from typing import Iterable

import galsim as gs
import gauss2d as g2
import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    GaussianConfig,
    ParameterConfig,
    SersicConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.utils import get_params_uniq
import matplotlib.pyplot as plt
import numpy as np

names_params_ellipse = ["sigma_x", "sigma_y", "rho"]
names_params_ellipse_psf = ["psf_" + x for x in names_params_ellipse]
names_params_gauss = ["cen_x", "cen_y", "flux"] + names_params_ellipse
num_params_gauss = len(names_params_gauss)
order_params_gauss = {name: idx for idx, name in enumerate(names_params_gauss)}


def get_model(
    fluxes: dict[g2f.Channel, float], shape, sigma_xs, sigma_ys, rhos, nsers, order: int = 4
) -> g2f.Model:
    cens = g2f.CentroidParameters(shape[0] / 2.0, shape[1] / 2.0)
    components = []
    for sigma_x, sigma_y, rho, nser in zip(sigma_xs, sigma_ys, rhos, nsers):
        component, *_ = SersicConfig(
            size_x=ParameterConfig(value_initial=sigma_x, fixed=True),
            size_y=ParameterConfig(value_initial=sigma_y, fixed=True),
            rho=ParameterConfig(value_initial=rho, fixed=True),
            sersicindex=SersicIndexParameterConfig(value_initial=nser, fixed=True),
            order=order,
        ).make_component(centroid=cens, channels=fluxes.keys())
        for (channel, flux), param in zip(fluxes.items(), get_params_uniq(component, nonlinear=False)):
            param.value = flux
            param.fixed = True
        components.append(component)
    source = g2f.Source(components)
    img = g2.ImageD(np.zeros(shape, dtype=float))
    mask = g2.ImageB(np.ones(shape, dtype=bool))
    data = g2f.Data([g2f.Observation(img, img, mask, channel) for channel in fluxes.keys()])
    psf, *_ = GaussianConfig(
        size_x=ParameterConfig(value_initial=0.0, fixed=True),
        size_y=ParameterConfig(value_initial=0.0, fixed=True),
        rho=ParameterConfig(value_initial=0.0, fixed=True),
    ).make_component(centroid=g2f.CentroidParameters(0.0, 0.0), channels=[g2f.Channel.NONE])
    psfmodels = tuple((g2f.PsfModel([psf]) for _ in range(len(fluxes))))
    model = g2f.Model(data, psfmodels, [source])
    return model


def get_setup(
    xdim=15,
    ydim=15,
    r_major=1,
    axrat=1,
    angle=0,
    nsub=1,
    do_output=False,
    do_like=False,
    do_residual=False,
    do_grad=False,
    do_jac=False,
    noise=1e-2,
):
    cmds = [
        "import gauss2d as g2",
        "import numpy as np",
        f"xdim={xdim}",
        f"ydim={ydim}",
        f"centroid = g2.Centroid({xdim}/2, {ydim}/2)",
        "kernel = g2.Gaussian(centroid=g2.Centroid(0, 0)," "ellipse=g2.Ellipse(sigma_x=0., sigma_y=0))",
        "ellipse = g2.Ellipse(g2.EllipseMajor("
        f"r_major={r_major}*g2.M_SIGMA_HWHM, axrat={axrat}, angle={angle}, degrees=True))",
        f"source = g2.Gaussian(centroid=centroid, ellipse=ellipse,"
        f" integral=g2.GaussianIntegralValue(1/{nsub}))",
        f"gaussians = g2.ConvolvedGaussians([g2.ConvolvedGaussian(source, kernel) for _ in range({nsub})])",
    ]
    img = "g2.ImageD"
    arr = "g2.ImageArrayD"
    cmds.extend(
        [
            f"data = {img}(data=g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim).data"
            f" + np.random.normal(scale={noise}, size=[{ydim}, {xdim}]))",
            f"sigma_inv = {img}(data=1./np.array([[{noise}]]))",
        ]
        if (do_like or do_residual or do_grad or do_jac)
        else ["data, sigma_inv = None, None"]
    )
    cmds.append(f'output = {f"{img}(n_rows={ydim}, n_cols={xdim})" if do_output else "None"}')
    cmds.append(f'residual = {f"{img}(n_rows={ydim}, n_cols={xdim})" if do_residual else "None"}')
    # TODO: Find a better way to set n_params=6
    args_grad = (
        f"{arr}([{img}(n_rows={ydim}, n_cols={xdim}) for _ in range(6*{nsub})])"
        if do_jac
        else (f"{arr}([{img}(n_rows=1, n_cols={nsub*6})])" if do_grad else None)
    )
    cmds.append(f"grads = {args_grad}")
    if do_output or do_like or do_residual or do_grad or do_jac:
        args = ", ".join(
            [f"{x}={x}" for x in ("gaussians", "data", "sigma_inv", "output", "residual", "grads")]
        )
        cmds.append(f"evaluator = g2.GaussianEvaluatorD({args})")
    return "; ".join(cmds)


def gaussian_test(
    xdim: int = 49,
    ydim: int = 51,
    reffs: Iterable[float] | None = None,
    angs: Iterable[float] | None = None,
    axrats: Iterable[float] | None = None,
    nbenchmark: int = 0,
    nsub=1,
    do_like: bool = True,
    do_residual: bool = False,
    do_grad: bool = False,
    do_jac: bool = False,
    do_meas_modelfit: bool = False,
    plot=False,
) -> list[dict]:
    """Test and/or benchmark different Gaussian evaluation methods.

    Parameters
    ----------
    xdim
        The x-axis dimensions of the image.
    ydim
        The y-axis dimensions of the image.
    reffs
        Iterable of effective radii.
    angs
        Iterable of position angles in degrees CCW from +x.
    axrats
        Iterable of major-to-minor axis ratios.
    nbenchmark
        Number of times to repeat function evaluation for benchmarking.
    nsub
        Number of (identical) Gaussians to evaluate (as for a GMM).
    do_like
        Whether to evaluate the likelihood.
    do_residual
        Whether to evaluate the residual.
    do_grad
        Whether to evaluate the likelihood gradient.
    do_jac
        Whether to evaluate the model Jacobian.
    do_meas_modelfit
        Whether to test meas_modelfit's code.
    plot
        Whether to plot.

    Returns
    -------
    results
        List of dicts with results for each combination of parameters.

    Usage
    -----
    Example usage:

    test = gaussian_test(nbenchmark=1000)
    for x in test:
        print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))
    """
    if reffs is None:
        reffs = [2.0, 5.0]
    if angs is None:
        angs = np.linspace(0, 90, 7)
    if axrats is None:
        axrats = [1, 0.5, 0.2, 0.1, 0.01]
    results = []
    hasgs = find_spec("galsim") is not None
    num_params = nsub * 6
    if hasgs:
        import galsim as gs

    centroid = g2.Centroid(xdim / 2, ydim / 2)
    kernel = g2.Gaussian(centroid=g2.Centroid(0, 0), ellipse=g2.Ellipse(sigma_x=0.0, sigma_y=0))

    cmd_func = "g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim,)"
    cmd_obj = "evaluator.loglike_pixel()"

    functions = {
        "eval": (cmd_func, {"nsub": nsub}),
        "output": (cmd_obj, {"nsub": nsub, "do_output": True}),
    }
    if do_like:
        functions["like"] = (cmd_obj, {"nsub": nsub, "do_like": True})
    if do_residual:
        functions["residual"] = (cmd_obj, {"nsub": nsub, "do_residual": True})
    if do_grad:
        functions["grad"] = (cmd_obj, {"nsub": nsub, "do_grad": True})
    if do_jac:
        functions["jac"] = (cmd_obj, {"nsub": nsub, "do_jac": True})

    for key, (cmd, kwargs_setup) in functions.items():
        setup = get_setup(**kwargs_setup)
        print(f'Evaluating with {key}: "{setup}; {cmd}"')
        exec("; ".join((setup, cmd)), locals(), locals())

    for reff in reffs:
        for ang in angs:
            for axrat in axrats:
                ellipse = g2.Ellipse(
                    g2.EllipseMajor(r_major=reff * g2.M_SIGMA_HWHM, axrat=axrat, angle=ang, degrees=True)
                )
                source = g2.Gaussian(
                    centroid=centroid, ellipse=ellipse, integral=g2.GaussianIntegralValue(1 / nsub)
                )
                gaussians = g2.ConvolvedGaussians([g2.ConvolvedGaussian(source, kernel) for _ in range(nsub)])
                gaussmpf = g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim).data
                result = "Ran make"
                if hasgs:
                    gaussgs = (
                        gs.Gaussian(flux=1, half_light_radius=reff * np.sqrt(axrat))
                        .shear(q=axrat, beta=ang * gs.degrees)
                        .drawImage(nx=xdim, ny=ydim, scale=1, method="no_pixel")
                        .array
                    )
                    if plot:
                        fig, ax = plt.subplots(ncols=3)
                        ax[0].imshow(np.log10(gaussmpf))
                        ax[1].imshow(np.log10(gaussgs))
                        ax[2].imshow(gaussgs - gaussmpf)
                        plt.show()
                    gstonew = np.sum(np.abs(gaussmpf - gaussgs))
                    result += f"; GalSim/new residual=({gstonew:.3e})"
                if nbenchmark > 0:
                    times = {
                        key: np.min(
                            timeit.repeat(
                                cmd,
                                setup=get_setup(xdim, ydim, reff, axrat, ang, **kwargs_setup),
                                repeat=nbenchmark,
                                number=1,
                            )
                        )
                        for key, (cmd, kwargs_setup) in functions.items()
                    }
                    if do_meas_modelfit:
                        ang_rad = np.deg2rad(ang)
                        for key in ("dev", "exp"):
                            times[f"mmf-{key}"] = np.min(
                                timeit.repeat(
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
                                    repeat=nbenchmark,
                                    number=1,
                                )
                            )
                    if hasgs:
                        times["GalSim"] = np.min(
                            timeit.repeat(
                                f'x=profile.drawImage(nx={xdim}, ny={ydim}, scale=1, method="no_pixel").array',
                                setup=f"import galsim as gs; profile = gs.Gaussian(flux=1,"
                                f" half_light_radius={reff*np.sqrt(axrat)}).shear("
                                f"q={axrat}, beta={ang}*gs.degrees)",
                                repeat=nbenchmark,
                                number=1,
                            )
                        )
                    result += (
                        f";{'/'.join(times.keys())} times=("
                        f"{','.join(['{:.3e}'.format(x) for x in times.values()])})"
                    )
                    results.append(
                        {
                            "string": result,
                            "xdim": xdim,
                            "ydim": ydim,
                            "reff": reff,
                            "axrat": axrat,
                            "ang": ang,
                        }
                    )
    return results


def gradient_test(
    xdim: int = 5,
    ydim: int = 4,
    flux: float = 1e4,
    reff: float = 2,
    axrat: float = 0.5,
    angle: float = 0,
    bg: float = 1e3,
    reff_psf: float = 0,
    axrat_psf: float = 0.95,
    angle_psf: float = 0,
    n_psfs: int = 1,
    printout: bool = False,
    plot: bool = False,
):
    """Benchmark and test accuracy of gradient evaluations.

    Parameters
    ----------
    xdim
        The x-axis dimensions of the image.
    ydim
        The y-axis dimensions of the image.
    flux
        Total source flux.
    reff
        Major-axis effective radius in pixels.
    axrat
        Axis ratio (minor/major).
    angle
        Position angle in degrees.
    bg
        Background value per pixel.
    reff_psf
        PSF major-axis effective radius in pixels.
    axrat_psf
        PSF axis ratio (minor/major).
    angle_psf
        PSF position angle in degrees.
    n_psfs
        Number of PSF gaussians.
    printout
        Whether to print detailed results.
    plot
        Whether to plot results.
    """
    if n_psfs > 1:
        raise ValueError("n_psfs>1 not yet supported")

    cen_x, cen_y = xdim / 2.0, ydim / 2.0
    # Keep this in units of sigma, not re==FWHM/2
    source = g2.EllipseMajor(reff * g2.M_SIGMA_HWHM, axrat, angle, degrees=True)
    source_g = g2.Ellipse(source)
    psf = g2.EllipseMajor(reff_psf * g2.M_SIGMA_HWHM, axrat_psf, angle_psf, degrees=True)
    psf_g = g2.Ellipse(psf)
    conv = g2.EllipseMajor(source_g.make_convolution(psf_g), degrees=True)

    source = g2.Gaussian(
        centroid=g2.Centroid(x=cen_x, y=cen_y), ellipse=source_g, integral=g2.GaussianIntegralValue(flux)
    )
    values = (
        source.centroid.x,
        source.centroid.y,
        source.integral.value,
        source.ellipse.sigma_x,
        source.ellipse.sigma_y,
        source.ellipse.rho,
    )

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

    gaussians = g2.ConvolvedGaussians(
        [
            g2.ConvolvedGaussian(
                source=source,
                kernel=g2.Gaussian(
                    centroid=g2.Centroid(x=xdim / 2, y=ydim / 2),
                    ellipse=psf_g,
                    integral=g2.GaussianIntegralValue(1.0 / n_psfs),
                ),
            )
            for _ in range(n_psfs)
        ]
    )
    model = g2.make_gaussians_pixel_D(gaussians, n_rows=ydim, n_cols=xdim).data

    if printout:
        sum_img = np.sum(model)
        print(f"Modelsum = {sum_img:5e} vs flux {flux:.5e} ({(flux-sum_img)/sum_img*100.:.2f}% missing)")
        print(f"Source: {source}")
        print(f"PSF: {psf}")
        print(f"Convolved: {conv}")
        print("reff, axrat, ang:", (conv.r_major, conv.axrat, conv.angle))

    data = g2.ImageD(np.random.poisson(model + bg) - bg)
    sigma_inv = g2.ImageD(np.array([[1 / bg]]))
    output = g2.ImageD(np.zeros_like(data.data))
    n_params = 6
    grads = g2.ImageArrayD([g2.ImageD(np.zeros((n_psfs, n_params)))])

    # Compute the log likelihood and gradients
    evaluator_i = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, output=output)
    ll_i = evaluator_i.loglike_pixel()

    evaluator_g = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, grads=grads)
    ll_g = evaluator_g.loglike_pixel()

    jacobian = np.zeros([ydim, xdim, n_params * n_psfs])
    jacobian_arr = g2.ImageArrayD([g2.ImageD(jacobian[:, :, idx].view()) for idx in range(n_params * n_psfs)])
    evaluator_j = g2.GaussianEvaluatorD(gaussians, data=data, sigma_inv=sigma_inv, grads=jacobian_arr)
    evaluator_j.loglike_pixel()

    dxs = [1e-6, 1e-6, flux * 1e-6, 1e-8, 1e-8, 1e-8]
    dlls = np.zeros(n_params)
    diffabs = np.zeros(n_params)
    format_param_name = f"<{max(len(param_name) for param_name in names_params_gauss)}"

    for i, dxi in enumerate(dxs):
        dx = np.zeros(n_params)
        dx[i] = dxi
        value = values[i]
        set_param(source, i, value + dxi)
        # Note that there's no option to return the log likelihood and the Jacobian - the latter skips
        # computing the former for efficiency, assuming that you won't need it
        llnewg, llnewi = (evaltor.loglike_pixel() for evaltor in (evaluator_g, evaluator_i))

        # It's actually the jacobian of the residual
        findiff = -(output.data - model) / dxi * sigma_inv.data
        jacparam = jacobian[:, :, i]
        diffabs[i] = np.sum(np.abs(findiff - jacparam))
        if plot:
            fig, axes = plt.subplots(nrows=2, ncols=3)
            fig.suptitle(f"{names_params_gauss[i]} gradients")
            axes[0][0].imshow(model)
            axes[0][0].set_title("Model")
            axes[0][1].imshow(output.data)
            axes[0][1].set_title("Model (modified)")
            axes[1][0].imshow(findiff)
            axes[1][0].set_title(f"Finite difference (dx={dxi:.2e})")
            axes[1][1].imshow(jacparam)
            axes[1][1].set_title("Exact Jacobian")
            pcdiff = 100 * (1 - jacparam / findiff)
            pcdiff[np.abs(jacparam - findiff) < 10 * dxi] = 0
            axes[1][2].imshow(pcdiff)
            axes[1][2].set_title("Percent difference")
            plt.show()
        if printout:
            print(
                f"{names_params_gauss[i]:{format_param_name}} LL, LLnew1, llnew12 = {ll_g:.4e}, {llnewg:.4e}"
                f", {llnewi:.4e}; LLdiff={llnewg-ll_g:.4e} with dx={dxi:.3e}"
            )
        dlls[i] = (llnewi - ll_i) / dxi
        set_param(source, i, value)

    grads_data = grads.at(0).data
    # Gradient evaluation is additive, therefore it has been summed 1 + len(dxs) times
    grads_data = grads_data / (1.0 + len(dxs))
    return grads_data, dlls, diffabs


def mgsersic_test(
    xdim: int | None = None,
    ydim: int | None = None,
    flux: float = 1e4,
    reff: float = 3,
    nser: float = 1,
    axrat: float = 1,
    angle: float = 0,
    plot=False,
    do_galsim: bool = False,
    do_meas_modelfit: bool = False,
):
    """Evaluate multi-Gaussian Sersics and compare to the true profile.

    Parameters
    ----------
    xdim
        The x-axis dimensions of the image.
    ydim
        The y-axis dimensions of the image.
    flux
        Total source flux.
    reff
        Effective radius in pixels.
    nser
        Sersic index.
    axrat
        Axis ratio (minor/major).
    angle
        Position angle in degrees.
    plot
        Whether to plot results.
    do_galsim
        Whether to evaluate with galsim.
    do_meas_modelfit
        Whether to evaluate with meas_modelfit.
    """
    if xdim is None and ydim is None:
        xdim, ydim = 15, 15
    else:
        if ydim is None:
            ydim = xdim
        elif xdim is None:
            xdim = ydim
    shape = (xdim, ydim)
    band = "i"
    channel = g2f.Channel.get(band)

    ell = g2.Ellipse(g2.EllipseMajor(r_major=reff, axrat=axrat, angle=angle, degrees=True))
    model_mpf = get_model(
        {channel: flux},
        (xdim, ydim),
        sigma_xs=[ell.sigma_x],
        sigma_ys=[ell.sigma_y],
        rhos=[ell.rho],
        nsers=[nser],
    )

    model_mpf.setup_evaluators(model_mpf.EvaluatorMode.image)
    model_mpf.evaluate()
    img_ref = model_mpf.outputs[0].data
    imgs = {"mpf": img_ref}

    if do_meas_modelfit:
        is_exp = nser == 1
        model = "lux" if is_exp else ("luv" if nser == 4 else None)
        if model:
            from lsst.afw.geom.ellipses import Axes, Ellipse
            from lsst.geom import Point2D
            from lsst.shapelet import RadialProfile

            basis = RadialProfile.get(model).getBasis(6 if is_exp else 8, 4 if is_exp else 8)
            xc, yc = xdim / 2.0 - 0.5, ydim / 2.0 - 0.5
            ellipse = Ellipse(Axes(a=reff, b=reff), Point2D(xc, yc))
            msf = basis.makeFunction(ellipse, np.array([flux]))
            img_mmf = np.zeros(shape)
            msf.evaluate().addToImage(img_mmf)
            imgs["mmf"] = flux * img_mmf
    if do_galsim:
        sersic = gs.Sersic(n=nser, half_light_radius=reff * math.sqrt(axrat), flux=flux).shear(
            q=axrat, beta=angle * gs.degrees
        )
        gaussians = model_mpf.gaussians(channel)
        gaussian = None
        for idx in range(len(gaussians)):
            gauss_mpf = gaussians.at(idx)
            ell_major = g2.EllipseMajor(gauss_mpf.ellipse)
            gauss_i = gs.Gaussian(
                sigma=ell_major.r_major * math.sqrt(axrat), flux=gauss_mpf.integral_value
            ).shear(q=axrat, beta=angle * gs.degrees)
            if gaussian:
                gaussian += gauss_i
            else:
                gaussian = gauss_i
        for obj, name_obj in ((sersic, "ser"), (gaussian, "mgser")):
            for method, name_method in (("real_space", "real"), ("no_pixel", "nopix")):
                img_tmp = gs.ImageD(np.zeros(shape))
                obj.drawImage(image=img_tmp, method="real_space", scale=1.0)
                imgs[f"gs_{name_obj}_{name_method}"] = img_tmp.array

    keys = tuple(imgs.keys())

    diffs = {key: imgs[key] - img_ref for key in keys[1:]}
    nrows = len(keys) - 1
    if plot:
        fig, axes = plt.subplots(nrows=nrows, ncols=3)
        for idx, key in enumerate(keys[1:]):
            axes_row = axes[idx] if nrows > 1 else axes
            axes_row[0].imshow(np.log10(imgs[key]))
            axes_row[0].set_title(f"log10({key})")
            diff = diffs[key]
            axes_row[1].imshow(diff)
            axes_row[1].set_title(f"{key}-{keys[0]}")
            axes_row[2].imshow(diff / img_ref)
            axes_row[2].set_title(f"({key}-{keys[0]})/{keys[0]}")
        plt.show()
    return diffs
