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

import csv
import galsim as gs
import gauss2d as g2
import gauss2d.fit as g2f
import io
import logging
import matplotlib.pyplot as plt
from multiprofit.limits import limits_ref, Limits
from multiprofit.multigaussianapproxprofile import MultiGaussianApproximationComponent
import multiprofit.objects as mpfobj
import multiprofit.priors as mpfpri
from multiprofit.transforms import transforms_ref
import multiprofit.utils as mpfutil
import numpy as np
from scipy import stats
import sys
import time
import traceback
from typing import Dict, List, NamedTuple


class ImageEmpty:
    shape = (0, 0)

    def __init__(self, shape=(0, 0)):
        self.shape = shape


class ModelSpec(NamedTuple):
    name: str
    model: str
    fixedparams: str = ""
    initparams: str = ""
    inittype: str = ""
    psfmodel: str = ""
    psfpixel: bool = True
    unlimitedparams: str = ""
    values_init: dict = None
    values_init_psf: dict = None


class ModelFits(NamedTuple):
    fits: List[dict]
    modeltype: str = ""


class MomentResult(NamedTuple):
    fluxes: dict
    moments_by_name: dict
    values_min: dict
    values_max: dict
    num_pix_img: int


class PlotInfo(NamedTuple):
    figaxes: Dict[str, mpfobj.FigAxes] = {}
    do_plot_as_column: bool = False
    num_rows: int = 0
    num_cols: int = 0
    title: str = None


# For priors
def norm_logpdf_mean(x, mean=0., scale=1.):
    return stats.norm.logpdf(x - mean, scale=scale)


def truncnorm_logpdf_mean(x, mean=0., scale=1., a=-np.inf, b=np.inf):
    return stats.truncnorm.logpdf(x - mean, scale=scale, a=a, b=b)


def is_ratioparam(param):
    return isinstance(param, g2f.ProperFractionParameterD)


def is_fluxparam(param):
    return isinstance(param, g2f.IntegralParameterD) or is_ratioparam(param)


def make_FluxParameter(is_fluxes_fracs=False, *args, **kwargs):
    if is_fluxes_fracs:
        param = g2f.ProperFractionParameterD(*args, **kwargs)
        limits = kwargs.get('limits')
        if limits is None or not ((limits.min >= 0) and (limits.max <= 1)):
            param.limits = limits_ref['fraction']
    else:
        param = g2f.IntegralParameterD(*args, **kwargs)
    return param


# Return both is_flux and is_ratio
def is_fluxparam_ratio(param):
    is_ratio_param = is_ratioparam(param)
    return (is_ratio_param or is_fluxparam(param)), is_ratio_param


def get_param_default(
        param, value=None, profile=None, fixed=False, use_sersic_logit=True,
        is_multigauss=False, return_value=False
):
    """ Get a reasonable default instantiation of a parameter by name.

    :param param: str; parameter name.
    :param value: float; initial value.
    :param profile: str; a profile name to interpret profile-dependent params like "slope".
    :param fixed: bool; whether the param should be set to fixed.
    :param use_sersic_logit: bool; whether to use a logit transform on the Sersic index. Ignored if profile
        is not "sersic".
    :param is_multigauss: bool; whether the profile is a Gaussian mixture approximation.
    :param return_value: bool; whether to only return the value instead of a Parameter
    :return: multiprofit.objects.Parameter initialized as specified, or float value if return_value is True.
    """
    transform = None
    limits = None

    if param == "n_ser":
        param = "slope"
        profile = "sersic"
    if param == "slope":
        if profile == "moffat":
            transform = transforms_ref["inverse"]
            limits = limits_ref["con"]
            obj = g2f.MoffatConcentrationParameterD
        elif profile == "sersic":
            if use_sersic_logit:
                if is_multigauss:
                    transform = transforms_ref["logit_multigauss"]
                    limits = limits_ref["n_ser_multigauss"]
                else:
                    transform = transforms_ref["logit_sersic"]
                    limits = limits_ref["n_ser"]
            else:
                transform = transforms_ref["log10"]
                limits = limits_ref["n_ser"]
            obj = g2f.SersicIndexParameterD
    elif param == "sigma_x" or param == "sigma_y":
        transform = transforms_ref["log10"]
        obj = g2f.SigmaXParameterD if param == "sigma_x" else g2f.SigmaYParameterD
    elif param == "rho":
        transform = transforms_ref["logit_rho"]
        limits = limits_ref["rho"]
        obj = g2f.RhoParameterD
    elif param == "r_scale":
        transform = transforms_ref['log10']
        obj = g2f.RadiusScaleParameterD
    else:
        raise ValueError(f"Unknown parameter type '{param}'")

    kwargs = dict(fixed=fixed, limits=limits, transform=transform)
    if value is not None:
        kwargs['value'] = value
    param = obj(**kwargs)

    if return_value:
        return param.value
    return param


def get_components(profile, fluxes, values=None, is_fluxes_fracs=True):
    if values is None:
        values = {}
    bands = list(fluxes.keys())
    band_ref = bands[0]
    for band in bands:
        if is_fluxes_fracs:
            fluxfracs_band = fluxes[band]
            fluxfracs_band = np.array(fluxfracs_band)
            sum_fluxfracs_band = np.sum(fluxfracs_band)
            if any(np.logical_not(fluxfracs_band > 0)) or not sum_fluxfracs_band < 1:
                raise RuntimeError('fluxfracs_band={} has elements not > 0 or sum {} < 1'.format(
                    fluxfracs_band, sum_fluxfracs_band))
            if len(fluxfracs_band) == 0:
                fluxfracs_band = np.ones(1)
            else:
                fluxfracs_band /= np.concatenate([np.array([1.0]), 1-np.cumsum(fluxfracs_band[:-1])])
                fluxfracs_band = np.append(fluxfracs_band, 1)
            fluxes[band] = fluxfracs_band
        num_comps_band = len(fluxes[band])
        if band == band_ref:
            num_comps = num_comps_band
        elif num_comps_band != num_comps:
            raise RuntimeError(
                'get_components for profile {} has num_comps[{}]={} != num_comps[{}]={}'.format(
                    profile, num_comps_band, band, num_comps, band_ref)
            )
    components = []
    is_gaussian = profile == "gaussian"
    is_multi_gaussian_sersic = profile.startswith('mgsersic')
    if is_multi_gaussian_sersic:
        order = np.int(profile.split('mgsersic')[1])
        profile = "sersic"
        if 'n_ser' in values:
            values['n_ser'] = np.zeros_like(values['n_ser'])

    transform = transforms_ref["logit"] if is_fluxes_fracs else transforms_ref["log10"]
    for compi in range(num_comps):
        is_last = compi == (num_comps - 1)
        param_fluxescomp = [
            make_FluxParameter(
                value=fluxes[band][compi], label=band,
                transform=transform, fixed=is_last,
                is_fluxes_fracs=is_fluxes_fracs,
            )
            for band in bands
        ]
        params = [
            get_param_default(
                param, valueslice[compi], profile, fixed=False, is_multigauss=is_multi_gaussian_sersic)
            for param, valueslice in values.items() if not ((param == "slope") and is_gaussian)
        ]
        params_ellipse_type = {}
        params_other_type = []
        for param in params:
            if param.name == 'sigma_x' or param.name == 'sigma_y' or param.name == 'rho':
                params_ellipse_type[param.name] = param
            else:
                params_other_type.append(param)
        sigma_x, sigma_y, rho = [params_ellipse_type.get(p) for p in ['sigma_x', 'sigma_y', 'rho']]
        if any([p is None for p in [sigma_x, sigma_y, rho]]):
            raise RuntimeError('At least one of sigma_x,sigma_y,rho = {},{},{} is None from values {}'
                               'and [p.name for p in params]={}'.format(
                                    sigma_x, sigma_y, rho, values, [p.name for p in params]))
        params_ellipse = mpfobj.EllipseParameters(
            params_ellipse_type['sigma_x'], params_ellipse_type['sigma_y'], params_ellipse_type['rho'])
        if is_multi_gaussian_sersic:
            components.append(MultiGaussianApproximationComponent(
                param_fluxescomp, params_ellipse=params_ellipse, profile=profile,
                parameters=params_other_type, order=order))
        else:
            components.append(mpfobj.EllipticalParametricComponent(
                param_fluxescomp, params_ellipse=params_ellipse, profile=profile,
                parameters=params_other_type))

    return components


# TODO: Fix per-component offset_xy as it's not actually working
def get_model(
    fluxes_by_band, model_string, size_image, sigma_xs=None, sigma_ys=None, rhos=None, slopes=None,
    fluxfracs=None, offset_xy=None, fit_background=False, name_model="", namesrc="", n_exposures=1,
    engine="galsim", engineopts=None, convertfluxfracs=False,
    repeat_ellipse=False, logger=None
):
    """
    Convenience function to get a multiprofit.objects.model with a single source with components with
    reasonable default parameters and transforms.

    :param fluxes_by_band: Dict; key=band: value=np.array of fluxes per component if fluxfracs is None else
        source flux
    :param model_string: String; comma-separated list of 'component_type:number'
    :param size_image: Float[2]; the x- and y-size of the image
    :param sigma_xs: Float[num_components]; Ellipse covariance sigma_x of each component
    :param sigma_ys: Float[num_components]; Ellipse covariance sigma_y of each component
    :param rhos: Float[num_components]; Ellipse covariance rho of each component
    :param slopes: Float[num_components]; Profile shape (e.g. Sersic n) of each components
    :param fluxfracs: Float[num_components]; The flux fraction for each component
    :param offset_xy: Float[2][num_components]; The x-y offsets relative to source center of each component
    :param fit_background: bool; whether to fit a flat background level per band
    :param name_model: String; a name for this model
    :param namesrc: String; a name for the source
    :param n_exposures: Int > 0; the number of exposures in each band.
    :param engine: String; the rendering engine to pass to the multiprofit.objects.Model.
    :param engineopts: Dict; the rendering options to pass to the multiprofit.objects.Model.
    :param convertfluxfracs: Boolean; should the model have absolute fluxes per component instead of ratios?
    :param repeat_ellipse: Boolean; is there only one set of values in sigma_xs, sigma_ys, rhos?
        If so, re-use the provided value for each component.
    :param logger: logging.Logger; a logger to print messages
    :return:
    """
    bands = list(fluxes_by_band.keys())
    model_strings = model_string.split(",")

    profiles = {}
    num_comps = 0
    for description_model in model_strings:
        # TODO: What should be done about modifiers?
        profile_modifiers = description_model.split("+")
        profile, num_comps_prof = profile_modifiers[0].split(":")
        num_comps_prof = np.int(num_comps_prof)
        profiles[profile] = num_comps_prof
        num_comps += num_comps_prof
    try:
        none_all = np.repeat(None, num_comps)
        sigma_xs = np.array(sigma_xs) if sigma_xs is not None else none_all
        sigma_ys = np.array(sigma_ys) if sigma_ys is not None else none_all
        rhos = np.array(rhos) if rhos is not None else none_all
        slopes = np.array(slopes) if slopes is not None else none_all
        if fluxfracs is not None:
            fluxfracs = np.array(fluxfracs)
        # TODO: Verify lengths identical to bandscount
    except Exception as error:
        raise error

    # TODO: Figure out how this should work in multiband
    cen_x, cen_y = [x / 2.0 for x in size_image]
    if offset_xy is not None:
        cen_x += offset_xy[0]
        cen_y += offset_xy[1]
    if n_exposures > 0:
        exposures = []
        for band in bands:
            for _ in range(n_exposures):
                exposures.append(
                    mpfobj.Exposure(
                        band, image=np.zeros(shape=size_image), mask_inverse=None, error_inverse=None))
        data = mpfobj.Data(exposures)
    else:
        data = None

    modelastro = mpfobj.AstrometricModel(
        cen_x=g2f.CentroidXParameterD(value=cen_x, limits=Limits(min=0., max=size_image[0])),
        cen_y=g2f.CentroidYParameterD(value=cen_y, limits=Limits(min=0., max=size_image[1])),
    )
    components = []

    if fluxfracs is None:
        fluxfracs = np.repeat(1.0/num_comps, num_comps)

    compnum = 0
    for profile, num_profiles in profiles.items():
        comprange = range(compnum, compnum + num_profiles)
        # TODO: Review whether this should change to support band-dependent fluxfracs
        fluxfracs_comp = [fluxfracs[i] for i in comprange][:-1]
        fluxfracs_comp = {band: fluxfracs_comp for band in bands}
        comprangeellipse = range(0, 1) if repeat_ellipse else comprange
        values = {
            "sigma_x": sigma_xs[comprangeellipse],
            "sigma_y": sigma_ys[comprangeellipse],
            "rho": rhos[comprangeellipse],
        }
        if repeat_ellipse:
            for key, value in values.items():
                values[key] = np.repeat(value, num_profiles)
        values["slope"] = slopes[comprange]
        components += get_components(profile, fluxfracs_comp, values)
        if len(components) != num_profiles:
            raise RuntimeError('get_components returned {}/{} expected profiles'.format(
                len(components), num_profiles))
        compnum += num_profiles
    param_fluxes = [
        g2f.IntegralParameterD(
            value=np.clip(fluxes_by_band[band], 1e-16, np.Inf),
            label=band, transform=transforms_ref["log10"], fixed=False,
        )
        for bandi, band in enumerate(bands)
    ]
    modelphoto = mpfobj.PhotometricModel(components, param_fluxes)
    if convertfluxfracs:
        modelphoto.convert_param_fluxes(use_fluxfracs=False, transform=transforms_ref['log10'])
    sources = [mpfobj.Source(modelastro, modelphoto, namesrc)]
    if fit_background:
        # The small positive value is a bit of a hack to get the initial linear fit to work
        # (otherwise the initial background model is zero and nnls can't do anything with it)
        param_fluxes_bg = [
            g2f.IntegralParameterD(value=1e-9, label=band, fixed=False)
            for bandi, band in enumerate(bands)
        ]
        background = mpfobj.Background(param_fluxes_bg)
        modelastro = mpfobj.AstrometricModel(
            cen_x=g2f.CentroidXParameterD(value=cen_x, limits=Limits(min=0., max=size_image[0]), fixed=True),
            cen_y=g2f.CentroidYParameterD(value=cen_y, limits=Limits(min=0., max=size_image[1]), fixed=True),
        )
        modelphoto = mpfobj.PhotometricModel([background])
        sources.append(mpfobj.Source(modelastro, modelphoto, namesrc))
    model = mpfobj.Model(sources, data, engine=engine, engineopts=engineopts, name=name_model, logger=logger)
    return model


# Convenience function to evaluate a model and optionally plot with title, returning chi map only
def evaluate_model(model, plot=False, title=None, **kwargs):
    """
    Convenience function to evaluate a model and optionally at a title to the plot.
    :param model: multiprofit.Model
    :param plot: Boolean; generate plot?
    :param title: String; title to add on top of the plot.
    :param kwargs: Dict; additional keyword arguments to pass to model.evaluate().
    :return: Chi maps for each exposure.
    """
    _, _, chis, _ = model.evaluate(plot=plot, **kwargs)

    if plot:
        if title is not None:
            plt.suptitle(title)
        #plt.show(block=False)
    return chis


# Convenience function to fit a model. kwargs are passed on to evaluate_model
def fit_model(
    model=None, modeller=None, modellib="scipy", modellibopts=None, do_print_final=True,
    print_step_interval=100, plot=False, do_linear=True, params_adjusted=None, logger_modeller=None,
    kwargs_fit=None,
    **kwargs
):
    """
    Convenience function to fit a model with reasonable defaults.
    :param model: multiprofit.Model; default: modeller.model
    :param modeller: multiprofit.Modeller; default: new Modeller.
    :param modellib: String; the modelling library to use if modeller is None.
    :param modellibopts: Dict; options to pass to the modeller if modeller is None.
    :param do_print_final: Boolean; print the final parameter values?
    :param print_step_interval: Integer; step interval between printing.
    :param plot: Boolean; plot final fit?
    :param do_linear: Boolean; do linear fit?
    :param params_adjusted: Dict [multiprofit.objects.Parameter, float]; parameter-offset pairs for
        parameters that were adjusted (with limits) during fitting. Offset values are subtracted from
        untransformed fit value, then returned after storage in the return value.
    :param logger_modeller: logging.Logger; a logger to initialize the modeller with if it modeller is None.
    :param kwargs_fit: Dict; additional keyword arguments to pass to modeller.fit().
    :param kwargs: Dict; passed to evaluate_model() after fitting is complete (e.g. plotting options).
    :return: Tuple of modeller.fit and modeller.
    """
    if modeller is None:
        modeller = mpfobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts,
                                   logger=logger_modeller)
    elif model is None:
        raise ValueError('fit_model must be passed a non-None model or modeller')
    else:
        model = modeller.model
    if kwargs_fit is None:
        kwargs_fit = {}
    if params_adjusted is None:
        params_adjusted = {}
    fit = modeller.fit(
        do_print_final=do_print_final, print_step_interval=print_step_interval, do_linear=do_linear,
        **kwargs_fit
    )
    # Conveniently sets the parameters to the right values too
    # TODO: Find a better way to ensure chis are returned than setting do_draw_image=True
    chis = evaluate_model(model, plot=plot, param_values=fit["params_best"], do_draw_image=True, **kwargs)
    fit["chisqred"] = mpfutil.get_chisqred(chis)
    params = model.get_parameters()
    for item in ['params_bestall', 'params_bestalltransformed', 'params_allfixed']:
        fit[item] = []
    idx_free = 0
    for param in params:
        param_adjust = params_adjusted.get(param)
        value_untrans = param.value
        if param_adjust:
            limits_param = param.limits
            param.limits = None
            value_untrans -= param_adjust
            fit["params_best"][idx_free] = value_untrans
        fit["params_bestall"].append(value_untrans)
        fit["params_bestalltransformed"].append(param.value_transformed)
        fit["params_allfixed"].append(param.fixed)
        if param_adjust:
            param.value = value_untrans
            param.limits = limits_param
            idx_free += 1

    return fit, modeller


def fit_psf_model(
    modeltype, imgpsf, engines, band, fits_model_psf=None, error_inverse=None, modellib="scipy",
    modellibopts=None, plot=False, name_model=None, label=None, redo=True,
    skip_fit=False, logger=None, **kwargs
):
    """Fit a single PSF model to an image.

    :param modeltype: str; the type of model.
    :param imgpsf: array-like; PSF image (normalized).
    :param engines: iterable [tuple] of str (engine name) and dict (engine options) pairs
    :param band: str; the photometric filter/passband.
    :param fits_model_psf: dict; existing results returned by this function.
    :param error_inverse: array-like; uncertainty (sigma)
    :param modellib: string; Model fitting library
    :param modellibopts: dict; Model fitting library options
    :param plot: bool; whether to plot
    :param name_model: str; a name for this model
    :param redo: bool; whether to redo existing fits
    :param skip_fit: bool; whether to skip fitting and only setup the model
    :param logger: logging.Logger; a logger to print messages
    :param kwargs: Additional keyword arguments to pass to `fit_model`
    :return: fits_model_psf; dict keyed by engine containing:
    """
    if fits_model_psf is None:
        fits_model_psf = {}
    if name_model is None:
        name_model = modeltype
    # Fit the PSF
    num_comps = np.int(modeltype.split(":")[1])
    for engine, engineopts in engines.items():
        if engine not in fits_model_psf:
            fits_model_psf[engine] = {}
        if redo or name_model not in fits_model_psf[engine]:
            model = get_psfmodel(engine, engineopts, num_comps, band, modeltype, imgpsf,
                                 error_inverse=error_inverse, logger=logger)
            fits_model_psf[engine][name_model] = {}
        else:
            model = fits_model_psf[engine][name_model]['modeller'].model
        model.name = '.'.join(['PSF', band, name_model])
        fit_done = 'fit' in fits_model_psf[engine][name_model]

        if (redo or not fit_done) and not skip_fit:
            fits_model_psf[engine][name_model]['fit'], fits_model_psf[engine][name_model]['modeller'] = \
                fit_model(
                    model, modellib=modellib, modellibopts=modellibopts, plot=plot, name_model=label,
                    logger_modeller=logger, **kwargs
                )
        else:
            params_free = model.get_parameters(fixed=False)
            if not fit_done:
                fits_model_psf[engine][name_model]['fit'] = {
                    'name_params': [param.name for param in params_free],
                }
            fits_model_psf[engine][name_model]['modeller'] = mpfobj.Modeller(
                model, modellib, modellibopts=modellibopts)
            if plot:
                exposure = model.data.exposures[band][0]
                is_empty = isinstance(exposure.image, ImageEmpty)
                if is_empty:
                    set_exposure(model, band, image=imgpsf, error_inverse=error_inverse)
                if skip_fit:
                    values = None
                else:
                    fit = fits_model_psf[engine][name_model]['fit']
                    values = fit.get('params_best')
                    if values is None:
                        raise RuntimeError(f'No valid PSF fit found in fit structure: {fit}')
                evaluate_model(
                    model, param_values=values, plot=plot, title=kwargs.get('plot'),
                    name_model=kwargs.get('label'), figaxes=kwargs.get('figaxes'),
                    row_figure=kwargs.get('row_figure')
                )
                if is_empty:
                    set_exposure(model, band, image='empty')

    return fits_model_psf


def get_init_from_moments(
    exposures_psfs, estimate_moments=True, flux_min_obj=1e-3, flux_min_img=None, sigma_min=1e-2, logger=None,
    **kwargs
):
    """Get model initial parameters from multi-band exposures assumed to contain a single object.

    :param exposures_psfs: iterable of tuple; multiprofit.objects.Exposure, multiprofit.objects.PSF pairs.
    :param estimate_moments: bool; whether to estimate moments. If False, will only return limits
    :param flux_min_obj: float; minimum estimated flux for the object.
    :param flux_min_img: float; minimum total flux in the image required for moment estimation
    :param sigma_min: float; minimum sigma_x/y to return

    :param logger: logging.Logger; a logger to print messages
    :param kwargs: dict; additional keyword args to pass to mpfutil.estimate_ellipse
    :return: result: MomentResult with estimated moments
    """
    bands = {}
    fluxes = {}
    num_pix_img = None
    cens = {'cen_x': 0., 'cen_y': 0.}
    moments = None
    num_exposures_measured = 0

    for exposure, psf in exposures_psfs:
        band = exposure.band
        img_exp = exposure.image
        num_pix_img_exp = img_exp.shape
        if num_pix_img is None:
            num_pix_img = num_pix_img_exp
        elif num_pix_img_exp != num_pix_img:
            raise RuntimeError(
                f'get_init_from_moments image (band={band} shape={num_pix_img_exp} '
                f'not same as first={num_pix_img}')
        if estimate_moments and (band not in bands):
            if flux_min_img is None or np.sum(img_exp) > flux_min_img:
                if psf is None:
                    deconvolution_params = None
                else:
                    # First key is engine, next is model
                    psf = next(iter(next(iter(psf.values())).values()))['object']
                    fluxfracs = [p for p in psf.model.get_parameters(fixed=True, free=True)
                                 if is_ratioparam(p)]
                    comps = psf.model.modelphotometric.components
                    if len(fluxfracs) != len(comps):
                        raise RuntimeError(f'PSF model len(fluxes)={len(fluxes)} != len(comps)={len(comps)}')
                    flux_factor = 1
                    p_xx, p_yy, p_xy = 0., 0., 0.
                    for fluxfrac, comp in zip(fluxfracs, comps):
                        flux_comp = flux_factor*fluxfrac.value
                        ell = comp.params_ellipse
                        p_x, p_y, p_rho = (param.value
                                           for param in (ell.sigma_x, ell.sigma_y, ell.rho))
                        p_x, p_y = (p*g2.M_SIGMA_HWHM for p in (p_x, p_y))
                        p_xx += flux_comp * p_x * p_x
                        p_yy += flux_comp * p_y * p_y
                        p_xy += flux_comp * p_rho * p_x * p_y
                        flux_factor -= flux_comp
                    deconvolution_params = p_xx, p_yy, p_xy
                moments_band, cen_x_band, cen_y_band = mpfutil.estimate_ellipse(
                    img_exp, return_cens=True, validate=False, sigma_inverse=exposure.get_sigma_inverse(),
                    deconvolution_params=deconvolution_params, return_as_params=True, **kwargs)
                if logger:
                    logger.debug(f'Got moments_band={moments_band} cens={cen_x_band, cen_y_band} subbing'
                                 f' deconv_params={deconvolution_params}')
                cens['cen_x'] += cen_x_band
                cens['cen_y'] += cen_y_band
                # TODO: subtract PSF moments from object
                if moments is None:
                    moments = np.array(moments_band)
                else:
                    moments += np.array(moments_band)
                num_exposures_measured += 1
            bands[exposure.band] = None
        # TODO: Figure out what to do if given multiple exposures per band (TBD if we want to)
        fluxes[band] = np.clip(
            np.sum(img_exp[exposure.mask_inverse] if exposure.mask_inverse is not None else img_exp),
            flux_min_obj, np.Inf)

    if estimate_moments:
        moments_by_name = {
            name_param: value for name_param, value in zip(
                mpfobj.names_params_ellipse,
                g2.Ellipse(g2.Covariance(*(moments/num_exposures_measured))).xyr,
            )
        }

        cens['cen_x'] /= num_exposures_measured
        cens['cen_y'] /= num_exposures_measured

        # moments_by_name is in units of HWHM; convert to sigma TODO: verify this
        moments_by_name['sigma_x'] = np.max((g2.M_HWHM_SIGMA*moments_by_name['sigma_x'], sigma_min))
        moments_by_name['sigma_y'] = np.max((g2.M_HWHM_SIGMA*moments_by_name['sigma_y'], sigma_min))
    else:
        moments_by_name = {name_param: 0 for name_param in mpfobj.names_params_ellipse}

    moments_by_name.update(cens)

    num_pix_img = None if num_pix_img is None else np.flip(num_pix_img, axis=0)
    sigma_max = np.Inf if num_pix_img is None else np.sqrt(np.sum((num_pix_img/2.)**2))
    values_max = {"sigma_x": sigma_max, "sigma_y": sigma_max}
    values_min = {}
    for band in fluxes.keys():
        values_min["flux_" + band] = 1e-6 * fluxes[band]
        values_max["flux_" + band] = 100 * fluxes[band]

    return MomentResult(
        fluxes=fluxes, moments_by_name=moments_by_name, values_min=values_min, values_max=values_max,
        num_pix_img=num_pix_img,
    )


def _get_param_name(param):
    is_flux, is_fluxrat = is_fluxparam_ratio(param)
    return (
        param.name if not is_flux else f'flux{("ratio" if is_fluxrat else "")}_{param.label}',
        is_flux, is_fluxrat
    )


def _get_param_info(param, flag_params_fixed):
    name_param, is_flux, is_fluxrat = _get_param_name(param)
    is_fixed = param.fixed
    if name_param in flag_params_fixed or (is_flux and 'flux' in flag_params_fixed):
        is_fixed = True
    # TODO: Figure out a better way to reset modifiers to be free
    elif name_param == 'r_scale':
        is_fixed = False
    return name_param, is_flux, is_fluxrat, is_fixed


def _reset_params_fixed(model, params_fixed_default, params_inherit, **kwargs):
    """Reset parameter fixed status to model defaults.

    :param model: multiprofit.objects.Model to retrieve parameters from.
    :param params_fixed_default: iterable of bool; default fixed status for each param.
    :param params_inherit: set [str]; names of params that inherit values and therefore should not be reset.
    :param kwargs: Dict of additional kwargs to pass to model.get_parameters.
    """
    params = model.get_parameters(**kwargs)
    if len(params) != len(params_fixed_default):
        raise RuntimeError(f'len(params)={len(params)} != len(params_fixed_default)='
                           f'{len(params_fixed_default)}')
    for param, fixed in zip(params, params_fixed_default):
        if param.name not in params_inherit:
            param.fixed = fixed


def fit_galaxy(
        exposures_psfs, modelspecs, plot=False, name=None, models=None,
        fits_by_engine=None, logger=None, kwargs_moments=None, **kwargs
):
    """Convenience function to fit a galaxy given some exposures with PSFs.

    :param exposures_psfs: Iterable of tuple(mpfobj.Exposure, dict; key=psftype: value=mpfobj.PSF)
    :param modelspecs: Model specifications as returned by get_modelspecs
    :param plot: bool; whether to plot
    :param name: string; Name of the model for plot labelling
    :param models: dict; key=model name: value=mpfobj.Model
    :param fits_by_engine: dict; same format as return value
    :param logger: logging.Logger; a logger to print messages and be passed to model(ler)s
    :param kwargs_moments: dict; additional keyword arguments to pass to get_init_from_moments
    :param kwargs: dict; additional keyword arguments to pass to fit_galaxy_model

    :return: fits_by_engine: dict[str, FitResult]
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if logger not in kwargs:
        kwargs['logger'] = logger
    if kwargs_moments is None:
        kwargs_moments = {}
    estimate_moments = any(modelinfo.inittype == 'moments' for modelinfo in modelspecs)
    init_moments = get_init_from_moments(exposures_psfs, estimate_moments=estimate_moments, **kwargs_moments)
    bands = list(init_moments.fluxes.keys())
    if logger:
        logger.debug(f"Bands: {bands}; Moment init.: {init_moments.moments_by_name}")
    engine = 'galsim'
    engines = {
        engine: {
            "gsparams": gs.GSParams(
                kvalue_accuracy=1e-2, integration_relerr=1e-2, integration_abserr=1e-3,
                maximum_fft_size=32768)}
    }
    title = name if plot else None

    models = {} if (models is None) else models
    fits_by_engine = {} if ((models is None) or (fits_by_engine is None)) else fits_by_engine

    for engine, engineopts in engines.items():
        if engine not in fits_by_engine:
            fits_by_engine[engine] = {}
        fits_engine = fits_by_engine[engine]

        if plot:
            num_rows, num_cols = len(modelspecs), 0
            figaxes = {}
            for band in bands + (['multi'] if len(bands) > 1 else []):
                num_cols = 5
                # Change to landscape
                figure, axes = plt.subplots(nrows=min([num_cols, num_rows]), ncols=max([num_cols, num_rows]))
                if num_rows > num_cols:
                    axes = np.transpose(axes)
                # This keeps things consistent with the nrows>1 case
                if num_rows == 1:
                    axes = np.array([axes])
                if title is not None:
                    plt.suptitle(f"{title} {engine} model")
                figaxes[band] = mpfobj.FigAxes(figure=figure, axes=axes)
            plotinfo = PlotInfo(
                figaxes=figaxes, do_plot_as_column=num_rows > num_cols, num_rows=num_rows, num_cols=num_cols)
        else:
            plotinfo = PlotInfo()
        params_adjusted = {}

        for idx_model, modelinfo in enumerate(modelspecs):
            fit_galaxy_model(
                modelinfo, bands, exposures_psfs, init_moments, engine=engine, engineopts=engineopts,
                fits=fits_engine, models=models, modelspecs_prior=modelspecs[0:idx_model], plot=plot,
                params_adjusted=params_adjusted, plotinfo=plotinfo, idx_model=idx_model,
                **kwargs,
            )
        if plot:
            do_figure = len(bands) > 1
            for figaxis in plotinfo.figaxes.values():
                figure = figaxis.figure
                if do_figure:
                    plt.figure(figure.number)
                plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96,
                                    wspace=0.02, hspace=0.15)
            plt.show(block=False)

    return fits_by_engine, models


def fit_galaxy_model(
        modelinfo: ModelSpec, bands, exposures_psfs, init_moments, engine=None, engineopts=None,
        modellib=None, modellibopts=None, fits: Dict[str, ModelFits] = None,
        models=None, modelspecs_prior=None, redo=False,
        params_adjusted=None, logger=None, plot=False, plotinfo: PlotInfo=None, idx_model=None, name=None,
        img_plot_maxs=None, img_multi_plot_max=None, weights_band=None,
        do_fit_fluxfracs=False, fit_background=False, print_step_interval=100,
        print_exception=True, prior_specs=None, skip_fit=False, background_sigma_add=None,
        replace_data_by_model=False,
):
    """

    :param modelinfo: ModelSpec; the model specification
    :param modellib: string; Model fitting library
    :param modellibopts: dict; Model fitting library options
    :param fits: dict; Previous fit results
    :param redo: bool; Redo any pre-existing fits in fits_by_engine
    :param plot: bool; whether to plot
    :param name: str; Name of the source for plotting/logging
    :param img_plot_maxs: dict; key=band: value=float (Maximum value when plotting images in this band)
    :param img_multi_plot_max: float; Maximum value of summed images when plotting multi-band.
    :param weights_band: dict; key=band: value=float (Multiplicative weight when plotting multi-band RGB)
    :param do_fit_fluxfracs: bool; fit component flux ratios instead of absolute fluxes?
    :param fit_background: bool; whether to fit a flat background level per band
    :param print_step_interval: int; number of steps to run before printing output
    :param print_exception: bool; whether to print the first exception encountered and the stack trace
    :param prior_specs: dict; prior specifications.
    :param skip_fit: bool; whether to skip fitting and only setup the model
    :param background_sigma_add: float; factor to multiply the sky background prior standard deviation before
        adding to the prior's mean (which is otherwise zero). Default zero.
    :param replace_data_by_model: bool; whether to replace the real data by the initial model
    :return:
    """
    if fits is None:
        fits = {}
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.level = 21
    if modellib is None:
        modellib = "scipy"
    if modelspecs_prior is None:
        modelspecs_prior = []
    if params_adjusted is None:
        params_adjusted = {}
    if prior_specs is None:
        prior_specs = {}
    kwargs_fit = {'replace_data_by_model': replace_data_by_model}
    use_modellib_default = modellibopts is None
    name_model = modelinfo.name
    modeltype = modelinfo.model
    model_default = get_model(
        init_moments.fluxes, modeltype, init_moments.num_pix_img, [init_moments.moments_by_name["sigma_x"]],
        [init_moments.moments_by_name["sigma_y"]], [init_moments.moments_by_name["rho"]],
        engine=engine, engineopts=engineopts, convertfluxfracs=not do_fit_fluxfracs,
        fit_background=fit_background, repeat_ellipse=True, name_model=name_model, logger=logger
    )
    params_fixed_default = [param.fixed for param in model_default.get_parameters(free=True, fixed=True)]
    exists_model = modeltype in models
    model = model_default if not exists_model else models[modeltype]
    if not exists_model:
        models[modeltype] = model
    is_psf_pixelated = modelinfo.psfpixel
    name_psf = modelinfo.psfmodel + ("_pixelated" if is_psf_pixelated else "")
    params_unlimited = modelinfo.unlimitedparams
    params_unlimited = {name_param for name_param in params_unlimited.split(";")} if params_unlimited\
        else {}

    if is_psf_pixelated:
        can_do_no_pixel = True
        for source in model.sources:
            if can_do_no_pixel:
                for component in source.modelphotometric.components:
                    if not component.is_gaussian_mixture():
                        can_do_no_pixel = False
                        break
            else:
                break
        if can_do_no_pixel:
            engineopts["drawmethod"] = mpfobj.draw_method_pixel[engine]
    model.data.exposures = {band: [] for band in bands}
    values_init_psf = modelinfo.values_init_psf

    for exposure, psfs in exposures_psfs:
        exposure.psf = psfs[engine][name_psf]['object']
        model.data.exposures[exposure.band].append(exposure)
        if values_init_psf is not None:
            values_init_psf_band = values_init_psf.get(exposure.band)
            if values_init_psf_band is not None:
                params_psf = exposure.psf.model.get_parameters(fixed=False, free=True)
                if len(params_psf) != len(values_init_psf_band):
                    raise RuntimeError(
                        f'len(params_psf)={len(params_psf)} != len(values_init_psf_band)='
                        f'{len(values_init_psf_band)} (band={exposure.band})'
                    )
                for param, (name_init, value) in zip(params_psf, values_init_psf_band):
                    __validate_param_name(param.name, name_init)
                    param.value = value

    do_plot_multi = plot and len(bands) > 1
    if skip_fit:
        fits_model = []
    else:
        fits_model = fits.get(name_model)
        if fits_model is None:
            fits_model = ModelFits(modeltype=modeltype, fits=[])
            fits[name_model] = fits_model
        if fits_model.fits is None:
            fits_model.fits = []
        fits_model = fits_model.fits
    # For printing parameter values when plotting
    params_postfix_name_model = []
    do_init = redo or len(fits_model) == 0

    if do_init:
        # Parse default overrides from model spec
        flag_param_keys = ['inherit', 'modify']
        flag_params = {key: [] for key in flag_param_keys}
        for flag, values in (('fixed', modelinfo.fixedparams), ('init', modelinfo.initparams)):
            flag_params[flag] = {}
            if values:
                for flag_value in values.split(";"):
                    if flag == "fixed":
                        flag_params[flag][flag_value] = None
                    elif flag == "init":
                        value = flag_value.split("=")
                        # TODO: improve input handling here or just revamp the whole system later
                        if value[1] in flag_param_keys:
                            flag_params[value[1]].append(value[0])
                        else:
                            value_split = [np.float(x) for x in value[1].split(',')]
                            flag_params[flag][value[0]] = value_split

        # Initialize model from estimate of moments (size/ellipticity) or from another fit
        inittype = modelinfo.inittype
        init_with_values = inittype == 'values'
        guesstype = None
        params_fixed, params_init = flag_params['fixed'], flag_params['init']
        values_init = modelinfo.values_init

        # Need to set fixed parameters to fixed to get the right list of free params to init
        # Other init methods expect default values... I think.
        if init_with_values:
            _reset_params_fixed(
                model, params_fixed_default, flag_params['inherit'], fixed=True, modifiers=False)

        # Model may depend on earlier fits to initialize some of the values and fit only a subset
        init_hybrid = values_init is not None and inittype != 'values'

        if inittype == 'moments':
            if logger:
                logger.debug(f'Initializing from moments: {init_moments.moments_by_name}')
            for param in model.get_parameters(fixed=False):
                value = init_moments.moments_by_name.get(param.name)
                if value is not None:
                    param.value = value
        else:
            model, guesstype = init_model(
                model, modeltype, inittype, models, modelspecs_prior, fits, bands=bands,
                params_inherit=flag_params['inherit'], params_modify=flag_params['modify'],
                params_fixed=params_fixed, values_init=values_init
            )

        # Already done above in this case
        if not init_with_values:
            _reset_params_fixed(
                model, params_fixed_default, flag_params['inherit'], fixed=True, modifiers=False)
        # Now actually apply the overrides and the hardcoded maxima
        times_matched = {}

        for param in model.get_parameters(fixed=True):
            name_param, is_flux, is_fluxrat, param.fixed = _get_param_info(param, params_fixed)
            is_fluxrat = is_ratioparam(param)
            is_bg = param.name == 'background'

            # Passing complete init values will override individual params
            if (not init_with_values or param.fixed) and name_param in params_init:
                if name_param not in times_matched:
                    times_matched[name_param] = 0
                # If there's only one input value, assume it applies to all instances of this param
                idx_paraminit = (0 if len(params_init[name_param]) == 1 else
                                 times_matched[name_param])
                param.value = params_init[name_param][idx_paraminit]
                times_matched[name_param] += 1

            # Try to set a hard limit on params that need them with a logit transform
            # This way even methods that don't respect bounds will have to until the transformed
            # value reaches +/-inf, at least
            # The modelspec can override this for e.g. free Gaussian mixtures
            if (name_param in init_moments.values_max) and (not is_bg) and not (
                    name_param in params_unlimited):
                value_min = 0 if name_param not in init_moments.values_min else \
                    init_moments.values_min[name_param]
                value_max = init_moments.values_max[name_param]
                param.limits = g2f.LimitsD(min=value_min, max=value_max)

            # Reset non-finite free param values
            # This occurs e.g. at the limits of a logit transformed param
            if not param.fixed:
                value_param = param.value
                value_param_transformed = param.value_transformed
                if not np.isfinite(value_param_transformed):
                    # Get the next float in the direction of inf if -inf else -inf
                    # This works for nans too, otherwise we could use -value_param
                    # TODO: Deal with nans explicitly - they may not be recoverable
                    direction = -np.inf * np.sign(value_param_transformed)
                    # This is probably excessive but this ought to allow for a non-zero init. gradient
                    for _ in range(100):
                        value_param = np.nextafter(value_param, direction)
                    param.value = value_param
            if not np.isfinite(param.value):
                raise RuntimeError(f"Initialized param={param} to non-finite value")

            # This has to come after resetting param fixed status above
            if plot and not param.fixed:
                if name_param == "n_ser" or is_fluxrat:
                    params_postfix_name_model += [('{:.2f}', param)]

        values_param = np.array([x.value for x in model.get_parameters(
            fixed=True)])
        if not all(np.isfinite(values_param)):
            raise RuntimeError(f'Not all params finite for model {name_model}', values_param)

        if init_hybrid:
            init_model_from_values(model, values_init)

        # Setup priors
        if prior_specs:
            # Value is whether prior is applied per component or not
            priors_avail_per = {'shape': True, 'cen_x': False, 'cen_y': False}
            model.priors = []
            priors_comp = {}
            priors_src = {}
            prior_background = prior_specs.get('background')
            # Adjust the input background prior values
            if prior_background is not None and background_sigma_add is not None:
                for band, bg_prior_values in prior_background.items():
                    bg_prior_values['mean'] += background_sigma_add*bg_prior_values['stddev']

            for name_prior, params_prior_type in prior_specs.items():
                if name_prior != 'background':
                    prior_type = priors_avail_per.get(name_prior)
                    if prior_type is None:
                        raise RuntimeError(f'Unknown prior type {name_prior}')
                    (priors_comp if prior_type else priors_src)[name_prior] = params_prior_type

            for src in model.sources:
                all_gauss = all(comp.is_gaussian() for comp in src.modelphotometric.components)
                # TODO: There should be a better way of getting source-specific parameters
                # This hack will do for now
                params_src = {p.name: p for p in src.modelastrometric.get_parameters()}
                for name_prior, params_prior_type in priors_src.items():
                    params_prior = params_prior_type[all_gauss]
                    param = params_src[name_prior]
                    transformed = params_prior_type.get('transformed', False)
                    mean = params_prior.get(
                        'mean', param.value_transformed if transformed else param.value,
                    )
                    model.priors.append(
                        mpfpri.GaussianLsqPrior(param, mean=mean, **params_prior)
                    )
                for comp in src.modelphotometric.components:
                    if isinstance(comp, mpfobj.Background):
                        if prior_background:
                            for band, param_flux in comp.fluxes_dict.items():
                                model.priors.append(
                                    mpfpri.GaussianLsqPrior(param_flux, **prior_background[band])
                                )
                                # Adjust the initial background value if a constant was added
                                if background_sigma_add and param_flux not in params_adjusted:
                                    flux_add = background_sigma_add*prior_background[band]['stddev']
                                    param_flux.value = param_flux.value + flux_add
                                    params_adjusted[param_flux] = flux_add
                    else:
                        params_comp = {p.name: p for p in comp.get_parameters()}
                        for name_prior, params_prior_type in priors_comp.items():
                            params_prior = params_prior_type[all_gauss]
                            if name_prior == 'shape':
                                model.priors.append(mpfpri.ShapeLsqPrior(comp.params_ellipse, **params_prior))
                            else:
                                param = params_comp[name_prior]
                                transformed = params_prior.get('transformed', False)
                                mean = params_prior.get(
                                    'mean',
                                    param.value_transformed if transformed else param.value,
                                )
                                model.priors.append(
                                    mpfpri.GaussianLsqPrior(param, mean=mean, **params_prior)
                                )

        if skip_fit:
            # Make a dummy fit result in case it's needed for subsequent model initialization
            params_all = model.get_parameters(fixed=True)
            params_free = model.get_parameters(fixed=False)
            fits[name_model] = ModelFits(
                fits=[{
                    'name_params': [param.name for param in params_free],
                    'params': params_all,
                    'params_allfixed': [param.fixed for param in params_all],
                    'params_bestall': [param.value for param in params_all],
                    'chisqred': 1.,
                }],
                modeltype=modeltype,
            )
        else:
            if logger:
                logger.debug(f"Fitting model {name_model} of type {modeltype} with engine {engine}")
            model.name = name_model
            sys.stdout.flush()
            if guesstype is not None:
                init_model_by_guessing(model, guesstype, bands, nguesses=3)

            try:
                if background_sigma_add is not None:
                    # Add a flat background level to the image so that the fit background is >0
                    for exposure, _ in exposures_psfs:
                        added_bg = background_sigma_add*prior_background[exposure.band]['stddev']
                        if added_bg != 0:
                            exposure.image += added_bg
                            exposure.meta['bg_const_added'] = added_bg
                do_second = len(model.sources[0].modelphotometric.components) > 1 or \
                    not use_modellib_default
                if use_modellib_default:
                    modellibopts = {
                        "algo": ("lbfgs" if modellib == "pygmo" else "L-BFGS-B") if do_second else
                        ("neldermead" if modellib == "pygmo" else "Nelder-Mead")
                    }
                    if modellib == "scipy":
                        modellibopts['options'] = {'maxfun': 1e4}
                do_second = do_second and not model.can_do_fit_leastsq
                fit1, modeller = fit_model(
                    model, modellib=modellib, modellibopts=modellibopts, do_print_final=True,
                    print_step_interval=print_step_interval, plot=plot and not do_second,
                    do_plot_multi=do_plot_multi, figaxes=plotinfo.figaxes,
                    row_figure=idx_model, do_plot_as_column=plotinfo.do_plot_as_column,
                    name_model=name_model, params_postfix_name_model=params_postfix_name_model,
                    img_plot_maxs=img_plot_maxs, img_multi_plot_max=img_multi_plot_max,
                    weights_band=weights_band, kwargs_fit=kwargs_fit, logger_modeller=logger,
                )
                fits_model.append(fit1)
                if do_second:
                    if use_modellib_default:
                        modeller.modellibopts["algo"] = "neldermead" if modellib == "pygmo" else \
                            "Nelder-Mead"
                    fit2, _ = fit_model(
                        model, modeller, do_print_final=True,
                        print_step_interval=print_step_interval, plot=plot,
                        do_plot_multi=do_plot_multi, figaxes=plotinfo.figaxes,
                        row_figure=idx_model, do_plot_as_column=plotinfo.do_plot_as_column,
                        name_model=name_model, params_postfix_name_model=params_postfix_name_model,
                        img_plot_maxs=img_plot_maxs, img_multi_plot_max=img_multi_plot_max,
                        weights_band=weights_band, do_linear=False,
                    )
                    fits_model.append(fit2)
            except Exception as e:
                trace = traceback.format_exc()
                if print_exception:
                    print(f"Error fitting galaxy {name}: {e}; traceback: {trace}")
                fits_model.append((e, trace))

    if plot and (skip_fit or (not redo and fits_model is not None)):
        if background_sigma_add is not None:
            for exposure, _ in exposures_psfs:
                if 'bg_const_added' not in exposure.meta:
                    added_bg = background_sigma_add*prior_background[exposure.band][1]
                    if added_bg != 0:
                        exposure.image += added_bg
                        exposure.meta['bg_const_added'] = added_bg
        if skip_fit:
            values_best = (None,)
        else:
            values_best = fits.get(name_model)
            if values_best is not None:
                values_best = values_best.fits
                if values_best is not None and len(values_best) > 0:
                    values_best = values_best[-1]['params_bestalltransformed']

        if values_best is not None:
            if plotinfo.title is not None:
                plt.suptitle(plotinfo.title)
            model.evaluate(
                plot=plot, name_model=name_model,
                params_postfix_name_model=params_postfix_name_model,
                figaxes=plotinfo.figaxes, row_figure=idx_model,
                do_plot_as_column=plotinfo.do_plot_as_column, do_plot_multi=do_plot_multi,
                img_plot_maxs=img_plot_maxs, img_multi_plot_max=img_multi_plot_max,
                weights_band=weights_band
            )
    if background_sigma_add is not None:
        for exposure, _ in exposures_psfs:
            added_bg = exposure.meta.get('bg_const_added', 0)
            if added_bg != 0:
                exposure.image -= added_bg
                del exposure.meta['bg_const_added']
    return fits, params_adjusted


def get_psf_models(modelspecs):
    """ Return a set of PSF model specifications.

    :param modelspecs: List of dict, each a galaxy model specifications as returned by get_modelspecs.
    :return: set of tuples of PSF model type string (e.g. "gaussian:1") and pixelization boolean
    """
    return set([(x.psfmodel, x.psfpixel) for x in modelspecs])


def fit_psf_exposures(
    exposures_psfs, model_psfs, bands, results=None, sampling=1., shrink=0., engine=None, engineopts=None,
    logger=None, redo=True, plot=False, title=None, print_step_interval=None, skip_fit=False
):
    """ Fit the PSFs associated with exposures.

    :param exposures_psfs: List of tuples (multiprofit.object.Exposure, nparray with PSF image)
    :param model_psfs: set of list; PSF model specifications as returned by get_psf_models.
    :param bands: List of bands
    :param results: Dict with similar structure as return value.
    :param sampling: float; sampling factor - fit sizes will be divided by this value.
    :param shrink: float; Length in pixels to subtract from sigma_{x,y} in quadrature for future source fits.
    :param engine: String; the rendering engine to pass to fit_model.
    :param engineopts: Dict; the rendering options to pass to fit_model.
    :param logger: logging.Logger; a logger to print messages.
    :param redo: bool; Redo any pre-existing fits in fits_by_engine?
    :param plot: Boolean; generate plots?
    :param title: String; title for plots
    :param print_step_interval: int; fit step interval to pass to fit_model.
    :return: dict by exposure index containing a dict by engine of the result returned by fit_psf.

    """
    figaxes = None
    row = None
    resample = sampling != 1.
    resize = shrink > 0
    if plot:
        num_psfs = 0
        for modeltype_psf, _ in model_psfs:
            num_psfs += modeltype_psf != "empirical"
        num_psfs *= len(bands)
        if num_psfs > 1:
            ncols = 5
            figure, axes = plt.subplots(nrows=min([ncols, num_psfs]), ncols=max([ncols, num_psfs]))
            if num_psfs > ncols:
                axes = np.transpose(axes)
            figaxes = mpfobj.FigAxes(figure=figure, axes=axes)
            figaxes = {band: figaxes for band in bands}
            row = 0
    if results is None:
        results = {}
    if engine is None:
        engine = "galsim"
    if engineopts is None:
        engineopts = {
            "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
        }
    any_skipped = False
    for idx, (exposure, psf) in enumerate(exposures_psfs):
        band = exposure.band
        if band in bands:
            if idx not in results:
                results[idx] = {engine: {}}
            for modeltype_psf, is_psf_pixelated in model_psfs:
                name_psf = modeltype_psf + ("_pixelated" if is_psf_pixelated else "")
                label = modeltype_psf + (" pix." if is_psf_pixelated else "") + " PSF"
                if modeltype_psf == "empirical":
                    # TODO: Check if this works
                    results[idx][engine][name_psf] = {'object': mpfobj.PSF(
                        band=band, engine=engine, image=psf.image.array)}
                else:
                    engineopts["drawmethod"] = mpfobj.draw_method_pixel[engine] if is_psf_pixelated else None
                    has_fit = name_psf in results[idx][engine]
                    do_fit = redo or not has_fit
                    if do_fit or plot:
                        if logger and do_fit:
                            logger.debug('Fitting PSF band={} model={} (not in {})'.format(
                                band, name_psf, results[idx][engine].keys()))
                        results[idx] = fit_psf_model(
                            modeltype_psf, psf.image.array if (
                                    hasattr(psf, "image") and hasattr(psf.image, "array")) else (
                                psf if isinstance(psf, np.ndarray) else None),
                            {engine: engineopts}, band=band, fits_model_psf=results[idx], plot=plot,
                            name_model=name_psf, label=label, title=title, figaxes=figaxes,
                            row_figure=row, redo=do_fit, print_step_interval=print_step_interval,
                            do_linear=False, logger=logger, skip_fit=skip_fit
                        )

                        if do_fit or 'object' not in results[idx][engine][name_psf]:
                            model_psf = results[idx][engine][name_psf]['modeller'].model.sources[0]
                            results[idx][engine][name_psf]['object'] = mpfobj.PSF(
                                band=band, engine=engine, model=model_psf,
                                is_model_pixelated=is_psf_pixelated)

                        if resample or resize:
                            model_psf = results[idx][engine][name_psf]['modeller'].model.sources[0]
                            for param in (p for p in model_psf.get_parameters(fixed=True, free=True)
                                          if p.name.startswith('sigma')):
                                sigma = param.value
                                param.value = np.nanmax((1e-3, np.sqrt((sigma/sampling)**2 - shrink**2)))
                                sigma_new = param.value
                                if logger:
                                    logger.debug(
                                        f'Changed {param.name} value from {sigma:.5e} to {sigma_new:.5e}'
                                        f' (PSF sampling={sampling}, shrink={shrink})')

                        if plot and row is not None:
                            row += 1

            exposures_psfs[idx] = (exposure, results[idx])
        else:
            any_skipped = True
    return results, any_skipped


def fit_galaxy_exposures(
        exposures_psfs, bands, modelspecs, results=None, plot=False, name_fit=None, redo=False,
        engine=None, engineopts=None, psf_sampling=1, psf_shrink=0, redo_psfs=False, skip_fit_psf=False,
        reset_images=False, loggerPsf=None, **kwargs
):
    """
    Fit a set of exposures and accompanying PSF images in the given bands with the requested model
    specifications.

    :param exposures_psfs: List of tuples (multiprofit.object.Exposure, nparray with PSF image)
    :param bands: List of bands
    :param modelspecs: List of dicts; as in get_modelspecs().
    :param results: Dict with similar structure as return value.
    :param plot: Boolean; generate plots?
    :param name_fit: String; name of the galaxy/image to use as a title in plots
    :param redo: bool; Redo any pre-existing fits in fits_by_engine?
    :param engine: String; the rendering engine to pass to fit_model.
    :param engineopts: Dict; the rendering options to pass to fit_model.
    :param psf_sampling: float; sampling factor for the PSF - fit sizes will be divided by this value.
    :param psf_shrink: float; Length in pixels to subtract from PSF size_{x,y} in quadrature before fitting
        PSF-convolved models
    :param redo_psfs: Boolean; Redo any pre-existing PSF fits in results?
    :param reset_images: Boolean; whether to reset all images in data objects to EmptyImages before returning
    :param loggerPsf: logging.Logger; a logger to print messages for PSF fitting
    :param kwargs: dict; keyword: value arguments to pass on to fit_galaxy()
    :return: results: dict containing the following values:
        fits: dict by modelspec name of fit results
        models: dict by model type name of mpfobj.Model
        psfs: dict by PSF model name containing similar model/fit info
        metadata: dict containing general, model-independent metadata such as the filters used in the fit
    """
    if loggerPsf is None:
        loggerPsf = kwargs.get('logger', logging.getLogger(__name__))
    if results is None:
        results = {}
    if engine is None:
        engine = "galsim"
    if engineopts is None:
        engineopts = {
            "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
        }
    metadata = {"bands": bands}
    # Having worked out what the image, psf and variance map are, fit PSFs and images
    psfs = results['psfs'] if 'psfs' in results else {}
    model_psfs = get_psf_models(modelspecs)
    psfs, any_skipped = fit_psf_exposures(
        exposures_psfs, model_psfs, bands, results=psfs, engine=engine, engineopts=engineopts,
        sampling=psf_sampling, shrink=psf_shrink, redo=redo_psfs, logger=loggerPsf,
        plot=plot, title=name_fit, print_step_interval=np.Inf, skip_fit=skip_fit_psf
    )
    if plot:
        plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
    fits_by_engine = None if 'fits' not in results else results['fits']
    models = None if 'models' not in results else results['models']
    kwargs['redo'] = redo
    fits, models = fit_galaxy(
        exposures_psfs if not any_skipped else [x for x in exposures_psfs if x[0].band in bands],
        modelspecs=modelspecs, name=name_fit, plot=plot, models=models, fits_by_engine=fits_by_engine,
        **kwargs
    )
    if reset_images:
        for idx, psfs_by_engine in psfs.items():
            if engine in psfs_by_engine:
                for modeltype_psf, psf in psfs_by_engine[engine].items():
                    set_exposure(psf['modeller'].model, exposures_psfs[idx][0].band, image='empty')
        for name_model, model in models.items():
            for band in bands:
                set_exposure(model, band, image='empty')
            model.do_fit_leastsq_cleanup(bands)
        for engine, modelfitinfo in fits.items():
            for name_model, modelfits in modelfitinfo.items():
                if 'fits' in modelfits:
                    for fit in modelfits["fits"]:
                        fit["fitinfo"]["log"] = None
                        # Don't try to pickle pygmo problems for some reason I forget
                        if hasattr(fit["result"], "problem"):
                            fit["result"]["problem"] = None
    results = {'fits': fits, 'models': models, 'psfs': psfs, 'metadata': metadata}
    return results


def get_psfmodel(
        engine, engineopts, num_comps, band, model, image, error_inverse=None, ratios_size=None,
        factor_sigma=1, logger=None):
    sigma_x, sigma_y, rho = g2.Ellipse(
        g2.Covariance(*mpfutil.estimate_ellipse(image, return_as_params=True))
    ).xyr
    if logger:
        logger.debug(f'PSF init. mom. sig_x, sig_y, rho = ({sigma_x}, {sigma_y}, {rho})')
    sigma_x *= g2.M_SIGMA_HWHM
    sigma_y *= g2.M_SIGMA_HWHM
    sigma_xs = np.repeat(sigma_x, num_comps)
    sigma_ys = np.repeat(sigma_y, num_comps)
    rhos = np.repeat(rho, num_comps)

    if ratios_size is None and num_comps > 1:
        log_ratio = np.log(np.cbrt(num_comps))
        ratios_size = np.exp(np.linspace(-log_ratio, log_ratio, num_comps))
    if num_comps > 1:
        for idx, (ratio, sigma_x, sigma_y) in enumerate(zip(ratios_size, sigma_xs, sigma_ys)):
            sigma_xs[idx], sigma_ys[idx] = ratio*sigma_x, ratio*sigma_y

    model = get_model(
        {band: 1}, model, np.flip(image.shape, axis=0), sigma_xs, sigma_ys, rhos,
        fluxfracs=mpfutil.normalize(2.**(-np.arange(num_comps))), engine=engine, engineopts=engineopts,
        logger=logger)
    for param in model.get_parameters(fixed=False):
        is_flux, is_ratio = is_fluxparam_ratio(param)
        param.fixed = is_flux and not is_ratio
    set_exposure(model, band, image=image, error_inverse=error_inverse, factor_sigma=factor_sigma)
    return model


def get_modelspecs(filename) -> List[ModelSpec]:
    """
    Read a model specification file, the format of which is to be described.

    :param filename: String; path to the model specification file.
    :return: modelspecs; list of dicts of key specification name: value specification value.
    """
    if filename is None:
        rows = io.StringIO("\n".join([
            ",".join(["name", "model", "fixedparams", "initparams", "inittype", "psfmodel", "psfpixel"]),
            ",".join(["gausspix", "sersic:1", "n_ser", "n_ser=0.5", "moments", "gaussian:2", "T"]),
            ",".join(["gauss", "sersic:1",  "n_ser", "n_ser=0.5", "moments", "gaussian:2", "F"]),
            ",".join(["gaussinitpix ", "sersic:1", "n_ser", "", "gausspix", "gaussian:2", "F"]),
        ]))
    with rows if filename is None else open(filename, 'r') as filecsv:
        rows = [row for row in csv.reader(filecsv)]
    header = rows[0]
    header_set = set(header)
    if len(header_set) != len(header):
        raise RuntimeError(f"Modelspecs from filename={filename} header={header} contains duplicates")
    if not header_set.issubset(set(ModelSpec._fields)):
        raise RuntimeError(f"Modelspecs from filename={filename} header={header}"
                           f" not subset of {ModelSpec._fields}")
    modelspecs = []
    fieldtypes = list(ModelSpec._field_types[x] for x in header)
    for row in rows[1:]:
        kwargs = {}
        for idx, name in enumerate(header):
            value = row[idx]
            type_var = fieldtypes[idx]
            value = value if type_var is str else (mpfutil.str2bool(value) if type_var is bool else None)
            if value is None:
                raise RuntimeError(f'Got invalid modelspec value None from type_var={type_var}')
            kwargs[name] = value
        modelspecs.append(ModelSpec(**kwargs))
    # TODO: Validate modelspecs
    return modelspecs


def init_model_from_model_fits(model, modelfits, fluxfracs=None):
    # TODO: Come up with a better structure for parameter
    # TODO: Move to utils as a generic init model from other model(s) method
    chisqreds = [value['chisqred'] for value in modelfits]
    model_best = chisqreds.index(min(chisqreds))
    if fluxfracs is None:
        fluxfracs = 1./np.array(chisqreds)
        fluxfracs = fluxfracs/np.sum(fluxfracs)
    if not len([s for s in model.sources if not s.is_sky()]) == 1:
        raise RuntimeError("Can't init model with multiple non-background sources from fits")
    has_fluxfracs = len(model.sources[0].modelphotometric.fluxes) > 0
    if has_fluxfracs:
        total = 1
        for i, frac in enumerate(fluxfracs):
            fluxfracs[i] = frac/total
            total -= frac
        fluxfracs[-1] = 1.0
    fits_best = modelfits[model_best]
    if model.logger:
        model.logger.debug(f'Initializing from best model={fits_best["name"]} w/fluxfracs: {fluxfracs}')
    paramtree_best = fits_best['paramtree']
    fluxcens_init = paramtree_best[0][1][0] + paramtree_best[0][0]
    fluxcens_init_values = [p.value for p in fluxcens_init]
    fluxes_init = []
    comps_init = []
    for modelfit in modelfits:
        paramtree = modelfit['paramtree']
        for param, value in zip(modelfit['params'], modelfit['values_param']):
            param.value = value
        sourcefluxes = paramtree[0][1][0]
        if (len(sourcefluxes) > 0) != has_fluxfracs:
            raise RuntimeError('Can\'t init model with has_fluxfracs={} and opposite for model fit')
        for iflux, flux in enumerate(sourcefluxes):
            is_flux, is_ratio = is_fluxparam_ratio(flux)
            if is_flux and not is_ratio:
                fluxes_init.append(flux)
            else:
                raise RuntimeError(
                    "paramtree[0][1][0][{}] (type={}) isFluxParameter={} and/or is_ratio".format(
                        iflux, type(flux), is_flux))
        for comp in paramtree[0][1][1:-1]:
            comps_init.append([(param, param.value) for param in comp])
    params = model.get_parameters(fixed=True, flatten=False)
    # Assume one source
    params_src = params[0]
    flux_comps = params_src[1]
    fluxcens = flux_comps[0] + params_src[0]
    # The first list is fluxes
    comps = [comp for comp in flux_comps[1:-1]]
    # Check if fluxcens all length three with a total flux parameter and two centers named cen_x and cen_y
    # TODO: More informative errors; check fluxes_init
    bands = {flux.label: True for flux in fluxes_init}
    num_bands = len(bands)
    for name, fluxcen in {"init": fluxcens_init, "new": fluxcens}.items():
        num_fluxcens_expect = 2 + num_bands
        err_fluxcens = len(fluxcens) != num_fluxcens_expect
        err_fluxcens_init = len(fluxcens_init) != num_fluxcens_expect
        error_msg = None if not (err_fluxcens or err_fluxcens_init) else \
            '{} len(fluxcen{})={} != {}=(2 x,y cens + num_bands={})'.format(
                name,
                'init' if err_fluxcens_init else '', len(fluxcens) if err_fluxcens else len(fluxcens_init),
                num_fluxcens_expect, num_bands)
        if error_msg is not None:
            raise RuntimeError(error_msg)
        for idx_band in range(num_bands):
            is_flux, is_ratio = is_fluxparam_ratio(fluxcen[0])
            if not is_flux or is_ratio:
                raise RuntimeError(f"{name} fluxcen[0] (type={type(fluxcen[0])}) isFluxParameter={is_flux} "
                                   f"or is_ratio")
        if not (fluxcen[num_bands].name == "cen_x" and fluxcen[num_bands+1].name == "cen_y"):
            raise RuntimeError(f"{name}[{num_bands}:{num_bands+1}] names=({fluxcen[num_bands].name},"
                               f"{fluxcen[num_bands+1].name}) not ('cen_x','cen_y')")
    for param_to_set, value_init in zip(fluxcens, fluxcens_init_values):
        param_to_set.value = value_init
    # Check if num_comps equal
    if len(comps) != len(comps_init):
        raise RuntimeError(f"Model {model.name} has {len(comps)} components but prereqs "
                           f"{[x['modeltype'] for x in modelfits]} have a total of {len(comps_init)}")
    for idx_comp, (comp_set, comp_init) in enumerate(zip(comps, comps_init)):
        if len(comp_set) != len(comp_init):
            # TODO: More informative error plz
            raise RuntimeError(
                f'[len(compset)={len(comp_set)}, len(comp_init)={len(comp_init)}, '
                f'len(fluxfracs)={len(fluxfracs)}] not identical')
        for param_to_set, (param_init, value) in zip(comp_set, comp_init):
            is_flux, is_ratio = is_fluxparam_ratio(param_to_set)
            if is_flux:
                if has_fluxfracs:
                    if not is_ratioparam(param_to_set):
                        raise RuntimeError('Component flux parameter is not ratio but should be')
                    param_to_set.value = fluxfracs[idx_comp]
                else:
                    if is_ratioparam(param_to_set):
                        raise RuntimeError('Component flux parameter is ratio but shouldn\'t be')
                    # Note this means that the new total flux will be some weighted sum of the best fits
                    # for each model that went into this, which may not be ideal. Oh well!
                    param_to_set.value = param_init.value*fluxfracs[idx_comp]
            else:
                if type(param_to_set) != type(param_init):
                    # TODO: finish this
                    raise RuntimeError("Param types don't match")
                if param_to_set.name != param_init.name:
                    # TODO: warn or throw?
                    pass
                param_to_set.value = value


coeffs_init_guess = {
    'gauss2exp': ([-7.6464e+02, 2.5384e+02, -3.2337e+01, 2.8144e+00, -4.0001e-02], (0.005, 0.12)),
    'g2ev': ([-1.0557e+01, 1.6120e+01, -9.8877e+00, 4.0207e+00, -2.1059e-01], (0.05, 0.45)),
    'exp2dev': ([2.0504e+01, -1.3940e+01, 9.2510e-01, 2.2551e+00, -6.9540e-02], (0.02, 0.38)),
}


def init_model_by_guessing(model, guesstype, bands, nguesses=5):
    if nguesses > 0:
        time_init = time.time()
        do_sersic = guesstype == 'gauss2ser'
        guesstype_init = guesstype
        guesstypes = ['gauss2exp', 'g2ev'] if guesstype == 'gauss2ser' else [guesstype]
        like_init = None
        values_best = None
        for guesstype in guesstypes:
            if guesstype in coeffs_init_guess:
                if like_init is None:
                    like_init = model.evaluate()[0]
                    like_best = like_init
                params = model.get_parameters(fixed=False)
                names = ['sigma_x', 'sigma_y', 'rho']
                params_init = {name: [] for name in names + ['n_ser'] + ['flux_' + band for band in bands]}
                for param in params:
                    init = None
                    is_sersic = do_sersic and param.name == 'n_ser'
                    if is_fluxparam(param) and not (param.name == 'background'):
                        init = params_init['flux_' + param.label]
                    elif is_sersic or param.name in names:
                        init = params_init[param.name]
                    if init is not None:
                        init.append((param, param.value))
                    if is_sersic:
                        # TODO: This will change everything to exp/dev which is not really intended
                        # TODO: Decide if this should work for multiple components
                        param.value = 1. if guesstype == 'gauss2exp' else 4.
                num_params_init = len(params_init['rho'])
                # TODO: Ensure that fluxes and sizes come from the same component - this check is insufficient
                for band in bands:
                    if num_params_init != len(params_init['flux_' + band]):
                        raise RuntimeError('len(flux_{})={} != len(rho)={}; params_init={}'.format(
                            band, len(params_init['flux_' + band]), num_params_init, params_init))
                coeffs, xrange = coeffs_init_guess[guesstype]
                values_x = np.linspace(xrange[0], xrange[1], nguesses)
                values_y = np.polyval(coeffs, values_x)
                # For every pair of fluxes & sizes, take a guess
                for idx_param in range(num_params_init):
                    param_fluxes = [params_init['flux_' + band][idx_param] for band in bands]
                    param_sig_x = params_init['sigma_x'][idx_param]
                    param_sig_y = params_init['sigma_y'][idx_param]
                    params_to_set = param_fluxes + [param_sig_x, param_sig_y]
                    if do_sersic:
                        params_to_set += [params_init['n_ser'][idx_param]]
                    for x, y in zip(values_x, values_y):
                        for (param, value), ratiolog in [(param_flux, x) for param_flux in param_fluxes] + \
                                                        [(param_sig_x, y), (param_sig_y, y)]:
                            value_new = value*10**ratiolog
                            value_max = param.limits.max
                            param.value = np.min((0.9*value_max, value_new))
                        like = model.evaluate()[0]
                        if like > like_best:
                            like_best = like
                            values_best = {p[0]: p[0].value_transformed for p in params_to_set}
                if do_sersic:
                    for param_values in params_init.values():
                        for param, value in param_values:
                            param.value = value
        if values_best:
            for param, value in values_best.items():
                param.value_transformed = value
        if model.logger:
            model.logger.debug(
                f"Model '{model.name}' init by guesstype={guesstype_init} took {time.time() - time_init:.3e}s"
                f" to change like from {like_init} to {like_best}"
            )


def __validate_param_name(name_param, name_init):
    name_postfix = name_init.split('_')
    prefix = "sigma_" if (len(name_postfix) > 1) and (name_postfix[-2] == "sigma") else ""
    name_postfix = f'{prefix}{name_postfix[-1]}'
    if name_postfix == 'instFlux':
        name_postfix = 'flux'
    if not name_postfix.startswith(name_param):
        raise RuntimeError(f"Can't set param with name_postfix='{name_postfix}' !.startswith"
                           f" name_postfix='{name_param}' (shortened from name_init'{name_init}')")


def init_model_from_values(model, values_init, free=True, fixed=False, params_fixed=None):
    """

    :param model: `multiprofit.objects.Model` to set params for
    :param values_init: iterable [tuple[str, float]]; param name/value pairs
    :param free:
    :param fixed:
    :return: No return
    """
    if params_fixed is None:
        params_fixed = {}
    params = [param for param in model.get_parameters(free=free, fixed=fixed)
              if not _get_param_info(param, params_fixed)[3]]
    if not values_init or (len(values_init) != len(params)):
        raise RuntimeError(f"Can't init {len(params)} free params with {len(values_init)} values; "
                           f"params={params}, values_init={values_init}, params_fixed={params_fixed}")
    for param, (name_init, value) in zip(params, values_init):
        __validate_param_name(param.name, name_init)
        param.value = value


def init_model(
    model: mpfobj.Model,
    modeltype: str,
    inittype: str,
    models,
    modelinfo_comps: List[ModelSpec],
    fits_engine: Dict[str, ModelFits],
    bands=None,
    params_inherit=None, params_modify=None, params_fixed=None, values_init=None
):
    """
    Initialize a multiprofit.objects.Model of a given modeltype with a method inittype.

    :param model: A multiprofit.objects.Model.
    :param modeltype: String; a valid model type, as defined in TODO: define it somewhere.
    :param inittype: String; a valid initialization type, as defined in TODO: define it somewhere.
    :param models: Dict; key modeltype: value existing multiprofit.objects.Model.
        TODO: review if/when this is necessary.
    :param modelinfo_comps: Model specifications to map onto individual components of the model,
        e.g. to initialize a two-component model from two single-component fits.
    :param fits_engine: Dict; key=init/model type: value=FitResult.
    :param bands: String[]; a list of bands to pass to get_profiles when calling get_multigaussians.
    :param params_inherit: Inherited params object to pass to get_multigaussians.
    :param params_modify: Modified params object to pass to get_multigaussians.
    :param values_init: Initial untransformed parameter values. Required if inittype is 'values', else ignored
    :return: A multiprofit.objects.Model initialized as requested; it may be the original model or a new one.
    """

    logger = model.logger
    guesstype = None
    if inittype.startswith("guess"):
        guesstype = inittype.split(':')
        inittype = guesstype[1]
        guesstype = guesstype[0].split("guess")[1]
    if inittype.startswith("best"):
        if inittype == "best":
            name_modelcomps = []

            # Loop through all previous models and add ones of the same type
            for modelinfo_comp in modelinfo_comps:
                if modelinfo_comp.model == modeltype:
                    name_modelcomps.append(modelinfo_comp.name)
        else:
            # TODO: Check this more thoroughly
            name_modelcomps = inittype.split(":")[1].split(";")
        chisqreds = [fits_engine[name_modelcomp].fits[-1]["chisqred"] for name_modelcomp in name_modelcomps]
        inittype = name_modelcomps[np.argmin(chisqreds)]
    elif inittype == "values":
        init_model_from_values(model, values_init, params_fixed=params_fixed)
    else:
        inittype = inittype.split(';')
        if len(inittype) > 1:
            # Example:
            # mg8devexppx,mgsersic8:2,n_ser,"n_ser=4,1",mg8dev2px;mg8exppx,gaussian:3,T
            # ... means init two mgsersic8 profiles from some combination of the m8dev and mg8exp fits
            modelfits = []
            for initname in inittype:
                fitresult = fits_engine[initname]
                fit_last = fitresult.fits[-1]
                type_model = fitresult.modeltype
                model_init = models[type_model]
                modelfits.append({
                    'values_param': fit_last['params_bestall'],
                    'paramtree': model_init.get_parameters(fixed=True, flatten=False),
                    'params': model_init.get_parameters(fixed=True),
                    'chisqred': fit_last['chisqred'],
                    'modeltype': type_model,
                    'name': initname,
                })
            init_model_from_model_fits(model, modelfits)
            inittype = None
        else:
            inittype = inittype[0]
            if inittype not in fits_engine:
                raise RuntimeError("Model={} can't find reference={} "
                                   "to initialize from".format(modeltype, inittype))

    fit_init = fits_engine.get(inittype)
    has_fit_init = fit_init is not None

    if logger:
        logger.debug(f'Init model name={model.name} type-{modeltype} using inittype={inittype};'
                     f' hasinitfit={has_fit_init}')
    if has_fit_init:
        values_param_init = fit_init.fits[-1]["params_bestall"]
        # TODO: Find a more elegant method to do this
        inittype_mod = fit_init.modeltype.split('+')
        inittype_split = inittype_mod[0].split(':')
        inittype_order = None if not inittype_split[0].startswith('mgsersic') else \
            np.int(inittype_split[0].split('mgsersic')[1])
        if inittype_order is not None:
            if inittype_order not in MultiGaussianApproximationComponent.weights['sersic']:
                raise RuntimeError('Inittype {} has unimplemented order {} not in {}'.format(
                    inittype, inittype_order, MultiGaussianApproximationComponent.weights['sersic'].keys()))
        modeltype_base = modeltype.split('+')[0].split(':')
        is_mg_to_gauss = (
                inittype_order is not None and inittype_split[1].isdecimal() and
                modeltype_base[0] == 'gaussian' and len(modeltype_base) > 1 and modeltype_base[1].isdecimal()
                and np.int(modeltype_base[1]) == inittype_order*np.int(inittype_split[1])
        )
        if is_mg_to_gauss:
            num_components = np.repeat(inittype_order, inittype_split[1])
            num_sources = len(model.sources)
            model_new = model
            model = models[fit_init.modeltype]
            components_new = []
        if logger:
            logger.debug(f"Initializing from best model={inittype}"
                         f"{' (MGA to {} GMM)'.format(num_components) if is_mg_to_gauss else ''}")
        # For mgtogauss, first we turn the mgsersic model into a true GMM
        # Then we take the old model and get the parameters that don't depend on components (mostly source
        # centers) and set those as appropriate
        for i in range(1+is_mg_to_gauss):
            params_all = model.get_parameters(fixed=True, modifiers=not is_mg_to_gauss)
            if logger and is_mg_to_gauss:
                logger.debug(f"Param values init: {values_param_init}")
                logger.debug(f"Param names:       {[x.name for x in params_all]}")
            if len(values_param_init) != len(params_all):
                raise RuntimeError('len(values_param_init)={} != len(params)={}, params_all={}'.format(
                    len(values_param_init), len(params_all), [x.name for x in params_all]))
            for param, value in zip(params_all, values_param_init):
                # The logic here is that we can't start off an MG Sersic at n=0.5 since it's just one Gauss.
                # It's possible that a Gaussian mixture is better than an n<0.5 fit, so start it close to 0.55
                # Note that get_components (called by get_multigaussians) will ignore input values of n_ser
                # This prevents having a multigaussian model with components having n>0.5 (bad!)
                if is_mg_to_gauss and param.name == 'n_ser' and value <= 0.55:
                    value = 0.55
                param.value = value
            if logger and is_mg_to_gauss:
                logger.debug(f"Param values: {[param.value for param in model.get_parameters()]}")
            # Set the ellipse parameters fixed the first time through
            # The second time through, uh, ...? TODO Remember what happens
            if is_mg_to_gauss and i == 0:
                is_sky = num_sources*[False]
                for idx_src in range(num_sources):
                    is_sky[idx_src] = model.sources[idx_src].is_sky()
                    if not is_sky[idx_src]:
                        components_new.append(get_multigaussians(
                            model.sources[idx_src].get_profiles(bands=bands, engine='libprofit'),
                            params_inherit=params_inherit, params_modify=params_modify,
                            num_components=num_components, source=model_new.sources[idx_src]))

                        components_old = model.sources[idx_src].modelphotometric.components
                        for modeli in [model, model_new]:
                            modeli.sources[idx_src].modelphotometric.components = []
                        values_param_init = [param.value
                                             for param in model.get_parameters(fixed=True)]
                        if not np.all(np.isfinite(values_param_init)):
                            raise RuntimeError(f'values_param_init={values_param_init} not all finite')
                        model.sources[idx_src].modelphotometric.components = components_old
                model = model_new
        if is_mg_to_gauss:
            idx_new = 0
            for idx_src in range(num_sources):
                if not is_sky[idx_src]:
                    model.sources[idx_src].modelphotometric.components = components_new[idx_new]
                    idx_new += 1

    return model, guesstype


# Convenience function to set an exposure object with optional defaults for the sigma (variance) map
# Can be used to nullify an exposure before saving to disk, for example
def set_exposure(model, band, index=0, image=None, error_inverse=None, psf=None, mask=None, meta=None,
                 factor_sigma=1):
    if band not in model.data.exposures:
        model.data.exposures[band] = [mpfobj.Exposure(band=band, image=None)]
    exposure = model.data.exposures[band][index]
    is_empty_image = image == "empty"
    exposure.image = image if not is_empty_image else ImageEmpty(exposure.image.shape)
    if is_empty_image:
        exposure.error_inverse = exposure.image
    else:
        if psf is None and image is not None and error_inverse is None:
            img_sigma = np.sqrt(np.var(image))
            exposure.error_inverse = 1.0/(factor_sigma*img_sigma)
            exposure.is_error_sigma = True
        else:
            exposure.error_inverse = error_inverse
        if np.ndim(exposure.error_inverse) == 0:
            exposure.error_inverse = np.array([[exposure.error_inverse]])
    exposure.psf = psf
    exposure.mask = mask
    exposure.meta = {} if meta is None else meta
    return model


# TODO: Figure out multi-band operation here
def get_multigaussians(profiles, params_inherit=None, params_modify=None, num_components=None, source=None):
    """
    Get Gaussian component objects from profiles that are multi-Gaussian approximations (to e.g. Sersic)

    :param profiles: Dict; key band: value: profiles as formatted by mpf.objects.model.get_profiles()
    :param params_inherit: List of parameter names for Gaussians to inherit the values of (e.g. sigmas, rho)
    :param params_modify: List of parameter names modifying the Gaussian (e.g. r_scale)
    :param num_components: Array of ints specifying the number of Gaussian components in each physical
        component. Defaults to the number of Gaussians used to represent profiles, i.e. each Gaussian has
        independent ellipse parameters.
    :param source:
    :return: List of new Gaussian components
    """
    bands = {}
    for profile in profiles:
        for band in profile.keys():
            bands[band] = True
    bands = list(bands)
    band_ref = bands[0]
    params = ('flux', 'sigma_x', 'sigma_y', 'rho', 'n_ser')
    values = {}
    fluxes = {}

    for band in bands:
        # Keep these as lists to make the check against values[band_ref] below easier (no need to call np.all)
        values[band] = {
            ('slope' if (name == 'n_ser') else name): [profile[band][name] for profile in profiles]
            for name in params
        }
        # mag_to_flux needs a numpy array
        fluxes[band] = np.array(values[band]['flux'])
        # Ensure that only band-independent parameters are compared
        del values[band]['flux']
        if not values[band] == values[band_ref]:
            raise RuntimeError('values[{}]={} != values[{}]={}; band-dependent values unsupported'.format(
                band, values[band], band_ref, values[band_ref]))

    # Comparing dicts above is easier with list elements but we want np arrays in the end
    values = {key: np.array(value) for key, value in values[band_ref].items()}

    # These are the Gaussian components
    components_gauss = get_components('gaussian', fluxes, values=values, is_fluxes_fracs=False)
    num_components_gauss = len(components_gauss)
    if (params_inherit is not None or params_modify is not None) and num_components_gauss > 1:
        if params_inherit is None:
            params_inherit = []
        if params_modify is None:
            params_modify = []
        if num_components is None:
            num_components = [num_components_gauss]
        num_components = np.array(num_components)
        if np.sum(num_components) != num_components_gauss:
            raise ValueError(
                'Number of Gaussian components={} != total number of Gaussian sub-components in physical '
                'components={}; list={}'.format(num_components_gauss, np.sum(num_components), num_components))

        # Inheritee has every right to be a word
        component_init = 0
        params_modify_comps = {}
        if params_modify is not None:
            modifiers = source.modelphotometric.modifiers
            num_components_old = len(num_components)
            for name_param in params_modify:
                modifiers_of_type = [modifier for modifier in modifiers if modifier.name == name_param]
                num_modifiers = len(modifiers_of_type)
                if num_modifiers == 0:
                    if name_param == 'r_scale':
                        params_modify_comps[name_param] = [
                            get_param_default(name_param, value=1, fixed=False)
                            for _ in range(num_components_old)
                        ]
                        for param in params_modify_comps[name_param]:
                            source.modelphotometric.modifiers.append(param)
                elif num_modifiers == num_components_old:
                    params_modify_comps[name_param] = modifiers_of_type
                    if name_param == 'r_scale':
                        for modifier in modifiers_of_type:
                            modifier.value = 1
                else:
                    raise RuntimeError('Expected to find {} modifiers of type {} but found {}'.format(
                        num_components_old, name_param, num_modifiers
                    ))
        for idx_comp, num_comps_to_add in enumerate(num_components):
            params_inheritees = {
                param.name: param for param in components_gauss[component_init].get_parameters()
                if param.name in params_inherit}
            for param in params_inheritees.values():
                param.inheritors = set()
            params_modify_comp = [paramoftype[idx_comp] for paramoftype in params_modify_comps.values()]
            for offset in range(num_comps_to_add):
                comp = components_gauss[component_init+offset]
                for param in comp.get_parameters():
                    if offset > 0 and param.name in params_inherit:
                        # This param will map onto the first component's param
                        param.fixed = True
                        params_inheritees[param.name].inheritors.add(param)
                    if param.name == 'rho':
                        param.modifiers.update(params_modify_comp)
            component_init += num_comps_to_add

    return components_gauss
