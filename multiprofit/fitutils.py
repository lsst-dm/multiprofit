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

from collections import OrderedDict
import csv
import galsim as gs
import io
import logging
import matplotlib.pyplot as plt
from multiprofit.ellipse import Ellipse
from multiprofit.limits import limits_ref, Limits
from multiprofit.multigaussianapproxprofile import MultiGaussianApproximationComponent
import multiprofit.objects as mpfobj
from multiprofit.transforms import transforms_ref
import multiprofit.utils as mpfutil
import numpy as np
from scipy import stats
import sys
import time
import traceback


class ImageEmpty:
    shape = (0, 0)

    def __init__(self, shape=(0, 0)):
        self.shape = shape


# For priors
def norm_logpdf_mean(x, mean=0., scale=1.):
    return stats.norm.logpdf(x - mean, scale=scale)


def truncnorm_logpdf_mean(x, mean=0., scale=1., a=-np.inf, b=np.inf):
    return stats.truncnorm.logpdf(x - mean, scale=scale, a=a, b=b)


def is_fluxratio(param):
    return isinstance(param, mpfobj.FluxParameter) and param.is_fluxratio


def get_param_default(param, value=None, profile=None, fixed=False, is_value_transformed=False,
                      use_sersic_logit=True, is_multigauss=False):
    transform = transforms_ref["none"]
    limits = limits_ref["none"]
    name = param
    if param == "slope":
        if profile == "moffat":
            name = "con"
            transform = transforms_ref["inverse"]
            limits = limits_ref["coninverse"]
            if value is None:
                value = 2.5
        elif profile == "sersic":
            name = "nser"
            if use_sersic_logit:
                if is_multigauss:
                    transform = transforms_ref["logitmultigausssersic"]
                    limits = limits_ref["nsermultigauss"]
                else:
                    transform = transforms_ref["logitsersic"]
            else:
                transform = transforms_ref["log10"]
                limits = limits_ref["nserlog10"]
            if value is None:
                value = 0.5
    elif param == "sigma_x" or param == "sigma_y":
        transform = transforms_ref["log10"]
    elif param == "rho":
        transform = transforms_ref["logitrho"]
        limits = limits_ref["logitrho"]
    elif param == "rscale":
        transform = transforms_ref['log10']

    if value is None:
        # TODO: Improve this (at least check limits)
        value = 0.
    elif not is_value_transformed:
        value = transform.transform(value)

    param = mpfobj.Parameter(name, value, "", limits=limits,
                             transform=transform, transformed=True, fixed=fixed)
    return param


def get_components(profile, fluxes, values=None, is_values_transformed=False, is_fluxes_fracs=True):
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
        if 'nser' in values:
            values['nser'] = np.zeros_like(values['nser'])

    transform = transforms_ref["logit"] if is_fluxes_fracs else transforms_ref["log10"]
    for compi in range(num_comps):
        is_last = compi == (num_comps - 1)
        param_fluxescomp = [
            mpfobj.FluxParameter(
                band, "flux", transform.transform(fluxes[band][compi]), None, limits=limits_ref["none"],
                transform=transform, fixed=is_last, is_fluxratio=is_fluxes_fracs, transformed=True)
            for band in bands
        ]
        params = [
            get_param_default(
                param, valueslice[compi], profile, fixed=False,
                is_value_transformed=is_values_transformed, is_multigauss=is_multi_gaussian_sersic)
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
    fluxfracs=None, offset_xy=None, name_model="", namesrc="", n_exposures=1, engine="galsim",
    engineopts=None, is_values_transformed=False, convertfluxfracs=False, repeat_ellipse=False, logger=None
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
    :param name_model: String; a name for this model
    :param namesrc: String; a name for the source
    :param n_exposures: Int > 0; the number of exposures in each band.
    :param engine: String; the rendering engine to pass to the multiprofit.objects.Model.
    :param engineopts: Dict; the rendering options to pass to the multiprofit.objects.Model.
    :param is_values_transformed: Boolean; are the provided initial values above already transformed?
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
    cenx, ceny = [x / 2.0 for x in size_image]
    if offset_xy is not None:
        cenx += offset_xy[0]
        ceny += offset_xy[1]
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

    params_astrometry = [
        mpfobj.Parameter("cenx", cenx, "pix", Limits(lower=0., upper=size_image[0]),
                         transform=transforms_ref["none"]),
        mpfobj.Parameter("ceny", ceny, "pix", Limits(lower=0., upper=size_image[1]),
                         transform=transforms_ref["none"]),
    ]
    modelastro = mpfobj.AstrometricModel(params_astrometry)
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
        components += get_components(profile, fluxfracs_comp, values, is_values_transformed)
        if len(components) != num_profiles:
            raise RuntimeError('get_components returned {}/{} expected profiles'.format(
                len(components), num_profiles))
        compnum += num_profiles
    param_fluxes = [mpfobj.FluxParameter(
        band, "flux", np.log10(np.clip(fluxes_by_band[band], 1e-16, np.Inf)), None, limits=limits_ref["none"],
        transform=transforms_ref["log10"], transformed=True, prior=None, fixed=False, is_fluxratio=False)
        for bandi, band in enumerate(bands)
    ]
    modelphoto = mpfobj.PhotometricModel(components, param_fluxes)
    if convertfluxfracs:
        modelphoto.convert_param_fluxes(
            use_fluxfracs=False, transform=transforms_ref['log10'], limits=limits_ref["none"])
    source = mpfobj.Source(modelastro, modelphoto, namesrc)
    model = mpfobj.Model([source], data, engine=engine, engineopts=engineopts, name=name_model, logger=logger)
    return model


# Convenience function to evaluate a model and optionally plot with title, returning chi map only
def evaluate_model(model, plot=False, title=None, **kwargs):
    """
    Convenience function to evaluate a model and optionally at a title to the plot.
    :param model: multiprofit.Model
    :param plot: Boolean; generate plot?
    :param title: String; title to add on top of the plot.
    :param kwargs: Dict; additional arguments to pass to model.evaluate().
    :return: Chi maps for each exposure.
    """
    _, _, chis, _ = model.evaluate(plot=plot, **kwargs)

    if plot:
        if title is not None:
            plt.suptitle(title)
        #plt.show(block=False)
    return chis


# Convenience function to fit a model. kwargs are passed on to evaluate_model
def fit_model(model, modeller=None, modellib="scipy", modellibopts=None,
              do_print_final=True, print_step_interval=100, plot=False, do_linear=True, **kwargs):
    """
    Convenience function to fit a model with reasonable defaults.
    :param model: multiprofit.Model
    :param modeller: multiprofit.Modeller; default: new Modeller.
    :param modellib: String; the modelling library to use if modeller is None.
    :param modellibopts: Dict; options to pass to the modeller if modeller is None.
    :param do_print_final: Boolean; print the final parameter values?
    :param print_step_interval: Integer; step interval between printing.
    :param plot: Boolean; plot final fit?
    :param do_linear: Boolean; do linear fit?
    :param kwargs: Dict; passed to evaluate_model() after fitting is complete (e.g. plotting options).
    :return: Tuple of modeller.fit and modeller.
    """
    if modeller is None:
        modeller = mpfobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts)
    fit = modeller.fit(
        do_print_final=do_print_final, print_step_interval=print_step_interval, do_linear=do_linear)
    # Conveniently sets the parameters to the right values too
    # TODO: Find a better way to ensure chis are returned than setting do_draw_image=True
    chis = evaluate_model(model, plot=plot, param_values=fit["params_best"], do_draw_image=True, **kwargs)
    fit["chisqred"] = mpfutil.get_chisqred(chis)
    params = model.get_parameters()
    for item in ['params_bestall', 'params_bestalltransformed', 'params_allfixed']:
        fit[item] = []
    for param in params:
        fit["params_bestall"].append(param.get_value(transformed=False))
        fit["params_bestalltransformed"].append(param.get_value(transformed=True))
        fit["params_allfixed"].append(param.fixed)

    return fit, modeller


def fit_psf(modeltype, imgpsf, engines, band, fits_model_psf=None, error_inverse=None, modellib="scipy",
            modellibopts=None, plot=False, title='', name_model=None, label=None, do_print_final=True,
            print_step_interval=100, figaxes=(None, None), row_figure=None, redo=True, logger=None):
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
        if redo or 'fit' not in fits_model_psf[engine][name_model]:
            fits_model_psf[engine][name_model]['fit'], fits_model_psf[engine][name_model]['modeller'] = \
                fit_model(
                    model, modellib=modellib, modellibopts=modellibopts, do_print_final=do_print_final,
                    print_step_interval=print_step_interval, plot=plot, title=title, name_model=label,
                    figure=figaxes[0], axes=figaxes[1], row_figure=row_figure, do_linear=False
                )
        elif plot:
            exposure = model.data.exposures[band][0]
            is_empty = isinstance(exposure.image, ImageEmpty)
            if is_empty:
                set_exposure(model, band, image=imgpsf, error_inverse=error_inverse)
            evaluate_model(
                model, param_values=fits_model_psf[engine][name_model]['fit']['params_best'],
                plot=plot, title=title, name_model=label, figure=figaxes[0], axes=figaxes[1],
                row_figure=row_figure)
            if is_empty:
                set_exposure(model, band, image='empty')

    return fits_model_psf


# Engine is galsim; TODO: add options
def fit_galaxy(
        exposures_psfs, modelspecs, modellib=None, modellibopts=None, plot=False, name=None, models=None,
        fits_by_engine=None, redo=False, img_plot_maxs=None, img_multi_plot_max=None, weights_band=None,
        do_fit_fluxfracs=False, print_step_interval=100, logger=None, flux_min=1e-3
):
    """
    Convenience function to fit a galaxy given some exposures with PSFs.

    :param exposures_psfs: Iterable of tuple(mpfobj.Exposure, dict; key=psftype: value=mpfobj.PSF)
    :param modelspecs: Model specifications as returned by get_modelspecs
    :param modellib: string; Model fitting library
    :param modellibopts: dict; Model fitting library options
    :param plot: bool; whether to plot
    :param name: string; Name of the model for plot labelling
    :param models: dict; key=model name: value=mpfobj.Model
    :param fits_by_engine: dict; same format as return value
    :param redo: bool; Redo any pre-existing fits in fits_by_engine
    :param img_plot_maxs: dict; key=band: value=float (Maximum value when plotting images in this band)
    :param img_multi_plot_max: float; Maximum value of summed images when plotting multi-band.
    :param weights_band: dict; key=band: value=float (Multiplicative weight when plotting multi-band RGB)
    :param do_fit_fluxfracs: bool; fit component flux ratios instead of absolute fluxes?
    :param print_step_interval: int; number of steps to run before printing output
    :param logger: logging.Logger; a logger to print messages and be passed to model(ler)s

    :return: fits_by_engine: dict; key=engine: value=dict; key=name_model: value=dict;
        key='fits': value=array of fit results, key='modeltype': value =
        fits_by_engine[engine][name_model] = {"fits": fits, "modeltype": modeltype}
        , models: tuple of complicated structures:

        modelinfos: dict; key=model name: value=dict; TBD
        models: dict; key=model name: value=mpfobj.Model
        psfmodels: dict: TBD
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    bands = OrderedDict()
    fluxes = {}
    num_pix_img = None
    name_params_moments_init = mpfobj.names_params_ellipse
    cens = dict(cenx=0, ceny=0)
    moments_by_name = {name_param: 0 for name_param in name_params_moments_init}
    num_exposures_measured = 0
    for exposure, _ in exposures_psfs:
        band = exposure.band
        img_exp = exposure.image
        num_pix_img_exp = img_exp.shape
        if num_pix_img is None:
            num_pix_img = num_pix_img_exp
        elif num_pix_img_exp != num_pix_img:
            'fit_galaxy exposure image shape={} not same as first={}'.format(num_pix_img_exp, num_pix_img)
        if band not in bands:
            if np.sum(img_exp) > 0:
                moments, cenx, ceny = mpfutil.estimate_ellipse(img_exp, return_cens=True, validate=False)
                cens['cenx'] += cenx
                cens['ceny'] += ceny
                # TODO: subtract PSF moments from object
                for name_param, value in zip(name_params_moments_init,
                                             Ellipse.covar_matrix_as(moments, params=False)):
                    moments_by_name[name_param] += value
                num_exposures_measured += 1
            bands[exposure.band] = None
        # TODO: Figure out what to do if given multiple exposures per band (TBD if we want to)
        fluxes[band] = np.clip(np.sum(
            img_exp[exposure.mask_inverse] if exposure.mask_inverse is not None else img_exp), flux_min,
            np.Inf)
    num_pix_img = np.flip(num_pix_img, axis=0)
    for params in [cens, moments_by_name]:
        for name_param in params:
            params[name_param] /= num_exposures_measured
    moments_by_name = {name_param: value for name_param, value in zip(
        name_params_moments_init,
        Ellipse.covar_terms_as(*moments_by_name.values(), matrix=False, params=True))}
    logger.info(f"Bands: {bands}; Moment init.: {moments_by_name}")
    engine = 'galsim'
    engines = {
        engine: {
            "gsparams": gs.GSParams(
                kvalue_accuracy=1e-2, integration_relerr=1e-2, integration_abserr=1e-3,
                maximum_fft_size=32768)}
    }
    title = name if plot else None
    if modellib is None:
        modellib = "scipy"

    values_max = {
        "sigma_x": np.sqrt(np.sum((num_pix_img/2.)**2)),
        "sigma_y": np.sqrt(np.sum((num_pix_img/2.)**2)),
    }
    values_min = {}
    for band in bands:
        values_min["flux_" + band] = 1e-6 * fluxes[band]
        values_max["flux_" + band] = 100 * fluxes[band]
    models = {} if (models is None) else models
    params_fixed_default = {}
    fits_by_engine = {} if ((models is None) or (fits_by_engine is None)) else fits_by_engine
    use_modellib_default = modellibopts is None
    for engine, engineopts in engines.items():
        if engine not in fits_by_engine:
            fits_by_engine[engine] = {}
        fits_engine = fits_by_engine[engine]
        if plot:
            num_rows, num_cols = len(modelspecs), 0
            figures = {}
            axes_list = {}
            for band in list(bands) + (['multi'] if len(bands) > 1 else []):
                num_cols = 5
                # Change to landscape
                figure, axes = plt.subplots(nrows=min([num_cols, num_rows]), ncols=max([num_cols, num_rows]))
                if num_rows > num_cols:
                    axes = np.transpose(axes)
                # This keeps things consistent with the nrows>1 case
                if num_rows == 1:
                    axes = np.array([axes])
                if title is not None:
                    plt.suptitle(title + " {} model".format(engine))
                figures[band] = figure
                axes_list[band] = axes
            if len(bands) == 1:
                figures = figures[band]
                axes_list = axes_list[band]
            do_plot_as_column = num_rows > num_cols
        else:
            figures = None
            axes_list = None
            do_plot_as_column = None
        for modelidx, modelinfo in enumerate(modelspecs):
            name_model = modelinfo["name"]
            modeltype = modelinfo["model"]
            model_default = get_model(
                fluxes, modeltype, num_pix_img, [moments_by_name["sigma_x"]],
                [moments_by_name["sigma_y"]], [moments_by_name["rho"]],
                engine=engine, engineopts=engineopts, convertfluxfracs=not do_fit_fluxfracs,
                repeat_ellipse=True, name_model=name_model, logger=logger
            )
            params_fixed_default[modeltype] = [
                param.fixed for param in model_default.get_parameters(fixed=True)]
            exists_model = modeltype in models
            model = model_default if not exists_model else models[modeltype]
            if not exists_model:
                models[modeltype] = model
            is_psf_pixelated = mpfutil.str2bool(modelinfo["psfpixel"])
            name_psf = modelinfo["psfmodel"] + ("_pixelated" if is_psf_pixelated else "")
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
            for exposure, psfs in exposures_psfs:
                exposure.psf = psfs[engine][name_psf]['object']
                model.data.exposures[exposure.band].append(exposure)
            do_plot_multi = plot and len(bands) > 1
            if not redo and name_model in fits_by_engine[engine] and \
                    'fits' in fits_by_engine[engine][name_model]:
                if plot:
                    values_best = fits_engine[name_model]['fits'][-1]['params_bestalltransformed']
                    # TODO: consider how to avoid code repetition here and below
                    params_postfix_name_model = []
                    for param, value in zip(model.get_parameters(fixed=True), values_best):
                        param.set_value(value, transformed=True)
                        if (param.name == "nser" and (
                                not param.fixed or param.get_value(transformed=False) != 0.5)) or \
                            param.name == "re" or (is_fluxratio(param) and param.get_value(
                                transformed=False) < 1):
                            params_postfix_name_model += [('{:.2f}', param)]
                    if title is not None:
                        plt.suptitle(title)
                    model.evaluate(
                        plot=plot, name_model=name_model, params_postfix_name_model=params_postfix_name_model,
                        figure=figures, axes=axes_list, row_figure=modelidx,
                        do_plot_as_column=do_plot_as_column, do_plot_multi=do_plot_multi,
                        img_plot_maxs=img_plot_maxs, img_multi_plot_max=img_multi_plot_max,
                        weights_band=weights_band)
            else:
                # Parse default overrides from model spec
                flag_param_keys = ['inherit', 'modify']
                flag_params = {key: [] for key in flag_param_keys}
                for flag in ['fixed', 'init']:
                    flag_params[flag] = {}
                    values = modelinfo[flag + 'params']
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
                inittype = modelinfo['inittype']
                guesstype = None
                if inittype == 'moments':
                    logger.info('Initializing from moments')
                    for param in model.get_parameters(fixed=False):
                        value = moments_by_name.get(param.name)
                        if value is None:
                            value = cens.get(param.name)
                        if value is not None:
                            param.set_value(value, transformed=False)
                else:
                    model, guesstype = init_model(
                        model, modeltype, inittype, models, modelspecs[0:modelidx], fits_engine, bands=bands,
                        params_inherit=flag_params['inherit'], params_modify=flag_params['modify'])

                # Reset parameter fixed status
                for param, fixed in zip(model.get_parameters(fixed=True, modifiers=False),
                                        params_fixed_default[modeltype]):
                    if param.name not in flag_params['inherit']:
                        param.fixed = fixed
                # For printing parameter values when plotting
                params_postfix_name_model = []
                # Now actually apply the overrides and the hardcoded maxima
                times_matched = {}
                for param in model.get_parameters(fixed=True):
                    is_flux = isinstance(param, mpfobj.FluxParameter)
                    is_fluxrat = is_fluxratio(param)
                    name_param = param.name if not is_flux else (
                        'flux' + ('ratio' if is_fluxrat else '') + '_' + param.band)
                    if name_param in flag_params['fixed'] or (is_flux and 'flux' in flag_params['fixed']):
                        param.fixed = True
                    # TODO: Figure out a better way to reset modifiers to be free
                    elif name_param == 'rscale':
                        param.fixed = False
                    if name_param in flag_params["init"]:
                        if name_param not in times_matched:
                            times_matched[name_param] = 0
                        # If there's only one input value, assume it applies to all instances of this param
                        idx_paraminit = (0 if len(flag_params["init"][name_param]) == 1 else
                                         times_matched[name_param])
                        param.set_value(flag_params["init"][name_param][idx_paraminit],
                                       transformed=False)
                        times_matched[name_param] += 1
                    if plot and not param.fixed:
                        if name_param == "nser" or is_fluxrat:
                            params_postfix_name_model += [('{:.2f}', param)]
                    # Try to set a hard limit on params that need them with a logit transform
                    # This way even methods that don't respect bounds will have to until the transformed
                    # value reaches +/-inf, at least
                    if name_param in values_max:
                        value_min = 0 if name_param not in values_min else values_min[name_param]
                        value_max = values_max[name_param]
                        transform = param.transform.transform
                        param.limits = mpfobj.Limits(
                            lower=transform(value_min), upper=transform(value_max),
                            transformed=True)
                    # Reset non-finite free param values
                    # This occurs e.g. at the limits of a logit transformed param
                    if not param.fixed:
                        value_param = param.get_value(transformed=False)
                        value_param_transformed = param.get_value(transformed=True)
                        if not np.isfinite(value_param_transformed):
                            # Get the next float in the direction of inf if -inf else -inf
                            # This works for nans too, otherwise we could use -value_param
                            # TODO: Deal with nans explicitly - they may not be recoverable
                            direction = -np.inf * np.sign(value_param_transformed)
                            # This is probably excessive but this ought to allow for a non-zero init. gradient
                            for _ in range(100):
                                value_param = np.nextafter(value_param, direction)
                            param.set_value(value_param, transformed=False)

                values_param = np.array([x.get_value(transformed=False) for x in model.get_parameters(
                    fixed=True)])
                if not all(np.isfinite(values_param)):
                    raise RuntimeError('Not all params finite for model {}:'.format(name_model), values_param)

                logger.info("Fitting model {:s} of type {:s} using engine {:s}".format(
                    name_model, modeltype, engine))
                model.name = name_model
                sys.stdout.flush()
                if guesstype is not None:
                    init_model_by_guessing(model, guesstype, bands, nguesses=3)
                try:
                    fits = []
                    do_second = len(model.sources[0].modelphotometric.components) > 1 or \
                        not use_modellib_default
                    if use_modellib_default:
                        modellibopts = {
                            "algo": ("lbfgs" if modellib == "pygmo" else "L-BFGS-B") if do_second else
                            ("neldermead" if modellib == "pygmo" else "Nelder-Mead")
                        }
                        if modellib == "scipy":
                            modellibopts['options'] = {'maxfun': 1e4}
                    fit1, modeller = fit_model(
                        model, modellib=modellib, modellibopts=modellibopts, do_print_final=True,
                        print_step_interval=print_step_interval, plot=plot and not do_second,
                        do_plot_multi=do_plot_multi, figure=figures, axes=axes_list, row_figure=modelidx,
                        do_plot_as_column=do_plot_as_column, name_model=name_model,
                        params_postfix_name_model=params_postfix_name_model,
                        img_plot_maxs=img_plot_maxs, img_multi_plot_max=img_multi_plot_max,
                        weights_band=weights_band
                    )
                    fits.append(fit1)
                    if do_second and not model.can_do_fit_leastsq:
                        if use_modellib_default:
                            modeller.modellibopts["algo"] = "neldermead" if modellib == "pygmo" else \
                                "Nelder-Mead"
                        fit2, _ = fit_model(
                            model, modeller, do_print_final=True, print_step_interval=print_step_interval,
                            plot=plot, do_plot_multi=do_plot_multi, figure=figures, axes=axes_list,
                            row_figure=modelidx, do_plot_as_column=do_plot_as_column, name_model=name_model,
                            params_postfix_name_model=params_postfix_name_model, img_plot_maxs=img_plot_maxs,
                            img_multi_plot_max=img_multi_plot_max, weights_band=weights_band, do_linear=False,
                        )
                        fits.append(fit2)
                    fits_by_engine[engine][name_model] = {"fits": fits, "modeltype": modeltype}
                except Exception as e:
                    print("Error fitting galaxy {}:".format(name))
                    print(e)
                    trace = traceback.format_exc()
                    print(trace)
                    fits_by_engine[engine][name_model] = e, trace
    if plot:
        if len(bands) > 1:
            for figure in figures.values():
                plt.figure(figure.number)
                plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
        else:
            plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
        plt.show(block=False)

    return fits_by_engine, models


def fit_galaxy_exposures(
        exposures_psfs, bands, modelspecs, results=None, plot=False, name_fit=None, redo=False,
        redo_psfs=False, reset_images=False, loggerPsf=None, **kwargs
):
    """
    Fit a set of exposures and accompanying PSF images in the given bands with the requested model
    specifications.

    :param exposures_psfs: List of tuples (multiprofit.object.Exposure, nparray with PSF image)
    :param bands: List of bands
    :param modelspecs: List of dicts; as in get_modelspecs().
    :param results:
    :param plot: Boolean; generate plots?
    :param name_fit: String; name of the galaxy/image to use as a title in plots
    :param redo: bool; Redo any pre-existing fits in fits_by_engine?
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
        loggerPsf = logging.getLogger(__name__)
    if results is None:
        results = {}
    metadata = {"bands": bands}
    # Having worked out what the image, psf and variance map are, fit PSFs and images
    psfs = results['psfs'] if 'psfs' in results else {}
    model_psfs = set([(x["psfmodel"], mpfutil.str2bool(x["psfpixel"])) for x in modelspecs])
    engine = 'galsim'
    engineopts = {
        "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
    }
    figure, axes = (None, None)
    row_psf = None
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
            row_psf = 0
    any_skipped = False
    for idx, (exposure, psf) in enumerate(exposures_psfs):
        band = exposure.band
        if band in bands:
            if idx not in psfs:
                psfs[idx] = {engine: {}}
            for modeltype_psf, is_psf_pixelated in model_psfs:
                name_psf = modeltype_psf + ("_pixelated" if is_psf_pixelated else "")
                label = modeltype_psf + (" pix." if is_psf_pixelated else "") + " PSF"
                if modeltype_psf == "empirical":
                    # TODO: Check if this works
                    psfs[idx][engine][name_psf] = {'object': mpfobj.PSF(
                        band=band, engine=engine, image=psf.image.array)}
                else:
                    engineopts["drawmethod"] = mpfobj.draw_method_pixel[engine] if is_psf_pixelated else None
                    has_fit = name_psf in psfs[idx][engine]
                    do_fit = redo_psfs or not has_fit
                    if do_fit or plot:
                        if do_fit:
                            loggerPsf.info('Fitting PSF band={} model={} (not in {})'.format(
                                band, name_psf, psfs[idx][engine].keys()))
                        psfs[idx] = fit_psf(
                            modeltype_psf, psf.image.array if (
                                    hasattr(psf, "image") and hasattr(psf.image, "array")) else (
                                psf if isinstance(psf, np.ndarray) else None),
                            {engine: engineopts}, band=band, fits_model_psf=psfs[idx], plot=plot,
                            name_model=name_psf, label=label, title=name_fit, figaxes=(figure, axes),
                            row_figure=row_psf, redo=do_fit, print_step_interval=np.Inf, logger=loggerPsf)
                        if do_fit or 'object' not in psfs[idx][engine][name_psf]:
                            psfs[idx][engine][name_psf]['object'] = mpfobj.PSF(
                                band=band, engine=engine,
                                model=psfs[idx][engine][name_psf]['modeller'].model.sources[0],
                                is_model_pixelated=is_psf_pixelated)
                        if plot and row_psf is not None:
                            row_psf += 1
            exposures_psfs[idx] = (exposure, psfs[idx])
        else:
            any_skipped = True
    if plot:
        plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
    fits_by_engine = None if 'fits' not in results else results['fits']
    models = None if 'models' not in results else results['models']
    kwargs['redo'] = redo
    fits, models = fit_galaxy(
        exposures_psfs if not any_skipped else [x for x in exposures_psfs if x[0].band in bands],
        modelspecs=modelspecs, name=name_fit, plot=plot, models=models, fits_by_engine=fits_by_engine,
        **kwargs)
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
    sigma_x, sigma_y, rho = Ellipse.covar_matrix_as(mpfutil.estimate_ellipse(image), params=True)
    sigma_xs = np.repeat(sigma_x, num_comps)
    sigma_ys = np.repeat(sigma_y, num_comps)
    rhos = np.repeat(rho, num_comps)

    if ratios_size is None and num_comps > 1:
        log_ratio = np.log(np.sqrt(num_comps))
        ratios_size = np.exp(np.linspace(-log_ratio, log_ratio, num_comps))
    if num_comps > 1:
        for idx, (ratio, sigma_x, sigma_y) in enumerate(zip(ratios_size, sigma_xs, sigma_ys)):
            sigma_xs[idx], sigma_ys[idx] = ratio*sigma_x, ratio*sigma_y

    model = get_model(
        {band: 1}, model, np.flip(image.shape, axis=0), sigma_xs, sigma_ys, rhos,
        fluxfracs=mpfutil.normalize(2.**(-np.arange(num_comps))), engine=engine, engineopts=engineopts,
        logger=logger)
    for param in model.get_parameters(fixed=False):
        param.fixed = isinstance(param, mpfobj.FluxParameter) and not param.is_fluxratio
    set_exposure(model, band, image=image, error_inverse=error_inverse, factor_sigma=factor_sigma)
    return model


def get_modelspecs(filename=None):
    """
    Read a model specification file, the format of which is to be described.

    :param filename: String; path to the model specification file.
    :return: modelspecs; list of dicts of key specification name: value specification value.
    """
    if filename is None:
        modelspecs = io.StringIO("\n".join([
            ",".join(["name", "model", "fixedparams", "initparams", "inittype", "psfmodel", "psfpixel"]),
            ",".join(["gausspix", "sersic:1", "nser", "nser=0.5", "moments", "gaussian:2", "T"]),
            ",".join(["gauss", "sersic:1",  "nser", "nser=0.5", "moments", "gaussian:2", "F"]),
            ",".join(["gaussinitpix ", "sersic:1", "nser", "", "gausspix", "gaussian:2", "F"]),
        ]))
    with modelspecs if filename is None else open(filename, 'r') as filecsv:
        modelspecs = [row for row in csv.reader(filecsv)]
    header = modelspecs[0]
    del modelspecs[0]
    modelspecs = [{name: modelspec[i] for i, name in enumerate(header)} for modelspec in modelspecs]
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
    if not len(model.sources) == 1:
        raise RuntimeError("Can't init model with multiple sources from fits")
    has_fluxfracs = len(model.sources[0].modelphotometric.fluxes) > 0
    if has_fluxfracs:
        total = 1
        for i, frac in enumerate(fluxfracs):
            fluxfracs[i] = frac/total
            total -= frac
        fluxfracs[-1] = 1.0
    model.logger.info('Initializing from best model={} w/fluxfracs: {}'.format(
        modelfits[model_best]['name'], fluxfracs))
    paramtree_best = modelfits[model_best]['paramtree']
    fluxcens_init = paramtree_best[0][1][0] + paramtree_best[0][0]
    # Get fluxes and components for init
    fluxes_init = []
    comps_init = []
    for modelfit in modelfits:
        paramtree = modelfit['paramtree']
        for param, value in zip(modelfit['params'], modelfit['values_param']):
            param.set_value(value, transformed=False)
        sourcefluxes = paramtree[0][1][0]
        if (len(sourcefluxes) > 0) != has_fluxfracs:
            raise RuntimeError('Can\'t init model with has_fluxfracs={} and opposite for model fit')
        for iflux, flux in enumerate(sourcefluxes):
            is_flux = isinstance(flux, mpfobj.FluxParameter)
            if is_flux and not flux.is_fluxratio:
                fluxes_init.append(flux)
            else:
                raise RuntimeError(
                    "paramtree[0][1][0][{}] (type={}) isFluxParameter={} and/or is_fluxratio".format(
                        iflux, type(flux), is_flux))
        for comp in paramtree[0][1][1:-1]:
            comps_init.append([(param, param.get_value(transformed=False)) for param in comp])
    params = model.get_parameters(fixed=True, flatten=False)
    # Assume one source
    params_src = params[0]
    flux_comps = params_src[1]
    fluxcens = flux_comps[0] + params_src[0]
    # The first list is fluxes
    comps = [comp for comp in flux_comps[1:-1]]
    # Check if fluxcens all length three with a total flux parameter and two centers named cenx and ceny
    # TODO: More informative errors; check fluxes_init
    bands = set([flux.band for flux in fluxes_init])
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
            is_flux = isinstance(fluxcen[0], mpfobj.FluxParameter)
            if not is_flux or fluxcen[0].is_fluxratio:
                raise RuntimeError("{} fluxcen[0] (type={}) isFluxParameter={} or is_fluxratio".format(
                    name, type(fluxcen[0]), is_flux))
        if not (fluxcen[num_bands].name == "cenx" and fluxcen[num_bands+1].name == "ceny"):
            raise RuntimeError("{}[{}:{}] names=({},{}) not ('cenx','ceny')".format(
                name, num_bands, num_bands+1, fluxcen[num_bands].name, fluxcen[num_bands+1].name))
    for param_to_set, param_init in zip(fluxcens, fluxcens_init):
        param_to_set.set_value(param_init.get_value(transformed=False), transformed=False)
    # Check if num_comps equal
    if len(comps) != len(comps_init):
        raise RuntimeError("Model {} has {} components but prereqs {} have a total of "
                           "{}".format(model.name, len(comps), [x['modeltype'] for x in modelfits],
                                       len(comps_init)))
    for idx_comp, (comp_set, comp_init) in enumerate(zip(comps, comps_init)):
        if len(comp_set) != len(comp_init):
            # TODO: More informative error plz
            raise RuntimeError(
                '[len(compset)={}, len(comp_init)={}, len(fluxfracs)={}] not identical'.format(
                    len(comp_set), len(comp_init), len(fluxfracs)
                ))
        for param_to_set, (param_init, value) in zip(comp_set, comp_init):
            if isinstance(param_to_set, mpfobj.FluxParameter):
                if has_fluxfracs:
                    if not param_to_set.is_fluxratio:
                        raise RuntimeError('Component flux parameter is not ratio but should be')
                    param_to_set.set_value(fluxfracs[idx_comp], transformed=False)
                else:
                    if param_to_set.is_fluxratio:
                        raise RuntimeError('Component flux parameter is ratio but shouldn\'t be')
                    # Note this means that the new total flux will be some weighted sum of the best fits
                    # for each model that went into this, which may not be ideal. Oh well!
                    param_to_set.set_value(param_init.get_value(transformed=False)*fluxfracs[idx_comp],
                                          transformed=False)
            else:
                if type(param_to_set) != type(param_init):
                    # TODO: finish this
                    raise RuntimeError("Param types don't match")
                if param_to_set.name != param_init.name:
                    # TODO: warn or throw?
                    pass
                param_to_set.set_value(value, transformed=False)


coeffs_init_guess = {
    'gauss2exp': ([-7.6464e+02, 2.5384e+02, -3.2337e+01, 2.8144e+00, -4.0001e-02], (0.005, 0.12)),
    'gauss2dev': ([-1.0557e+01, 1.6120e+01, -9.8877e+00, 4.0207e+00, -2.1059e-01], (0.05, 0.45)),
    'exp2dev': ([2.0504e+01, -1.3940e+01, 9.2510e-01, 2.2551e+00, -6.9540e-02], (0.02, 0.38)),
}


def init_model_by_guessing(model, guesstype, bands, nguesses=5):
    if nguesses > 0:
        time_init = time.time()
        do_sersic = guesstype == 'gauss2ser'
        guesstype_init = guesstype
        guesstypes = ['gauss2exp', 'gauss2dev'] if guesstype == 'gauss2ser' else [guesstype]
        like_init = None
        values_best = None
        for guesstype in guesstypes:
            if guesstype in coeffs_init_guess:
                if like_init is None:
                    like_init = model.evaluate()[0]
                    like_best = like_init
                params = model.get_parameters(fixed=False)
                names = ['sigma_x', 'sigma_y', 'rho']
                params_init = {name: [] for name in names + ['nser'] + ['flux_' + band for band in bands]}
                for param in params:
                    init = None
                    is_sersic = do_sersic and param.name == 'nser'
                    if isinstance(param, mpfobj.FluxParameter):
                        init = params_init['flux_' + param.band]
                    elif is_sersic or param.name in names:
                        init = params_init[param.name]
                    if init is not None:
                        init.append((param, param.get_value(transformed=False)))
                    if is_sersic:
                        # TODO: This will change everything to exp/dev which is not really intended
                        # TODO: Decide if this should work for multiple components
                        param.set_value(1. if guesstype == 'gauss2exp' else 4., transformed=False)
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
                        params_to_set += [params_init['nser'][idx_param]]
                    for x, y in zip(values_x, values_y):
                        for (param, value), ratiolog in [(param_flux, x) for param_flux in param_fluxes] + \
                                                        [(param_sig_x, y), (param_sig_y, y)]:
                            value_new = value*10**ratiolog
                            if not np.isfinite(value_new):
                                raise RuntimeError('Init by guessing tried to set non-finite value_new={}'
                                                   'from value={} and ratiolog={}'.format(
                                                        value_new, value, ratiolog))
                            param.set_value(value_new, transformed=False)
                        like = model.evaluate()[0]
                        if like > like_best:
                            like_best = like
                            values_best = {p[0]: p[0].get_value(transformed=True) for p in params_to_set}
                if do_sersic:
                    for param_values in params_init.values():
                        for param, value in param_values:
                            param.set_value(value, transformed=False)
        if values_best:
            for param, value in values_best.items():
                param.set_value(value, transformed=True)
        model.logger.info("Model '{}' init by guesstype={} took {:.3e}s to change like from {} to {}".format(
            model.name, guesstype_init, time.time() - time_init, like_init, like_best))


def init_model(model, modeltype, inittype, models, modelinfo_comps, fits_engine, bands=None,
               params_inherit=None, params_modify=None):
    """
    Initialize a multiprofit.objects.Model of a given modeltype with a method inittype.

    :param model: A multiprofit.objects.Model.
    :param modeltype: String; a valid model type, as defined in TODO: define it somewhere.
    :param inittype: String; a valid initialization type, as defined in TODO: define it somewhere.
    :param models: Dict; key modeltype: value existing multiprofit.objects.Model.
        TODO: review if/when this is necessary.
    :param modelinfo_comps: Model specifications to map onto individual components of the model,
        e.g. to initialize a two-component model from two single-component fits.
    :param fits_engine: Dict; key=engine: value=dict with fit results.
    :param bands: String[]; a list of bands to pass to get_profiles when calling get_multigaussians.
    :param params_inherit: Inherited params object to pass to get_multigaussians.
    :param params_modify: Modified params object to pass to get_multigaussians.
    :return: A multiprofit.objects.Model initialized as requested; it may be the original model or a new one.
    """
    # TODO: Refactor into function
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
                if modelinfo_comp["model"] == modeltype:
                    name_modelcomps.append(modelinfo_comp['name'])
        else:
            # TODO: Check this more thoroughly
            name_modelcomps = inittype.split(":")[1].split(";")
        chisqreds = [fits_engine[name_modelcomp]["fits"][-1]["chisqred"]
                     for name_modelcomp in name_modelcomps]
        inittype = name_modelcomps[np.argmin(chisqreds)]
    else:
        inittype = inittype.split(';')
        if len(inittype) > 1:
            # Example:
            # mg8devexppx,mgsersic8:2,nser,"nser=4,1",mg8dev2px;mg8exppx,gaussian:3,T
            # ... means init two mgsersic8 profiles from some combination of the m8dev and mg8exp fits
            modelfits = [{
                'values_param': fits_engine[initname]['fits'][-1]['params_bestall'],
                'paramtree': models[fits_engine[initname]['modeltype']].get_parameters(
                    fixed=True, flatten=False),
                'params': models[fits_engine[initname]['modeltype']].get_parameters(fixed=True),
                'chisqred': fits_engine[initname]['fits'][-1]['chisqred'],
                'modeltype': fits_engine[initname]['modeltype'],
                'name': initname, }
                for initname in inittype
            ]
            init_model_from_model_fits(model, modelfits)
            inittype = None
        else:
            inittype = inittype[0]
            if inittype not in fits_engine:
                # TODO: Fail or fall back here?
                raise RuntimeError("Model={} can't find reference={} "
                                   "to initialize from".format(modeltype, inittype))
    has_fit_init = inittype and 'fits' in fits_engine[inittype]
    logger.info('Model {} using inittype={}; hasinitfit={}'.format(modeltype, inittype, has_fit_init))
    if has_fit_init:
        values_param_init = fits_engine[inittype]["fits"][-1]["params_bestall"]
        # TODO: Find a more elegant method to do this
        inittype_mod = fits_engine[inittype]['modeltype'].split('+')
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
            model = models[fits_engine[inittype]['modeltype']]
            components_new = []
        logger.info(f"Initializing from best model={inittype}"
                    f"{' (MGA to {} GMM)'.format(num_components) if is_mg_to_gauss else ''}")
        # For mgtogauss, first we turn the mgsersic model into a true GMM
        # Then we take the old model and get the parameters that don't depend on components (mostly source
        # centers) and set those as appropriate
        for i in range(1+is_mg_to_gauss):
            params_all = model.get_parameters(fixed=True, modifiers=not is_mg_to_gauss)
            if is_mg_to_gauss:
                logger.info(f"Param values init: {values_param_init}")
                logger.info(f"Param names:       {[x.name for x in params_all]}")
            if len(values_param_init) != len(params_all):
                raise RuntimeError('len(values_param_init)={} != len(params)={}, params_all={}'.format(
                    len(values_param_init), len(params_all), [x.name for x in params_all]))
            for param, value in zip(params_all, values_param_init):
                # The logic here is that we can't start off an MG Sersic at n=0.5 since it's just one Gauss.
                # It's possible that a Gaussian mixture is better than an n<0.5 fit, so start it close to 0.55
                # Note that get_components (called by get_multigaussians) will ignore input values of nser
                # This prevents having a multigaussian model with components having n>0.5 (bad!)
                if is_mg_to_gauss and param.name == 'nser' and value <= 0.55:
                    value = 0.55
                param.set_value(value, transformed=False)
            if is_mg_to_gauss:
                logger.info(f"Param values: "
                            f"{[param.get_value(transformed=False)for param in model.get_parameters()]}")
            # Set the ellipse parameters fixed the first time through
            # The second time through, uh, ...? TODO Remember what happens
            if is_mg_to_gauss and i == 0:
                for idx_src in range(num_sources):
                    components_new.append(get_multigaussians(
                        model.sources[idx_src].get_profiles(bands=bands, engine='libprofit'),
                        params_inherit=params_inherit, params_modify=params_modify,
                        num_components=num_components, source=model_new.sources[idx_src]))
                    components_old = model.sources[idx_src].modelphotometric.components
                    for modeli in [model, model_new]:
                        modeli.sources[idx_src].modelphotometric.components = []
                    values_param_init = [param.get_value(transformed=False)
                                         for param in model.get_parameters(fixed=True)]
                    model.sources[idx_src].modelphotometric.components = components_old
                model = model_new
        if is_mg_to_gauss:
            for idx_src in range(len(components_new)):
                model.sources[idx_src].modelphotometric.components = components_new[idx_src]
    return model, guesstype


# Convenience function to set an exposure object with optional defaults for the sigma (variance) map
# Can be used to nullify an exposure before saving to disk, for example
def set_exposure(model, band, index=0, image=None, error_inverse=None, psf=None, mask=None, meta=None,
                 factor_sigma=1):
    if band not in model.data.exposures:
        model.data.exposures[band] = [mpfobj.Exposure(band=band, image=None)]
    exposure = model.data.exposures[band][index]
    is_empty_image = image is "empty"
    exposure.image = image if not is_empty_image else ImageEmpty(exposure.image.shape)
    if is_empty_image:
        exposure.error_inverse = exposure.image
    else:
        if psf is None and image is not None and error_inverse is None:
            img_sigma = np.sqrt(np.var(image))
            exposure.error_inverse = 1.0/(factor_sigma*img_sigma)
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
    :param params_inherit: List of parameter names modifying the Gaussian (e.g. rscale)
    :param num_components: Array of ints specifying the number of Gaussian components in each physical
        component. Defaults to the number of Gaussians used to represent profiles, i.e. each Gaussian has
        independent ellipse parameters.
    :param source:
    :return: List of new Gaussian components
    """
    bands = set()
    for profile in profiles:
        for band in profile.keys():
            bands.add(band)
    bands = list(bands)
    band_ref = bands[0]
    params = ['mag', 'sigma_x', 'sigma_y', 'rho', 'nser']
    values = {}
    fluxes = {}
    for band in bands:
        # Keep these as lists to make the check against values[band_ref] below easier (no need to call np.all)
        values[band] = {
            'slope' if name == 'nser' else name: [profile[band][name] for profile in profiles]
            for name in params
        }
        # mag_to_flux needs a numpy array
        fluxes[band] = mpfutil.mag_to_flux(np.array(values[band]['mag']))
        # Ensure that only band-independent parameters are compared
        del values[band]['mag']
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
                    if name_param == 'rscale':
                        params_modify_comps[name_param] = [
                            get_param_default(name_param, value=1, is_value_transformed=False, fixed=False)
                            for _ in range(num_components_old)
                        ]
                        for param in params_modify_comps[name_param]:
                            source.modelphotometric.modifiers.append(param)
                elif num_modifiers == num_components_old:
                    params_modify_comps[name_param] = modifiers_of_type
                    if name_param == 'rscale':
                        for modifier in modifiers_of_type:
                            modifier.set_value(1, transformed=False)
                else:
                    raise RuntimeError('Expected to find {} modifiers of type {} but found {}'.format(
                        num_components_old, name_param, num_modifiers
                    ))
        for idx_comp, num_comps_to_add in enumerate(num_components):
            params_inheritees = {
                param.name: param for param in components_gauss[component_init].get_parameters()
                if param.name in params_inherit}
            for param in params_inheritees.values():
                param.inheritors = []
            params_modify_comp = [paramoftype[idx_comp] for paramoftype in params_modify_comps.values()]
            for offset in range(num_comps_to_add):
                comp = components_gauss[component_init+offset]
                for param in comp.get_parameters():
                    if offset > 0 and param.name in params_inherit:
                        # This param will map onto the first component's param
                        param.fixed = True
                        params_inheritees[param.name].inheritors.append(param)
                    if param.name == 'sigma_x' or param.name == 'sigma_y':
                        param.modifiers += params_modify_comp
            component_init += num_comps_to_add

    return components_gauss

