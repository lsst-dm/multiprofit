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

from abc import ABCMeta, abstractmethod
import astropy.visualization as apvis
import copy
import galsim as gs
import logging
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprofit as mpf
import multiprofit.asinhstretchsigned as mpfasinh
from multiprofit.ellipse import Ellipse
import multiprofit.gaussutils as mpfgauss
from multiprofit.limits import Limits
from multiprofit.transforms import Transform
import multiprofit.utils as mpfutil
import numpy as np
import scipy.stats as spstats
import scipy.optimize as spopt
import seaborn as sns
import sys
import time

try:
    import pygmo as pg
except ImportError:
    pg = None

try:
    import pyprofit as pyp
except ImportError:
    pyp = None


# TODO: Make this a class?
def get_gsparams(engineopts=None):
    return engineopts.get("gsparams", gs.GSParams()) if engineopts is not None else gs.GSParams()


draw_method_pixel = {
    "galsim": "no_pixel",
    "libprofit": "rough",
}


# TODO: Implement WCS
# The smart way for this to work would be to specify sky coordinates and angular sizes for objects. This
# way you could give a model some exposures with WCS and it would automagically convert to pixel coordinates.
class Exposure:
    """
        A class to hold an image, sigma map, bad pixel mask and reference to a PSF model/image.
        TODO: Decide whether this should be mutable and implement getters/setters if so; or use afw class
    """
    def get_sigma_inverse(self):
        return self.error_inverse if self.is_error_sigma else np.sqrt(self.error_inverse)

    def get_var_inverse(self):
        return self.error_inverse if not self.is_error_sigma else self.error_inverse**2

    def __init__(self, band, image, mask_inverse=None, error_inverse=None, psf=None, use_mask_inverse=None,
                 meta=None, is_error_sigma=True):
        if psf is not None and not isinstance(psf, PSF):
            raise TypeError("Exposure (band={}) PSF type={:s} not instanceof({:s})".format(
                band, type(psf), type(PSF)))
        self.band = band
        self.image = image
        args_extra = {
            'error_inverse': error_inverse,
            'mask_inverse': mask_inverse,
        }
        for key, value in args_extra.items():
            is_error = key == 'error_inverse'
            if value is not None and not (value.shape == image.shape or (is_error and value.shape == (1,1))):
                raise ValueError('Exposure input {:s} shape={} not same as image.shape={}{}'.format(
                    key, value.shape, image.shape, ' or == (1, 1)'))
        self.mask_inverse = mask_inverse
        self.error_inverse = error_inverse
        self.psf = psf
        self.use_mask_inverse = use_mask_inverse
        self.meta = dict() if meta is None else meta
        self.is_error_sigma = is_error_sigma


class Data:
    """
        A class that holds exposures, mostly for convenience to pass to a model.
        Currently a dict keyed by band with items lists of exposures.
        Also enforces that all exposures are the same size right now, assuming that they're pixel matched.
        This will be relaxed once WCS is implemented.
        TODO: Should this use multiband exposures? Dunno
    """
    def __init__(self, exposures):
        self.exposures = {}
        self.nx = None
        self.ny = None
        for i, exposure in enumerate(exposures):
            if not isinstance(exposure, Exposure):
                raise TypeError(
                    "exposure[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(exposure), type(Exposure))
                )
            else:
                if self.nx is None:
                    self.nx = exposure.image.shape[0]
                    self.ny = exposure.image.shape[1]
                else:
                    if exposure.image.shape[0] != self.nx or exposure.image.shape[1] != self.ny:
                        raise ValueError(
                            "Mismatch in exposure[{:d}] (band={:s}] shape={} vs expected {}".format(
                                i, band, exposure.image.shape, (self.nx, self.ny)))
                band = exposure.band
                if band not in self.exposures:
                    self.exposures[band] = []
                self.exposures[band].append(exposure)


class PSF:
    """
        Either a model or an image. The model can be non-pixellated - i.e. a function describing the PDF of
        photons scattering to a given position from their origin - or pixellated - i.e. a function
        describing the convolution of the PSF with a pixel at a given position.

        Has convenience functions to convert to/from galsim format.
    """

    def get(self, use_model=None):
        if use_model is None:
            use_model = self.use_model
        if use_model:
            return self.model
        return self.get_image()

    def get_image_shape(self):
        if self.image is None:
            return None
        if isinstance(self.image, gs.InterpolatedImage):
            return self.image.image.array.shape
        return self.image.shape

    # TODO: support rescaling of the PSF if it's a galsim interpolatedimage?
    def get_image(self, engine=None, size=None, engineopts=None):
        if engine is None:
            engine = self.engine
        if size is None and self.model is None:
            size = self.get_image_shape()
        if self.image is None or self.get_image_shape() != size:
            if self.model is None:
                raise RuntimeError("Can't get new PSF image without a model")
            exposure = Exposure(self.band, np.zeros(shape=size), None, None)
            data = Data(exposures=[exposure])
            model = Model(sources=[self.model], data=data)
            # TODO: Think about the consequences of making a new astrometry vs resetting the old one
            # It's necessary because the model needs to be centered and it might not be
            astrom = model.sources[0].modelastrometric
            model.sources[0].modelastrometric = AstrometricModel([
                Parameter("cenx", value=size[0]/2.),
                Parameter("ceny", value=size[1]/2.),
            ])
            model.evaluate(keep_images=True, get_likelihood=False)
            self.image = data.exposures[self.band][0].meta["img_model"]
            model.sources[0].modelastrometric = astrom

        # TODO: There's more torturous logic needed here if we're to support changing engines on the fly
        if engine != self.engine:
            if engine == "galsim":
                gsparams = get_gsparams(engineopts)
                self.image = gs.InterpolatedImage(gs.ImageD(self.image, scale=1), gsparams=gsparams)
            else:
                if self.engine == "galsim":
                    self.image = self.image.image.array
            self.engine = engine
        return self.image

    def __init__(self, band, image=None, model=None, engine=None, use_model=False, is_model_pixelated=False):
        self.band = band
        self.model = model
        self.image = image
        self.is_model_pixelated = is_model_pixelated
        if model is None:
            if image is None:
                raise ValueError("PSF must be initialized with either a model or engine but both are none")
            if use_model:
                raise ValueError("PSF use_model==True but no model specified")
            if (engine == "galsim") and (isinstance(image, gs.InterpolatedImage) or
                                         isinstance(image, gs.Image) or (
                                            hasattr(image, "array") and isinstance(image.array, np.ndarray))):
                self.engine = engine
            else:
                if not isinstance(image, np.ndarray):
                    raise ValueError("PSF image must be an ndarray or galsim.Image/galsim.InterpolatedImage"
                                     " if using galsim")
                self.engine = "libprofit"
                self.image = self.get_image(engine=engine)
        else:
            if image is not None:
                raise ValueError("PSF initialized with a model cannot be initialized with an image as well")
            if not isinstance(model, Source):
                raise ValueError("PSF model (type={:s}) not instanceof({:s})".format(
                    type(model), type(Source)))
            self.engine = engine
        self.use_model = use_model


names_params_ellipse = ['sigma_x', 'sigma_y', 'rho']
names_params_ellipse_psf = ['psf_' + x for x in names_params_ellipse]
names_params_gauss = ["cenx", "ceny", "flux"] + names_params_ellipse
num_params_gauss = len(names_params_gauss)
order_params_gauss = {name: idx for idx, name in enumerate(names_params_gauss)}


def gaussian_profiles_to_matrix(profiles):
    es = names_params_ellipse
    ep = names_params_ellipse_psf
    rv = np.array(
        [
            # Look through ellipse_src and ellipse_psf first
            # Otherwise, look for sigma_x, sigma_y, etc
            # There is a compelling reason for doing it this way... probably.
            np.array(
                [p['cenx'], p['ceny'], p['flux']] +
                (
                    list(p["ellipse_src"]()) if "ellipse_src" in p else
                    [p[es[i]] if es[i] in p else 0 for i in range(3)]
                ) + (
                    list(p["ellipse_psf"]()) if "ellipse_psf" in p else
                    [p[ep[i]] if ep[i] in p else 0 for i in range(3)]
                )
            )
            for p in profiles if not p.get("background")
        ]
    )
    for idx in (3, 4, 6, 7):
        rv[:, idx] = mpfgauss.reff_to_sigma(rv[:, idx])
    return rv


def _sidecolorbar(axis, figure, img, vertical=True, do_show_labels=True):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right' if vertical else 'bottom', size='5%', pad=0.05)
    cbar = figure.colorbar(img, cax=cax, ax=axis, orientation='vertical' if vertical else 'horizontal')
    if not do_show_labels:
        (cbar.ax.set_yticklabels if vertical else cbar.ax.set_xticklabels)([])
    return cbar


class Model:
    likefuncs = {
        "normal": "normal",
        "gaussian": "normal",
        "student-t": "t",
        "t": "t",
    }
    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in cls.ENGINES:
            raise ValueError("Unknown Model rendering engine {:s}".format(engine))

    @classmethod
    def _label_figureaxes(
            cls, axes, chisqred, name_model='Model', description_model=None, label_img='Image',
            is_first_model=False, is_last_model=False, do_plot_as_column=False, label_diff_postfix=None,
            label_chi_postfix=None):
        if label_diff_postfix is None:
            label_diff_postfix = ''
        if label_chi_postfix is None:
            label_chi_postfix = ''
        (axes[0].set_title if do_plot_as_column else axes[0].set_ylabel)(label_img)
        # Check if the name_model is informative as it's redundant otherwise
        if name_model != "Model":
            (axes[1].set_title if do_plot_as_column else axes[1].set_ylabel)(name_model)
        if description_model is not None:
            (axes[2].set_title if do_plot_as_column else axes[2].set_ylabel)(description_model)
        label_chisq = r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred)
        (axes[3].set_title if do_plot_as_column else axes[3].set_ylabel)(label_chisq)
        axes[4].set_xlabel(r'$\chi=$(Data-Model)/$\sigma$')
        if do_plot_as_column:
            # TODO: What to do here?
            if not (is_first_model or is_last_model):
                for i in range(1, 5):
                    axes[i].tick_params(labelleft=False)
        else:
            axes[4].yaxis.tick_right()
            for i in range(1, 5):
                if i != 4:
                    axes[i].set_yticklabels([])
                axes[i].yaxis.set_label_position('right')
                if not is_last_model:
                    axes[i].set_xticklabels([])
        if is_first_model:
            labels = ["Data", "Model", "Data-Model" + label_diff_postfix,
                      r'$\chi=$(Data-Model)/$\sigma$' + label_chi_postfix, 'PDF']
            for axis, label in enumerate(labels):
                (axes[axis].set_ylabel if do_plot_as_column or axis == 4 else axes[axis].set_title)(label)


    @classmethod
    def _plot_chi_hist(cls, chi, axis):
        good_chi = chi[np.isfinite(chi)]
        n_bins = 2*np.max([20, np.int(np.round(len(good_chi)/50))])
        sns.distplot(good_chi, bins=n_bins, ax=axis,
                     hist_kws={"log": True, "histtype": "step"},
                     kde_kws={"kernel": "tri", "gridsize": n_bins//2}).set(
            xlim=(-5, 5), ylim=(1e-4, 1)
        )
        x = np.linspace(-5., 5., int(1e4) + 1, endpoint=True)
        axis.plot(x, spstats.norm.pdf(x))


    @classmethod
    def _plot_exposures_color(
            cls, images, img_models, chis, figaxes, bands=None, band_string=None,
            max_img=None, name_model='Model', description_model=None, params_postfix_name_model=None,
            is_first_model=True, is_last_model=True, do_plot_as_column=False, origin_img='bottom',
            weight_per_img=None, asinhscale=16, imgdiffscale=0.05):
        if bands is None:
            bands = []
        if band_string is None:
            band_string = ','.join(bands) if bands is not None else ''
        if params_postfix_name_model is not None:
            if description_model is None:
                description_model = ''
            description_model += Model._formatmodelparams(params_postfix_name_model, bands)
        # TODO: verify lengths
        axes = figaxes[1]
        images_all = [np.copy(images), np.copy(img_models)]
        for images_of_type in images_all:
            for idx, img in enumerate(images_of_type):
                images_of_type[idx] = np.clip(img, 0, np.Inf)
            if weight_per_img is not None:
                for img, weight in zip(images_of_type, weight_per_img):
                    img *= weight
        if max_img is None:
            max_img = np.max([np.max(np.sum(imgs)) for imgs in images_all])
        for i, images_of_type in enumerate(images_all):
            rgb = apvis.make_lupton_rgb(images_of_type[0], images_of_type[1], images_of_type[2],
                                        stretch=max_img/asinhscale, Q=asinhscale)
            axes[i].imshow(rgb, origin=origin_img)
        (axes[0].set_title if do_plot_as_column else axes[0].set_ylabel)(band_string)
        # TODO: Verify if the image limits are working as intended
        imgsdiff = [data-model for data, model in zip(images_all[0], images_all[1])]
        rgb = apvis.make_lupton_rgb(imgsdiff[0], imgsdiff[1], imgsdiff[2], minimum=-max_img*imgdiffscale,
                                    stretch=imgdiffscale*max_img/asinhscale, Q=asinhscale)
        axes[2].imshow(rgb, origin=origin_img)
        # Check if the name_model is informative as it's redundant otherwise
        if name_model != "Model":
            (axes[1].set_title if do_plot_as_column else axes[1].set_ylabel)(name_model)
        if description_model is not None:
            (axes[2].set_title if do_plot_as_column else axes[2].set_ylabel)(description_model)
        # The chi (data-model)/error map clipped at +/- 10 sigma
        rgb = np.clip(np.stack(chis, axis=2)/20 + 0.5, 0, 1)
        axes[3].imshow(rgb, origin=origin_img)
        # Residual histogram compared to a normal distribution
        chi = np.array([])
        for chi_band in chis:
            chi = np.append(chi, chi_band)
        Model._plot_chi_hist(chi, axes[4])
        chisqred = np.sum(chi*chi)/len(chi)
        Model._label_figureaxes(axes, chisqred, name_model=name_model, description_model=description_model,
                                label_img=band_string, is_first_model=is_first_model,
                                is_last_model=is_last_model, do_plot_as_column=do_plot_as_column,
                                label_diff_postfix=' (lim. +/- {}*max(image)'.format(imgdiffscale),
                                label_chi_postfix=r' ($\pm 10\sigma$)')

    # Takes an iterable of tuples of formatstring, param and returns a sensibly formatted summary string
    # TODO: Should probably be tuples of param, formatstring
    # TODO: Add options for the definition of 'sensibly formatted'
    @classmethod
    def _formatmodelparams(cls, modelparamsformat, bands=None):
        if bands is None:
            bands = {}
        description_models = {}
        description_params = {'nser': 'n', 'reff': 'r'}
        for formatstring, param in modelparamsformat:
            is_flux = isinstance(param, FluxParameter)
            is_fluxratio = is_flux and param.is_fluxratio
            if '=' not in formatstring:
                name_param = 'f' if is_fluxratio else (
                    description_params[param.name] if param.name in description_params else param.name)
                value = formatstring
            else:
                name_param, value = formatstring.split('=')
            if name_param not in description_models:
                description_models[name_param] = []
            if not is_fluxratio or not param.fixed and param.band in bands:
                description_models[name_param].append(value.format(param.get_value(transformed=False)))
        # Show the flux ratio if there is only one (otherwise it's painful to work out)
        # Show other parameters if there <= 3 of them otherwise the string is too long
        # TODO: Make a more elegant solution for describing models
        description_model = ';'.join(
            [name_param + ':' + ','.join(values) for name_param, values in description_models.items()
             if len(values) <= (1 + 2 * (name_param != 'f'))])
        return description_model

    def _do_fit_leastsq_prep(self, bands, engine, engineopts):
        """
        Determine if this model can do non-linear least squares fits (currently only for Gaussian mixtures),
        and if so set up the necessary data structures.
        :param bands: List[string] of filters. All exposures with these filters will be prepared.
        :param engine: A valid rendering engine
        :param engineopts: dict; engine options.
        :return: tuple[grad_param_maps, sersic_param_maps, can_do_fit_leastsq]:
            grad_param_maps: A 2D ndarray associating each Gaussian parameter with the index of the
                free fit parameter controlling its value (if any).
            sersic_param_maps: As grad_param_maps but specifically for Sersic index parameters.
            can_do_fit_leastsq: If False, then both param_maps are set to None.
        """
        return_none = (None, None, False, None, None, None)
        if engine == 'galsim' and not (
                engineopts is not None and engineopts.get("drawmethod") == draw_method_pixel["galsim"]):
            return return_none
        params = self.get_parameters(fixed=True)
        fixed = np.zeros(len(params))
        n_params_free = 0
        num_sersic_free = 0
        idx_params = {param: 0 for param in params}
        n_priors = 0 if not self.priors else np.sum([len(p) for p in self.priors])
        for idx, param in enumerate(params):
            fixed[idx] = param.fixed
            if not param.fixed:
                idx_params[param] = n_params_free + 1
                n_params_free += 1
                if param.name == "nser":
                    num_sersic_free += 1
        profiles = self.get_profiles(bands=bands, engine="libprofit")
        grad_param_maps = {
            band: (
                np.zeros((len(profiles), len(order_params_gauss)), dtype=np.uint),
                [],
            )
            for band in bands
        }
        sersic_param_maps = {
            band: None if not num_sersic_free > 0 else {
                idx: idx_params[p[band]['param_nser']]
                for idx, p in enumerate(profiles) if 'param_nser' in p[band]
            }
            for band in bands
        }
        datasize = 0
        order_flux = order_params_gauss['flux']
        # Sets aren't ordered. AAAAAAH.
        param_flux_added = {}
        for band in bands:
            for exposure in self.data.exposures[band]:
                exposure.meta['index_jacobian_start'] = datasize
                datasize += exposure.image.size
                exposure.meta['index_jacobian_end'] = datasize

            gpmb_indices, gpmb_params = grad_param_maps[band]

            for idx, profile in enumerate(profiles):
                gpmb_indices_row = gpmb_indices[idx, :]
                profile_band = profile[band]
                if "profile" not in profile_band or not profile_band["can_do_fit_leastsq"]:
                    return return_none
                param_flux = profile_band["param_flux"]
                params_profile = profile_band["params"]
                params_profile["flux"] = param_flux
                params_to_append = []
                param_flux_ratio = profile_band.get("param_flux_ratio")
                param_flux_ratio_isnt_none = param_flux_ratio is not None
                if param_flux_ratio_isnt_none and param_flux not in param_flux_added:
                    param_flux_added[param_flux] = True
                    if not param_flux_ratio.fixed:
                        n_params_free += 1
                        idx_params[param_flux] = n_params_free
                is_bg = profile.get('background')
                # TODO: Make this {'flux': 0} if is_bg when this works
                orders_profile = {} if is_bg else order_params_gauss
                for name_param, idx_param in orders_profile.items():
                    param = params_profile[name_param]
                    if name_param == 'flux' and param_flux_ratio_isnt_none and not param_flux_ratio.fixed:
                        gpmb_indices_row[order_flux] = idx_params[param_flux_ratio]
                    else:
                        gpmb_indices_row[idx_param] = idx_params[param]
                    params_to_append.append(param)
                gpmb_params.append(params_to_append)

        datasize += n_priors
        n_params_jac = n_params_free + 1
        jacobian_prior, residuals_prior = None, None
        if self.jacobian is None or self.jacobian.shape != (datasize, n_params_jac):
            self.jacobian = np.zeros((datasize + n_priors, n_params_jac))
            jacobian_prior = self.jacobian[datasize:, ].view()
            jacobian_prior.shape = (n_priors, n_params_jac)
        if self.residuals is None or self.residuals.shape != (datasize,):
            self.residuals = np.zeros(datasize + n_priors)
            residuals_prior = self.residuals[datasize:].view()
        for band in bands:
            for exposure in self.data.exposures[band]:
                start, end = (exposure.meta[f'index_jacobian_{pos}'] for pos in ('start', 'end'))
                exposure.meta['jacobian_img'] = self.jacobian[start:end, ].view()
                exposure.meta['jacobian_img'].shape = (
                    exposure.image.shape[0], exposure.image.shape[1], n_params_jac)
                exposure.meta['residual_img'] = self.residuals[start:end].view()
                exposure.meta['residual_img'].shape = exposure.image.shape
        for src in self.sources:
            for comp in src.modelphotometric.components:
                if hasattr(comp, 'do_return_all_profiles'):
                    comp.do_return_all_profiles = True
        return grad_param_maps, sersic_param_maps, True, jacobian_prior, residuals_prior, idx_params

    def do_fit_leastsq_cleanup(self, bands):
        self.grad_param_maps = None
        self.param_maps = {"sersic": None}
        for band in bands:
            for exposure in self.data.exposures[band]:
                for item in ["jacobian", "residual"]:
                    if item in exposure.meta:
                        del exposure.meta[item]
        self.jacobian = None
        self.residuals = None
        self.jacobian_prior = None
        self.residuals_prior = None

    def __validate_jacobian_param(
            self, param, idx_param, exposure, image, jacobian, scale, engine, engineopts, meta_model,
            dx_ratio=1e-6, do_raise=True, do_plot_if_failed=False, dx_min=1e-8, dx_max=10, dlldxs=None):
        """

        :param param: mpfObj.Parameter; param to validate jacobian for.
        :param idx_param: int; index of parameter in jacobian matrix.
        :param exposure: mpfObj.Exposure; exposure to validate jacobian on.
        :param image: 2darray; the model image for the current parameters.
        :param jacobian: 3darray; the jacobian matrix.
        :param scale: float; the spatial pixel scale. Unimplemented.
        :param engine: A valid rendering engine
        :param engineopts: dict; engine options.
        :param meta_model: dict; model metadata as returned by _get_image_model_exposure_setup.
        :param dx_ratio: float; the desired dx/x for finite differencing.
        :param do_raise: bool; raise an exception on failure?
        :param do_plot_if_failed: bool; plot on failure?
        :param dx_min: float; minimum absolute value for dx.
        :param dx_max: float; maximum absolute value for dx.
        :param dlldxs: tuple(float, list-like); the original log-likelihood, and a list to append dloglike/dx.
            Forces an additional evaluation of the model loglike(x+dx).
        :return: True if successful/False otherwise.
        """
        value = param.get_value(transformed=True)
        dx = np.clip(value*dx_ratio, dx_min, dx_max)
        # Allow for roundoff errors - it need not be such a strict limit
        dx_min_allowed = 0.99*dx_min
        for sign in [1, -1]:
            value_new = value + dx*sign
            param.set_value(value_new, transformed=True)
            dx_actual = param.get_value(transformed=True) - value
            if sign*dx_actual >= sign*dx_min_allowed:
                break
        if sign*dx_actual < dx_min_allowed:
            raise RuntimeError(f"Failed to get param {param} |dx={dx_actual}| within "
                               f"({dx_min_allowed}, {dx_max}) for validating jacobian (sign={sign})")
        dx = dx_actual
        # This might happen if the parameter is near a limit
        dparam = param.get_transform_derivative()
        profiles_new, meta_model_new, engine_new, engineopts_new, _ = \
            self._get_image_model_exposure_setup(
                exposure, engine=engine, engineopts=engineopts)
        errmsg = ""
        for name_item, (item_new, item_old) in {
            'meta_model': (meta_model_new, meta_model),
            'engine': (engine_new, engine),
            'engineopts': (engineopts_new, engineopts),
        }.items():
            if name_item == "meta_model":
                if meta_model.keys() != meta_model_new.keys():
                    errmsg += f"Got different meta_model keys validating jacobian: " \
                        f"new={meta_model_new.keys()} vs old={meta_model.keys()}"
                else:
                    meta_model_diff = {}
                    for key in meta_model:
                        if key != "profiles" and meta_model[key] != meta_model_new[key]:
                            meta_model_diff[key] = (meta_model[key], meta_model_new[key])
                    if meta_model_diff:
                        errmsg += f"Got different meta_model entries validating jacobian: " \
                            f"{meta_model_diff}"
            elif item_new != item_old:
                errmsg += f"Got different {name_item} validating jacobian: " \
                    f"new={item_new} vs old={item_old}"
        if errmsg != "":
            raise RuntimeError(errmsg)
        image_new, _, _, likelihood_new = self.get_image_model_exposure(
            exposure, profiles=profiles_new, meta_model=meta_model_new, engine=engine,
            engineopts=engineopts, do_draw_image=True, scale=scale, get_likelihood=True,
            verify_derivative=True)
        # (data - model_new)/sigma - (data - model)/sigma = -(model_new - model)/sigma
        grad_findif = -exposure.get_sigma_inverse()*(image_new - image)/dx
        grad_jac = jacobian[:, :, idx_param+1]
        passed = np.isclose(grad_findif, grad_jac, rtol=1e-3, atol=1e-4)
        if np.all(passed):
            if dlldxs is not None:
                dlldxs[1].append((likelihood_new - dlldxs[0])/dx)
            param.set_value(value, transformed=True)
        else:
            param.set_value(value, transformed=True)
            if do_plot_if_failed:
                ncols = 3
                plots = (
                    (np.log10(image), f'Model(x)'),
                    (np.log10(image_new), f'Model(x+dx)'),
                    (passed+0., 'Passed'),
                    (grad_findif, 'Fin. Dif.'),
                    (grad_jac, 'Jacobian'),
                    (grad_jac/grad_findif, 'Jac./Fin.Dif. ratio'),
                )
                fig, axes = plt.subplots(ncols=ncols, nrows=int(np.ceil(len(plots)/ncols)))
                for idx_ax, (img_plot, title) in enumerate(plots):
                    ax_plot = axes[idx_ax // ncols][idx_ax % ncols]
                    ax_plot.imshow(img_plot)
                    ax_plot.set_title(title)
                plt.suptitle(f'Failed jac. {param.name}={value:.3e} dx={dx:.3e}')
                plt.show()
            if do_raise:
                raise RuntimeError(
                    f'jacobian verification failed on param {param} value_new={value_new:.6e} and '
                    f'value_true={param.get_value(transformed=False):.6e} with dx={dx} idx={idx_param+1} '
                    f'dparam={dparam}'
                )
            return False
        return True

    def estimate_uncertainties(self, params_best, eps=1e-8):
        if not self.can_do_fit_leastsq:
            return None
        else:
            return None
            n_params = len(params_best)
            grad = np.zeros(n_params)
            hessian = np.zeros((n_params, n_params))

    @staticmethod
    def _compute_loglike_exposure(likelihood, exposure, is_likelihood_log=True):
        like_const = exposure.meta.get("like_const", 0. if is_likelihood_log else 1.)
        if is_likelihood_log:
            likelihood = likelihood + like_const
        else:
            likelihood = np.exp(likelihood)*like_const
        return likelihood

    @staticmethod
    def _get_background(profiles, band):
        has_bg = 0
        flux_bg = 0.
        for p in profiles:
            if p[band].get('background'):
                has_bg += 1
                flux_bg += p[band]['flux']
        if has_bg > 1:
            raise RuntimeError("Can't handle multiple background components")
        return (has_bg == 1), np.array([flux_bg], dtype=np.float64) if has_bg else None

    def evaluate(self, param_values=None, data=None, bands=None, engine=None, engineopts=None,
                 params_transformed=True, get_likelihood=True, is_likelihood_log=True, keep_likelihood=False,
                 keep_images=False, keep_models=False, plot=False,
                 do_plot_multi=False, figure=None, axes=None, row_figure=None, name_model="Model",
                 description_model=None, params_postfix_name_model=None, do_draw_image=False, scale=1,
                 clock=False, do_plot_as_column=False,
                 img_plot_maxs=None, img_multi_plot_max=None, weights_band=None,
                 do_fit_linear_prep=False, do_fit_leastsq_prep=False, do_fit_nonlinear_prep=False,
                 do_jacobian=False, do_verify_jacobian=False,
                 do_compare_likelihoods=False, grad_loglike=None,
                 debug=False):
        """
        Evaluate a model, plot and/or benchmark, and optionally return the likelihood and derived images.

        :param param_values: ndarray; optional new values for all model free parameters in the order returned
            by get_parameters. Defaults to use existing values.
        :param data: multiprofit.data; optional override to evaluate model on a different set of data. May
            not work as expected unless the data has exposures of the same size and in the same order as
            the default (self.data).
        :param bands: iterable; bandpass filters to use for evaluating the model. Defaults to use all bands
            in data.
        :param engine: A valid rendering engine
        :param engineopts: dict; engine options.
        :param params_transformed: bool; are the values in params already transformed?
        :param get_likelihood: bool; return the model likelihood?
        :param is_likelihood_log: bool; return the natural logarithm of the likelihood?
        :param keep_likelihood: bool; store each exposure's likelihood in its metadata?
        :param keep_images: bool; store each exposure's model image in its metadata?
        :param keep_models: bool; store each exposure's model specification in its metadata?
        :param plot: bool; plot the model and residuals for each exposure?
        :param do_plot_multi: bool; plot colour images if fitting multiband? Ignored otherwise.
        :param figure: matplotlib figure handle. If data has multiple bands, must be a dict keyed by band.
        :param axes: iterable of matplotlib axis handles. If data has multiple bands, must be a dict keyed by
            band.
        :param row_figure: non-negative integer; the index of the axis handle to plot this model on.
        :param name_model: string; a name for the model to appear in plots.
        :param description_model: string; a description of the model to appear in plots.
        :param params_postfix_name_model: iterable of multiprofit.Parameter; parameters whose values should
            appear in the plot (if possible).
        :param do_draw_image: bool; draw (evaluate) the model image for each exposure? May be overruled if
            get_likelihood or plot is True.
        :param scale: float; Spatial scale factor for drawing images. User beware; this may not work
            properly and should be replaced by exposure WCS soon.
        :param clock: bool; benchmark model evaluation?
        :param do_plot_as_column: bool; are the plots arranged vertically?
        :param img_plot_maxs: dict; key band: value maximum for image/model plots.
        :param img_multi_plot_max: float; maximum for multiband image/model plots.
        :param weights_band: dict; key band: weight for scaling each band in multiband plots.
        :param do_fit_linear_prep: bool; do prep work to fit a linear model?
        :param do_fit_leastsq_prep: bool; do prep work for a least squares fit?
        :param do_fit_nonlinear_prep: bool; do prep work to fit a non-linear model?
            Ignored if do_leastsq_prep and leastsq prep succeeds.
        :param do_jacobian: bool; evaluate the Jacobian?
        :param do_verify_jacobian: bool; verify the Jacobian if do_fit_leastsq_prep by comparing to
            finite differencing
        :param do_compare_likelihoods: bool; compare likelihoods from C++ vs Python (if both exist)
        :param grad_loglike: ndarray[float]; values for the gradient of the loglikehood vs fit params
        :param debug: bool; whether to do debug tests/checks
        :return: likelihood: float; the (log) likelihood
            param_values: ndarray of floats; parameter values
            chis: list of residual images for each fit exposure, where chi = (data-model)/sigma
            times: Dict of floats, with time in
        """
        if grad_loglike is not None:
            do_jacobian = False
            do_verify_jacobian = False
        times = {}
        time_now = time.time() if clock else None
        if engine is None:
            engine = self.engine
        Model._checkengine(engine)
        if engineopts is None:
            engineopts = self.engineopts
        if data is None:
            data = self.data

        params = self.get_parameters(free=True, fixed=False)
        if param_values is None:
            param_values = [param.value for param in params]
        else:
            num_params = len(param_values)
            if not len(params) == num_params:
                raise RuntimeError("Length of parameters[{:d}] != # of free param_values=[{:d}]".format(
                    len(params), num_params
                ))
            for paramobj, value_param in zip(params, param_values):
                paramobj.set_value(value_param, transformed=params_transformed)

        if get_likelihood:
            likelihood = 1.
            if is_likelihood_log:
                likelihood = 0.
        else:
            likelihood = None

        if bands is None:
            bands = data.exposures.keys()
        if plot:
            num_plots = 0
            is_single_model = figure is None or axes is None or row_figure is None
            if is_single_model:
                for band in bands:
                    num_plots += len(data.exposures[band])
                num_rows = num_plots + do_plot_multi
                num_rows_band = {band: num_rows for band in bands}
                figure, axes = plt.subplots(nrows=num_rows, ncols=5, figsize=(10, 2*num_rows), dpi=100)
                if num_plots == 1:
                    axes.shape = (1, 5)
                row_figure = 0
            else:
                num_bands = len(bands)
                num_rows_band = {band: (axes[band] if num_bands > 1 else axes).shape[0] for band in bands}
            figaxes = (figure, axes)
            is_first_model = row_figure is None or row_figure == 0
            is_last_model_band = {
                band: row_figure is None or axes is None or (row_figure + 1) == num_rows_band[band]
                for band in bands
            }
        else:
            figaxes = None
            is_first_model = None
            is_last_model_band = None
        if plot and (figaxes is None or any(x is None for x in figaxes)):
            raise RuntimeError("Plot is true but there are None figaxes: {}".format(figaxes))
        chis = []
        imgs_clip = []
        models_clip = []
        if do_plot_multi:
            weight_per_img = []
        if do_fit_leastsq_prep:
            self.grad_param_maps, self.param_maps['sersic'], self.can_do_fit_leastsq,\
                self.jacobian_prior, self.residuals_prior, self.params_prior_jacobian = \
                self._do_fit_leastsq_prep(bands, engine, engineopts)
            if not self.can_do_fit_leastsq:
                do_fit_leastsq_prep = False
        if not do_fit_leastsq_prep and do_fit_nonlinear_prep and self.priors:
            self.residuals_prior = np.zeros(np.sum([len(p) for p in self.priors]))
        if clock:
            times["setup"] = time.time() - time_now
            time_now = time.time()

        for band in bands:
            is_last_model = is_last_model_band[band] if is_last_model_band is not None else None
            # TODO: Check band
            for idx_exposure, exposure in enumerate(data.exposures[band]):
                profiles, meta_model, engine, engineopts, times = self._get_image_model_exposure_setup(
                    exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
                if self.can_do_fit_leastsq and not (
                        'is_all_gaussian' in meta_model and meta_model['is_all_gaussian']):
                    raise RuntimeError('Model should be able to do least-squares fit but is not gauss mix.')

                # Get the model image and/or likelihood for this exposure only
                image, model, times_model, likelihood_exposure = self.get_image_model_exposure(
                    exposure, profiles=profiles, meta_model=meta_model, engine=engine,
                    engineopts=engineopts, do_draw_image=do_draw_image or plot or do_compare_likelihoods,
                    scale=scale, clock=clock, times=times, do_fit_linear_prep=do_fit_linear_prep,
                    do_fit_leastsq_prep=do_fit_leastsq_prep, do_fit_nonlinear_prep=do_fit_nonlinear_prep,
                    do_jacobian=do_jacobian, get_likelihood=get_likelihood,
                    is_likelihood_log=is_likelihood_log, grad_loglike=grad_loglike, debug=debug)

                if clock:
                    name_exposure = '_'.join([band, str(idx_exposure)])
                    times['_'.join([name_exposure, 'modeltotal'])] = time.time() - time_now
                    for name_time, value_time in times_model.items():
                        times['_'.join([name_exposure, name_time])] = value_time
                    time_now = time.time()
                if keep_images:
                    exposure.meta["img_model"] = np.array(image)
                if keep_models:
                    exposure.meta["model"] = model
                if plot:
                    if do_plot_multi:
                        figaxes = (figure[band], axes[band][row_figure])
                    else:
                        figaxes = (figure, axes[row_figure])

                gaussians = None
                has_bg, background = None, None

                if do_fit_leastsq_prep and do_verify_jacobian and 'jacobian_img' in exposure.meta:
                    jacobian = exposure.meta['jacobian_img']

                    has_bg, background = self._get_background(profiles, band)
                    gaussians = gaussian_profiles_to_matrix([p[band] for p in profiles])
                    grad = np.zeros((exposure.image.shape[0], exposure.image.shape[1],
                                     num_params_gauss*(len(profiles) - has_bg) + has_bg))
                    sigma_inv = exposure.get_sigma_inverse()

                    # The point of this is to fill in the grad array
                    likelihood_new = mpfgauss.loglike_gaussians_pixel(
                        data=exposure.image, sigma_inv=sigma_inv, gaussians=gaussians,
                        to_add=False, grad=grad, background=background)

                    if likelihood_exposure is None:
                        likelihood_exposure = self._compute_loglike_exposure(
                            likelihood_new, exposure, is_likelihood_log=is_likelihood_log)

                    dlldxs = []
                    failed = []

                    for idx_param, param in enumerate(params):
                        passed = False
                        for dx_ratio, do_raise in [(1e-7, False), (1e-5, True)]:
                            if self.__validate_jacobian_param(
                                param, idx_param, exposure, image, jacobian, scale, engine, engineopts,
                                meta_model, dx_ratio=dx_ratio, do_raise=False,
                                do_plot_if_failed=plot and do_raise, dlldxs=(likelihood_new, dlldxs)
                            ):
                                passed = True
                                break
                        if not passed:
                            failed.append(str(param))
                    if failed:
                        self.logger.warning(f'Failed jacobian validation: {failed}')
                    self.logger.info(f'DLLs: [{", ".join(f"{dlldx:.3e}" for dlldx in dlldxs)}]')

                missing_likelihood = (get_likelihood or do_compare_likelihoods) and (
                    likelihood_exposure is None)
                if (plot and image is None) or missing_likelihood:
                    if has_bg is None:
                        has_bg, background = self._get_background(profiles, band)
                    likelihood_new = mpfgauss.loglike_gaussians_pixel(
                        data=exposure.image, sigma_inv=exposure.get_sigma_inverse(),
                        gaussians=gaussians if gaussians is not None else gaussian_profiles_to_matrix(
                            [p[band] for p in profiles]),
                        background=background, output=image)
                    likelihood_new = self._compute_loglike_exposure(
                        likelihood_new, exposure, is_likelihood_log=is_likelihood_log)
                    if not np.isclose(likelihood_new, likelihood_exposure):
                        raise RuntimeError(
                            f'get_exposure_likelihood={likelihood_exposure:.6e} !close to '
                            f'loglike_gaussians_pixel={likelihood_new:.6e} (diff='
                            f'{likelihood_exposure - likelihood_new:.6e})'
                        )
                    if missing_likelihood:
                        likelihood_exposure = likelihood_new

                if plot or do_compare_likelihoods:
                    # Also generate chi and clipped images for plotting
                    likelihood_validate, chi, img_clip, model_clip = \
                        self.get_exposure_likelihood(
                            exposure, image, log=is_likelihood_log, figaxes=figaxes,
                            name_model=name_model, description_model=description_model,
                            params_postfix_name_model=params_postfix_name_model,
                            do_plot_as_column=do_plot_as_column, is_first_model=is_first_model,
                            is_last_model=is_last_model, max_img=img_plot_maxs[band] if (
                                img_plot_maxs is not None and band in img_plot_maxs) else None
                        )
                elif do_draw_image and exposure.error_inverse is not None:
                    chi = (image - exposure.image)*exposure.get_sigma_inverse()
                else:
                    chi = None

                if do_compare_likelihoods:
                    if not np.isclose(likelihood_validate, likelihood_exposure):
                        raise RuntimeError(
                            f'get_exposure_likelihood={likelihood_exposure:.6e} !close to '
                            f'loglike_gaussians_pixel={likelihood_validate:.6e} (diff='
                            f'{likelihood_exposure - likelihood_validate:.6e})'
                        )

                if get_likelihood or plot:
                    if clock:
                        times['_'.join([name_exposure, 'like'])] = time.time() - time_now
                        time_now = time.time()
                    if keep_likelihood:
                        exposure.meta["likelihood"] = likelihood_exposure
                        exposure.meta["is_likelihood_log"] = is_likelihood_log
                    if is_likelihood_log:
                        likelihood += likelihood_exposure
                    else:
                        likelihood *= likelihood_exposure
                    chis.append(chi)
                    if plot:
                        if not do_plot_multi:
                            row_figure += 1
                        if do_plot_multi:
                            imgs_clip.append(img_clip)
                            models_clip.append(model_clip)
                            weight_per_img.append(
                                weights_band[band] if (weights_band is not None and band in weights_band)
                                else 1)
        # Color images! whooo
        if plot:
            if do_plot_multi:
                if is_single_model:
                    Model._plot_exposures_color(
                        imgs_clip, models_clip, chis, (figure, axes[row_figure]),
                        bands=bands, name_model=name_model, description_model=description_model,
                        params_postfix_name_model=params_postfix_name_model, is_first_model=is_first_model,
                        is_last_model=is_last_model, do_plot_as_column=do_plot_as_column,
                        max_img=img_multi_plot_max, weight_per_img=weight_per_img)
                else:
                    Model._plot_exposures_color(
                        imgs_clip, models_clip, chis, (figure['multi'], axes['multi'][row_figure]),
                        bands=bands, name_model=name_model, description_model=description_model,
                        params_postfix_name_model=params_postfix_name_model, is_first_model=is_first_model,
                        is_last_model=is_last_model, do_plot_as_column=do_plot_as_column,
                        max_img=img_multi_plot_max, weight_per_img=weight_per_img)
                row_figure += 1
        if clock:
            self.logger.info(','.join([f'{name}={value:.2e}' for name, value in times.items()]))
        return likelihood, param_values, chis, times

    def get_exposure_likelihood(
            self, exposure, img_model, log=True, likefunc=None, figaxes=None, max_img=None, min_img=None,
            name_model="Model", description_model=None, params_postfix_name_model=None, is_first_model=True,
            is_last_model=True, do_plot_as_column=False, origin_img='bottom', norm_img_diff=None,
            norm_chi=None
    ):
        if likefunc is None:
            likefunc = self.likefunc
        has_mask = exposure.mask_inverse is not None
        sigma_inv = exposure.get_sigma_inverse()
        if figaxes is not None:
            if params_postfix_name_model is not None:
                if description_model is None:
                    description_model = ""
                description_model += Model._formatmodelparams(params_postfix_name_model, {exposure.band})

            axes = figaxes[1]
            if has_mask:
                xlist = np.arange(img_model.shape[1])
                ylist = np.arange(img_model.shape[0])
                x, y = np.meshgrid(xlist, ylist)
                z = exposure.mask_inverse
            if max_img is None:
                if has_mask:
                    max_img = np.max([np.max(exposure.image[exposure.mask_inverse]),
                                     np.max(img_model[exposure.mask_inverse])])
                else:
                    max_img = np.max([np.max(exposure.image), np.max(img_model)])
            if min_img is None:
                min_img = np.min([0, np.min(exposure.image), np.min(img_model)])
            # The original image and model image
            norm = apvis.ImageNormalize(vmin=min_img, vmax=max_img, stretch=apvis.AsinhStretch(1e-2))
            do_show_labels = is_last_model
            for i, img in enumerate([exposure.image, img_model]):
                img_handle = axes[i].imshow(img, cmap='gray', origin=origin_img, norm=norm)
                _sidecolorbar(axes[i], figaxes[0], img_handle, vertical=do_plot_as_column,
                              do_show_labels=do_show_labels)
                if has_mask:
                    axes[i].contour(x, y, z, colors='green')
            # The difference map
            chi = exposure.image - img_model
            if norm_img_diff is None:
                diff_abs_max = np.max(exposure.image)/10
                norm_img_diff = apvis.ImageNormalize(vmin=-diff_abs_max, vmax=diff_abs_max,
                                                     stretch=mpfasinh.AsinhStretchSigned(0.1))
            imgdiff = axes[2].imshow(chi, cmap='gray', origin=origin_img, norm=norm_img_diff)
            _sidecolorbar(
                axes[2], figaxes[0], imgdiff, vertical=do_plot_as_column, do_show_labels=do_show_labels)
            if has_mask:
                axes[2].contour(x, y, z, colors='green')
            # The chi (data-model)/error map
            chi *= sigma_inv
            chisqred = mpfutil.get_chisqred([chi[exposure.mask_inverse] if has_mask else chi])
            if norm_chi is None:
                chi_abs_max = np.max(np.abs(chi))
                if chi_abs_max < 1:
                    chi_abs_max = np.ceil(chi_abs_max*100)/100
                else:
                    chi_abs_max = 10
                norm_chi = apvis.ImageNormalize(vmin=-chi_abs_max, vmax=chi_abs_max,
                                                stretch=mpfasinh.AsinhStretchSigned(0.1))
            img_chi = axes[3].imshow(chi, cmap='RdYlBu_r', origin=origin_img, norm=norm_chi)
            _sidecolorbar(
                axes[3], figaxes[0], img_chi, vertical=do_plot_as_column, do_show_labels=do_show_labels)
            if has_mask:
                axes[3].contour(x, y, z, colors="green")
                chi = chi[exposure.mask_inverse]
            Model._plot_chi_hist(chi, axes[4])
            Model._label_figureaxes(axes, chisqred, name_model=name_model, description_model=description_model,
                                    label_img='Band={}'.format(exposure.band), is_first_model=is_first_model,
                                    is_last_model=is_last_model, do_plot_as_column=do_plot_as_column)
        else:
            if has_mask:
                chi = (exposure.image[exposure.mask_inverse] - img_model[exposure.mask_inverse]) * \
                    sigma_inv[exposure.mask_inverse]
            else:
                chi = (exposure.image - img_model)*sigma_inv

        if likefunc == "t":
            variance = chi.var()
            dof = 2. * variance / (variance - 1.)
            dof = max(min(dof, float('inf')), 0)
            likelihood = np.sum(spstats.t.logpdf(chi, dof))
        elif likefunc == "normal":
            likelihood = np.sum(spstats.norm.logpdf(chi)
                                + np.log(sigma_inv[exposure.mask_inverse] if has_mask else sigma_inv))
        else:
            raise ValueError("Unknown likelihood function {:s}".format(self.likefunc))

        if not log:
            likelihood = np.exp(likelihood)

        return likelihood, chi, exposure.image, img_model

    def _get_image_model_exposure_setup(
            self, exposure, engine=None, engineopts=None, clock=False, times=None):
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        Model._checkengine(engine)
        if engine == "galsim":
            gsparams = get_gsparams(engineopts)
        band = exposure.band
        if clock:
            time_now = time.time()
            if times is None:
                times = {}
        else:
            times = None
        has_psf = exposure.psf is not None
        is_psf_pixelated = has_psf and exposure.psf.model is not None and exposure.psf.is_model_pixelated
        is_profiles_gaussian = [comp.is_gaussian_mixture()
                                for comps in [src.modelphotometric.components for src in self.sources]
                                for comp in comps]
        is_all_gaussian = not has_psf or (
            exposure.psf.model is not None and
            all([comp.is_gaussian_mixture() for comp in exposure.psf.model.modelphotometric.components]))
        is_any_gaussian = is_all_gaussian and any(is_profiles_gaussian)
        is_all_gaussian = is_all_gaussian and all(is_profiles_gaussian)
        # Use fast, efficient Gaussian evaluators only if everything's a Gaussian mixture model
        use_fast_gauss = (engineopts is None or (engineopts.get("use_fast_gauss") is not False)) and (
            is_all_gaussian and (
                is_psf_pixelated or (
                    not has_psf and
                    # Disabling these for fitting with a PSF because we don't want discontinuous likelihoods
                    # in fitting because the drawing method has changed, but it's fine for PSF fitting
                    engineopts is not None and (
                        (engine == 'galsim' or engine == 'libprofit') and
                        engineopts.get("drawmethod") == draw_method_pixel[engine]
                    )
                )
            )
        )
        is_all_fast_gauss = False
        if use_fast_gauss:
            if engine != 'libprofit':
                engine = 'libprofit'
            engineopts = {"drawmethod": draw_method_pixel[engine]}
            if is_all_gaussian:
                is_all_fast_gauss = True
                engineopts["get_profile_skip_covar"] = True
        profiles = self.get_profiles(bands=[band], engine=engine, engineopts=engineopts)
        if clock:
            times['get_profiles'] = time.time() - time_now
            time_now = time.time()

        if profiles:
            # Do analytic convolution of any Gaussian model with the Gaussian PSF
            # This turns each resolved profile into N_PSF_components Gaussians
            if is_any_gaussian and (engineopts is None or
                                    (engineopts.get("do_analytic_convolution") is not False)):
                is_libprofit = engine == "libprofit"
                if clock:
                    times['model_all_gauss_eig'] = 0
                names_params = names_params_ellipse
                ellipse_srcs = [
                    (
                        Ellipse(*[profile[band][var] for var in names_params])
                        if profile[band].get("nser") == 0.5
                        else None,
                        profile[band]
                    )
                    for profile in (profiles if is_libprofit else self.get_profiles(
                        bands=[band], engine="libprofit"))
                ]
                if not has_psf:
                    if is_libprofit:
                        for idx_src, (ellipse_src, profile) in enumerate(ellipse_srcs):
                            profile["ellipse_src"] = ellipse_src
                            profile["idx_src"] = idx_src
                else:
                    profiles_new = []
                    profiles_psf = exposure.psf.model.get_profiles(bands=[band], engine="libprofit")
                    fluxfracs = []
                    ellipses_psf = []
                    for idx_psf, profile_psf in enumerate(profiles_psf):
                        fluxfracs.append(profile_psf[band]["flux"])
                        ellipses_psf.append(Ellipse(*[profile_psf[band][var] for var in names_params]))
                    for idx_src, (ellipse_src, profile_src) in enumerate(ellipse_srcs):
                        for idx_psf, (fluxfrac, ellipse_psf) in enumerate(zip(fluxfracs, ellipses_psf)):
                            if ellipse_src is not None:
                                profile = copy.copy(profile_src)
                                if is_all_fast_gauss or is_libprofit:
                                    # Needed because each PSF component will loop over the same profile object
                                    profile["flux"] *= fluxfrac
                                    profile["idx_src"] = idx_src
                                    profile["ellipse_src"] = ellipse_src
                                    profile["ellipse_psf"] = ellipse_psf
                                if not is_all_fast_gauss:
                                    convolved = ellipse_src.convolve(ellipse_psf, new=True)
                                    reff, axrat, ang = mpfgauss.covar_to_ellipse(convolved)
                                    if is_libprofit:
                                        profile["mag"] = mpfutil.flux_to_mag(fluxfrac*profile["flux"])
                                        profile["re"] = reff
                                        profile["axrat"] = axrat
                                        profile["ang"] = ang
                                    else:
                                        profile["profile_gs"] = gs.Gaussian(
                                            flux=profile["flux"]*fluxfrac,
                                            fwhm=2*reff*sqrt(axrat), gsparams=gsparams)
                                        profile["shear"] = gs.Shear(q=axrat, beta=ang*gs.degrees)
                                        profile["offset"] = gs.PositionD(profile["cenx"], profile["ceny"])
                                profile["pointsource"] = True
                                profile["resolved"] = True
                                profile["fluxfrac_psf"] = fluxfrac
                                profile = {band: profile}
                            else:
                                profile = profiles[idx_src]
                            # TODO: Remember what the point of this check is
                            if ellipse_src is not None or idx_psf == 0:
                                profiles_new.append(profile)
                    profiles = profiles_new
                if clock:
                    times['model_all_gauss'] = time.time() - time_now
                    time_now = time.time()

        if clock:
            times['model_setup'] = time.time() - time_now

        meta_model = {
            'is_all_gaussian': is_all_gaussian,
            'has_psf': has_psf,
            'profiles': profiles,
            'is_psf_pixelated': is_psf_pixelated,
            'use_fast_gauss': use_fast_gauss,
        }
        return profiles, meta_model, engine, engineopts, times

    def get_image_model_exposure(
            self, exposure, profiles=None, meta_model=None, engine=None, engineopts=None, do_draw_image=True,
            scale=1, clock=False, times=None, do_fit_linear_prep=False, do_fit_leastsq_prep=False,
            do_fit_nonlinear_prep=False, do_jacobian=False, get_likelihood=False, is_likelihood_log=True,
            verify_derivative=False, grad_loglike=None, debug=False):
        """
            Draw model image for one exposure with one PSF

            Returns the image and the engine-dependent model used to draw it.
            If get_likelihood == True, it will try to get the likelihood
        """
        # TODO: Rewrite this completely at some point as validating inputs is a pain
        if profiles is None or meta_model is None:
            profiles, meta_model, engine, engineopts, times = self._get_image_model_exposure_setup(
                exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
        likelihood = None
        ny, nx = exposure.image.shape
        band = exposure.band
        image = None
        has_psf = meta_model['has_psf']
        if do_fit_linear_prep:
            exposure.meta['img_model_fixed'] = None
            exposure.meta['img_models_params_free'] = []

        if clock:
            times = meta_model['times'] if 'times' in meta_model else []
            time_now = time.time()
            if times is None:
                times = {}

        if engine == "galsim":
            gsparams = get_gsparams(engineopts)

        if do_draw_image:
            image = np.zeros_like(exposure.image)

        model = {}

        # TODO: Do this in a smarter way
        if meta_model['use_fast_gauss']:
            profiles_left = []
            profiles_to_draw = []
            profiles_to_fit_linear_ids = []
            profiles_to_fit_linear = {}
            do_prep_both = do_fit_linear_prep and do_fit_leastsq_prep
            profiles_to_draw_now = []
            level_bg, order_bg = None, None
            for profile in profiles:
                params = profile[band]
                # Verify that it's a Gaussian
                is_bg = profile.get('background', False)
                if is_bg or ('profile' in params and params['profile'] == 'sersic' and 'nser' in params and
                             params['nser'] == 0.5):
                    if do_fit_linear_prep and not params['param_flux'].fixed:
                        id_param_flux = id(params['param_flux'])
                        if id_param_flux not in profiles_to_fit_linear:
                            profiles_to_fit_linear_ids.append(id_param_flux)
                            profiles_to_fit_linear[id_param_flux] = ([], params['param_flux'])
                        profiles_to_fit_linear[id_param_flux][0].append(profile)
                    else:
                        profiles_to_draw += [profile]
                    if do_prep_both:
                        profiles_to_draw_now += [profile]
                else:
                    profiles_left += [profile]
                if is_bg:
                    # TODO: Should not have to assume that background is last and all bands free
                    level_bg = params['flux']
                    order_bg = params['order_rev']
            profiles = profiles_left
            # TODO: Explain this for a future user and/or self
            # Normally, we'd skip the rest of this if do_fit_linear_prep is true and profiles_to_draw is empty
            # That would be when all profiles are free
            if not do_prep_both:
                profiles_to_draw_now = profiles_to_draw
            model['multiprofit'] = {key: value for key, value in zip(
                ['profiles' + x for x in ['todraw', 'left', 'linearfit']],
                [profiles_to_draw, profiles_left, profiles_to_fit_linear]
            )}
            # If these are the only profiles, just get the likelihood
            get_like_only = (get_likelihood or do_jacobian) and profiles_to_draw_now \
                and not profiles_left and (not profiles_to_fit_linear or do_fit_leastsq_prep)
            if get_like_only:
                # TODO: Do this in a prettier way while avoiding recalculating loglike_gaussian_pixel
                sigma_inv = exposure.get_sigma_inverse()
                if 'like_const' not in exposure.meta:
                    mask = exposure.mask_inverse if exposure.mask_inverse is not None else (sigma_inv > 0)
                    exposure.meta['like_const'] = np.sum(np.log(sigma_inv[mask]/sqrt(2.*np.pi)))
                    if exposure.error_inverse.size == 1:
                        exposure.meta['like_const'] *= nx*ny
                    self.logger.info(f"Setting exposure.meta['like_const']={exposure.meta['like_const']}")
                do_grad_any = do_jacobian or do_fit_leastsq_prep
                profiles_band_bad = 0
                profiles_band = []
                profiles_band_idx = []
                background = mpfgauss.zero_double
                for idx_profile, profile in enumerate(profiles_to_draw_now):
                    profile_band = profile[band]
                    if np.isinf(profile_band["flux"]) and not do_grad_any:
                        profiles_band_bad += 1
                    elif profile.get('background', False):
                        background = np.array([profile_band['flux']], dtype=np.float64)
                    else:
                        profiles_band_idx.append(idx_profile)
                        profiles_band.append(profile_band)
                num_profiles = len(profiles_band)
                profile_matrix = gaussian_profiles_to_matrix(profiles_band)
                has_grad_param_map = self.grad_param_maps is not None and band in self.grad_param_maps
                if do_jacobian and not has_grad_param_map:
                    raise RuntimeError("Can't compute jacobian; self.grad_param_maps is {} and [band]={} "
                                       "(did you run with do_fit_leastsq_prep first?)".format(
                                           type(self.grad_param_maps), False if self.grad_param_maps is None
                                           else self.grad_param_maps.get(band)))
                grad_params_indices, grad_params_obj = self.grad_param_maps[band] if has_grad_param_map else \
                    (None, None)
                sersic_param_indices = self.param_maps['sersic'][band] if has_grad_param_map else None
                if do_grad_any:
                    if grad_loglike:
                        grad = grad_loglike
                        do_jacobian = False
                    else:
                        grad = exposure.meta['jacobian_img']
                        if do_fit_leastsq_prep:
                            grad.fill(0)
                            if level_bg is not None:
                                grad[:, :, -1-order_bg] = -sigma_inv
                    if do_fit_leastsq_prep:
                        grad_param_map = np.zeros((num_profiles, num_params_gauss), dtype=np.uint64)
                        grad_param_factor = np.zeros((num_profiles, num_params_gauss))
                        sersic_param_map = []
                        sersic_param_factor = []
                    else:
                        # TODO: Think about sanity checking here
                        grad_param_map = exposure.meta['grad_param_map']
                        grad_param_factor = exposure.meta['grad_param_factor']
                        sersic_param_map = exposure.meta['sersic_param_map']
                        sersic_param_factor = exposure.meta['sersic_param_factor']
                    idx_sersic = 0
                    param_flux_ratios = {}
                    for idx_profile, profile in enumerate(profiles_band):
                        flux_profile = profile["flux"]
                        # This is a source profile, not a logical source which might have multiple profiles
                        # TODO: Fix nomenclature. profile -> profile_sub?
                        param_flux = profile["param_flux"]
                        flux_src = param_flux.get_value(transformed=False)
                        weight = flux_profile/flux_src
                        idx_src = profile["idx_src"]
                        grad_param_map[idx_profile, :] = grad_params_indices[idx_src, :]
                        # Special handling of profiles with flux ratios happens on the python side
                        # ... for better or worse. It could be done in C++ if needed.
                        param_flux_ratio = profile.get('param_flux_ratio')
                        param_flux_ratio_is_none = param_flux_ratio is None
                        for idx_array, idx_param in enumerate(grad_param_map[idx_profile, :]):
                            param = grad_params_obj[idx_src][idx_array]
                            is_sigma = param.name == "sigma_x" or param.name == "sigma_y"
                            if idx_param > 0:
                                is_flux = isinstance(param, FluxParameter)
                                weight_grad = weight if is_flux and param_flux_ratio_is_none else 1.
                                # The function returns df/dg, where g is the Gaussian parameter
                                # We want df/dp, where p(g) is the fit parameter
                                # df/dp = df/dg * dg/dp = df/dg / dp/dg
                                # Skip the derivative if this profile has a flux ratio - it'll be added later
                                derivative = param.get_transform_derivative(
                                    verify=verify_derivative, rtol=1e-3) if not (
                                    is_flux and not param_flux_ratio_is_none) else None
                                if derivative is not None:
                                    weight_grad /= derivative
                                # TODO: Revisit for fitting rscale
                                if is_sigma:
                                    weight_grad *= mpfgauss.reff_to_sigma(profile.get("sigma_factor", 1))
                            else:
                                weight_grad = 0
                            grad_param_factor[idx_profile, idx_array] = weight_grad
                        param_nser = profile.get('param_nser')
                        if param_nser is not None and not param_nser.fixed:
                            dweight_dn = profile.get("dweight_dn")
                            dreff_dn = profile.get("dreff_dn")
                            if dweight_dn is not None and dreff_dn is not None:
                                # L = flux_src*frac_psf*weight
                                # dL/dn = dL/dweight * dweight/dn = flux_src*frac_psf*dweight_dn
                                # frac_psf = flux_profile/flux_src/weight
                                # df/dn = df/dL*dL/dn
                                # sig = sig_src*sigma_factor
                                # dsig/dn = dsig/dsigma_factor * dsigma_factor/dn = sig_src*dsigma_factor/dn
                                # sig_src = sig_profile/sigma_factor
                                dreff_dn /= mpfgauss.sigma_to_reff(profile["sigma_factor"])
                                derivative = profile["param_nser"].get_transform_derivative(
                                    verify=True, rtol=1e-3)
                                if derivative is not None:
                                    dweight_dn /= derivative
                                    dreff_dn /= derivative
                                ellipse = profile['ellipse_src']
                                factors = [dweight_dn*flux_src*profile.get("fluxfrac_psf", 1.),
                                           dreff_dn*ellipse.sigma_x,
                                           dreff_dn*ellipse.sigma_y]
                                if not all([np.isfinite(factor) for factor in factors]):
                                    raise RuntimeError("Sersic param factors: {} not all finite".format(
                                        factors
                                    ))
                                indices = [idx_profile, sersic_param_indices[idx_src]]
                                if do_fit_leastsq_prep:
                                    sersic_param_map.append(indices)
                                    sersic_param_factor.append(factors)
                                else:
                                    sersic_param_map[idx_sersic, :] = indices
                                    sersic_param_factor[idx_sersic, :] = factors
                                idx_sersic += 1
                            else:
                                # TODO: Compare idx and sersic_param_map[idx_sersic, 0]?
                                pass
                        # TODO: Should be done in prep since it won't change
                        if param_flux_ratio is not None:
                            if param_flux not in param_flux_ratios:
                                param_flux_ratios[param_flux] = []
                            param_flux_ratios[param_flux].append((param_flux_ratio, idx_profile, profile))
                    if do_fit_leastsq_prep:
                        sersic_param_map = np.array(sersic_param_map, dtype=np.uint64)
                        sersic_param_factor = np.array(sersic_param_factor)
                        exposure.meta['grad_param_map'] = grad_param_map
                        exposure.meta['grad_param_factor'] = grad_param_factor
                        exposure.meta['sersic_param_map'] = sersic_param_map
                        exposure.meta['sersic_param_factor'] = sersic_param_factor
                        profiles_ref, meta_modelref, _, _, _ = self._get_image_model_exposure_setup(
                            exposure, engine=engine, engineopts=engineopts)
                    if profiles_band_bad > 0:
                        profiles_band_idx = np.array(profiles_band_idx)
                        grad_param_map = grad_param_map[profiles_band_idx, :]
                        grad_param_factor = grad_param_factor[profiles_band_idx, :]
                        sersic_param_map = sersic_param_map[profiles_band_idx, :]
                        sersic_param_factor = sersic_param_factor[profiles_band_idx, :]
                else:
                    grad = mpfgauss.zeros_double
                    grad_param_map = mpfgauss.zeros_uint64
                    grad_param_factor = mpfgauss.zeros_double
                    sersic_param_map = mpfgauss.zeros_uint64
                    sersic_param_factor = mpfgauss.zeros_double
                # Don't output the image if do_fit_linear_prep is true - the next section will do that
                residual = exposure.meta['residual_img'] if do_jacobian or do_fit_leastsq_prep else \
                    mpfgauss.zeros_double
                output = image if do_draw_image else mpfgauss.zeros_double

                likelihood_free = mpf.loglike_gaussians_pixel(
                    data=exposure.image, sigma_inv=sigma_inv, gaussians=profile_matrix,
                    x_min=0, x_max=nx, y_min=0, y_max=ny, to_add=False, output=output, residual=residual,
                    grad=grad, grad_param_map=grad_param_map, grad_param_factor=grad_param_factor,
                    sersic_param_map=sersic_param_map, sersic_param_factor=sersic_param_factor,
                    background=background
                )
                if debug:
                    pass
                    #print('pm', exposure.meta['like_const'], likelihood_free)#, profile_matrix)
                likelihood = exposure.meta['like_const'] + likelihood_free
                # If computing the jacobian, will skip evaluating the likelihood and instead return the
                # residual image for fitting
                if do_grad_any or do_fit_leastsq_prep:
                    order_flux = order_params_gauss['flux']
                    # Turn all of the gradients w.r.t. Gaussian fluxes into total flux and flux ratio values
                    if do_fit_leastsq_prep:
                        grad_param_fixed = []
                    for param_flux, ratios in param_flux_ratios.items():
                        flux = param_flux.get_value(transformed=False)
                        flux_remaining = np.zeros(1+len(ratios))
                        flux_ratios_neg_p_one = np.zeros(len(ratios))
                        flux_remaining[0] = flux
                        is_flux_fixed = param_flux.fixed
                        for ratio_iter, (param_flux_ratio, idx_ratio, profile) in enumerate(ratios):
                            idx_component = grad_param_map[idx_ratio, order_flux]
                            flux_ratio = param_flux_ratio.get_value(transformed=False)
                            flux_ratios_neg_p_one[ratio_iter] = 1 - flux_ratio
                            flux_remaining[ratio_iter + 1] = flux_remaining[ratio_iter] * (1-flux_ratio)
                            if not is_flux_fixed:
                                # Cheat a little and use the extra residual entry in index zero for temp space
                                # ... I was considering removing it before this
                                grad[:, :, 0] += 1/grad[:, :, idx_component]
                            # dFn/dfn = Fn/fn = flux_remaining[n]*fn/fn
                            grad[:, :, idx_component] *= flux_remaining[ratio_iter]
                            # The Jacobian of the Nth component contributes to the Jacobian of fluxrats[0:N-1]
                            for idx_shift in range(ratio_iter):
                                idx_previous = idx_ratio - idx_shift - 1
                                idx_grad_previous = grad_param_map[idx_previous, order_flux]
                                grad[:, :, idx_grad_previous] -= grad[:, :, idx_component] * (
                                    flux_ratio/flux_ratios_neg_p_one[idx_previous])
                        # Divide all free parameters by their transform derivatives
                        for param_flux_ratio, idx_ratio, _ in ratios:
                            idx_component = grad_param_map[idx_ratio, order_flux]
                            if not param_flux_ratio.fixed:
                                grad[:, :, idx_component] /= param_flux_ratio.get_transform_derivative(
                                    verify=True, rtol=1e-3)
                        # idx_component is now the one for the final ratio which is always fixed
                        if is_flux_fixed:
                            if do_fit_leastsq_prep:
                                grad_param_fixed.append(idx_component)
                        else:
                            grad[:, :, idx_component] = \
                                1/(grad[:, :, 0]*param_flux.get_transform_derivative(verify=True, rtol=1e-3))
                            grad[:, :, 0].fill(0)
                    if do_fit_leastsq_prep:
                        self.grad_param_free = np.arange(1, grad.shape[2])
                        if grad_param_fixed:
                            self.grad_param_free = np.setdiff1d(self.grad_param_free, grad_param_fixed)
                elif not is_likelihood_log:
                    likelihood = np.exp(likelihood)
            if (not get_like_only) or do_fit_linear_prep or do_fit_leastsq_prep:
                if profiles_to_draw:
                    image = mpf.make_gaussians_pixel(
                        gaussians=gaussian_profiles_to_matrix(
                            [profile[band] for profile in profiles_to_draw]),
                        x_min=0, x_max=nx, y_min=0, y_max=ny, dim_x=nx, dim_y=ny)
                    for profile in profiles_to_draw:
                        background = 0
                        if profile[band].get('background'):
                            if background:
                                raise RuntimeError("Ended up with >1 backgrounds to draw")
                            background = profile[band].get('flux')
                        # TODO: Also fix this if backgrounds get more complicated than constant
                        if background:
                            image += background
                    if do_fit_linear_prep:
                        exposure.meta['img_model_fixed'] = np.copy(image)
                # Ensure identical order until all dicts are ordered
                for id_flux in profiles_to_fit_linear_ids:
                    profiles_flux, param_flux = profiles_to_fit_linear[id_flux]
                    # Draw all of the profiles together if there are multiple; otherwise
                    # make_gaussian_pixel is faster
                    if len(profiles_flux) > 1:
                        gaussians = gaussian_profiles_to_matrix([profile[band] for profile in profiles_flux])
                        imgprofiles = mpf.make_gaussians_pixel(gaussians, 0, nx, 0, ny, nx, ny)
                    else:
                        profile = profiles_flux[0]
                        params = profile[band]

                        if profile.get('background'):
                            # TODO: Use this or an alternative if background models become more complex
                            # than just a flat background
                            # imgprofiles = mpfgauss.loglike_gaussians_pixel()
                            imgprofiles = np.full((ny, nx), params['flux'])
                        else:
                            imgprofiles = np.array(mpf.make_gaussian_pixel_covar(
                                params['cenx'], params['ceny'], params['flux'],
                                mpfgauss.reff_to_sigma(params['sigma_x']),
                                mpfgauss.reff_to_sigma(params['sigma_y']),
                                params['rho'], 0, nx, 0, ny, nx, ny))
                    if do_fit_linear_prep:
                        exposure.meta['img_models_params_free'] += [(imgprofiles, param_flux)]
                    if not do_draw_image:
                        if image is None:
                            image = np.copy(imgprofiles)
                        else:
                            image += imgprofiles

        # If there are any profiles left, render them with libprofit/galsim
        # In principle one could evaluate Gaussian mixtures with multiprofit code and render other models
        # with the specified engine, although it's really best to just stick to Gaussian mixtures
        # (unless you really want to use an empirical PSF model)
        if profiles:
            if engine == 'libprofit':
                if do_fit_linear_prep:
                    profiles_free = []
                    param_fluxs = []
                profiles_pro = {}
                for profile in profiles:
                    profile = profile[band]
                    profile_type = profile["profile"]
                    if profile_type not in profiles_pro:
                        profiles_pro[profile_type] = []
                    del profile["profile"]
                    if not profile["pointsource"]:
                        profile["convolve"] = has_psf
                        if not profile["resolved"]:
                            raise RuntimeError("libprofit can't handle non-point sources that aren't resolved"
                                               "(i.e. profiles with size==0)")
                    del profile["pointsource"]
                    del profile["resolved"]
                    # TODO: Find a better way to do this
                    for coord in ["x", "y"]:
                        name_old = "cen" + coord
                        profile[coord + "cen"] = profile[name_old]
                        del profile[name_old]
                    # libprofit uses the irritating GALFIT convention of up = 0 degrees
                    profile["ang"] -= 90
                    do_linear_profile = do_fit_linear_prep and profile["param_flux"].fixed
                    if do_linear_profile:
                        profiles_free += [{}]
                        param_fluxs += [profile["param_flux"]]
                    del profile["param_flux"]
                    profile_dict = profiles_free[:-1] if do_linear_profile else profiles_pro
                    profile_dict[profile_type] += [profile]

                profit_model = {
                    'width': nx,
                    'height': ny,
                    'magzero': 0.0,
                    'profiles': profiles_pro,
                }
                if meta_model['is_psf_pixelated']:
                    profit_model['rough'] = True
                if has_psf:
                    shape = exposure.psf.get_image_shape()
                    if shape is None:
                        shape = [1 + np.int(x/2) for x in np.floor([nx, ny])]
                    profit_model["psf"] = exposure.psf.get_image(engine, size=shape, engineopts=engineopts)

                if exposure.use_mask_inverse is not None:
                    profit_model['calcmask'] = exposure.use_mask_inverse
                if clock:
                    times['model_setup'] = time.time() - time_now
                    time_now = time.time()
                if do_draw_image or get_likelihood:
                    image_pro = np.array(pyp.make_model(profit_model)[0])
                    if image is None:
                        image = image_pro
                    else:
                        image += image_pro
                    if do_fit_linear_prep:
                        if exposure.meta['img_model_fixed'] is None:
                            exposure.meta['img_model_fixed'] = np.copy(image)
                        else:
                            exposure.meta['img_model_fixed'] += image
                        for param_flux, profiles_free in zip(param_fluxs, profiles_free):
                            profit_model['profiles'] = profiles_free
                            image_profile = np.array(pyp.make_model(profit_model)[0])
                            image += image_profile
                            exposure.meta['img_models_params_free'].append((image_profile, param_flux))
                model[engine] = profit_model
            elif engine == "galsim":
                if clock:
                    time_setup_gs = time.time()
                    times['model_setup_profile_gs'] = 0
                model[engine] = None
                # The logic here is to attempt to avoid GalSim's potentially excessive memory usage when
                # performing FFT convolution by splitting the profiles up into big and small ones
                # Perfunctory experiments have shown that this can reduce the required FFT size
                profiles_gs = {key: {False: [], True: []} for key in ["small", "big", "linearfitprep"]}
                cen_img = gs.PositionD(nx/2., ny/2.)
                for profile in profiles:
                    profile = profile[band]
                    if not profile["pointsource"] and not profile["resolved"]:
                        profile["pointsource"] = True
                        raise RuntimeError("Converting small profiles to point sources not implemented yet")
                        # TODO: Finish this
                    else:
                        profile_gs = profile["profile_gs"].shear(
                            profile["shear"]).shift(
                            profile["offset"] - cen_img)
                    # TODO: Revise this when image scales are taken into account
                    do_linear_profile = do_fit_linear_prep and not profile["param_flux"].fixed
                    bin_size_name = "linearfitprep" if do_linear_profile else (
                        "big" if profile_gs.original.half_light_radius > 1 else "small")
                    convolve = has_psf and not profile["pointsource"]
                    profiles_gs[bin_size_name][convolve] += [
                        (profile_gs, profile["param_flux"]) if do_linear_profile else profile_gs
                    ]

                if has_psf:
                    psf_gs = exposure.psf.model
                    if psf_gs is None:
                        profiles_psf = exposure.psf.get_image(engine=engine)
                    else:
                        psf_gs = psf_gs.get_profiles(bands=[band], engine=engine)
                        profiles_psf = None
                        for profile in psf_gs:
                            profile = profile[band]
                            profile_gs = profile["profile_gs"].shear(profile["shear"])
                            # TODO: Think about whether this would ever be necessary
                            #.shift(profile["offset"] - cen_img)
                            if profiles_psf is None:
                                profiles_psf = profile_gs
                            else:
                                profiles_psf += profile_gs
                else:
                    if any([value[True] for key, value in profiles_gs.items()]):
                        raise RuntimeError("Model (band={}) has profiles to convolve but no PSF, "
                                           "profiles are: {}".format(exposure.band, profiles_gs))
                # TODO: test this, and make sure that it works in all cases, not just gauss. mix
                # Has a PSF and it's a pixelated analytic model, so all sources must use no_pixel
                if meta_model['is_psf_pixelated']:
                    method = "no_pixel"
                else:
                    if (self.engineopts is not None and "drawmethod" in self.engineopts and
                            self.engineopts["drawmethod"] is not None):
                        method = self.engineopts["drawmethod"]
                    else:
                        method = "fft"
                if clock:
                    times['model_setup_profile_gs'] = time.time() - time_setup_gs
                has_psfimage = has_psf and psf_gs is None
                if clock:
                    times['model_setup'] = time.time() - time_now
                    time_now = time.time()
                # GalSim has special profile objects which can be added together for efficient simultaneous
                # evaluation and convolution
                for profile_type, profiles_of_type in profiles_gs.items():
                    for convolve, profiles_gs_bin in profiles_of_type.items():
                        if profiles_gs_bin:
                            # Pixel convolution is included with PSF images so no_pixel should be used
                            # TODO: Implement oversampled convolution here (or use libprofit?)
                            # Otherwise, flux conservation may be very poor
                            method_bin = 'no_pixel' if convolve and has_psfimage else method
                            is_profile_linear = profile_type == 'linearfitprep'
                            profiles_to_draw = profiles_gs_bin if is_profile_linear else [profiles_gs_bin[0]]
                            if not is_profile_linear:
                                for profile_to_add in profiles_gs_bin[1:]:
                                    profiles_to_draw[0] += profile_to_add
                            for profile_to_draw in profiles_to_draw:
                                if is_profile_linear:
                                    param_flux = profile_to_draw[1]
                                    profile_to_draw = profile_to_draw[0]
                                if convolve:
                                    profile_to_draw = gs.Convolve(
                                        profile_to_draw, profiles_psf, gsparams=gsparams)
                                if model[engine] is None:
                                    model[engine] = profile_to_draw
                                else:
                                    model[engine] += profile_to_draw
                                if do_draw_image or get_likelihood or do_fit_linear_prep:
                                    try:
                                        image_gs = profile_to_draw.drawImage(
                                            method=method_bin, nx=nx, ny=ny, scale=scale).array
                                    # Somewhat ugly hack - catch RunTimeErrors which are usually excessively
                                    # large FFTs and then try to evaluate the model in real space or give
                                    # up if it's any other error
                                    except RuntimeError:
                                        try:
                                            if method == "fft":
                                                image_gs = profile_to_draw.drawImage(
                                                    method='real_space', nx=nx, ny=ny, scale=scale).array
                                            else:
                                                raise
                                        except Exception:
                                            raise
                                    except Exception:
                                        print("Exception attempting to draw image from profiles:",
                                              profile_to_draw)
                                        raise
                                    if do_fit_linear_prep:
                                        if is_profile_linear:
                                            exposure.meta['img_models_params_free'].append(
                                                (np.copy(image_gs), param_flux))
                                        else:
                                            if exposure.meta['img_model_fixed'] is None:
                                                exposure.meta['img_model_fixed'] = np.copy(image_gs)
                                            else:
                                                exposure.meta['img_model_fixed'] += image_gs
                                    if image is None:
                                        image = image_gs
                                    else:
                                        image += image_gs
            else:
                error_msg = "engine is None" if engine is None else (
                        "engine type " + engine + " not implemented")
                raise RuntimeError(error_msg)

        if clock:
            times['modeldraw'] = time.time() - time_now

        if image is not None:
            sum_not_finite = np.sum(~np.isfinite(image))
            if sum_not_finite > 0:
                raise RuntimeError(f"{type(self)}.get_image_model_exposure() got "
                                   f"{sum_not_finite:d}/{np.prod(image.shape):d} non-finite pixels from"
                                   f" params {params}")

        return image, model, times, likelihood

    def get_limits(self, free=True, fixed=True, transformed=True):
        """
        Get parameter limits.

        :param free: Bool; return limits for free parameters?
        :param fixed: Bool; return limits for fixed parameters?
        :param transformed: Bool; return transformed limit values?
        :return:
        """
        params = self.get_parameters(free=free, fixed=fixed)
        return [param.get_limits(transformed=transformed) for param in params]

    def get_profiles(self, **kwargs):
        """
        Get engine-dependent representations of profiles to be rendered.

        :param kwargs: keyword arguments passed to Source.get_profiles()
        :return: List of profiles
        """
        if 'engine' not in kwargs or kwargs['engine'] is None:
            kwargs['engine'] = self.engine
        if 'engineopts' not in kwargs or kwargs['engine'] is None:
            kwargs['engineopts'] = self.engineopts
        self._checkengine(kwargs['engine'])
        profiles = []
        for src in self.sources:
            profiles += src.get_profiles(**kwargs)
        return profiles

    def get_parameters(self, free=True, fixed=True, flatten=True, modifiers=True):
        params = []
        for src in self.sources:
            params_src = src.get_parameters(free=free, fixed=fixed, flatten=flatten, modifiers=modifiers)
            if flatten:
                params += params_src
            else:
                params.append(params_src)
        return params

    def get_param_names(self, free=True, fixed=True):
        names = []
        for i, src in enumerate(self.sources):
            name_src = src.name
            if name_src == "":
                name_src = str(i)
            names += [".".join([name_src, name_param]) for name_param in
                      src.get_param_names(free=free, fixed=fixed)]
        return names

    def get_prior_value(self, log=True):
        if self.priors:
            idx_params = self.params_prior_jacobian
            prior_log = 0
            idx_prior = 0
            for prior in self.priors:
                prior_value, residuals, jacobians = prior.calc_residual(calc_jacobian=True)
                prior_log += prior_value
                n_res = len(residuals)
                sli = slice(idx_prior, idx_prior + n_res)
                try:
                    self.residuals_prior[sli] = residuals
                except Exception:
                    print(f"Failed to set prior residual for prior {prior}")
                    raise
                for param, jac_param in jacobians.items():
                    idx_param = idx_params.get(param)
                    if idx_param is not None:
                        self.jacobian_prior[sli, idx_param] += jac_param
                idx_prior += n_res
            return prior_log if log else 10**prior_log
        else:
            return 0. if log else 1.

    def get_likelihood(self, params=None, data=None, log=True, **kwargs):
        likelihood = self.evaluate(params, data, is_likelihood_log=log, **kwargs)[0]
        return likelihood

    def __init__(self, sources, data=None, likefunc="normal", engine=None, engineopts=None, name="",
                 logger=None, priors=None):
        if engine is None:
            engine = "libprofit"
        for i, source in enumerate(sources):
            if not isinstance(source, Source):
                raise TypeError(
                    "Model source[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(source), type(Source))
                )
        if data is not None and not isinstance(data, Data):
            raise TypeError(
                "Model data (type={:s}) is not an instance of {:s}".format(
                    str(type(data)), str(Data))
            )
        Model._checkengine(engine)
        if logger is None:
            logger = logging.getLogger(__name__)
        self.can_do_fit_leastsq = False
        self.data = data
        self.engine = engine
        self.engineopts = engineopts
        self.grad_param_maps = None
        self.grad_param_free = None
        self.jacobian = None
        self.jacobian_prior = None
        self.param_maps = {'sersic': None}
        self.likefunc = likefunc
        self.name = name
        self.logger = logger
        self.residuals = None
        self.residuals_prior = None
        self.sources = sources
        self.priors = priors
        self.params_prior_jacobian = None


class ModellerPygmoUDP:
    def fitness(self, x):
        return [-self.modeller.evaluate(x, returnlponly=True, timing=self.timing)]

    def get_bounds(self):
        return self.bounds_lower, self.bounds_upper

    def gradient(self, x):
        # TODO: Fix this; it doesn't actually work now that the import is conditional
        # Putting an import statement here is probably a terrible idea
        return pg.estimate_gradient(self.fitness, x)

    def __init__(self, modeller, bounds_lower, bounds_upper, timing=False):
        self.modeller = modeller
        self.bounds_lower = bounds_lower
        self.bounds_upper = bounds_upper
        self.timing = timing

    def __deepcopy__(self, memo):
        """
            This is some rather ugly code to make a deep copy of a modeller without copying the model's
            data, which should be shared between all copies of a model (and its modeller).
        """
        modelself = self.modeller.model
        model = Model(sources=copy.deepcopy(modelself.sources, memo=memo), data=modelself.data,
                      likefunc=copy.copy(modelself.likefunc), engine=copy.copy(modelself.engine),
                      engineopts=copy.copy(modelself.engineopts))
        modeller = Modeller(model=model, modellib=copy.copy(self.modeller.modellib),
                            modellibopts=copy.copy(self.modeller.modellibopts))
        modeller.fitinfo = copy.copy(self.modeller.fitinfo)
        copied = self.__class__(modeller=modeller, bounds_lower=copy.copy(self.bounds_lower),
                                bounds_upper=copy.copy(self.bounds_upper), timing=copy.copy(self.timing))
        memo[id(self)] = copied
        return copied


class Modeller:
    """
        A class that does the fitting for a Model.
        Some useful things it does is define optimization functions called by libraries and optionally
        store info that they don't track (mainly per-iteration info, including parameter values,
        running time, separate priors and likelihoods rather than just the objective, etc.).
    """
    def evaluate(self, params_free=None, timing=False, returnlog=True, returnlponly=False, **kwargs):

        if timing:
            time_init = time.time()
        # TODO: Clarify that setting do_draw_image = True forces likelihood evaluation with both C++/Python
        #  (if possible)
        likelihood = self.model.get_likelihood(params_free, **kwargs)
        # Must come second if do_fit_leastsq_prep is True and not previously called
        prior = self.model.get_prior_value() if (likelihood is not None) else None

        # return LL, LP, etc.
        if returnlponly:
            rv = likelihood + prior
            if not returnlog:
                rv = np.exp(rv)
        else:
            if not returnlog:
                likelihood = np.exp(likelihood)
                prior = np.exp(prior)
            rv = likelihood, prior
        if timing:
            tstep = time.time() - time_init
            rv += tstep
        log_step = "print_step_interval" in self.fitinfo and self.fitinfo["print_step_interval"] is not None
        loginfo = {
            "params": params_free,
            "likelihood": likelihood,
            "prior": prior,
        }
        if timing:
            loginfo["time"] = tstep
            loginfo["time_init"] = time_init
        if "log" in self.fitinfo:
            self.fitinfo["log"].append(loginfo)
        if log_step:
            stepnum = len(self.fitinfo["log"])
            if stepnum % self.fitinfo["print_step_interval"] == 0:
                self.logger.info(f"Step {stepnum}: rv={rv}, loginfo={loginfo}")
        return rv

    def fit(self, params_init=None, do_print_final=True, timing=None, walltime_max=np.Inf,
            print_step_interval=None, do_linear=True, do_linear_only=False, debug=False):
        """
        Fit the Model to the data and return a dict with assorted fit information.

        :param params_init: iterable of floats; initial transformed parameter values.
            Default None (existing values will be used).
        :param do_print_final: bool; whether to print status messages when finished.
        :param timing: bool; whether to track and return elapsed time.
        :param walltime_max: float; maximum allowed walltime for fit in seconds. Only supported by some
            pygmo algorithms.
        :param print_step_interval: int; number of steps to run before printing another status message.
        :param do_linear: bool; whether to do a linear fit for free component amplitudes first.
        :param do_linear_only: bool; whether to skip the non-linear fit for non-linear problems.
        :param debug: bool; whether to run additional diagnostic tests in model evaluation.

        :return: dict with output.
        """
        self.fitinfo["log"] = []
        self.fitinfo["print_step_interval"] = print_step_interval

        self.fitinfo["n_eval_func"] = 0
        self.fitinfo["n_eval_grad"] = 0
        self.fitinfo["n_timings"] = 0 if timing is None else len(timing)

        params_free = self.model.get_parameters(fixed=False)
        if params_init is None:
            params_init = np.array([param.get_value(transformed=True) for param in params_free])

        name_params = [param.name for param in params_free]
        is_flux_params_free = [isinstance(param, FluxParameter) for param in params_free]
        is_any_flux_free = any(is_flux_params_free)
        # Is this a linear problem? True iff all free params are fluxes
        is_linear = all(is_flux_params_free)
        # Do a linear fit at all? No if all fluxes are fixed
        do_linear = do_linear and is_any_flux_free
        # TODO: The mechanism here is TBD
        do_linear_only = do_linear and do_linear_only

        # If fitting with GalSim, ensure that the fast gaussian code is not used to avoid inconsistent
        # likelihoods at n=0.5
        if self.model.engine == "galsim" and (self.model.engineopts.get("use_fast_gauss") is not False):
            any_not_gauss = False
            for source in self.model.sources:
                if not any_not_gauss:
                    for comp in source.modelphotometric.components:
                        if not (comp.is_gaussian_mixture() and not (
                                isinstance(comp, EllipticalParametricComponent) and
                                comp.profile == "sersic" and not comp.parameters["nser"].fixed)
                        ):
                            any_not_gauss = True
                            break
            if any_not_gauss:
                self.model.engineopts["use_fast_gauss"] = False
                self.model.engineopts["do_analytic_convolution"] = False

        time_start = time.time()
        likelihood = self.evaluate(
            params_init, do_fit_linear_prep=do_linear, do_fit_leastsq_prep=True, do_fit_nonlinear_prep=True,
            do_verify_jacobian=not do_linear_only, do_compare_likelihoods=True, debug=debug)
        likelihood_init = likelihood
        self.logger.info(f"Param names   : {name_params}")
        self.logger.info(f"Initial params (transformed): {params_init}")
        self.logger.info(f"Initial params: {[param.get_value(transformed=False) for param in params_free]}")
        self.logger.info(f"Initial likelihood in t={time.time() - time_start:.3e}: {likelihood}")
        sys.stdout.flush()

        do_nonlinear = not (do_linear_only or (is_linear and do_linear and self.model.can_do_fit_leastsq))

        time_run = 0.0
        # TODO: This should be an input argument
        bands = np.sort(list(self.model.data.exposures.keys()))
        if do_linear:
            # If this isn't a linear problem, fix all free non-flux parameters and do a linear fit first
            if not is_linear:
                for param in params_free:
                    if not isinstance(param, FluxParameter):
                        param.fixed = True
            self.logger.info("Beginning linear fit")
            time_init = time.time()
            datasizes = []
            params = []
            for band in bands:
                params_band = None
                for exposure in self.model.data.exposures[band]:
                    datasizes.append(np.sum(exposure.mask_inverse)
                                     if exposure.mask_inverse is not None else
                                     exposure.image.size)
                    params_exposure = [x[1] for x in exposure.meta['img_models_params_free']]
                    if params_band is None:
                        params_band = params_exposure
                    elif params_band != params_exposure:
                        raise RuntimeError('Got different linear fit params '
                                           'in two exposures: {} vs {}'.format(params_band, params_exposure))
                if params_band is not None:
                    params += params_band

            num_params = len(params)
            datasize = np.sum(datasizes)
            # Matrix of vectors for each variable component
            x = np.zeros([datasize, num_params])
            y = np.zeros(datasize)
            idx_begin = 0
            idx_exposure = 0
            idx_param = 0
            for band in bands:
                exposures = self.model.data.exposures[band]
                idx_free = None
                for exposure in exposures:
                    idx_end = idx_begin + datasizes[idx_exposure]
                    maskinv = exposure.mask_inverse
                    sigma_inv = exposure.get_sigma_inverse()
                    if maskinv is not None:
                        sigma_inv = sigma_inv[maskinv]
                    for idx_free, (img_free, _) in enumerate(exposure.meta['img_models_params_free']):
                        x[idx_begin:idx_end, idx_param + idx_free] = (
                            img_free if maskinv is None else img_free[maskinv]).flat * sigma_inv
                    img = exposure.image
                    img_fixed = exposure.meta['img_model_fixed']
                    y[idx_begin:idx_end] = (
                        (img if maskinv is None else img[maskinv]) if img_fixed is None else
                        (img - img_fixed if maskinv is None else img[maskinv] - img_fixed[maskinv])
                    ).flat * sigma_inv
                    idx_begin = idx_end
                    idx_exposure += 1
                if exposures and idx_param:
                    idx_param += idx_free + 1

            fluxratios_to_print = None
            fitmethods = {
                'scipy.optimize.nnls': [None],
                # TODO: Confirm that nnls really performs best
                #'scipy.optimize.lsq_linear': ['bvls'],
                #'numpy.linalg.lstsq': [None],#, 1e-3, 1, 100],
                #'fastnnls.fnnls': [None],
            }
            values_init = [param.get_value(transformed=False) for param in params]
            reset = True

            for method, params_to_fit in fitmethods.items():
                for fitparam in params_to_fit:
                    if method == 'scipy.optimize.nnls':
                        fluxratios = spopt.nnls(x, y)[0]
                    elif method == 'scipy.optimize.lsq_linear':
                        fluxratios = spopt.lsq_linear(x, y, bounds=(0.01, np.Inf), method=fitparam).x
                    elif method == 'numpy.linalg.lstsq':
                        fluxratios = np.linalg.lstsq(x, y, rcond=fitparam)[0]
                    elif method == 'fastnnls.fnnls':
                        from fastnnls import fnnls
                        y = x.T.dot(y)
                        x = x.T.dot(x)
                        fluxratios = fnnls(x, y)
                    else:
                        raise ValueError('Unknown linear fit method ' + method)

                    values_new = []
                    for fluxratio, param, value_init in zip(fluxratios, params, values_init):
                        ratio = np.max([1e-2, fluxratio])
                        # TODO: See if there is a better alternative to setting values outside transform range
                        # Perhaps leave an option to change the transform to log10?
                        value_set = None
                        for frac in np.linspace(1, 0, 10+1):
                            value_new = value_init*(frac*ratio + 1. - frac)
                            param.set_value(value_new, transformed=False)
                            if np.isfinite(param.get_value(transformed=False)):
                                values_new.append(value_new)
                                value_set = value_new
                                break
                        if value_set is None:
                            values_new.append(value_init)

                    likelihood_new = self.evaluate()

                    if likelihood_new[0] > likelihood[0]:
                        fluxratios_to_print = fluxratios
                        method_best = method
                        likelihood = likelihood_new
                        values_init = values_new
                        reset = False
                    else:
                        reset = True

            if reset:
                for value_init, param in zip(values_init, params):
                    param.set_value(value_init, transformed=False)

            self.logger.info("Model '{}' linear fit elapsed time: {:.3e}".format(
                self.model.name, time.time() - time_init))
            if fluxratios_to_print is None:
                self.logger.info("Linear fit failed to improve on initial parameters")
                likelihood_new = self.evaluate()
                if np.abs(likelihood_new[0] - likelihood_init[0]) > 1e-2:
                    sigma_inv = exposure.get_sigma_inverse()
                    like_const_new = np.sum(np.log(sigma_inv[exposure.mask_inverse]/sqrt(2.*np.pi)))
                    raise RuntimeError(
                        f'likelihood_new={likelihood_new[0]:.6e} != likelihood_init={likelihood_init[0]:.6e}'
                        f' reset={reset} like_const_new={like_const_new} vs exp={exposure.meta["like_const"]}'
                        f' lens={(len(fluxratios), len(params), len(values_init))}'
                    )
            else:
                params_init = np.array([param.get_value(transformed=True) for param in params_free])
                params_init_true = np.array([param.get_value(transformed=False) for param in params_free])
                self.logger.info(f"Final loglike, logprior: {likelihood}")
                self.logger.info(f"New initial parameters from method {method_best}: {params_init}")
                self.logger.info(f"New initial parameter values from method {method_best}: "
                                 f"{params_init_true}")
                self.logger.info(f"Linear flux ratios: {fluxratios_to_print}")
            if not is_linear:
                for param in params_free:
                    param.fixed = False

        # Skip non-linear fit if asked to or if it's a linear least squares problem,
        # for which it's unnecessary
        # TODO: Review if/when priors are changed - linear fits won't be able to handle non-linear priors
        # (even if they're Gaussian, e.g. if applied to transforms/non-linear combinations of params)
        do_fit_leastsq_cleanup = False
        if do_nonlinear:
            limits = self.model.get_limits(fixed=False, transformed=True)
            algo = self.modellibopts["algo"] if "algo" in self.modellibopts else (
                "neldermead" if self.modellib == "pygmo" else "Nelder-Mead")
            if self.modellib == "scipy":
                if self.model.can_do_fit_leastsq:
                    for band, exposures in self.model.data.exposures.items():
                        for exposure in exposures:
                            if not exposure.is_error_sigma:
                                exposure.error_inverse = np.sqrt(exposure.error_inverse)
                                exposure.is_error_sigma = True

                    def residuals(params_i, modeller):
                        modeller.evaluate(params_i, timing=timing, get_likelihood=False, do_jacobian=True)
                        modeller.fitinfo["n_eval_func"] += 1
                        return modeller.model.residuals

                    def jacobian(params_i, modeller):
                        modeller.fitinfo["n_eval_grad"] += 1
                        return modeller.model.jacobian[:, modeller.model.grad_param_free]

                    bounds = ([x[0] for x in limits], [x[1] for x in limits])
                    time_init = time.time()
                    result = spopt.least_squares(residuals, params_init, jac=jacobian, args=(self,),
                                                 bounds=bounds)
                    time_run += time.time() - time_init
                    params_best = result.x
                    do_fit_leastsq_cleanup = True
                else:
                    def neg_like_model(params_i, modeller):
                        modeller.fitinfo["n_eval_func"] += 1
                        return -modeller.evaluate(params_i, timing=timing, returnlponly=True)
                    time_init = time.time()
                    result = spopt.minimize(neg_like_model, params_init, method=algo, bounds=np.array(limits),
                                            options={} if 'options' not in self.modellibopts else
                                            self.modellibopts['options'], args=(self,))
                    time_run += time.time() - time_init
                    params_best = result.x

            elif self.modellib == "pygmo":
                algocmaes = algo == "cmaes"
                algonlopt = not algocmaes
                if algocmaes:
                    uda = pg.cmaes()
                elif algonlopt:
                    uda = pg.nlopt(algo)
                    uda.ftol_abs = 1e-3
                    if np.isfinite(walltime_max) and walltime_max > 0:
                        uda.maxtime = walltime_max
                    nloptopts = ["stopval", "ftol_rel", "ftol_abs", "xtol_rel", "xtol_abs", "maxeval"]
                    for nloptopt in nloptopts:
                        if nloptopt in self.modellibopts and self.modellibopts[nloptopt] is not None:
                            setattr(uda, nloptopt, self.modellibopts[nloptopt])

                algo = pg.algorithm(uda)
                #        algo.extract(pg.nlopt).ftol_rel = 1e-6
                if algonlopt:
                    algo.extract(pg.nlopt).ftol_abs = 1e-3

                if "verbosity" in self.modellibopts and self.modellibopts["verbosity"] is not None:
                    algo.set_verbosity(self.modellibopts["verbosity"])
                limitslower = np.zeros(len(limits))
                limitsupper = np.zeros(len(limits))
                for i, limit in enumerate(limits):
                    limitslower[i] = limit[0]
                    limitsupper[i] = limit[1]

                udp = ModellerPygmoUDP(modeller=self, bounds_lower=limitslower, bounds_upper=limitsupper,
                                       timing=timing)
                problem = pg.problem(udp)
                pop = pg.population(prob=problem, size=0)
                if algocmaes:
                    npop = 5
                    npushed = 0
                    while npushed < npop:
                        try:
                            #pop.push_back(init + np.random.normal(np.zeros(np.sum(data.to_fit)),
                            #                                      data.sigmas[data.to_fit]))
                            npushed += 1
                        except:
                            pass
                else:
                    pop.push_back(params_init)
                time_init = time.time()
                result = algo.evolve(pop)
                time_run += time.time() - time_init
                params_best = result.champion_x
                # TODO: Increment n_evals
            else:
                raise RuntimeError("Unknown optimization library " + self.modellib)
            likelihood = self.evaluate(params_best, debug=debug)
            if do_print_final:
                self.logger.info(
                    "Model '{}' nonlinear fit elapsed time: {:.3e} after {},{} function,gradient "
                    "evaluations ({} logged)".format(
                        self.model.name, time_run, self.fitinfo["n_eval_func"],
                        self.fitinfo["n_eval_grad"],
                        (len(timing) if timing is not None else 0) - self.fitinfo["n_timings"]))
                self.logger.info(f"Final loglike, logprior: {likelihood}")
                self.logger.info(f"Param names (fit): "
                                 f"{','.join(['{:11s}'.format(i) for i in name_params])}")
                self.logger.info(f"Values transfo.:   " 
                                 f"{','.join(['{:+1.4e}'.format(i) for i in params_best])}")
                params_all = self.model.get_parameters(fixed=True)
                self.logger.info(f"Param names (all): "
                                 f"{','.join(['{:11s}'.format(p.name) for p in params_all])}")
                values_all = ','.join(['{:+.4e}'.format(p.get_value(transformed=False)) for p in params_all])
                self.logger.info(f"Values untransfo.: {values_all}")
        else:
            params_best = params_init
            result = None
            time_run += time.time() - time_init

        if do_fit_leastsq_cleanup:
            self.model.do_fit_leastsq_cleanup(bands)

        result = {
            "fitinfo": self.fitinfo.copy(),
            "params": self.model.get_parameters(),
            "name_params": name_params,
            "likelihood": likelihood,
            "likelihood_init": likelihood_init,
            "n_eval_func": self.fitinfo["n_eval_func"],
            "n_eval_grad": self.fitinfo["n_eval_grad"],
            "params_best": params_best,
            "result": result,
            "time": time_run,
            "uncertainties": self.model.estimate_uncertainties(params_best),
        }
        return result

    # TODO: Should constraints be implemented?
    def __init__(self, model, modellib, modellibopts=None, constraints=None, logger=None):
        self.model = model
        self.modellib = modellib
        self.modellibopts = modellibopts if modellibopts is not None else {}
        self.constraints = constraints
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Scratch space
        self.fitinfo = {}


class Source:
    """
        A model of a source, like a galaxy or star, or even the PSF (TBD).
    """
    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    def get_parameters(self, free=None, fixed=None, flatten=True, modifiers=True, time=None):
        astrometry = self.modelastrometric.get_position(time)
        paramobjects = [
            self.modelastrometric.get_parameters(free, fixed, time),
            self.modelphotometric.get_parameters(free, fixed, flatten=flatten, modifiers=modifiers,
                                                 astrometry=astrometry)
        ]
        params = []
        for paramobject in paramobjects:
            if flatten:
                params += paramobject
            else:
                params.append(paramobject)
        return params

    # TODO: Finish this
    def get_param_names(self, free=None, fixed=None):
        return self.modelastrometric.get_param_names(free, fixed) + \
            self.modelphotometric.get_parameters(free, fixed)

    def get_profiles(self, engine, bands, time=None, engineopts=None):
        """

        :param bands: List of bands
        :param engine: Valid rendering engine
        :param engineopts: Dict of engine options
        :return:
        """
        self._checkengine(engine)
        cenx, ceny = self.modelastrometric.get_position(time=time)
        return self.modelphotometric.get_profiles(
            engine=engine, bands=bands, cenx=cenx, ceny=ceny,
            params={k: v for k, v in self.modelastrometric.params.items()},
            time=time, engineopts=engineopts)

    def is_sky(self):
        return all([isinstance(c, Background) for c in self.modelphotometric.components])

    def __init__(self, modelastrometry, modelphotometry, name=""):
        self.name = name
        self.modelphotometric = modelphotometry
        self.modelastrometric = modelastrometry


class PhotometricModel:
    def convert_param_fluxes(self, use_fluxfracs, **kwargs):
        if use_fluxfracs and self.fluxes:
            raise RuntimeError("Tried to convert model to use flux fracs but self.fluxes not empty, "
                               "so it must already be using flux fracs")
        profiles = self.get_profiles(engine='libprofit', bands=self.fluxes.keys(), cenx=0, ceny=0)
        for profiles_comp, comp in zip(profiles, self.components):
            if not isinstance(comp, Background):
                for i, flux in enumerate(comp.fluxes):
                    profile_band = profiles_comp[flux.band]
                    if flux.is_fluxratio is use_fluxfracs:
                        raise RuntimeError(
                            'Tried to convert component with is_fluxratio={} already == use_fluxfracs={}'.format(
                                flux.is_fluxratio, use_fluxfracs
                            ))
                    if use_fluxfracs:
                        if flux.band in self.fluxes:
                            self.fluxes[flux.band].append(flux)
                        else:
                            self.fluxes[flux.band] = np.array([flux])
                    else:
                        flux = FluxParameter(
                            band=flux.band, value=0, name=flux.name, is_fluxratio=False,
                            unit=self.fluxes[flux.band].unit, **kwargs)
                        flux.set_value(profile_band['flux'], transformed=False)
                        comp.fluxes[i] = flux
                comp.fluxes_dict = {flux.band: flux for flux in comp.fluxes}
        if use_fluxfracs:
            # Convert fluxes to fractions
            for band, fluxes in self.fluxes.items():
                values = np.zeros(len(fluxes))
                fixed = []
                units = []
                for idx, flux in enumerate(fluxes):
                    values[idx] = flux.get_value(transformed=False)
                    fixed[idx] = flux.fixed
                    units[idx] = flux.units
                if all(fixed):
                    fixed = True
                elif not any(fixed):
                    fixed = False
                else:
                    raise RuntimeError('Fixed={} invalid; must be all true or all false'.format(fixed))
                total = np.sum(values)
                idx_last = len(fluxes) - 1
                for idx, flux in enumerate(fluxes):
                    is_last = idx == idx_last
                    value = 1 if is_last else values[i]/total
                    fluxfrac = FluxParameter(
                        band=flux.band, value=value, is_fluxratio=True, unit=None,
                        fixed=True if is_last else fixed, **kwargs)
                    fluxes[idx] = fluxfrac
                    total -= values[idx]
                self.fluxes[band] = FluxParameter(
                    band=band, value=np.sum(values), fixed=fixed, name=flux.name, is_fluxratio=False,
                    **kwargs)
        else:
            self.fluxes = {}

    def get_parameters(self, free=True, fixed=True, flatten=True, modifiers=True, astrometry=None):
        params_flux = [flux for flux in self.fluxes.values() if
                       (flux.fixed and fixed) or (not flux.fixed and free)]
        params = params_flux if flatten else [params_flux]
        for comp in self.components:
            params_comp = comp.get_parameters(free=free, fixed=fixed)
            if flatten:
                params += params_comp
            else:
                params.append(params_comp)
        if modifiers:
            modifiers = ([param for param in self.modifiers if
                          (param.fixed and fixed) or (not param.fixed and free)])
            if flatten:
                params += modifiers
            else:
                params.append(modifiers)
        return params

    # TODO: Determine how the astrometric model is supposed to interact here
    def get_profiles(self, engine, bands, cenx, ceny, params=None, time=None, engineopts=None):
        """
        :param engine: Valid rendering engine
        :param bands: List of bands
        :param cenx: X coordinate
        :param ceny: Y coordinate
        :param params: Dict of potentially relevant parameters (e.g. controlling cenx/y)
        :param time: A time for variable sources. Not implemented yet.
        :param engineopts: Dict of engine options
        :return: List of dicts by band
        """
        # TODO: Check if this should skip entirely instead of adding a None for non-included bands
        if bands is None:
            bands = self.fluxes.keys()
        flux_by_band = {band: self.fluxes[band].get_value(transformed=False) if
                        band in self.fluxes else None for band in bands}
        profiles = []
        for comp in self.components:
            profiles += comp.get_profiles(
                flux_by_band, engine, cenx, ceny, params, engineopts=engineopts)
        for band, param_flux in self.fluxes.items():
            params_fluxratio = []
            for idx, profile in enumerate(profiles):
                profile_band = profile[band]
                profile_band["param_flux_ratio"] = profile_band["param_flux"]
                profile_band["param_flux"] = param_flux
                if idx > 0:
                    params_fluxratio.append(profile_band["param_flux"])
                profile_band["param_flux_ratios"] = params_fluxratio
                profile_band["param_flux_idx"] = idx
        for flux in comp.fluxes:
            if flux.is_fluxratio and flux.get_value(transformed=False) != 1.:
                raise RuntimeError('Non-unity flux ratio for final component')
        return profiles

    def __init__(self, components, fluxes=None, modifiers=None):
        for i, comp in enumerate(components):
            if not isinstance(comp, Component):
                raise TypeError("PhotometricModel component[{:d}](type={:s}) "
                                "is not an instance of {:s}".format(
                    i, type(comp), type(Component)))
        if fluxes is None:
            fluxes = []
        for i, flux in enumerate(fluxes):
            if not isinstance(flux, FluxParameter):
                raise TypeError("PhotometricModel flux[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(flux), type(FluxParameter)))
        bands_comps = [[flux.band for flux in comp.fluxes] for comp in components]
        # TODO: Check if component has a redundant mag or no specified flux ratio
        if not mpfutil.allequal(bands_comps):
            raise ValueError(
                "Bands of component fluxes in PhotometricModel components not all equal: {}".format(
                    bands_comps))
        bands_fluxes = [flux.band for flux in fluxes]
        if any([band not in bands_comps[0] for band in bands_fluxes]):
            raise ValueError("Bands of fluxes in PhotometricModel fluxes not all in fluxes of the "
                             "components: {} not all in {}".format(bands_fluxes, bands_comps[0]))
        self.components = components
        self.fluxes = {flux.band: flux for flux in fluxes}
        self.modifiers = [] if modifiers is None else modifiers


# TODO: Implement and use, with optional WCS attached?
class Position:
    def __init__(self, x, y):
        for key, value in {"x": x, "y": y}:
            if not isinstance(value, Parameter):
                raise TypeError("Position[{:s}](type={:s}) is not an instance of {:s}".format(
                    key, type(value), type(Parameter)))
        self.x = x
        self.y = y


class AstrometricModel:
    """
        The astrometric model for this source.
        TODO: Implement moving models, or at least think about how to do it
    """

    def get_parameters(self, free=True, fixed=True, time=None):
        return [value for value in self.params.values() if
                (value.fixed and fixed) or (not value.fixed and free)]

    def get_position(self, time=None):
        return self.params["cenx"].get_value(transformed=False), self.params["ceny"].get_value(
            transformed=False)

    def __init__(self, params):
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise TypeError("Param[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(param), type(Parameter)))
            # TODO: Check if component has a redundant mag or no specified flux ratio
        self.params = {param.name: param for param in params}


# TODO: Store position and/or astrometry
class Component(object, metaclass=ABCMeta):
    """
        A component of a source, which can be extended or not. This abstract class only stores fluxes.
        TODO: Implement shape model or at least alternative angle/axis ratio implementations (w/boxiness)
        It could be isophotal twisting, a 3D shape, etc.
    """
    optional = ["cenx", "ceny"]

    @abstractmethod
    def get_profiles(self, flux_by_band, engine, cenx, ceny, params=None, engineopts=None):
        """
            flux_by_band is a dict of bands with item flux in a linear scale or None if components independent
            Return is dict keyed by band with lists of engine-dependent profiles.
            galsim are GSObjects
            libprofit are dicts with a "profile" key
        """
        pass

    @abstractmethod
    def get_parameters(self, free=True, fixed=True):
        pass

    @abstractmethod
    def is_gaussian(self):
        pass

    @abstractmethod
    def is_gaussian_mixture(self):
        pass

    def __init__(self, fluxes, name=""):
        for i, param in enumerate(fluxes):
            if not isinstance(param, FluxParameter):
                raise TypeError(
                    "Component flux[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, str(type(param)), str(type(FluxParameter)))
                )
        self.fluxes = fluxes
        self.fluxes_dict = {flux.band: flux for flux in self.fluxes}
        self.name = name


class EllipseParameters(Ellipse):
    def get_parameters(self, free=True, fixed=True):
        return [value for value in [self.sigma_x, self.sigma_y, self.rho] if
                (value.fixed and fixed) or (not value.fixed and free)]

    def get_profile(self):
        return {
            'sigma_x': self.get_sigma_x(),
            'sigma_y': self.get_sigma_y(),
            'rho': self.get_rho(),
            'params': {
                'sigma_x': self.sigma_x,
                'sigma_y': self.sigma_y,
                'rho': self.rho,
            }
        }

    def get(self):
        return self.get_sigma_x(), self.get_sigma_y(), self.get_rho()

    def get_radius(self):
        return sqrt(self.get_sigma_x()**2 + self.get_sigma_y()**2)

    def get_sigma_x(self):
        return self.sigma_x.get_value(transformed=False)

    def get_sigma_y(self):
        return self.sigma_y.get_value(transformed=False)

    def get_rho(self):
        return self.rho.get_value(transformed=False)

    def _set_sigma_x(self, x):
        self.sigma_x.set_value(x, transformed=False)

    def _set_sigma_y(self, x):
        self.sigma_y.set_value(x, transformed=False)

    def _set_rho(self, x):
        self.rho.set_value(x, transformed=False)

    def __init__(self, sigma_x, sigma_y, rho):
        isparams = [isinstance(x, Parameter) for x in [sigma_x, sigma_y, rho]]
        if not all(isparams):
            raise TypeError("Not all {} parameters are {} ({})".format(
                self.__class__, Parameter.__class__, [x.__class__ for x in [sigma_x, sigma_y, rho]]
            ))
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        self._check()


class EllipticalComponent(Component):
    axrat_min = 1e-8

    def get_profiles(self, flux_by_band, engine, cenx, ceny, params=None, engineopts=None):
        profile = self.params_ellipse.get_profile()
        if params is not None:
            profile["params"].update(params)
        for modifier in self.params_ellipse.rho.modifiers:
            if modifier.name == "rscale":
                factor = modifier.get_value(transformed=False)
                profile["sigma_x"] *= factor
                profile["sigma_y"] *= factor
                profile["params"]["rscale"] = modifier
        return [profile]

    def get_parameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if
                (value.fixed and fixed) or (not value.fixed and free)] + \
               self.params_ellipse.get_parameters(free=free, fixed=fixed)

    def is_gaussian(self):
        return False

    def is_gaussian_mixture(self):
        return False

    def __init__(self, fluxes, params_ellipse, name=""):
        super().__init__(fluxes, name=name)
        if not isinstance(params_ellipse, EllipseParameters):
            raise TypeError("params_ellipse type {} must be a {}".format(
                type(params_ellipse), EllipseParameters))
        self.params_ellipse = params_ellipse


class EllipticalParametricComponent(EllipticalComponent):
    """
        Class for any profile with a (generalized) ellipse shape.
        TODO: implement boxiness for libprofit; not sure if galsim does generalized ellipses?
    """
    profiles_avail = ["gaussian", "moffat", "sersic"]
    # TODO: Consider adopting gs's flexible methods of specifying re, fwhm, etc.
    mandatory = {
        "gaussian": [],
        "moffat": ["con"],
        "sersic": ["nser"],
    }
    sizenames = {
        "gaussian": "hwhm",
        "moffat": "hwhm",
        "sersic": "re",
    }

    ENGINES = ["galsim", "libprofit"]
    splinelogeps = 1e-12

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    # TODO: Should the parameters be stored as a dict? This method is the only reason why it's useful now
    def is_gaussian(self):
        return (
            self.profile == "gaussian"
            or (self.profile == "sersic" and self.parameters["nser"].get_value() == 0.5)
            or (self.profile == "moffat" and np.isinf(self.parameters["con"].get_value()))
        )

    def is_gaussian_mixture(self):
        return self.is_gaussian()

    def get_parameters(self, free=True, fixed=True):
        return super().get_parameters(free=free, fixed=fixed) + \
            [value for value in self.parameters.values() if
                (value.fixed and fixed) or (not value.fixed and free)]

    def get_profiles(self, flux_by_band, engine, cenx, ceny, params=None, engineopts=None):
        """
        :param flux_by_band: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param engineopts: Dict of engine options
        :return: Dict by band with list of profiles
        """
        self._checkengine(engine)
        is_gaussian = self.is_gaussian()
        if params is None:
            params = {}

        for band in flux_by_band.keys():
            if band not in self.fluxes_dict:
                raise ValueError(
                    "Asked for EllipticalComponent (profile={:s}, name={:s}) model for band={:s} not in "
                    "bands with fluxes {}".format(self.profile, self.name, band, self.fluxes_dict))

        profiles = {}
        profile_base = super().get_profiles(flux_by_band, engine, cenx, ceny, params, engineopts)[0]
        skip_covar = engineopts is not None and engineopts.get("get_profile_skip_covar", False)
        radius, axrat, ang = (np.Inf, None, None) if skip_covar else \
            mpfgauss.covar_to_ellipse(self.params_ellipse)
        for band in flux_by_band.keys():
            flux_param_band = self.fluxes_dict[band]
            profile = profile_base.copy()
            flux = flux_param_band.get_value(transformed=False)
            if flux_param_band.is_fluxratio:
                if not 0 <= flux <= 1:
                    raise ValueError(f"flux ratio not 0 <= {flux} <= 1")
                flux_value = flux*flux_by_band[band]
                flux_by_band[band] *= (1.0-flux)
                flux = flux_value
            if not skip_covar:
                profile["axrat"] = axrat
                profile["ang"] = ang
            profile['can_do_fit_leastsq'] = self.profile == "gaussian"
            for param in self.parameters.values():
                profile[param.name] = param.get_value(transformed=False)

            if not flux >= 0:
                raise ValueError(f"flux {flux}!>=0")
            if not skip_covar:
                if not 0 < profile["axrat"] <= 1:
                    if profile["axrat"] <= __class__.axrat_min:
                        profile["axrat"] = __class__.axrat_min
                    else:
                        raise ValueError("axrat {} ! >0 and <=1".format(profile["axrat"]))

            cens = {"cenx": cenx, "ceny": ceny}

            # Does this profile have a non-zero size?
            # TODO: Review cutoff - it can't be zero for galsim or it will request huge FFTs
            resolved = not radius < 0*(engine == "galsim")
            for key, value in cens.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = np.copy(value)
            if resolved:
                if engine == "galsim":
                    axrat = profile["axrat"]
                    axrat_sqrt = sqrt(axrat)
                    gsparams = get_gsparams(engineopts)
                    if is_gaussian:
                        profile_gs = gs.Gaussian(
                            flux=flux, fwhm=2*radius*axrat_sqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "sersic":
                        if profile["nser"] < 0.3 or profile["nser"] > 6.2:
                            self.logger.warning("Sersic n {:.3f} not >= 0.3 and <= 6.2; "
                                                "GalSim could fail.".format(profile["nser"]))
                        profile_gs = gs.Sersic(
                            flux=flux, n=profile["nser"],
                            half_light_radius=radius*axrat_sqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "moffat":
                        profile_gs = gs.Moffat(
                            flux=flux, beta=profile["con"],
                            fwhm=2*radius*axrat_sqrt,
                            gsparams=gsparams
                        )
                    profile = {
                        "profile_gs": profile_gs,
                        "shear": gs.Shear(q=axrat, beta=profile["ang"]*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    }
                elif engine == "libprofit":
                    profile[__class__.sizenames[self.profile]] = radius
                    profile["flux"] = flux
                    # TODO: Review this. It might not be a great idea because Sersic != Moffat integration
                    # libprofit should be able to handle Moffats with infinite con
                    if self.profile != "sersic" and self.is_gaussian():
                        profile["profile"] = "sersic"
                        profile["nser"] = 0.5
                        if self.profile == "moffat":
                            del profile["con"]
                        elif self.profile != "gaussian":
                            raise RuntimeError("No implentation for turning profile {} into gaussian".format(
                                profile["profile"]))
                    else:
                        profile["profile"] = self.profile
                else:
                    raise ValueError("Unimplemented rendering engine {:s}".format(engine))
            else:
                profile["flux"] = flux
            # This profile is part of a point source *model*
            profile["pointsource"] = False
            profile["resolved"] = resolved
            profile["param_flux"] = flux_param_band
            profiles[band] = profile
        return [profiles]

    @classmethod
    def _checkparameters(cls, parameters, profile):
        mandatory = {param: False for param in EllipticalParametricComponent.mandatory[profile]}
        name_params_needed = mandatory.keys()
        name_params = [param.name for param in parameters]
        errors = []
        # Not as efficient as early exit if true but oh well
        if len(name_params) > len(set(name_params)):
            errors.append("Parameters array not unique")
        # Check if these parameters are known (in mandatory)
        for param in parameters:
            if isinstance(param, FluxParameter):
                errors.append("Param {:s} is {:s}, not {:s}".format(param.name, type(FluxParameter),
                                                                    type(Parameter)))
            if param.name in name_params_needed:
                mandatory[param.name] = True
            elif param.name not in Component.optional:
                errors.append("Unknown param {:s}".format(param.name))

        for name_param, found in mandatory.items():
            if not found:
                errors.append("Missing mandatory param {:s}".format(name_param))
        if errors:
            error_msg = "Errors validating params of component (profile={}):\n".format(profile) + \
                       "\n".join(errors) + "\nPassed params:" + str(parameters)
            raise ValueError(error_msg)

    def __init__(self, fluxes, params_ellipse, name="", profile="sersic", parameters=None):
        super().__init__(fluxes, params_ellipse, name=name)
        if profile not in __class__.profiles_avail:
            raise ValueError("Profile type={:s} not in available: ".format(profile) + str(
                __class__.profiles_avail))
        self._checkparameters(parameters, profile)
        self.profile = profile
        self.parameters = {param.name: param for param in parameters}


class PointSourceProfile(Component):

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in cls.ENGINES:
            raise ValueError(f"Unknown {type(cls)} rendering engine {engine}")

    @classmethod
    def is_gaussian(cls):
        return True

    @classmethod
    def is_gaussian_mixture(cls):
        return True

    def get_parameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if
                (value.fixed and fixed) or (not value.fixed and free)]

    # TODO: default PSF could be unit image?
    def get_profiles(self, flux_by_band, engine, cenx, ceny, params=None, psf=None):
        """
        Get engine-dependent representations of profiles to be rendered.

        :param flux_by_band: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param psf: A PSF (required, despite the default)
        :return: List of engine-dependent profiles
        """
        self._checkengine(engine)
        if not isinstance(psf, PSF):
            raise TypeError("psf type {} must be a {}".format(type(psf), PSF))

        for band in flux_by_band.keys():
            if band not in self.fluxes_dict:
                raise ValueError(
                    "Called PointSourceProfile (name={:s}) get_profiles() for band={:s} not in "
                    "bands with fluxes {}", self.name, band, self.fluxes_dict)

        # TODO: Think of the best way to do this
        # TODO: Ensure that this is getting copies - it isn't right now
        profiles = psf.model.get_profiles(engine=engine, bands=flux_by_band.keys())
        for band in flux_by_band.keys():
            flux = self.fluxes_dict[band].get_value(transformed=False)
            for profile in profiles[band]:
                if engine == "galsim":
                    profile["profile"].flux *= flux
                elif engine == "libprofit":
                    profile["flux"] = flux
                profile["pointsource"] = True
            else:
                raise ValueError("Unimplemented PointSourceProfile rendering engine {:s}".format(engine))
        return profiles

    def __init__(self, fluxes, name=""):
        super().__init__(fluxes=fluxes, name=name)


class Background(Component):

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in cls.ENGINES:
            raise ValueError(f"Unknown {type(cls)} rendering engine {engine}")

    @classmethod
    def is_gaussian(cls):
        return True

    @classmethod
    def is_gaussian_mixture(cls):
        return True

    def get_parameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if
                (value.fixed and fixed) or (not value.fixed and free)]

    def get_profiles(self, flux_by_band, engine, cenx, ceny, params=None, engineopts=None):
        """
        Get engine-dependent representations of profiles to be rendered.

        :param flux_by_band: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates. Ignored.
        :param ceny: Y center in image coordinates. Ignored.
        :return: List of engine-dependent profiles
        """
        self._checkengine(engine)

        profile = {"background": True}
        for band in flux_by_band.keys():
            if band not in self.fluxes_dict:
                raise ValueError(
                    "Called Background (name={:s}) get_profiles() for band={:s} not in "
                    "bands with fluxes {}", self.name, band, self.fluxes_dict)
            flux = self.fluxes_dict[band].get_value(transformed=False)
            if engine == "libprofit":
                profile[band] = {
                    "background": True,
                    "can_do_fit_leastsq": True,
                    "flux": flux,
                    "order_rev": self.band_order_rev[band],
                    "param_flux": self.fluxes_dict[band],
                    "params": {},
                    "pointsource": False,
                    "profile": "background",
                    "resolved": False,
                }
            elif engine == "galsim":
                pass
            else:
                raise ValueError("Unimplemented Background rendering engine {:s}".format(engine))
        return [profile]

    def __init__(self, fluxes, name=""):
        super().__init__(fluxes=fluxes, name=name)
        self.band_order_rev = {band: idx for idx, band in
                               enumerate(reversed(list(self.fluxes_dict.keys())))}


# TODO: This class needs loads of sanity checks and testing
class Parameter:
    """
        A parameter with all the info about itself that you would need when fitting.
    """
    def get_transform_derivative(self, return_finite_diff=False, dx=None, verify=False,
                                 verify_derivative_abs_max=1e6, dx_ratios=None, **kwargs):
        """
        :param return_finite_diff: bool; return a finite difference derivative if the transform doesn't have
            a derivative function.
        :param dx: float; a (small) delta value to use for finite differencing.
        :param verify: bool; verify the value by comparing to finite differencing?
        :param verify_derivative_abs_max: float; x value where verification skipped if np.abs(derivative) > x
        :param dx_ratios: float[]; iterable of ratios to set dx for finite differencing (signed),
            where dx = value*ratio (not transformed). Only used if dx is None.
            Default: [1e-4, -1e-4, 1e-6, 1e-6, 1e-8, -1e-8, 1e-10, -1e-10, 1e-12, -1e-12, 1e-14, -1e-14]
            Verification will test all values in order until at least one passes.
        :param kwargs: dict; args to pass to np.isclose when comparing derivative to finite difference e.g.
            atol, rtol
        :return:
        """
        if self.transform is None:
            return None
        if self.transform.derivative is None:
            if not return_finite_diff:
                return None
            else:
                value = self.get_value(transformed=False)
                if dx is None:
                    if dx_ratios is None:
                        raise ValueError('Finite difference derivative for parameter {} must specify one of'
                                         'dx or dx_ratios'.format(self))
                    dx = dx_ratios[0]*value
                return (self.transform(value + dx) - self.get_value(transformed=True))/dx
        value = self.get_value(transformed=False)
        derivative = self.transform.derivative(value)
        if verify:
            value_transformed = self.get_value(transformed=True)
            # Skip testing finite differencing if the derivative is very large
            # This might happen e.g. near the limits of the transformation
            # TODO: Check if better finite differencing is possible for large values
            if verify_derivative_abs_max is None:
                verify_derivative_abs_max = 1e8
            is_close = np.abs(derivative) > verify_derivative_abs_max
            if not is_close:
                if dx_ratios is None:
                    dx_ratios = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
                for ratio in dx_ratios:
                    dx = value*ratio
                    fin_diff = (self.transform(value + dx) - value_transformed)/dx
                    if not np.isfinite(fin_diff):
                        fin_diff = -(self.transform(value - dx) - value_transformed)/dx
                    is_close = np.isclose(derivative, fin_diff, **kwargs)
                    if is_close:
                        break
            if not is_close:
                raise RuntimeError(
                    'Param {}(t({})={}) derivative={:.8e} != last '
                    'finite diff.={:8e} with dx={} dx_abs_max={}'.format(
                        self.name, value, value_transformed, derivative, fin_diff, dx,
                        verify_derivative_abs_max
                    ))
        return derivative

    def get_value(self, transformed=False):
        if transformed and not self.transformed:
            return self.transform.transform(self.value)
        elif not transformed and self.transformed:
            return self.transform.reverse(self.value)
        return np.float64(self.value)

    def get_limits(self, transformed=False):
        lower = self.limits.lower
        upper = self.limits.upper
        if transformed and not self.limits.transformed:
            lower = self.transform.transform(lower)
            upper = self.transform.transform(upper)
        elif not transformed and self.limits.transformed:
            lower = self.transform.reverse(lower)
            upper = self.transform.reverse(upper)
        return lower, upper

    def set_value(self, value, transformed=False):
        if value is None:
            raise RuntimeError(f"Tried to set Parameter {self} to None")
        if transformed == self.limits.transformed:
            if self.limits is not None:
                value = self.limits.clip(value)
        else:
            if transformed:
                value = self.transform.reverse(value)
            else:
                value = self.transform.transform(value)
            if self.limits is not None:
                value = self.limits.clip(value)
        try:
            if np.isnan(value):
                raise RuntimeError(f"Tried to set Parameter {self} to nan")
            self.value = value
            # TODO: Error checking, etc. There are probably better ways to do this
            for param in self.inheritors:
                param.value = self.value
        except Exception:
            print(f"Failed to set {self} to value={value}")
            raise

    def __repr__(self):
        attrs = ', '.join([
            f'{var}={value!r}' for var, value in dict(
                name=self.name,
                value=self.value,
                unit=self.unit,
                limits=self.limits,
                transform=self.transform,
                transformed=self.transformed,
                fixed=self.fixed,
                inheritors=self.inheritors,
                modifiers=self.modifiers,
            ).items()
        ])
        return f'Parameter({attrs})'

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)

    def __init__(self, name, value, unit="", limits=None, transform=None, transformed=True,
                 fixed=False, inheritors=None, modifiers=None):
        if transform is None:
            transform = Transform()
        if limits is None:
            limits = Limits(transformed=transformed)
        if limits.transformed != transformed:
            raise ValueError("limits.transformed={} != Param[{:s}].transformed={}".format(
                limits.transformed, name, transformed
            ))
        self.fixed = fixed
        self.name = name
        self.unit = unit
        self.limits = limits
        self.transform = transform
        self.transformed = transformed
        # List of parameters that should inherit values from this one
        self.inheritors = [] if inheritors is None else inheritors
        # List of parameters that can modify this parameter's value - user decides what to do with them
        self.modifiers = [] if modifiers is None else modifiers
        self.value = None
        self.set_value(value, transformed=transformed)


class FluxParameter(Parameter):
    """
        A flux, magnitude or flux ratio, all of which one could conceivably fit.
        TODO: name seems a bit redundant, but I don't want to commit to storing the band as a string now
    """
    def __str__(self):
        attrs = ', '.join([
            f'{var}={value}' for var, value in dict(
                band=self.band,
                fixed=self.fixed,
                limits=self.limits,
                inheritors=self.inheritors,
                is_fluxratio=self.is_fluxratio,
                modifiers=self.modifiers,
                transform=self.transform,
                transformed=self.transformed,
                value=self.value,
            ).items()
        ])
        return f'FluxParameter (name={self.name}):({attrs})'

    def __init__(self, band, name, value, unit, limits, transform=None, transformed=True,
                 fixed=None, is_fluxratio=None):
        if is_fluxratio is None:
            is_fluxratio = False
        Parameter.__init__(self, name=name, value=value, unit=unit, limits=limits, transform=transform,
                           transformed=transformed, fixed=fixed)
        self.band = band
        self.is_fluxratio = is_fluxratio
