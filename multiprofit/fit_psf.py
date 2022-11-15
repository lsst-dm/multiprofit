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

from abc import ABC, abstractmethod
import astropy
from astropy.table import Table
import astropy.units as u
import logging
import gauss2d as g2
import gauss2d.fit as g2f
import numpy as np
import time
from typing import Any, Mapping, Type

import lsst.pex.config as pexConfig

from multiprofit.fit_catalog import CatalogExposureABC, CatalogFitterConfig, ColumnInfo
from multiprofit.modeller import LinearGaussians, make_psfmodel_null, Modeller
from multiprofit.psfmodel_utils import make_psf_source


class CatalogExposurePsfABC(CatalogExposureABC):
    @abstractmethod
    def get_psf_image(self, source: astropy.table.Row | Mapping[str, Any]) -> np.array:
        """Return a normalized, centered, odd-sized PSF image array."""


class CatalogPsfFitterConfig(CatalogFitterConfig):
    """Configuration for MultiProFit PSF image fitter."""
    sigmas = pexConfig.ListField[float](default=[1.5, 3], doc="Number of Gaussian components in PSF")

    def rebuild_psfmodel(self, params: astropy.table.Row | Mapping[str, Any]) -> g2f.PsfModel:
        """Rebuild a PSF model for a single source.

        Rebuilding currently means creating a new PsfModel object with all
        parameter values set to best fit values at the centroid of the source.

        Parameters
        ----------
        params : astropy.table.Row | typing.Mapping[str, typing.Any]
            A mapping with parameter values for the best-fit PSF model at the
            centroid of a single source.

        Returns
        -------
        psfmodel : `g2f.PsfModel`
            The rebuilt PSF model.
        """
        n_gaussians = len(self.sigmas)
        idx_gauss_max = n_gaussians - 1
        sigma_xs = [0.]*n_gaussians
        sigma_ys = [0.]*n_gaussians
        rhos = [0.]*n_gaussians
        fracs = [1.]*n_gaussians

        for idx in range(n_gaussians):
            sigma_xs[idx] = params[f"{self.prefix_column}comp{idx + 1}_sigma_x"]
            sigma_ys[idx] = params[f"{self.prefix_column}comp{idx + 1}_sigma_y"]
            rhos[idx] = params[f"{self.prefix_column}comp{idx + 1}_rho"]
            if idx != idx_gauss_max:
                fracs[idx] = params[f"{self.prefix_column}comp{idx + 1}_fluxfrac"]
        return g2f.PsfModel(
            make_psf_source(sigma_xs=sigma_xs, sigma_ys=sigma_ys, rhos=rhos, fracs=fracs).components
        )

    def schema(self) -> list[ColumnInfo]:
        """Return the schema as an ordered list of columns."""
        schema = super().schema()
        n_gaussians = len(self.sigmas)
        idx_gauss_max = n_gaussians - 1

        for idx_gauss in range(n_gaussians):
            prefix_comp = f"comp{idx_gauss + 1}_"
            columns_comp = [
                ColumnInfo(key=f'{prefix_comp}sigma_x', dtype='f8', unit=u.pix),
                ColumnInfo(key=f'{prefix_comp}sigma_y', dtype='f8', unit=u.pix),
                ColumnInfo(key=f'{prefix_comp}rho', dtype='f8', unit=u.pix),
            ]
            if idx_gauss != idx_gauss_max:
                columns_comp.append(ColumnInfo(key=f'{prefix_comp}fluxfrac', dtype='f8'))
            schema.extend(columns_comp)

        return schema

    def setDefaults(self):
        self.prefix_column = "mpf_psf_"


class CatalogPsfFitter:
    """Fit a Gaussian mixture model to a pixelated PSF image.

    Parameters
    ----------
    modeller : `multiprofit.Modeller`
        A Modeller instance to use for fitting.
    errors_expected : dict[Type[Exception], str]
        A dictionary keyed by an Exception type, with a string value of the
        flag column key to assign if this Exception is raised.

    Notes
    -----
    Any exceptions raised and not in errors_expected will be logged in a
    generic unknown_flag failure column.
    """
    def __init__(
        self,
        modeller: Modeller = None,
        errors_expected: dict[Type[Exception], str] = None,
    ):
        if modeller is None:
            modeller = Modeller()
        if errors_expected is None:
            errors_expected = {}
        self.errors_expected = errors_expected
        self.modeller = modeller

    @staticmethod
    def _get_data(img_psf: np.array, gain: float = 1e5) -> g2f.Data:
        """Build fittable Data from a normalized PSF image.

        Parameters
        ----------
        img_psf : `numpy.array`
            A normalized PSF image array.
        gain : float
            The number of counts in the image, used as a multiplicative
            factor for the inverse variance.

        Returns
        -------
        data : gauss2d.fit.Data
            A Data object that can be passed to a Model(ler).
        """
        # TODO: Improve these arbitrary definitions
        # Look at S/N of PSF stars?
        background = np.std(img_psf[img_psf < 2*np.abs(np.min(img_psf))])
        # Hacky fix; PSFs shouldn't have negative values but often do
        min_psf = np.min(img_psf)
        if not (background > -min_psf):
            background = -1.1 * min_psf
        img_sig_inv = np.sqrt(gain / (img_psf + background))
        return g2f.Data([
            g2f.Observation(
                channel=g2f.Channel.NONE,
                image=g2.ImageD(img_psf),
                sigma_inv=g2.ImageD(img_sig_inv),
                mask_inv=g2.ImageB(np.ones_like(img_psf)),
            )
        ])

    @staticmethod
    def _get_logger() -> logging.Logger:
        """Return a suitably-named and configured logger."""
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.level = logging.INFO

        return logger

    def fit(
        self,
        catexp: CatalogExposurePsfABC,
        config: CatalogPsfFitterConfig = None,
        logger: logging.Logger = None,
        **kwargs
    ) -> astropy.table.Table:
        """Fit a PSF models with MultiProFit.

        Each source has its PSF fit with a configureable Gaussian mixture PSF
        model, given a pixellated PSF image from the CatalogExposure.

        Parameters
        ----------
        catexp : `CatalogExposurePsfABC`
            An exposure to fit a model PSF at the position of all
            sources in the corresponding catalog.
        config: `CatalogPsfFitterConfig`
            Configuration settings for fitting and output.
        logger : `logging.Logger`
            The logger. Defaults to calling `_getlogger`.
        **kwargs
            Additional keyword arguments to pass to self.modeller.

        Returns
        -------
        catalog : `astropy.Table`
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        if config is None:
            config = CatalogPsfFitterConfig()
        if logger is None:
            logger = CatalogPsfFitter._get_logger()

        errors_expected = set(self.errors_expected.values())
        n_errors_expected = len(errors_expected)
        if n_errors_expected != len(errors_expected):
            raise ValueError(f"{self.errors_expected=} has duplicate values; they must be unique")
        if n_errors_expected != len(config.flag_errors):
            raise ValueError(f"len({self.errors_expected=}) != len({config.flag_errors=})")

        model_source = make_psf_source(sigma_xs=config.sigmas)
        n_gaussians = len(config.sigmas)
        params = tuple({x: None for x in model_source.parameters([], g2f.ParamFilter(fixed=False))})
        filter_flux = g2f.ParamFilter(nonlinear=False, channel=g2f.Channel.NONE)
        # Make an ordered set
        flux_total = tuple({x: None for x in model_source.parameters(paramfilter=filter_flux)}.keys())
        if len(flux_total) != 1:
            raise RuntimeError(f"len({flux_total=}) != 1; PSF model is badly-formed")
        flux_total = flux_total[0]
        filter_flux = g2f.ParamFilter(linear=False, channel=g2f.Channel.NONE, fixed=False)
        # TODO: Remove isinstance when channel filtering is fixed
        fluxfracs = tuple({x: None for x in model_source.parameters(paramfilter=filter_flux)
                           if isinstance(x, g2f.ProperFractionParameterD)}.keys())
        if len(fluxfracs) != (n_gaussians - 1):
            raise RuntimeError(f"len({fluxfracs=}) != {(n_gaussians - 1)=}; PSF model is badly-formed")
        gaussian = model_source.components[0]
        cenx, ceny = gaussian.parameters()[:2]
        if not (isinstance(cenx, g2f.CentroidXParameterD) and isinstance(ceny, g2f.CentroidYParameterD)):
            raise RuntimeError(f"{cenx=}, {ceny=} have unexpected non-centroid types")
        model_psf = make_psfmodel_null()

        catalog = catexp.get_catalog()
        n_rows = len(catalog)
        range_idx = range(n_rows)

        columns = config.schema()
        keys = [column.key for column in columns]
        prefix = config.prefix_column
        idx_flag_first = keys.index("unknown_flag")
        idx_var_first = keys.index("cen_x")
        columns_write = [f"{prefix}{col.key}" for col in columns[idx_var_first:]]
        dtypes = [(f'{prefix if col.key != "id" else ""}{col.key}', col.dtype) for col in columns]
        meta = {'config': config.toDict()}
        results = Table(data=np.full(n_rows, np.nan, dtype=dtypes), units=[x.unit for x in columns],
                        meta=meta)
        # Set nan-default flags to False instead
        for flag in columns[idx_flag_first:idx_var_first]:
            results[flag.key] = False

        # dummy size for first iteration
        size = 0

        for idx in range_idx:
            time_init = time.process_time()
            row = results[idx]
            source = catalog[idx]
            id_source = source[config.column_id]
            row[config.column_id] = id_source

            try:
                img_psf = catexp.get_psf_image(source)
                cenx.value, ceny.value = (x/2. for x in img_psf.shape[::-1])
                # Caches the jacobian residual if the kernel size is unchanged
                if img_psf.size != size:
                    jacobian, residual = None, None
                data = CatalogPsfFitter._get_data(img_psf)
                size = img_psf.size
                model = g2f.Model(data=data, psfmodels=[model_psf], sources=[model_source])

                if config.fit_linear:
                    flux_total.fixed = False
                    gaussians_linear = LinearGaussians.make(model_source, is_psf=True)
                    flux_total.fixed = True
                    result = self.modeller.fit_gaussians_linear(gaussians_linear, data[0])
                    result = list(result.values())[0]
                    # Re-normalize fluxes (hopefully close already)
                    result = result*np.array([
                        x[0].at(0).integral.value for x in gaussians_linear.gaussians_free
                    ])
                    result /= np.sum(result)
                    for idx_param, param in enumerate(fluxfracs):
                        param.value = result[idx_param]
                        result /= np.sum(result[idx_param + 1:])
                result_full = self.modeller.fit_model(
                    model, jacobian=jacobian, residual=residual, **kwargs
                )
                residual, jacobian = result_full.residual, result_full.jacobian
                results[f"{prefix}n_iter"][idx] = result_full.n_eval_func
                results[f"{prefix}time_eval"][idx] = result_full.time_eval
                results[f"{prefix}time_fit"][idx] = result_full.time_run
                results[f"{prefix}chisq_red"][idx] = np.sum(result_full.residual**2)/size

                for param, value, column in zip(params, result_full.params_best, columns_write):
                    param.value_transformed = value
                    results[column][idx] = param.value

                results[f"{prefix}time_full"][idx] = time.process_time() - time_init
            except Exception as e:
                column = self.errors_expected.get(e.__class__, "")
                if column:
                    row[f"{prefix}{column}"] = True
                    logger.info(f"{id_source=} PSF fit failed with not unexpected exception={e}")
                else:
                    row[f"{prefix}unknown_flag"] = True
                    logger.info(f"{id_source=} PSF fit failed with unexpected exception={e}")

        return results
