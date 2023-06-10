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
from dataclasses import dataclass, field
import logging
import gauss2d.fit as g2f
import numpy as np
import time
from typing import Any, Mapping, Sequence, Type

import lsst.pex.config as pexConfig

from .componentconfig import SersicConfig
from .fit_catalog import CatalogExposureABC, CatalogFitterConfig, ColumnInfo
from .modeller import FitInputsDummy, LinearGaussians, Modeller
from .transforms import transforms_ref


class CatalogExposureSourcesABC(CatalogExposureABC):
    @property
    def band(self) -> str:
        """Return the name of the exposure's passband (e.g. 'r')."""
        return self.channel.name

    # Note: not named band because that's usually a string
    @property
    @abstractmethod
    def channel(self) -> g2f.Channel:
        """Return the exposure's associated channel object."""

    @abstractmethod
    def get_psfmodel(self, params: Mapping[str, Any]) -> g2f.PsfModel:
        """Get the PSF model for a given source row.

        Parameters
        ----------
        params : Mapping[str, Any]
            A mapping with parameter values for the best-fit PSF model at the
            centroid of a single source.

        Returns
        -------
        psfmodel : `gauss2d.fit.PsfModel`
            A PsfModel object initialized with the best-fit parameters.
        """

    @abstractmethod
    def get_source_observation(self, source: Mapping[str, Any]) -> g2f.Observation:
        """Get the Observation for a given source row.

        Parameters
        ----------
        source : Mapping[str, Any]
            A mapping with any values needed to retrieve an observation for a
            single source.

        Returns
        -------
        observation : `gauss2d.fit.Observation`
            An Observation object with suitable data for fitting parametric
            models of the source.
        """


class CatalogSourceFitterConfig(CatalogFitterConfig):
    """Configuration for the MultiProFit profile fitter."""
    fit_cenx = pexConfig.Field[bool](default=True, doc="Fit x centroid parameter")
    fit_ceny = pexConfig.Field[bool](default=True, doc="Fit y centroid parameter")
    n_pointsources = pexConfig.Field[int](default=0, doc="Number of central point source components")
    prior_cenx_stddev = pexConfig.Field[float](default=0,
                                               doc="Prior std. dev. on x centroid (ignored if not >0)")
    prior_ceny_stddev = pexConfig.Field[float](default=0,
                                               doc="Prior std. dev. on y centroid (ignored if not >0)")

    # TODO: Verify that component names don't clash
    sersics = pexConfig.ConfigDictField(
        default={},
        doc="Sersic components",
        itemtype=SersicConfig,
        keytype=str,
        optional=False,
    )
    unit_flux = pexConfig.Field[str](default="", doc="Flux unit")

    def make_model_data(
        self,
        idx_source,
        catexps: list[CatalogExposureSourcesABC],
        model_priors: tuple[g2f.Source, list[g2f.Prior]] = None,
    ):
        if model_priors is None:
            model_source, priors, *_ = self.make_source([catexp.channel for catexp in catexps])
        else:
            model_source, priors = model_priors
        n_catexps = len(catexps)
        observations = [None]*n_catexps
        psfmodels = [None]*n_catexps

        for idx_catexp in range(n_catexps):
            catexp = catexps[idx_catexp]
            source = catexp.get_catalog()[idx_source]
            observations[idx_catexp] = catexp.get_source_observation(source)
            psfmodel = catexp.get_psfmodel(source)
            for param in psfmodel.parameters():
                param.fixed = True
            psfmodels[idx_catexp] = psfmodel

        data = g2f.Data(observations)
        model = g2f.Model(data=data, psfmodels=psfmodels, sources=[model_source], priors=priors)
        return model, data, psfmodels

    def make_source(
        self,
        channels: list[g2f.Channel],
    ) -> tuple[g2f.Source, list[g2f.Prior], g2f.LimitsD, g2f.LimitsD]:
        limits_x = g2f.LimitsD(min=0, max=np.Inf)
        limits_y = g2f.LimitsD(min=0, max=np.Inf)
        cenx = g2f.CentroidXParameterD(value=0, limits=limits_x, fixed=not self.fit_cenx)
        ceny = g2f.CentroidYParameterD(value=0, limits=limits_y, fixed=not self.fit_ceny)
        centroid = g2f.CentroidParameters(cenx, ceny)
        priors = []
        n_sersics = len(self.sersics)
        components = [None]*(self.n_pointsources + n_sersics)
        for idx in range(self.n_pointsources):
            components[idx] = g2f.GaussianComponent(
                centroid=centroid,
                ellipse=g2f.GaussianParametricEllipse(
                    sigma_x=g2f.SigmaXParameterD(0, fixed=True),
                    sigma_y=g2f.SigmaYParameterD(0, fixed=True),
                    rho=g2f.RhoParameterD(0, fixed=True),
                ),
                integral=g2f.LinearIntegralModel(
                    {channel: g2f.IntegralParameterD(1.0, transform=transforms_ref['log10'])
                     for channel in channels}
                ),
            )
        idx = self.n_pointsources
        for sersic in self.sersics.values():
            component, priors_comp = sersic.make_component(centroid=centroid, channels=channels)
            components[idx] = component
            priors.extend(priors_comp)
            idx += 1
        if self.prior_cenx_stddev > 0 and np.isfinite(self.prior_cenx_stddev):
            priors.append(
                g2f.GaussianPrior(centroid.x_param_ptr, 0, self.prior_cenx_stddev)
            )
        if self.prior_ceny_stddev > 0 and np.isfinite(self.prior_ceny_stddev):
            priors.append(
                g2f.GaussianPrior(centroid.y_param_ptr, 0, self.prior_ceny_stddev)
            )
        return g2f.Source(components), priors, limits_x, limits_y

    def schema(
        self,
        bands: list[str] = None,
    ) -> list[ColumnInfo]:
        if bands is None or not (len(bands) > 0):
            raise ValueError("PSF CatalogSourceFitter must provide at least one band")
        columns = super().schema(bands)
        for idx in range(self.n_pointsources):
            prefix_comp = f"ps{idx + 1}_"
            for band in bands:
                columns.append(ColumnInfo(key=f'{prefix_comp}{band}_flux', dtype='f8'))

        for idx, name_comp in enumerate(self.sersics.keys()):
            prefix_comp = f"{name_comp}_"
            columns_comp = [
                ColumnInfo(key=f'{prefix_comp}sigma_x', dtype='f8', unit=u.pix),
                ColumnInfo(key=f'{prefix_comp}sigma_y', dtype='f8', unit=u.pix),
                ColumnInfo(key=f'{prefix_comp}rho', dtype='f8', unit=u.pix),
            ]
            for band in bands:
                columns_comp.append(
                    ColumnInfo(key=f'{prefix_comp}{band}_flux', dtype='f8', unit=u.Unit(self.unit_flux))
                )
            columns.extend(columns_comp)

        return columns


@dataclass(kw_only=True)
class CatalogSourceFitterABC(ABC):
    """ Fit a Gaussian mixture source model to an image with a PSF model.

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

    modeller: Modeller = field(default_factory=Modeller)
    errors_expected: dict[Type[Exception], str] = field(default_factory=dict)

    @staticmethod
    def _get_logger():
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.level = logging.INFO

        return logger

    def fit(
        self,
        catalog_multi: Sequence,
        # TODO: Need to support catexps with band but no channel
        catexps: list[CatalogExposureSourcesABC],
        config: CatalogSourceFitterConfig = None,
        logger: logging.Logger = None,
        **kwargs
    ) -> astropy.table.Table:
        """Fit PSF-convolved source models with MultiProFit.

        Each source has a single PSF-convolved model fit, given PSF model
        parameters from a catalog, and a combination of initial source
        model parameters and a deconvolved source image from the
        CatalogExposureSources.

        Parameters
        ----------
        catalog_multi : typing.Sequence
            A multi-band source catalog to fit a model to.
        catexps : list[`CatalogExposureSourcesABC`]
            A list of (source and psf) catalog-exposure pairs
        config: `CatalogSourceFitterConfig`
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
            config = CatalogSourceFitterConfig()
        if logger is None:
            logger = self._get_logger()

        if len(self.errors_expected) != len(config.flag_errors):
            raise ValueError(f"{self.errors_expected=} keys not same len as {config.flag_errors=}")
        errors_bad = {}
        errors_recast = {}
        for error_name, error_type in self.errors_expected.items():
            if error_type in errors_recast:
                errors_bad[error_name] = error_type
            else:
                errors_recast[error_type] = error_name
        if errors_bad:
            raise ValueError(f"{self.errors_expected=} keys contain duplicates from {config.flag_errors=}")

        n_catexps = len(catexps)

        channels = {}
        for catexp in catexps:
            try:
                channel = catexp.channel
            except AttributeError:
                band = catexp.band
                if callable(band):
                    band = band()
                channel = g2f.Channel.get(band)
            if channel not in channels:
                channels[channel.name] = channel

        model_source, priors, limits_x, limits_y = config.make_source(channels=list(channels.values()))
        params = tuple({x: None for x in model_source.parameters([], g2f.ParamFilter(fixed=False))})

        n_rows = len(catalog_multi)
        range_idx = range(n_rows)

        columns = config.schema([channel.name for channel in channels.values()])
        keys = [column.key for column in columns]
        prefix = config.prefix_column
        idx_flag_first = keys.index("unknown_flag")
        idx_var_first = keys.index("cen_x")
        assert idx_var_first > idx_flag_first
        columns_write = [f"{prefix}{col.key}" for col in columns[idx_var_first:]]
        dtypes = [(f'{prefix if col.key != config.column_id else ""}{col.key}', col.dtype) for col in columns]
        meta = {'config': config.toDict()}
        results = Table(data=np.full(n_rows, 0, dtype=dtypes), units=[x.unit for x in columns],
                        meta=meta)

        # Validate that the columns are in the right order
        # assert because this is a logic error if it fails
        for idx in range(idx_flag_first, len(columns)):
            column = columns[idx]
            dtype = results.columns[idx].dtype
            if idx < idx_var_first:
                assert dtype == bool
                assert column.key.endswith("_flag")
            else:
                assert dtype == float

        # dummy size for first iteration
        size = 0
        fitInputs = FitInputsDummy()

        limits_flux = g2f.LimitsD(1e-6, 1e6, 'unreliable flux limits')
        transform_flux = g2f.LogitLimitedTransformD(limits_flux)

        for idx in range_idx:
            time_init = time.process_time()
            row = results[idx]
            source_multi = catalog_multi[idx]
            id_source = source_multi[config.column_id]
            row[config.column_id] = id_source

            try:
                model, data, psfmodels = config.make_model_data(
                    idx_source=idx, model_priors=(model_source, priors), catexps=catexps,
                )
                self.initialize_model(model, source_multi, limits_x=limits_x, limits_y=limits_y)

                # Caches the jacobian residual if the data size is unchanged
                # Note: this will need to change with priors
                # (data should report its own size)
                size_new = np.sum([datum.image.size for datum in data])
                if size_new != size:
                    fitInputs = None
                    size = size_new
                else:
                    fitInputs = fitInputs if not fitInputs.validate_for_model(model) else None

                if config.fit_linear:
                    for idx_catexp, catexp in enumerate(catexps):
                        gaussians_linear = LinearGaussians.make(model_source, channel=catexp.channel)
                        result = self.modeller.fit_gaussians_linear(gaussians_linear, data[idx_catexp],
                                                                    psfmodel=psfmodels[idx_catexp])
                        values = list(result.values())[0]
                        for (_, parameter), value in zip(gaussians_linear.gaussians_free, values):
                            if not (value > 0.01):
                                value = 0.01
                                parameter.transform = transform_flux
                                parameter.limits = limits_flux
                            parameter.value = value

                result_full = self.modeller.fit_model(model, fitinputs=fitInputs, **kwargs)
                fitInputs = result_full.inputs
                results[f"{prefix}n_iter"][idx] = result_full.n_eval_func
                results[f"{prefix}time_eval"][idx] = result_full.time_eval
                results[f"{prefix}time_fit"][idx] = result_full.time_run
                results[f"{prefix}chisq_red"][idx] = np.sum(fitInputs.residual**2)/size

                for param, value, column in zip(params, result_full.params_best, columns_write):
                    param.value_transformed = value
                    results[column][idx] = param.value

                results[f"{prefix}time_full"][idx] = time.process_time() - time_init
            except Exception as e:
                column = self.errors_expected.get(e.__class__, "")
                if column:
                    row[f"{prefix}{column}"] = True
                    logger.info(f"{id_source=} ({idx=}/{n_rows}) fit failed with known exception={e}")
                else:
                    row[f"{prefix}unknown_flag"] = True
                    logger.info(f"{id_source=} ({idx=}/{n_rows}) fit failed with unexpected exception={e}")
        return results

    @abstractmethod
    def initialize_model(
        self,
        model: g2f.Model,
        source: Mapping[str, Any],
        limits_x: g2f.LimitsD = None,
        limits_y: g2f.LimitsD = None,
    ) -> None:
        """Initialize a Model for a single source row.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model object to initialize.
        source : typing.Mapping[str, typing.Any]
            A mapping with fields expected to be populated in the
            corresponding source catalog for initialization.
        limits_x : `gauss2d.fit.LimitsD`
            Hard limits for the source's x centroid.
        limits_y : `gauss2d.fit.LimitsD`
            Hard limits for the source's y centroid.
        """
