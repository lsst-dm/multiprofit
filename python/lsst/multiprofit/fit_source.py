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

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Type

import astropy
import astropy.units as u
import gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np
import pydantic
from astropy.table import Table
from pydantic.dataclasses import dataclass

from .componentconfig import SersicConfig
from .fit_catalog import CatalogExposureABC, CatalogFitterConfig, ColumnInfo
from .modeller import FitInputsDummy, Modeller
from .transforms import transforms_ref
from .utils import get_params_uniq

__all__ = ["CatalogExposureSourcesABC", "CatalogSourceFitterConfig", "CatalogSourceFitterABC"]


class CatalogExposureSourcesABC(CatalogExposureABC):
    """Interface for a CatalogExposure for source modelling."""

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

    convert_cen_xy_to_radec = pexConfig.Field[bool](default=True, doc="Convert cen x/y params to RA/dec")
    fit_cen_x = pexConfig.Field[bool](default=True, doc="Fit x centroid parameter")
    fit_cen_y = pexConfig.Field[bool](default=True, doc="Fit y centroid parameter")
    n_pointsources = pexConfig.Field[int](default=0, doc="Number of central point source components")
    prior_cen_x_stddev = pexConfig.Field[float](
        default=0, doc="Prior std. dev. on x centroid (ignored if not >0)"
    )
    prior_cen_y_stddev = pexConfig.Field[float](
        default=0, doc="Prior std. dev. on y centroid (ignored if not >0)"
    )

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
        observations = [None] * n_catexps
        psfmodels = [None] * n_catexps

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
    ) -> tuple[
        g2f.Source,
        list[g2f.Prior],
        g2f.LimitsD,
        g2f.LimitsD,
        g2f.CentroidXParameterD,
        g2f.CentroidXParameterD,
    ]:
        limits_x = g2f.LimitsD(min=0, max=np.Inf)
        limits_y = g2f.LimitsD(min=0, max=np.Inf)
        cen_x = g2f.CentroidXParameterD(value=0, limits=limits_x, fixed=not self.fit_cen_x)
        cen_y = g2f.CentroidYParameterD(value=0, limits=limits_y, fixed=not self.fit_cen_y)
        centroid = g2f.CentroidParameters(cen_x, cen_y)
        priors = []
        n_sersics = len(self.sersics)
        components = [None] * (self.n_pointsources + n_sersics)
        for idx in range(self.n_pointsources):
            components[idx] = g2f.GaussianComponent(
                centroid=centroid,
                ellipse=g2f.GaussianParametricEllipse(
                    sigma_x=g2f.SigmaXParameterD(0, fixed=True),
                    sigma_y=g2f.SigmaYParameterD(0, fixed=True),
                    rho=g2f.RhoParameterD(0, fixed=True),
                ),
                integral=g2f.LinearIntegralModel(
                    [
                        (
                            channel,
                            g2f.IntegralParameterD(
                                1.0, transform=transforms_ref["log10"], label=f"Pt. Src {channel.name}-band"
                            ),
                        )
                        for channel in channels
                    ]
                ),
            )
        idx = self.n_pointsources
        for sersic in self.sersics.values():
            component, priors_comp = sersic.make_component(centroid=centroid, channels=channels)
            components[idx] = component
            priors.extend(priors_comp)
            idx += 1
        if self.prior_cen_x_stddev > 0 and np.isfinite(self.prior_cen_x_stddev):
            priors.append(g2f.GaussianPrior(centroid.x_param_ptr, 0, self.prior_cen_x_stddev))
        if self.prior_cen_y_stddev > 0 and np.isfinite(self.prior_cen_y_stddev):
            priors.append(g2f.GaussianPrior(centroid.y_param_ptr, 0, self.prior_cen_y_stddev))
        return g2f.Source(components), priors, limits_x, limits_y, cen_x, cen_y

    def schema(
        self,
        bands: list[str] = None,
    ) -> list[ColumnInfo]:
        if bands is None or not (len(bands) > 0):
            raise ValueError("PSF CatalogSourceFitter must provide at least one band")
        columns = super().schema(bands)
        suffixes = ("", "_err") if self.compute_errors else ("",)
        for suffix in suffixes:
            if suffix == "_err":
                if self.fit_cen_x:
                    columns.append(ColumnInfo(key=f"cen_x{suffix}", dtype="f8", unit=u.pix))
                if self.fit_cen_y:
                    columns.append(ColumnInfo(key=f"cen_y{suffix}", dtype="f8", unit=u.pix))
            for idx in range(self.n_pointsources):
                prefix_comp = f"ps{idx + 1}_"
                for band in bands:
                    columns.append(ColumnInfo(key=f"{prefix_comp}{band}_flux{suffix}", dtype="f8"))

            for idx, (name_comp, comp) in enumerate(self.sersics.items()):
                prefix_comp = f"{name_comp}_"
                columns_comp = [
                    ColumnInfo(key=f"{prefix_comp}reff_x{suffix}", dtype="f8", unit=u.pix),
                    ColumnInfo(key=f"{prefix_comp}reff_y{suffix}", dtype="f8", unit=u.pix),
                    ColumnInfo(key=f"{prefix_comp}rho{suffix}", dtype="f8", unit=u.pix),
                ]
                for band in bands:
                    columns_comp.append(
                        ColumnInfo(
                            key=f"{prefix_comp}{band}_flux{suffix}",
                            dtype="f8",
                            unit=u.Unit(self.unit_flux) if self.unit_flux else None,
                        )
                    )
                if not comp.sersicindex.fixed:
                    columns_comp.append(ColumnInfo(key=f"{prefix_comp}nser{suffix}", dtype="f8"))
                columns.extend(columns_comp)
            if self.convert_cen_xy_to_radec:
                columns.append(ColumnInfo(key=f"cen_ra{suffix}", dtype="f8", unit=u.deg))
                columns.append(ColumnInfo(key=f"cen_dec{suffix}", dtype="f8", unit=u.deg))
        return columns


@dataclass(kw_only=True)
class CatalogSourceFitterABC(ABC):
    """Fit a Gaussian mixture source model to an image with a PSF model.

    Notes
    -----
    Any exceptions raised and not in errors_expected will be logged in a
    generic unknown_flag failure column.
    """

    errors_expected: dict[Type[Exception], str] = pydantic.field(
        default_factory=dict,
        title="A dictionary of Exceptions with the name of the flag column key to fill if raised.",
    )
    modeller: Modeller = pydantic.field(
        default_factory=Modeller,
        title="A Modeller instance to use for fitting.",
    )

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
        **kwargs: Any,
    ) -> astropy.table.Table:
        """Fit PSF-convolved source models with MultiProFit.

        Each source has a single PSF-convolved model fit, given PSF model
        parameters from a catalog, and a combination of initial source
        model parameters and a deconvolved source image from the
        CatalogExposureSources.

        Parameters
        ----------
        catalog_multi
            A multi-band source catalog to fit a model to.
        catexps
            A list of (source and psf) catalog-exposure pairs.
        config
            Configuration settings for fitting and output.
        logger
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

        channels = self.get_channels(catexps)
        model_source, priors, limits_x, limits_y, cen_x, cen_y = config.make_source(
            channels=tuple(channels.values())
        )
        params = tuple(get_params_uniq(model_source, fixed=False))

        n_rows = len(catalog_multi)
        range_idx = range(n_rows)

        columns = config.schema([channel.name for channel in channels.values()])
        keys = [column.key for column in columns]
        prefix = config.prefix_column
        idx_flag_first = keys.index("unknown_flag")
        idx_flag_last = keys.index("cen_x")
        assert idx_flag_last > idx_flag_first
        idx_var_first = idx_flag_last + (not config.fit_cen_x) + (not config.fit_cen_x)
        idx_var_last = idx_var_first + len(params)
        column_cen_x, column_cen_y = (f"{prefix}cen_{v}" for v in ("x", "y"))
        columns_write = [f"{prefix}{col.key}" for col in columns[idx_var_first:idx_var_last]]
        n_radec = 2 * config.convert_cen_xy_to_radec
        columns_radec = [f"{prefix}{col.key}" for col in columns[idx_var_last : idx_var_last + n_radec]]
        idx_var_last += n_radec
        columns_write_err = [f"{prefix}{col.key}" for col in columns[idx_var_last : len(columns) - n_radec]]
        assert len(columns_write_err) == len(params)
        columns_radec_err = [f"{prefix}{col.key}" for col in columns[len(columns) - n_radec : len(columns)]]
        dtypes = [(f'{prefix if col.key != config.column_id else ""}{col.key}', col.dtype) for col in columns]
        meta = {"config": config.toDict()}
        results = Table(data=np.full(n_rows, 0, dtype=dtypes), units=[x.unit for x in columns], meta=meta)

        # Validate that the columns are in the right order
        # assert because this is a logic error if it fails
        for idx in range(idx_flag_first, len(columns)):
            column = columns[idx]
            dtype = results.columns[idx].dtype
            if idx < idx_flag_last:
                assert dtype == bool
                assert column.key.endswith("_flag")
            else:
                assert dtype == float

        # dummy size for first iteration
        size, size_new = 0, 0
        fitInputs = FitInputsDummy()
        plot = False
        errors_hessian = config.compute_errors == "INV_HESSIAN"
        errors_hessian_bestfit = config.compute_errors == "INV_HESSIAN_BESTFIT"
        compute_errors = errors_hessian or errors_hessian_bestfit

        kwargs_err_default = {
            True: {
                "options": g2f.HessianOptions(findiff_add=1e-3, findiff_frac=1e-3),
                "use_diag_only": config.compute_errors_no_covar,
            },
            False: {"options": g2f.HessianOptions(findiff_add=1e-6, findiff_frac=1e-6)},
        }

        for idx in range_idx:
            time_init = time.process_time()
            row = results[idx]
            source_multi = catalog_multi[idx]
            id_source = source_multi[config.column_id]
            row[config.column_id] = id_source

            try:
                model, data, psfmodels = config.make_model_data(
                    idx_source=idx,
                    model_priors=(model_source, priors),
                    catexps=catexps,
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

                # TODO: set limits_flux, transform_flux
                # (if not config.fit_linear_init)
                if config.fit_linear_init:
                    self.modeller.fit_model_linear(model=model, ratio_min=1e-3)

                result_full = self.modeller.fit_model(
                    model, fitinputs=fitInputs, config=config.config_fit, **kwargs
                )
                fitInputs = result_full.inputs
                results[f"{prefix}n_iter"][idx] = result_full.n_eval_func
                results[f"{prefix}time_eval"][idx] = result_full.time_eval
                results[f"{prefix}time_fit"][idx] = result_full.time_run

                # Set all params to best fit values
                # In case the optimizer doesn't
                for param, value, column in zip(params, result_full.params_best, columns_write):
                    param.value_transformed = value
                    results[column][idx] = param.value

                if config.fit_linear_final:
                    loglike_init, loglike_new = self.modeller.fit_model_linear(
                        model=model, ratio_min=0.01, validate=True
                    )
                    np.sum(loglike_new) - np.sum(loglike_init)
                    # TODO: See if it makes sense to set only flux params
                    for param, column in zip(params, columns_write):
                        results[column][idx] = param.value

                if not config.fit_cen_x:
                    results[column_cen_x][idx] = cen_x.value
                if not config.fit_cen_y:
                    results[column_cen_y][idx] = cen_y.value

                if config.convert_cen_xy_to_radec:
                    radec = self.get_model_radec(source_multi, cen_x.value, cen_y.value)
                    for value, column in zip(radec, columns_radec):
                        results[column][idx] = value

                if compute_errors:
                    errors = []
                    model_eval = model
                    errors_iter = None
                    if config.compute_errors_from_jacobian:
                        try:
                            errors_iter = np.sqrt(
                                self.modeller.compute_variances(
                                    model_eval,
                                    transformed=False,
                                    use_diag_only=config.compute_errors_no_covar,
                                )
                            )
                            errors.append((errors_iter, np.sum(~(errors_iter > 0))))
                        except Exception:
                            pass
                    if errors_iter is None:
                        img_data_old = []
                        if errors_hessian_bestfit:
                            # Model sans prior
                            model_eval = g2f.Model(
                                data=model.data, psfmodels=model.psfmodels, sources=model.sources
                            )
                            model_eval.setup_evaluators(evaluatormode=model.EvaluatorMode.image)
                            model_eval.evaluate()
                            for obs, output in zip(model_eval.data, model_eval.outputs):
                                img_data_old.append(obs.image.data.copy())
                                img = obs.image.data
                                img.flat = output.data.flat
                                # To make this a real bootstrap, could do this
                                # (but would need to iterate):
                                # + rng.standard_normal(img.size)*(
                                #   obs.sigma_inv.data.flat)

                        for return_negative in (False, True):
                            kwargs_err = kwargs_err_default[return_negative]
                            if errors and errors[-1][1] == 0:
                                break
                            try:
                                errors_iter = np.sqrt(
                                    self.modeller.compute_variances(
                                        model_eval, transformed=False, **kwargs_err
                                    )
                                )
                                errors.append((errors_iter, np.sum(~(errors_iter > 0))))
                            except Exception:
                                try:
                                    errors_iter = np.sqrt(
                                        self.modeller.compute_variances(
                                            model_eval,
                                            transformed=False,
                                            use_svd=True,
                                            **kwargs_err,
                                        )
                                    )
                                    errors.append((errors_iter, np.sum(~(errors_iter > 0))))
                                except Exception:
                                    pass

                        if errors_hessian_bestfit:
                            for obs, img_datum_old in zip(model.data, img_data_old):
                                obs.image.data.flat = img_datum_old.flat

                    if errors:
                        idx_min = np.argmax([err[1] for err in errors])
                        errors = errors[idx_min][0]
                        if plot:
                            errors_plot = np.clip(errors, 0, 1000)
                            errors_plot[~np.isfinite(errors_plot)] = 0
                            from .plots import ErrorValues, plot_loglike

                            try:
                                plot_loglike(model, errors={"err": ErrorValues(values=errors_plot)})
                            except Exception:
                                for param in params:
                                    param.fixed = False

                        for value, column in zip(errors, columns_write_err):
                            results[column][idx] = value
                        if config.convert_cen_xy_to_radec:
                            if not config.fit_cen_x and config.fit_cen_y:
                                # Note this will make naive tests that check
                                # for all-finite results fail
                                radec_err = np.nan, np.nan
                            else:
                                # TODO: improve this
                                # For one, it won't work right at RA=359.99...
                                # Could consider dividing by sqrt(2)
                                # but it would multiply out later
                                radec_err = np.array(
                                    self.get_model_radec(
                                        source_multi,
                                        cen_x.value + errors[0],
                                        cen_y.value + errors[1],
                                    )
                                )
                                radec_err -= radec
                            for value, column in zip(radec_err, columns_radec_err):
                                results[column][idx] = np.abs(value)

                results[f"{prefix}chisq_red"][idx] = np.sum(fitInputs.residual**2) / size
                results[f"{prefix}time_full"][idx] = time.process_time() - time_init
            except Exception as e:
                size = 0 if fitInputs is None else size_new
                column = self.errors_expected.get(e.__class__, "")
                if column:
                    row[f"{prefix}{column}"] = True
                    logger.info(f"{id_source=} ({idx=}/{n_rows}) fit failed with known exception={e}")
                else:
                    row[f"{prefix}unknown_flag"] = True
                    logger.info(
                        f"{id_source=} ({idx=}/{n_rows}) fit failed with unexpected exception={e}", exc_info=1
                    )
        return results

    def get_channels(
        self,
        catexps: list[CatalogExposureSourcesABC],
    ) -> dict[str, g2f.Channel]:
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
        return channels

    def get_model(
        self,
        idx_row: int,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config: CatalogSourceFitterConfig = None,
        results: astropy.table.Table = None,
    ) -> g2f.Model:
        """Reconstruct the model for a single row of a fit catalog.

        Parameters
        ----------
        idx_row
            The index of the row in the catalog.
        catalog_multi
            The multi-band catalog originally used for initialization.
        catexps
            The catalog-exposure pairs to reconstruct the model for.
        config
            The configuration used to generate sources.
            Default-initialized if None.
        results
            The corresponding best-fit parameter catalog to initialize
            parameter values from.

        Returns
        -------
        model
            The reconstructed model.
        """
        if config is None:
            config = CatalogSourceFitterConfig()

        if not idx_row >= 0:
            raise ValueError(f"{idx_row=} !>=0")
        if not len(catalog_multi) > idx_row:
            raise ValueError(f"{len(catalog_multi)=} !> {idx_row=}")
        if (results is not None) and not (len(results) > idx_row):
            raise ValueError(f"{len(results)=} !> {idx_row=}")

        channels = self.get_channels(catexps)
        model_source, priors, limits_x, limits_y, *_ = config.make_source(channels=list(channels.values()))
        source_multi = catalog_multi[idx_row]

        model, data, psfmodels = config.make_model_data(
            idx_source=idx_row,
            model_priors=(model_source, priors),
            catexps=catexps,
        )
        self.initialize_model(model, source_multi, limits_x=limits_x, limits_y=limits_y)

        if results is not None:
            params = get_params_uniq(model_source, fixed=False)
            columns = list(results.columns)
            row = results[idx_row]
            idx_col_start = columns.index(f"{config.prefix_column}cen_x")
            n_params = len(params)
            for param, column in zip(params, columns[idx_col_start : idx_col_start + n_params]):
                param.value = row[column]

        return model

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float) -> tuple[float, float]:
        return np.nan, np.nan

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
