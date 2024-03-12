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
from functools import cached_property
import logging
import time
from typing import Any, Iterable, Mapping, Sequence, Type

import astropy
from astropy.table import Table
import astropy.units as u
import gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass

from .componentconfig import Fluxes
from .fit_catalog import CatalogExposureABC, CatalogFitterConfig, ColumnInfo
from .modelconfig import ModelConfig
from .modeller import FitInputsDummy, Modeller
from .utils import FrozenArbitraryAllowedConfig, get_params_uniq

__all__ = [
    "CatalogExposureSourcesABC", "CatalogSourceFitterConfig", "CatalogSourceFitterConfigData",
    "CatalogSourceFitterABC",
]


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
    def get_source_observation(self, source: Mapping[str, Any], **kwargs: Any) -> g2f.Observation:
        """Get the Observation for a given source row.

        Parameters
        ----------
        source : Mapping[str, Any]
            A mapping with any values needed to retrieve an observation for a
            single source.
        **kwargs
            Additional keyword arguments not used during fitting.

        Returns
        -------
        observation : `gauss2d.fit.Observation`
            An Observation object with suitable data for fitting parametric
            models of the source.
        """


class CatalogSourceFitterConfig(CatalogFitterConfig):
    """Configuration for the MultiProFit profile fitter."""

    centroid_pixel_offset = pexConfig.Field[float](
        doc="Number to add to MultiProFit centroids (bottom-left corner is 0,0) to convert to catalog"
            " coordinates (e.g. set to -0.5 if the bottom-left corner is -0.5, -0.5)",
        default=0,
    )
    config_model = pexConfig.ConfigField[ModelConfig](doc="Source model configuration")
    convert_cen_xy_to_radec = pexConfig.Field[bool](default=True, doc="Convert cen x/y params to RA/dec")
    prior_cen_x_stddev = pexConfig.Field[float](
        default=0, doc="Prior std. dev. on x centroid (ignored if not >0)"
    )
    prior_cen_y_stddev = pexConfig.Field[float](
        default=0, doc="Prior std. dev. on y centroid (ignored if not >0)"
    )
    unit_flux = pexConfig.Field[str](default=None, doc="Flux unit", optional=True)

    def make_model_data(
        self,
        idx_row,
        catexps: list[CatalogExposureSourcesABC],
    ) -> tuple[g2f.Data, list[g2f.PsfModel]]:
        """Make data and psfmodels for a catalog row.

        Parameters
        ----------
        idx_row
            The index of the row in each catalog.
        catexps
            Catalog-exposure pairs to initialize observations from.

        Returns
        -------
        data
            The resulting data object.
        psfmodels
            A list of psfmodels, one per catexp.
        """
        observations = []
        psfmodels = []

        for catexp in catexps:
            source = catexp.get_catalog()[idx_row]
            observation = catexp.get_source_observation(source)
            if observation is not None:
                observations.append(observation)
                psfmodel = catexp.get_psfmodel(source)
                for param in get_params_uniq(psfmodel):
                    param.fixed = True
                psfmodels.append(psfmodel)

        data = g2f.Data(observations)
        return data, psfmodels

    def make_sources(
        self,
        channels: Iterable[g2f.Channel],
        source_fluxes: list[list[list[Fluxes]]] = None,
    ) -> tuple[
        list[g2f.Source],
        list[g2f.Prior],
        g2f.LimitsD,
        g2f.LimitsD,
    ]:
        n_sources = len(self.config_model.sources)
        if source_fluxes is None:
            source_fluxes = [None]*n_sources
            for idx, (config_source, component_group_fluxes) in enumerate(zip(
                    self.config_model.sources.values(), source_fluxes,
            )):
                component_group_fluxes = [
                    component_group.get_fluxes_default(
                        channels=channels,
                        component_configs=component_group.get_component_configs(),
                        is_fractional=component_group.is_fractional,
                    )
                    for component_group in config_source.component_groups.values()
                ]
                source_fluxes[idx] = component_group_fluxes
        else:
            if len(source_fluxes) != n_sources:
                raise ValueError(f"{len(source_fluxes)=} != {len(self.config_model.sources)=}")

        sources, priors = self.config_model.make_sources(
            component_group_fluxes_srcs=source_fluxes,
        )

        has_prior_x = self.prior_cen_x_stddev > 0 and np.isfinite(self.prior_cen_x_stddev)
        has_prior_y = self.prior_cen_y_stddev > 0 and np.isfinite(self.prior_cen_y_stddev)
        if has_prior_x or has_prior_y:
            for source in sources:
                for param in get_params_uniq(source, fixed=False):
                    if has_prior_x and isinstance(param, g2f.CentroidXParameterD):
                        priors.append(g2f.GaussianPrior(param.x_param_ptr, 0, self.prior_cen_x_stddev))
                    elif has_prior_y and isinstance(param, g2f.CentroidYParameterD):
                        priors.append(g2f.GaussianPrior(param.y_param_ptr, 0, self.prior_cen_y_stddev))

        return sources, priors

    def schema_configurable(self) -> list[ColumnInfo]:
        columns = []
        if self.config_fit.eval_residual:
            columns.append(ColumnInfo(key="n_eval_jac", dtype="i4"))
        if self.fit_linear_final:
            columns.append(ColumnInfo(key="delta_ll_fit_linear", dtype="f8"))
        return columns

    def schema(
        self,
        bands: list[str] = None,
    ) -> list[ColumnInfo]:
        if bands is None or not (len(bands) > 0):
            raise ValueError("CatalogSourceFitter must provide at least one band")
        schema = super().schema(bands)

        parameters = CatalogSourceFitterConfigData(
            config=self,
            channels=tuple((g2f.Channel.get(band) for band in bands)),
        ).parameters
        unit_size = u.Unit("pix")
        units = {
            g2f.IntegralParameterD: self.unit_flux,
            g2f.ReffXParameterD: unit_size,
            g2f.ReffYParameterD: unit_size,
            g2f.SizeXParameterD: unit_size,
            g2f.SizeYParameterD: unit_size,
        }
        idx_start = len(schema)
        schema.extend([
            ColumnInfo(key=key, dtype="f8", unit=units.get(type(param)))
            for key, param in parameters.items()
        ])
        if self.convert_cen_xy_to_radec:
            for key, param in parameters.items():
                suffix = "cen_ra" if isinstance(param, g2f.CentroidXParameterD) else (
                    "cen_dec" if isinstance(param, g2f.CentroidYParameterD) else None)
                if suffix is not None:
                    schema.append(ColumnInfo(key=f"{key[:-5]}{suffix}", dtype="f8", unit=u.deg))
        if self.compute_errors != "NONE":
            idx_end = len(schema)
            # Do not remove idx_end unless you like infinite recursion
            for column in schema[idx_start:idx_end]:
                schema.append(ColumnInfo(key=f"{column.key}_err", dtype=column.dtype, unit=column.unit))

        schema.extend(self.schema_configurable())
        return schema


@dataclass(frozen=True, kw_only=True, config=FrozenArbitraryAllowedConfig)
class CatalogSourceFitterConfigData:
    """Configuration data for a fitter that can initialize gauss2d.fit models
    and images thereof.

    This class relies on cached properties being computed once, mostly shortly
    after initialization. Therefore, it and the config field must be frozen to
    ensure that the model remains unchanged.
    """

    channels: list[g2f.Channel] = pydantic.Field(title="The list of channels")
    config: CatalogSourceFitterConfig = pydantic.Field(title="A CatalogSourceFitterConfig to be frozen")

    @pydantic.model_validator(mode="after")
    def validate_config(self):
        self.config.validate()

    @cached_property
    def components(self) -> tuple[g2f.Component]:
        sources = self.sources_priors[0]
        components = []
        for source in sources:
            components.extend(source.components)
        return components

    @cached_property
    def parameters(self) -> dict[str, g2f.ParameterD]:
        config_model = self.config.config_model
        idx_comp_first = 0
        has_prefix_source = config_model.has_prefix_source()
        n_channels = len(self.channels)
        parameters = {}

        for name_source, config_source in config_model.sources.items():
            prefix_source = f"{name_source}_" if has_prefix_source else ""
            has_prefix_group = config_source.has_prefix_group()

            for name_group, config_group in config_source.component_groups.items():
                prefix_group = f"{prefix_source}{name_group}_" if has_prefix_group else prefix_source
                multicen = len(config_group.centroids) > 1
                configs_comp = config_group.get_component_configs().items()

                for idx_comp_group, (name_comp, config_comp) in enumerate(configs_comp):
                    component = self.components[idx_comp_first + idx_comp_group]
                    label_size = config_comp.get_size_label()
                    prefix_comp = f"{prefix_group}{name_comp}{'_' if name_comp else ''}"
                    if multicen or (idx_comp_group == 0):
                        prefix_cen = prefix_comp if multicen else prefix_group
                        parameters[f"{prefix_cen}cen_x"] = component.centroid.x_param
                        parameters[f"{prefix_cen}cen_y"] = component.centroid.y_param
                    if not config_comp.size_x.fixed:
                        parameters[f"{prefix_comp}{label_size}_x"] = component.ellipse.size_x_param
                    if not config_comp.size_y.fixed:
                        parameters[f"{prefix_comp}{label_size}_y"] = component.ellipse.size_y_param
                    if not config_comp.rho.fixed:
                        parameters[f"{prefix_comp}rho"] = component.ellipse.rho_param
                    if not config_comp.flux.fixed:
                        # TODO: return this to component.integralmodel
                        # when binding for g2f.FractionalIntegralModel is fixed
                        params_flux = get_params_uniq(component, fixed=False, nonlinear=False)
                        if len(params_flux) != n_channels:
                            raise ValueError(f"{params_flux=} len={len(params_flux)} != {n_channels=}")
                        for channel, param_flux in zip(self.channels, params_flux):
                            parameters[f"{prefix_comp}{channel.name}_flux"] = param_flux
                    if hasattr(config_comp, "sersic_index") and not config_comp.sersic_index.fixed:
                        parameters[f"{prefix_comp}sersicindex"] = component.sersicindex_param

        return parameters

    @cached_property
    def sources_priors(self) -> tuple[tuple[g2f.Source], tuple[g2f.Prior]]:
        sources, priors = self.config.make_sources(channels=self.channels)
        return tuple(sources), tuple(priors)


@dataclass(kw_only=True, frozen=True, config=FrozenArbitraryAllowedConfig)
class CatalogSourceFitterABC(ABC):
    """Fit a Gaussian mixture source model to an image with a PSF model.

    Notes
    -----
    Any exceptions raised and not in errors_expected will be logged in a
    generic unknown_flag failure column.
    """

    errors_expected: dict[Type[Exception], str] = pydantic.Field(
        default_factory=dict,
        title="A dictionary of Exceptions with the name of the flag column key to fill if raised.",
    )
    modeller: Modeller = pydantic.Field(
        default_factory=Modeller,
        title="A Modeller instance to use for fitting.",
    )

    @staticmethod
    def _get_logger():
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.level = logging.INFO

        return logger

    def copy_centroid_errors(
        self,
        columns_cenx_err_copy: tuple[str],
        columns_ceny_err_copy: tuple[str],
        results: Table,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
    ):
        if columns_cenx_err_copy or columns_ceny_err_copy:
            raise RuntimeError(
                f"Fitter of {type(self)=} got {columns_cenx_err_copy=} and/or {columns_ceny_err_copy=}"
                f" but has not overriden copy_centroid_errors"
            )

    def fit(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
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
        config_data
            Configuration settings and data for fitting and output.
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
        if config_data is None:
            config_data = CatalogSourceFitterConfigData(
                config=CatalogSourceFitterConfig(),
                channels=[catexp.channel for catexp in catexps],
            )
        if logger is None:
            logger = self._get_logger()

        self.validate_fit_inputs(
            catalog_multi=catalog_multi,
            catexps=catexps,
            config_data=config_data,
            logger=logger,
            **kwargs
        )
        config = config_data.config

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
        model_sources, priors = config_data.sources_priors
        # TODO: If free Observation params are ever supported, make null Data
        # Because config_data knows nothing about the Observation(s)
        params = config_data.parameters
        values_init = {param: param.value for param in params.values() if param.free}
        prefix = config.prefix_column
        columns_param_fixed: dict[str, tuple[g2f.ParameterD, float]] = {}
        columns_param_free: dict[str, tuple[g2f.ParameterD, float]] = {}
        columns_param_flux: dict[str, g2f.IntegralParameterD] = {}
        params_radec = {}
        columns_err = []

        errors_hessian = config.compute_errors == "INV_HESSIAN"
        errors_hessian_bestfit = config.compute_errors == "INV_HESSIAN_BESTFIT"
        compute_errors = errors_hessian or errors_hessian_bestfit

        columns_cenx_err_copy = []
        columns_ceny_err_copy = []

        for key, param in params.items():
            key_full = f"{prefix}{key}"
            is_cenx = isinstance(param, g2f.CentroidXParameterD)
            is_ceny = isinstance(param, g2f.CentroidYParameterD)

            if compute_errors:
                if param.free:
                    columns_err.append(f"{key_full}_err")
                elif is_cenx:
                    columns_cenx_err_copy.append(f"{key_full}_err")
                elif is_ceny:
                    columns_ceny_err_copy.append(f"{key_full}_err")

            (columns_param_fixed if param.fixed else columns_param_free)[key_full] = (
                param,
                config_data.config.centroid_pixel_offset if (is_cenx or is_ceny) else 0,
            )
            if isinstance(param, g2f.IntegralParameterD):
                columns_param_flux[key_full] = param
            elif config.convert_cen_xy_to_radec:
                if is_cenx:
                    params_radec[f"{key_full[:-5]}"] = [param, None]
                elif is_ceny:
                    params_radec[f"{key_full[:-5]}"][1] = param

        if config.convert_cen_xy_to_radec:
            columns_params_radec = []
            columns_params_radec_err = []

            for key_base, (param_cen_x, param_cen_y) in params_radec.items():
                if param_cen_y is None:
                    raise RuntimeError(
                        f"Fitter failed to find corresponding cen_y param for {key_base=}; is it fixed?")
                columns_params_radec.append(
                    (f"{key_base}cen_ra", f"{key_base}cen_dec",  f"{key_base}cen_x",  f"{key_base}cen_y")
                )
                if compute_errors:
                    columns_params_radec_err.append(
                        (f"{key_base}cen_ra_err", f"{key_base}cen_dec_err", f"{key_base}cen_x",
                         f"{key_base}cen_y", f"{key_base}cen_x_err",  f"{key_base}cen_y_err")
                    )

        columns = config.schema([channel.name for channel in channels.values()])
        keys = [column.key for column in columns]
        idx_flag_first = keys.index("unknown_flag")
        idx_flag_last = next(iter(
            idx for idx, key in enumerate(keys[idx_flag_first:]) if key.endswith("cen_x")
        )) + idx_flag_first
        assert idx_flag_last > idx_flag_first

        n_rows = len(catalog_multi)
        dtypes = [(f'{prefix if col.key != config.column_id else ""}{col.key}', col.dtype) for col in columns]
        meta = {"config": config.toDict()}
        results = Table(
            data=np.full(n_rows, 0, dtype=dtypes), units=[x.unit for x in columns], meta=meta,
        )
        self.copy_centroid_errors(
            columns_cenx_err_copy=columns_cenx_err_copy,
            columns_ceny_err_copy=columns_ceny_err_copy,
            results=results,
            catalog_multi=catalog_multi,
            catexps=catexps,
            config_data=config_data,
        )

        # Validate that the columns are in the right order
        # assert because this is a logic error if it fails
        for idx in range(idx_flag_first, idx_flag_last):
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

        kwargs_err_default = {
            True: {
                "options": g2f.HessianOptions(findiff_add=1e-3, findiff_frac=1e-3),
                "use_diag_only": config.compute_errors_no_covar,
            },
            False: {"options": g2f.HessianOptions(findiff_add=1e-6, findiff_frac=1e-6)},
        }

        range_idx = range(n_rows)

        # TODO: Do this check with dummy data
        # data, psfmodels = config.make_model_data(
        #     idx_row=range_idx[0], catexps=catexps)
        # model = g2f.Model(data=data, psfmodels=psfmodels,
        #     sources=model_sources, priors=priors)
        # Remember to filter out fixed centroids from params
        # assert list(params.values()) == get_params_uniq(model, fixed=False)

        for idx in range_idx:
            time_init = time.process_time()
            row = results[idx]
            source_multi = catalog_multi[idx]
            id_source = source_multi[config.column_id]
            row[config.column_id] = id_source

            try:
                data, psfmodels = config.make_model_data(idx_row=idx, catexps=catexps)
                model = g2f.Model(data=data, psfmodels=psfmodels, sources=model_sources, priors=priors)
                self.initialize_model(
                    model, source_multi, catexps, values_init,
                    centroid_pixel_offset=config_data.config.centroid_pixel_offset,
                )

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
                    self.modeller.fit_model_linear(model=model, ratio_min=0.01)

                for observation in data:
                    observation.image.data[~np.isfinite(observation.image.data)] = 0

                result_full = self.modeller.fit_model(
                    model, fitinputs=fitInputs, config=config.config_fit, **kwargs
                )
                fitInputs = result_full.inputs
                results[f"{prefix}n_iter"][idx] = result_full.n_eval_func
                results[f"{prefix}time_eval"][idx] = result_full.time_eval
                results[f"{prefix}time_fit"][idx] = result_full.time_run
                if config.config_fit.eval_residual:
                    results[f"{prefix}n_eval_jac"][idx] = result_full.n_eval_jac

                # Set all params to best fit values
                # In case the optimizer doesn't
                for (key, (param, offset)), value in zip(columns_param_free.items(), result_full.params_best):
                    param.value_transformed = value
                    results[key][idx] = param.value + offset

                for key, (param, offset) in columns_param_fixed.items():
                    results[key][idx] = param.value + offset

                if config.fit_linear_final:
                    loglike_init, loglike_new = self.modeller.fit_model_linear(
                        model=model, ratio_min=0.01, validate=True
                    )
                    results[f"{prefix}delta_ll_fit_linear"][idx] = np.sum(loglike_new) - np.sum(loglike_init)
                    for column, param in columns_param_flux.items():
                        results[column][idx] = param.value

                if config.convert_cen_xy_to_radec:
                    for (key_ra, key_dec, key_cen_x, key_cen_y) in columns_params_radec:
                        # These will have been converted back if necessary
                        cen_x, cen_y = results[key_cen_x][idx], results[key_cen_y][idx]
                        radec = self.get_model_radec(source_multi, cen_x, cen_y)
                        results[key_ra][idx], results[key_dec][idx] = radec

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

                        for value, column_err in zip(errors, columns_err):
                            results[column_err][idx] = value
                        if config.convert_cen_xy_to_radec:
                            for (
                                key_ra_err, key_dec_err, key_cen_x, key_cen_y, key_cen_x_err, key_cen_y_err
                            ) in columns_params_radec_err:
                                cen_x, cen_y = results[key_cen_x][idx], results[key_cen_y][idx]
                                # TODO: improve this
                                # For one, it won't work right at RA=359.99...
                                # Could consider dividing by sqrt(2)
                                # but it would multiply out later
                                radec_err = np.array(
                                    self.get_model_radec(
                                        source_multi,
                                        cen_x + results[key_cen_x_err][idx],
                                        cen_y + results[key_cen_y_err][idx],
                                    )
                                )
                                radec_err -= radec
                                results[key_ra_err][idx], results[key_dec_err][idx] = np.abs(radec_err)

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
                        f"{id_source=} ({idx=}/{n_rows}) fit failed with unexpected exception={e}",
                        exc_info=1
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
        config_data: CatalogSourceFitterConfigData = None,
        results: astropy.table.Table = None,
        **kwargs: Any
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
        config_data
            The configuration used to generate sources.
            Default-initialized if None.
        results
            The corresponding best-fit parameter catalog to initialize
            parameter values from.
        **kwargs
            Additional keyword arguments to pass to initialize_model. Not
            used during fitting.

        Returns
        -------
        model
            The reconstructed model.
        """
        channels = self.get_channels(catexps)
        if config_data is None:
            config_data = CatalogSourceFitterConfigData(
                config=CatalogSourceFitterConfig,
                channels=list(channels.values()),
            )
        config = config_data.config

        if not idx_row >= 0:
            raise ValueError(f"{idx_row=} !>=0")
        if not len(catalog_multi) > idx_row:
            raise ValueError(f"{len(catalog_multi)=} !> {idx_row=}")
        if (results is not None) and not (len(results) > idx_row):
            raise ValueError(f"{len(results)=} !> {idx_row=}")

        model_sources, priors = config_data.sources_priors
        source_multi = catalog_multi[idx_row]

        data, psfmodels = config.make_model_data(
            idx_row=idx_row,
            catexps=catexps,
        )
        model = g2f.Model(data=data, psfmodels=psfmodels, sources=model_sources, priors=priors)
        self.initialize_model(model, source_multi, catexps, **kwargs)

        if results is not None:
            row = results[idx_row]
            for column, param in config_data.parameters.items():
                param.value = row[f"{config.prefix_column}{column}"]

        return model

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float) -> tuple[float, float]:
        return np.nan, np.nan

    @abstractmethod
    def initialize_model(
        self,
        model: g2f.Model,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        centroid_pixel_offset: float = 0,
        **kwargs: Any
    ) -> None:
        """Initialize a Model for a single source row.

        Parameters
        ----------
        model
            The model object to initialize.
        source
            A mapping with fields expected to be populated in the
            corresponding source catalog for initialization.
        catexps
            A list of (source and psf) catalog-exposure pairs.
        values_init
            Initial parameter values from the model configuration.
        centroid_pixel_offset
            The value of the offset required to convert pixel centroids from
            MultiProFit coordinates to catalog coordinates.
        **kwargs
            Additional keyword arguments that cannot be required for fitting.
        """

    @abstractmethod
    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        """Validate inputs to self.fit.

        This method is called before any fitting is done. It may be used for
        any purpose, including checking that the inputs are a particular
        subclass of the base classes.

        Parameters
        ----------
        catalog_multi
            A multi-band source catalog to fit a model to.
        catexps
            A list of (source and psf) catalog-exposure pairs.
        config_data
            Configuration settings and data for fitting and output.
        logger
            The logger. Defaults to calling `_getlogger`.
        **kwargs
            Additional keyword arguments to pass to self.modeller.
        """
        pass
