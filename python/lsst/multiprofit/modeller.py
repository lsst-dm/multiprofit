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
import logging
import time
from typing import Any

import gauss2d as g2
import gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass
import scipy.optimize as spopt

from .model_utils import make_image_gaussians, make_psfmodel_null
from .utils import ArbitraryAllowedConfig, get_params_uniq

try:
    # TODO: try importlib.util.find_spec
    import fastnnls  # noqa

    has_fastnnls = True
except ImportError:
    has_fastnnls = False


__all__ = [
    "InvalidProposalError",
    "fitmethods_linear",
    "LinearGaussians",
    "make_image_gaussians",
    "make_psfmodel_null",
    "FitInputsBase",
    "FitInputsDummy",
    "ModelFitConfig",
    "FitResult",
    "Modeller",
]


class InvalidProposalError(ValueError):
    """Error for an invalid parameter proposal."""


fitmethods_linear = {
    "scipy.optimize.nnls": {},
    "scipy.optimize.lsq_linear": {"bounds": (1e-5, np.Inf), "method": "bvls"},
    "numpy.linalg.lstsq": {"rcond": 1e-3},
}
if has_fastnnls:
    fitmethods_linear["fastnnls.fnnls"] = {}


@dataclass(frozen=True, kw_only=True, config=ArbitraryAllowedConfig)
class LinearGaussians:
    """Helper for linear least-squares fitting of Gaussian mixtures."""

    gaussians_fixed: g2.Gaussians = pydantic.Field(title="Fixed Gaussian components")
    gaussians_free: tuple[tuple[g2.Gaussians, g2f.ParameterD], ...] = pydantic.Field(
        title="Free Gaussian components"
    )

    @staticmethod
    def make(
        componentmixture: g2f.ComponentMixture,
        channel: g2f.Channel = None,
        is_psf: bool = False,
    ):
        """Make a LinearGaussians from a ComponentMixture.

        Parameters
        ----------
        componentmixture
            A component mixture to initialize Gaussians from.
        channel
            The channel all Gaussians are applicable for.
        is_psf
            Whether the components are a smoothing kernel.

        Returns
        -------
        lineargaussians
            A LinearGaussians instance initialized with the appropriate
            fixed/free gaussians.
        """
        if channel is None:
            channel = g2f.Channel.NONE
        components = componentmixture.components
        if len(components) == 0:
            raise ValueError(f"Can't get linear Source from {componentmixture=} with no components")

        gaussians_free = []
        gaussians_fixed = []

        for idx, component in enumerate(components):
            gaussians: g2.Gaussians = component.gaussians(channel)
            # TODO: Support multi-Gaussian components if sensible
            # The challenge would be in mapping linear param values back onto
            # non-linear IntegralModels
            if is_psf:
                n_g = len(gaussians)
                if n_g != 1:
                    raise ValueError(f"{component=} has {gaussians=} of len {n_g=}!=1")
            param_flux = component.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=channel))
            if len(param_flux) != 1:
                raise ValueError(f"Can't make linear source from {component=} with {len(param_flux)=}")
            param_flux: g2f.ParameterD = param_flux[0]
            if param_flux.fixed:
                gaussians_fixed.append(gaussians.at(0))
            else:
                gaussians_free.append((gaussians, param_flux))

        return LinearGaussians(
            gaussians_fixed=g2.Gaussians(gaussians_fixed), gaussians_free=tuple(gaussians_free)
        )


class FitInputsBase(ABC):
    """Interface for inputs to a model fit."""

    @abstractmethod
    def validate_for_model(self, model: g2f.Model) -> list[str]:
        """Check that this FitInputs is valid for a Model.

        Parameters
        ----------
        model
            The model to validate with.

        Returns
        -------
        errors
            A list of validation errors, if any.
        """


class FitInputsDummy(FitInputsBase):
    """A dummy FitInputs that always fails to validate.

    This class can be used to initialize a FitInputsBase that may be
    reassigned to a non-dummy derived instance in a loop.
    """

    def validate_for_model(self, model: g2f.Model) -> list[str]:
        return [
            "This is a dummy FitInputs and will never validate",
        ]


@dataclass(kw_only=True, config=ArbitraryAllowedConfig)
class FitInputs(FitInputsBase):
    """Model fit inputs for gauss2dfit."""

    jacobian: np.ndarray = pydantic.Field(None, title="The full Jacobian array")
    jacobians: list[list[g2.ImageD]] = pydantic.Field(
        title="Jacobian arrays (views) for each observation",
    )
    outputs_prior: tuple[g2.ImageD, ...] = pydantic.Field(
        title="Jacobian arrays (views) for each free parameter's prior",
    )
    residual: np.ndarray = pydantic.Field(title="The full residual (chi) array")
    residuals: list[g2.ImageD] = pydantic.Field(
        default_factory=list,
        title="Residual (chi) arrays (views) for each observation",
    )
    residuals_prior: g2.ImageD = pydantic.Field(
        title="Shared residual array for all Prior instances",
    )

    @classmethod
    def get_sizes(
        cls,
        model: g2f.Model,
    ):
        """Initialize Jacobian and residual arrays for a model.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model to initialize arrays for.
        """
        priors = model.priors
        n_prior_residuals = sum(len(p) for p in priors)
        params_free = tuple(get_params_uniq(model, fixed=False))
        n_params_free = len(params_free)
        # There's one extra validation array
        n_params_jac = n_params_free + 1
        if not (n_params_jac > 1):
            raise ValueError("Can't fit model with no free parameters")

        n_obs = len(model.data)
        shapes = np.zeros((n_obs, 2), dtype=int)
        ranges_params = [None] * n_obs

        for idx_obs in range(n_obs):
            observation = model.data[idx_obs]
            shapes[idx_obs, :] = (observation.image.n_rows, observation.image.n_cols)
            params = tuple(get_params_uniq(model, fixed=False, channel=observation.channel))
            n_params_obs = len(params)
            ranges_params_obs = [0] * (n_params_obs + 1)
            for idx_param in range(n_params_obs):
                ranges_params_obs[idx_param + 1] = params_free.index(params[idx_param]) + 1
            ranges_params[idx_obs] = ranges_params_obs

        n_free_first = len(ranges_params[0])
        assert all([len(rp) == n_free_first for rp in ranges_params[1:]])

        return n_obs, n_params_jac, n_prior_residuals, shapes

    @classmethod
    def from_model(
        cls,
        model: g2f.Model,
    ):
        """Initialize Jacobian and residual arrays for a model.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model to initialize arrays for.
        """
        n_obs, n_params_jac, n_prior_residuals, shapes = cls.get_sizes(model)
        n_data_obs = np.cumsum(np.prod(shapes, axis=1))
        n_data = n_data_obs[-1]
        size_data = n_data + n_prior_residuals
        shape_jacobian = (size_data, n_params_jac)
        jacobian = np.zeros(shape_jacobian)
        jacobians = [None] * n_obs
        outputs_prior = [None] * n_params_jac
        for idx in range(n_params_jac):
            outputs_prior[idx] = g2.ImageD(jacobian[n_data:, idx].view().reshape((1, n_prior_residuals)))

        residual = np.zeros(size_data)
        residuals = [None] * n_obs
        residuals_prior = g2.ImageD(residual[n_data:].reshape(1, n_prior_residuals))

        offset = 0
        for idx_obs in range(n_obs):
            shape = shapes[idx_obs, :]
            size_obs = shape[0] * shape[1]
            end = offset + size_obs
            jacobians_obs = [None] * n_params_jac
            for idx_jac in range(n_params_jac):
                jacobians_obs[idx_jac] = g2.ImageD(jacobian[offset:end, idx_jac].view().reshape(shape))
            jacobians[idx_obs] = jacobians_obs
            residuals[idx_obs] = g2.ImageD(residual[offset:end].view().reshape(shape))
            offset = end
            if offset != n_data_obs[idx_obs]:
                raise RuntimeError(f"Assigned {offset=} data points != {n_data_obs[idx_obs]=}")
        return cls(
            jacobian=jacobian,
            jacobians=jacobians,
            residual=residual,
            residuals=residuals,
            outputs_prior=outputs_prior,
            residuals_prior=residuals_prior,
        )

    def validate_for_model(self, model: g2f.Model) -> list[str]:
        n_obs, n_params_jac, n_prior_residuals, shapes = self.get_sizes(model)
        n_data = np.sum(np.prod(shapes, axis=1))
        size_data = n_data + n_prior_residuals
        shape_jacobian = (size_data, n_params_jac)

        errors = []

        if self.jacobian.shape != shape_jacobian:
            errors.append(f"{self.jacobian.shape=} != {shape_jacobian=}")

        if len(self.jacobians) != n_obs:
            errors.append(f"{len(self.jacobians)=} != {n_obs=}")

        if len(self.residuals) != n_obs:
            errors.append(f"{len(self.residuals)=} != {n_obs=}")

        if not errors:
            for idx_obs in range(n_obs):
                shape_obs = shapes[idx_obs, :]
                jacobian_obs = self.jacobians[idx_obs]
                n_jacobian_obs = len(jacobian_obs)
                if n_jacobian_obs != n_params_jac:
                    errors.append(f"len(self.jacobians[{idx_obs}])={n_jacobian_obs} != {n_params_jac=}")
                else:
                    for idx_jac in range(n_jacobian_obs):
                        if not all(jacobian_obs[idx_jac].shape == shape_obs):
                            errors.append(f"{jacobian_obs[idx_jac].shape=} != {shape_obs=}")
                if not all(self.residuals[idx_obs].shape == shape_obs):
                    errors.append(f"{self.residuals[idx_obs].shape=} != {shape_obs=}")

        shape_residual_prior = [1, n_prior_residuals]
        if len(self.outputs_prior) != n_params_jac:
            errors.append(f"{len(self.outputs_prior)=} != {n_params_jac=}")
        elif n_prior_residuals > 0:
            for idx in range(n_params_jac):
                if self.outputs_prior[idx].shape != shape_residual_prior:
                    errors.append(f"{self.outputs_prior[idx].shape=} != {shape_residual_prior=}")

        if n_prior_residuals > 0:
            if self.residuals_prior.shape != shape_residual_prior:
                errors.append(f"{self.residuals_prior.shape=} != {shape_residual_prior=}")

        return errors


class ModelFitConfig(pexConfig.Config):
    """Configuration for model fitting."""

    eval_residual = pexConfig.Field[bool](
        doc="Whether to evaluate the residual every iteration before the Jacobian, which can improve "
            "performance if most steps do not call the Jacobian function",
        default=True,
    )
    fit_linear_iter = pexConfig.Field[int](
        doc="The number of iterations to wait before performing a linear fit during optimization."
        " Default 0 disables the feature.",
        default=0,
    )

    def validate(self):
        if not self.fit_linear_iter >= 0:
            raise ValueError(f"{self.fit_linear_iter=} must be >=0")


@dataclass(kw_only=True, config=ArbitraryAllowedConfig)
class FitResult:
    """Results from a Modeller fit, including metadata."""

    # TODO: Why does setting default=ModelFitConfig() cause a circular import?
    config: ModelFitConfig = pydantic.Field(None, title="The configuration for fitting")
    inputs: FitInputs | None = pydantic.Field(None, title="The fit input arrays")
    result: Any | None = pydantic.Field(
        None,
        title="The result object of the fit, directly from the optimizer",
    )
    params_best: np.ndarray | None = pydantic.Field(
        None,
        title="The best-fit parameter array (un-transformed)",
    )
    n_eval_resid: int = pydantic.Field(0, title="Total number of self-reported residual function evaluations")
    n_eval_func: int = pydantic.Field(
        0, title="Total number of optimizer-reported fitness function evaluations"
    )
    n_eval_jac: int = pydantic.Field(
        0, title="Total number of optimizer-reported Jacobian function evaluations"
    )
    time_eval: float = pydantic.Field(0, title="Total runtime spent in model/Jacobian evaluation")
    time_run: float = pydantic.Field(0, title="Total runtime spent in fitting, excluding initial setup")


class Modeller:
    """Fit gauss2d.fit Model instances using Python optimizers.

    Parameters
    ----------
    logger : `logging.Logger`
        The logger. Defaults to calling `_getlogger`.
    """

    def __init__(self, logger=None):
        if logger is None:
            logger = self._get_logger()
        self.logger = logger

    @staticmethod
    def _get_logger():
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.level = logging.INFO

        return logger

    @staticmethod
    def compute_variances(model: g2f.Model, use_diag_only: bool = False, use_svd: bool = False, **kwargs):
        hessian = model.compute_hessian(**kwargs).data
        if use_diag_only:
            return -1 / np.diag(hessian)
        if use_svd:
            u, s, v = np.linalg.svd(-hessian)
            inverse = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
        else:
            inverse = np.linalg.inv(-hessian)
        return np.diag(inverse)

    @staticmethod
    def fit_gaussians_linear(
        gaussians_linear: LinearGaussians,
        observation: g2f.Observation,
        psfmodel: g2f.PsfModel = None,
        fitmethods: dict[str, dict[str, Any]] = None,
        plot: bool = False,
    ) -> dict[str, FitResult]:
        """Fit normalizations for a Gaussian mixture model.

        Parameters
        ----------
        gaussians_linear : LinearGaussians
            The Gaussian components - fixed or otherwise - to fit.
        observation : gauss2d.fit.Observation
            The observation to fit against.
        psfmodel : gauss2d.fit.PsfModel
            A PSF model for the observation, if fitting sources.
        fitmethods : dict[str, dict[str, Any]]
            A dictionary of fitting methods to employ, keyed by method name,
            with a value of a dict of options (kwargs) to pass on.
        plot : bool
            Whether to generate fit residual/diagnostic plots.

        Returns
        -------
        results : dict[str, FitResult]
            Fit results for each method, keyed by the fit method name.
        """
        if psfmodel is None:
            psfmodel = make_psfmodel_null()
        if fitmethods is None:
            fitmethods = {"scipy.optimize.nnls": fitmethods_linear["scipy.optimize.nnls"]}
        else:
            for fitmethod in fitmethods:
                if fitmethod not in fitmethods_linear:
                    raise ValueError(f"Unknown linear {fitmethod=}")
        n_params = len(gaussians_linear.gaussians_free)
        if not (n_params > 0):
            raise ValueError(f"!({len(gaussians_linear.gaussians_free)=}>0); can't fit with no free params")
        image = observation.image
        shape = image.shape
        coordsys = image.coordsys

        mask_inv = observation.mask_inv.data
        sigma_inv = observation.sigma_inv.data
        if mask_inv is None:
            size = np.prod(shape)
        else:
            sigma_inv = sigma_inv[mask_inv]
            size = np.sum(mask_inv)

        gaussians_psf = psfmodel.gaussians(g2f.Channel.NONE)
        if len(gaussians_linear.gaussians_fixed) > 0:
            image_fixed = make_image_gaussians(
                gaussians_source=gaussians_linear.gaussians_fixed,
                gaussians_kernel=gaussians_psf,
                n_rows=shape[0],
                n_cols=shape[1],
            ).data
            if mask_inv is not None:
                image_fixed = image_fixed[mask_inv]
        else:
            image_fixed = None

        x = np.zeros((size, n_params))

        params = [None] * n_params
        for idx_param, (gaussians_free, param) in enumerate(gaussians_linear.gaussians_free):
            image_free = make_image_gaussians(
                gaussians_source=gaussians_free,
                gaussians_kernel=gaussians_psf,
                n_rows=shape[0],
                n_cols=shape[1],
                coordsys=coordsys,
            ).data
            x[:, idx_param] = ((image_free if mask_inv is None else image_free[mask_inv]) * sigma_inv).flat
            params[idx_param] = param

        y = observation.image.data
        if plot:
            import matplotlib.pyplot as plt

            plt.imshow(y, origin="lower")
            plt.show()
        if mask_inv is not None:
            y = y[mask_inv]
        if image_fixed is not None:
            y -= image_fixed
        y = (y * sigma_inv).flat

        results = {}

        for fitmethod, kwargs in fitmethods.items():
            if fitmethod == "scipy.optimize.nnls":
                values = spopt.nnls(x, y)[0]
            elif fitmethod == "scipy.optimize.lsq_linear":
                kwargs = kwargs if kwargs is not None else fitmethods_linear[fitmethod]
                values = spopt.lsq_linear(x, y, **kwargs).x
            elif fitmethod == "numpy.linalg.lstsq":
                values = np.linalg.lstsq(x, y, **kwargs)[0]
            elif fitmethod == "fastnnls.fnnls":
                from fastnnls import fnnls

                y = x.T.dot(y)
                x = x.T.dot(x)
                values = fnnls(x, y)
            else:
                raise RuntimeError(f"Unknown linear {fitmethod=} not caught earlier (logic error)")
            results[fitmethod] = values
        return results

    def fit_model(
        self,
        model: g2f.Model,
        fitinputs: FitInputs | None = None,
        printout: bool = False,
        config: ModelFitConfig = None,
        **kwargs,
    ) -> FitResult:
        """Fit a model with a nonlinear optimizer.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model to fit.
        fitinputs : `FitInputs`
            An existing FitInputs with jacobian/residual arrays to reuse.
        printout : bool
            Whether to print diagnostic information.
        config : ModelFitConfig
            Configuration settings for model fitting.
        **kwargs
            Keyword arguments to pass to the optimizer.

        Returns
        -------
        result : FitResult
            The results from running the fitter.

        Notes
        -----
        The only supported fitter is scipy.optimize.least_squares.
        """
        if config is None:
            config = ModelFitConfig()
        config.validate()
        if fitinputs is None:
            fitinputs = FitInputs.from_model(model)
        else:
            errors = fitinputs.validate_for_model(model)
            if errors:
                newline = "\n"
                raise ValueError(f"fitinputs validation got errors:\n{newline.join(errors)}")

        def residual_func(params_new, model_jac, model_ll, params, result, jac):
            if not all(~np.isnan(params_new)):
                raise InvalidProposalError(f"optimizer for {model_ll=} proposed non-finite {params_new=}")
            try:
                for param, value in zip(params, params_new):
                    param.value_transformed = value
                    if not np.isfinite(param.value):
                        raise RuntimeError(f"{param=} set to (transformed) non-finite {value=}")
            except RuntimeError as e:
                raise InvalidProposalError(f"optimizer for {model_ll=} proposal generated error={e}")
            config_fit = result.config
            fit_linear_iter = config_fit.fit_linear_iter
            if (fit_linear_iter > 0) and ((result.n_eval_resid + 1) % fit_linear_iter == 0):
                self.fit_model_linear(model_ll, ratio_min=1e-6)
            time_init = time.process_time()
            if config_fit.eval_residual:
                model_ll.evaluate()
            else:
                model_jac.evaluate()
            result.time_eval += time.process_time() - time_init
            result.n_eval_resid += 1
            return -result.inputs.residual

        def jacobian_func(params_new, model_jac, model_ll, params, result, jac):
            if result.config.eval_residual:
                model_jac.evaluate()
            return jac

        if config.eval_residual:
            model_ll = g2f.Model(
                data=model.data, psfmodels=model.psfmodels, sources=model.sources, priors=model.priors,
            )
            model_ll.setup_evaluators(
                evaluatormode=g2f.Model.EvaluatorMode.loglike,
                residuals=fitinputs.residuals,
                residuals_prior=fitinputs.residuals_prior,
            )
        else:
            model_ll = None

        model.setup_evaluators(
            evaluatormode=g2f.Model.EvaluatorMode.jacobian,
            outputs=fitinputs.jacobians,
            residuals=fitinputs.residuals,
            outputs_prior=fitinputs.outputs_prior,
            residuals_prior=fitinputs.residuals_prior,
            print=printout,
            force=True,
        )

        params_free_sorted = tuple(get_params_uniq(model, fixed=False))
        offsets_params = dict(model.offsets_parameters())
        params_offsets = {v: k for (k, v) in offsets_params.items()}
        params_free = tuple(params_offsets[idx] for idx in range(1, len(offsets_params) + 1))
        jac = fitinputs.jacobian[:, 1:]
        # Assert that this is a view, otherwise this won't work
        assert id(jac.base) == id(fitinputs.jacobian)
        n_params_free = len(params_free)
        bounds = ([None] * n_params_free, [None] * n_params_free)
        params_init = [None] * n_params_free

        for idx, param in enumerate(params_free):
            limits = param.limits
            # If the transform has more restrictive limits, use those
            if hasattr(param.transform, "limits"):
                limits_transform = param.transform.limits
                n_within = limits.check(limits_transform.min) + limits.check(limits_transform.min)
                if n_within == 2:
                    limits = limits_transform
                elif n_within != 0:
                    raise ValueError(
                        f"{param=} {param.limits=} and {param.transform.limits=}"
                        f" intersect; one must be a subset of the other"
                    )
            bounds[0][idx] = param.transform.forward(limits.min)
            bounds[1][idx] = param.transform.forward(limits.max)
            if not limits.check(param.value):
                raise RuntimeError(f"{param=}.value_transformed={param.value} not within {limits=}")
            params_init[idx] = param.value_transformed

        results = FitResult(inputs=fitinputs, config=config)
        time_init = time.process_time()
        _ll_init = model.evaluate()
        x_scale_jac_clipped = np.clip(1.0/(np.sum(jac**2, axis=0)**0.5), 1e-5, 1e19)
        result_opt = spopt.least_squares(
            residual_func,
            params_init,
            jac=jacobian_func,
            bounds=bounds,
            args=(model, model_ll, params_free, results, jac),
            x_scale=x_scale_jac_clipped,
            **kwargs,
        )
        results.time_run = time.process_time() - time_init
        results.result = result_opt
        results.params_best = tuple(result_opt.x[offsets_params[param] - 1] for param in params_free_sorted)
        results.n_eval_func = result_opt.nfev
        results.n_eval_jac = result_opt.njev if result_opt.njev else 0
        return results

    # TODO: change to staticmethod if requiring py3.10+
    @classmethod
    def fit_model_linear(
        cls,
        model: g2f.Model,
        idx_obs: int = None,
        ratio_min: float = 0,
        validate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_data = len(model.data)
        n_sources = len(model.sources)
        if n_sources != 1:
            raise ValueError("fit_model_linear does not yet support models with >1 sources")
        if idx_obs is not None:
            if not ((idx_obs >= 0) and (idx_obs < n_data)):
                raise ValueError(f"{idx_obs=} not >=0 and < {len(model.data)=}")
            indices = range(idx_obs, idx_obs + 1)
        else:
            indices = range(n_data)

        if validate:
            model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike)
            loglike_init = model.evaluate()
        else:
            loglike_init = None
        values_init = {}
        values_new = {}

        for idx_obs in indices:
            obs = model.data[idx_obs]
            gaussians_linear = LinearGaussians.make(model.sources[0], channel=obs.channel)
            result = cls.fit_gaussians_linear(gaussians_linear, obs, psfmodel=model.psfmodels[idx_obs])
            values = list(result.values())[0]

            for (_, parameter), ratio in zip(gaussians_linear.gaussians_free, values):
                values_init[parameter] = float(parameter.value)
                if not (ratio >= ratio_min):
                    ratio = ratio_min
                value_new = max(ratio * parameter.value, parameter.limits.min)
                values_new[parameter] = value_new

        for parameter, value in values_new.items():
            # TODO: maybe just np.clip instead
            if not (value > parameter.limits.min):
                value = parameter.limits.min + (1e-5 if (parameter.limits.max == np.Inf) else (
                    0.02 * (parameter.limits.max - parameter.limits.min)
                ))
            elif not (value < parameter.limits.max):
                value = parameter.limits.min + 0.98*(parameter.limits.max - parameter.limits.min)
            parameter.value = value

        if validate:
            loglike_new = model.evaluate()
            if not (sum(loglike_new) > sum(loglike_init)):
                for parameter, value in values_init.items():
                    parameter.value = value
        else:
            loglike_new = None
        return loglike_init, loglike_new

    @staticmethod
    def make_components_linear(
        componentmixture: g2f.ComponentMixture,
    ) -> list[g2f.GaussianComponent]:
        """Make a list of fixed Gaussian components from a ComponentMixture.

        Parameters
        ----------
        componentmixture : `gauss2d.fit.ComponentMixture`
            A component mixture to create a component list for.

        Returns
        -------
        gaussians : `list[gauss2d.fit.GaussianComponent]`
            A list of Gaussians components with fixed parameters and values
            matching those in the original component mixture.
        """
        components = componentmixture.components
        if len(components) == 0:
            raise ValueError(f"Can't get linear Source from {componentmixture=} with no components")
        components_new = [None] * len(components)
        for idx, component in enumerate(components):
            gaussians = component.gaussians(g2f.Channel.NONE)
            # TODO: Support multi-Gaussian components if sensible
            # The challenge would be in mapping linear param values back onto
            # non-linear IntegralModels
            n_g = len(gaussians)
            if not n_g == 1:
                raise ValueError(f"{component=} has {gaussians=} of len {n_g=}!=1")
            gaussian = gaussians.at(0)
            component_new = g2f.GaussianComponent(
                g2f.GaussianParametricEllipse(
                    g2f.SigmaXParameterD(gaussian.ellipse.sigma_x, fixed=True),
                    g2f.SigmaYParameterD(gaussian.ellipse.sigma_y, fixed=True),
                    g2f.RhoParameterD(gaussian.ellipse.rho, fixed=True),
                ),
                g2f.CentroidParameters(
                    g2f.CentroidXParameterD(gaussian.centroid.x, fixed=True),
                    g2f.CentroidYParameterD(gaussian.centroid.y, fixed=True),
                ),
                g2f.LinearIntegralModel(
                    [
                        (g2f.Channel.NONE, g2f.IntegralParameterD(gaussian.integral.value)),
                    ]
                ),
            )
            components_new[idx] = component_new
        return components_new
