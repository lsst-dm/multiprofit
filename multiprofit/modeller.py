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

import gauss2d as g2
import gauss2d.fit as g2f
import logging
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass
import scipy.optimize as spopt
import time
from typing import Any

try:
    import fastnnls
    has_fastnnls = True
except ImportError:
    has_fastnnls = False


class InvalidProposalError(ValueError):
    pass


fitmethods_linear = {
    'scipy.optimize.nnls': {},
    'scipy.optimize.lsq_linear': {'bounds': (1e-5, np.Inf), 'method': 'bvls'},
    'numpy.linalg.lstsq': {'rcond': 1e-3},
}
if has_fastnnls:
    fitmethods_linear['fastnnls.fnnls'] = {}


class ArbitraryAllowedConfig:
    arbitrary_types_allowed = True
    extra = 'forbid'


@dataclass(frozen=True, kw_only=True, config=ArbitraryAllowedConfig)
class LinearGaussians:
    """Helper for linear least-squares fitting of Gaussian mixtures.
    """
    gaussians_fixed: g2.Gaussians = pydantic.Field(title="Fixed Gaussian components")
    gaussians_free: tuple[tuple[g2.Gaussians, g2f.Parameter], ...] = pydantic.Field(
        title="Free Gaussian components")

    @staticmethod
    def make(
        componentmixture: g2f.ComponentMixture,
        channel: g2f.Channel = None,
        is_psf: bool = False,
    ):
        """Make

        Parameters
        ----------
        componentmixture : gauss2d.fit.ComponentMixture
            A component mixture to initialize Gaussians from.
        channel : gauss2d.fit.Channel
            The channel all Gaussians are applicable for.
        is_psf : bool
            Whether the components are a smoothing kernel.

        Returns
        -------
        lineargaussians : `multiprofit.LinearGaussians`
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
            param_flux = component.parameters(
                paramfilter=g2f.ParamFilter(nonlinear=False, channel=channel)
            )
            if len(param_flux) != 1:
                raise ValueError(f"Can't make linear source from {component=} with {len(param_flux)=}")
            param_flux: g2f.Parameter = param_flux[0]
            if param_flux.fixed:
                gaussians_fixed.append(gaussians.at(0))
            else:
                gaussians_free.append((gaussians, param_flux))

        return LinearGaussians(gaussians_fixed=g2.Gaussians(gaussians_fixed),
                               gaussians_free=tuple(gaussians_free))


def make_image_gaussians(
    gaussians_source: g2.Gaussians,
    gaussians_kernel: g2.Gaussians | None = None,
    **kwargs,
) -> g2.ImageD:
    """Make an image array from a set of Gaussians.

    Parameters
    ----------
    gaussians_source : gauss2d.Gaussians
        Gaussians representing components of sources.
    gaussians_kernel : gauss2d.Gaussians
        Gaussians representing the smoothing kernel.
    kwargs
        Additional keyword arguments to pass to gauss2d.make_gaussians_pixel_D
        (i.e. image size, etc)

    Returns
    -------
    image : gauss2d.ImageD
        The rendered image of the given Gaussians.
    """
    if gaussians_kernel is None:
        gaussians_kernel = g2.Gaussians([g2.Gaussian()])
    n_gaussians_kernel = len(gaussians_kernel)
    n_gaussians_source = len(gaussians_source)
    gaussians = g2.ConvolvedGaussians([
        g2.ConvolvedGaussian(source=source, kernel=kernel)
        for source in (gaussians_source.at(idx) for idx in range(n_gaussians_source))
        for kernel in (gaussians_kernel.at(idx) for idx in range(n_gaussians_kernel))
    ])
    return g2.make_gaussians_pixel_D(gaussians=gaussians, **kwargs)


def make_psfmodel_null() -> g2f.PsfModel:
    """Make a default (null) PSF model.

    Returns
    -------
    model : gauss2d.fit.PsfModel
        A null PSF model consisting of a single, normalized, zero-size
        Gaussian.
    """
    return g2f.PsfModel(g2f.GaussianComponent.make_uniq_default_gaussians([0], True))


@dataclass(kw_only=True, config=ArbitraryAllowedConfig)
class FitInputs:
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
        params_free = Modeller.get_params_free(model)
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
            params = tuple({
                x: None for x in model.parameters(
                    paramfilter=g2f.ParamFilter(fixed=False, channel=observation.channel)
                )
            })
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
        jacobians = [None]*n_obs
        outputs_prior = [None]*n_params_jac
        for idx in range(n_params_jac):
            outputs_prior[idx] = g2.ImageD(jacobian[n_data:, idx].view().reshape((1, n_prior_residuals)))

        residual = np.zeros(size_data)
        residuals = [None]*n_obs
        residuals_prior = g2.ImageD(residual[n_data:].reshape(1, n_prior_residuals))

        offset = 0
        for idx_obs in range(n_obs):
            shape = shapes[idx_obs, :]
            size_obs = shape[0]*shape[1]
            end = offset + size_obs
            jacobians_obs = [None]*n_params_jac
            for idx_jac in range(n_params_jac):
                jacobians_obs[idx_jac] = g2.ImageD(jacobian[offset:end, idx_jac].view().reshape(shape))
            jacobians[idx_obs] = jacobians_obs
            residuals[idx_obs] = g2.ImageD(residual[offset:end].view().reshape(shape))
            offset = end
            if offset != n_data_obs[idx_obs]:
                raise RuntimeError(f"Assigned {offset=} data points != {n_data_obs[idx_obs]=}")
        return cls(
            jacobian=jacobian, jacobians=jacobians,
            residual=residual, residuals=residuals,
            outputs_prior=outputs_prior, residuals_prior=residuals_prior,
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
                if len(jacobian_obs) != n_params_jac:
                    errors.append(f"len(self.jacobians[{idx_obs}])={len(jacobian_obs)} != {n_params_jac=}")
                else:
                    for idx_jac in range(jacobian_obs):
                        if jacobian_obs[idx_jac].shape != shape_obs:
                            errors.append(f"{self.jacobians[idx_jac].shape=} != {shape_obs=}")
                if self.residuals[idx_obs].shape != shape_obs:
                    errors.append(f"{self.residuals[idx_obs].shape=} != {shape_obs=}")

        shape_residual_prior = (n_prior_residuals, n_params_jac)
        if len(self.outputs_prior) != n_params_jac:
            errors.append(f"{len(self.outputs_prior)=} != {n_params_jac=}")
        else:
            for idx in range(n_params_jac):
                if self.outputs_prior[idx].shape != shape_residual_prior:
                    errors.append(f"{self.outputs_prior[idx].shape=} != {shape_residual_prior=}")

        shape_residual_prior = (1, n_prior_residuals)
        if self.residuals_prior.shape != shape_residual_prior:
            errors.append(f"{self.residuals_prior.shape=} != {shape_residual_prior=}")

        return errors


@dataclass(kw_only=True, config=ArbitraryAllowedConfig)
class FitResult:
    """Results from a Modeller fit, including metadata.
    """
    inputs: FitInputs | None = pydantic.Field(None, title="The fit input arrays")
    result: Any | None = pydantic.Field(
        None,
        title="The result object of the fit, directly from the optimizer",
    )
    params_best: np.ndarray | None = pydantic.Field(
        None,
        title="The best-fit parameter array (un-transformed)",
    )
    n_eval_func: int = pydantic.Field(0, title="Total number of fitness function evaluations")
    n_eval_jac: int = pydantic.Field(0, title="Total number of Jacobian function evaluations")
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
            fitmethods = {'scipy.optimize.nnls': fitmethods_linear['scipy.optimize.nnls']}
        else:
            for fitmethod in fitmethods:
                if fitmethod not in fitmethods_linear:
                    raise ValueError(f"Unknown linear {fitmethod=}")
        n_params = len(gaussians_linear.gaussians_free)
        if not (n_params > 0):
            raise ValueError(f"!({len(gaussians_linear.gaussians_free)=}>0); can't fit with no free params")
        image = observation.image.data
        shape = image.shape

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
                n_rows=shape[0], n_cols=shape[1],
            ).data
            if mask_inv is not None:
                image_fixed = image_fixed[mask_inv]
        else:
            image_fixed = None

        x = np.zeros((size, n_params))

        params = [None]*n_params
        for idx_param, (gaussians_free, param) in enumerate(gaussians_linear.gaussians_free):
            image_free = make_image_gaussians(
                gaussians_source=gaussians_free,
                gaussians_kernel=gaussians_psf,
                n_rows=shape[0], n_cols=shape[1],
            ).data
            x[:, idx_param] = (
                (image_free if mask_inv is None else image_free[mask_inv])*sigma_inv
            ).flat
            params[idx_param] = param

        y = observation.image.data
        if plot:
            import matplotlib.pyplot as plt
            plt.imshow(y, origin='lower')
            plt.show()
        if mask_inv is not None:
            y = y[mask_inv]
        if image_fixed is not None:
            y -= image_fixed
        y = (y * sigma_inv).flat

        results = {}

        for fitmethod, kwargs in fitmethods.items():
            if fitmethod == 'scipy.optimize.nnls':
                values = spopt.nnls(x, y)[0]
            elif fitmethod == 'scipy.optimize.lsq_linear':
                kwargs = kwargs if kwargs is not None else fitmethods_linear[fitmethod]
                values = spopt.lsq_linear(x, y, **kwargs).x
            elif fitmethod == 'numpy.linalg.lstsq':
                values = np.linalg.lstsq(x, y, **kwargs)[0]
            elif fitmethod == 'fastnnls.fnnls':
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
        **kwargs
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
        kwargs
            Keyword arguments to pass to the optimizer.

        Returns
        -------
        result : FitResult
            The results from running the fitter.

        Notes
        -----
        The only supported fitter is scipy.optimize.least_squares.
        """
        if fitinputs is None:
            fitinputs = FitInputs.from_model(model)
        else:
            errors = fitinputs.validate_for_model(model)
            if errors:
                newline = "\n"
                raise ValueError(f"fitinputs validation got errors:\n{newline.join(errors)}")

        def residual_func(params_new, model, params, result):
            if not all(~np.isnan(params_new)):
                raise InvalidProposalError(f"optimizer for {model=} proposed non-finite {params_new=}")
            try:
                for param, value in zip(params, params_new):
                    param.value_transformed = value
            except RuntimeError as e:
                raise InvalidProposalError(f"optimizer for {model=} proposal generated error={e}")
            time_init = time.process_time()
            model.evaluate()
            result.time_eval += time.process_time() - time_init
            return result.inputs.residual

        def jacobian_func(params_new, model, params, result):
            return -result.inputs.jacobian[:, 1:]

        model.setup_evaluators(
            evaluatormode=g2f.Model.EvaluatorMode.jacobian,
            outputs=fitinputs.jacobians,
            residuals=fitinputs.residuals,
            outputs_prior=fitinputs.outputs_prior,
            residuals_prior=fitinputs.residuals_prior,
            print=printout,
            force=True,
        )

        params_free = self.get_params_free(model=model)
        n_params_free = len(params_free)
        bounds = ([None]*n_params_free, [None]*n_params_free)
        params_init = [None]*n_params_free

        for idx, param in enumerate(params_free):
            limits = param.transform.limits if hasattr(param.transform, 'limits') else param.limits
            bounds[0][idx] = param.transform.forward(limits.min)
            bounds[1][idx] = param.transform.forward(limits.max)
            if not (limits.min <= param.value <= limits.max):
                raise RuntimeError(f'{param=}.value={param.value_transforme} not within {limits=}')
            params_init[idx] = param.value_transformed

        results = FitResult(inputs=fitinputs)
        time_init = time.process_time()
        result_opt = spopt.least_squares(
            residual_func, params_init, jac=jacobian_func, bounds=bounds,
            args=(model, params_free, results), x_scale='jac',
            **kwargs
        )
        results.time_run = time.process_time() - time_init
        results.result = result_opt
        results.params_best = result_opt.x
        results.n_eval_func = result_opt.nfev
        results.n_eval_jac = result_opt.njev if result_opt.njev else 0
        return results

    @staticmethod
    def get_params_free(model: g2f.Model) -> tuple[g2f.Parameter]:
        """Get the list of free parameters for a model.

        Parameters
        ----------
        model : `gauss2d.fit.Model`
            The model to retrieve parameters for.

        Returns
        -------
        parameters : `tuple[gauss2d.fit.Parameter]`
            The list of free parameters.
        """
        return tuple({x: None for x in model.parameters(paramfilter=g2f.ParamFilter(fixed=False))})

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
                g2f.LinearIntegralModel({g2f.Channel.NONE: g2f.IntegralParameterD(gaussian.integral.value)}),
            )
            components_new[idx] = component_new
        return components_new

