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

from dataclasses import dataclass
import gauss2d as g2
import gauss2d.fit as g2f
import numpy as np
import scipy.optimize as spopt
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import fastnnls
    has_fastnnls = True
except ImportError:
    has_fastnnls = False

fitmethods_linear = {
    'scipy.optimize.nnls': {},
    'scipy.optimize.lsq_linear': {'bounds': (1e-5, np.Inf), 'method': 'bvls'},
    'numpy.linalg.lstsq': {'rcond': 1e-3},
}
if has_fastnnls:
    fitmethods_linear['fastnnls.fnnls'] = {}


@dataclass(frozen=True)
class LinearGaussians:
    gaussians_fixed: g2.Gaussians
    gaussians_free: Tuple[Tuple[g2.Gaussians, g2f.Parameter]]


def make_image_gaussians(
    gaussians_source: g2.Gaussians,
    gaussians_kernel: Optional[g2.Gaussians] = None,
    **kwargs,
) -> g2.ImageD:
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
    return g2f.PsfModel(g2f.GaussianComponent.make_uniq_default_gaussians([0], True))


def make_psf_linear_gaussians(componentmixture: g2f.ComponentMixture) -> LinearGaussians:
    components = componentmixture.components
    if not len(components) > 0:
        raise ValueError(f"Can't get linear Source from {source=} with no components")

    gaussians_free = []
    gaussians_fixed = []

    for idx, component in enumerate(components):
        gaussians = component.gaussians(g2f.Channel.NONE)
        # TODO: Support multi-Gaussian components if sensible
        # The challenge would be in mapping linear param values back onto
        # non-linear IntegralModels
        n_g = len(gaussians)
        if not n_g == 1:
            raise ValueError(f"{component=} has {gaussians=} of len {n_g=}!=1")
        param_flux = component.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=g2f.Channel.NONE))
        if len(param_flux) != 1:
            raise ValueError(f"Can't make linear source from {component=} with {len(param_flux)=}")
        param_flux = param_flux[0]
        if param_flux.fixed:
            gaussians_fixed.append(gaussians.at(0))
        else:
            gaussians_free.append((gaussians, param_flux))

    return LinearGaussians(gaussians_fixed=g2.Gaussians(gaussians_fixed), gaussians_free=tuple(gaussians_free))


class Modeller:
    @staticmethod
    def fit_gaussians_linear(
        gaussians_linear: LinearGaussians,
        observation: g2f.Observation,
        psfmodel: g2f.PsfModel = None,
        fitmethods: Dict[str, dict[str, Any]] = None,
        plot: bool = False,
    ):
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

        values_init = [param.value for param in params]

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
        jacobian: np.array = None,
        residual: np.array = None,
        printout: bool = False,
        **kwargs
    ):
        def residual_func(params_new, model, params, jacob, resid):
            for param, value in zip(params, params_new):
                param.value_transformed = value
            model.evaluate()
            return residual

        def jacobian_func(params_new, model, params, jacob, resid):
            return -jacobian

        n_priors = 0
        n_obs = len(model.data)
        n_rows = np.zeros(n_obs, dtype=int)
        n_cols = np.zeros(n_obs, dtype=int)
        datasizes = np.zeros(n_obs, dtype=int)
        ranges_params = [None] * n_obs
        params_free = list({x: None for x in model.parameters(paramfilter=g2f.ParamFilter(fixed=False))})

        # There's one extra validation array
        n_params_jac = len(params_free) + 1
        if not (n_params_jac > 1):
            raise ValueError("Can't fit model with no free parameters")

        for idx_obs in range(n_obs):
            observation = model.data[idx_obs]
            n_rows[idx_obs] = observation.image.n_rows
            n_cols[idx_obs] = observation.image.n_cols
            datasizes[idx_obs] = n_rows[idx_obs] * n_cols[idx_obs]
            params = list({
                x: None
                for x in model.parameters(paramfilter=g2f.ParamFilter(fixed=False, channel=observation.channel))
            })
            n_params_obs = len(params)
            ranges_params_obs = [0] * (n_params_obs + 1)
            for idx_param in range(n_params_obs):
                ranges_params_obs[idx_param + 1] = params_free.index(params[idx_param]) + 1
            ranges_params[idx_obs] = ranges_params_obs

        n_free_first = len(ranges_params[0])
        assert all([len(rp) == n_free_first for rp in ranges_params[1:]])

        datasize = np.sum(datasizes) + n_priors

        has_jacobian = jacobian is not None
        shape_jacobian = (datasize, n_params_jac)
        if has_jacobian:
            if jacobian.shape != datasize:
                raise ValueError(f"jacobian.shape={jacobian.shape} != shape_jacobian={shape_jacobian}")
        else:
            jacobian = np.zeros(shape_jacobian)
        jacobians = [None] * n_obs

        has_residual = residual is not None
        if has_residual:
            if residual.size != datasize:
                raise ValueError(f"residual.size={residual.shape} != datasize={datasize}")
        else:
            residual = np.zeros(datasize)
        residuals = [None] * n_obs
        # jacobian_prior = self.jacobian[datasize:, ].view()

        offset = 0
        for idx_obs in range(n_obs):
            size_obs = datasizes[idx_obs]
            end = offset + size_obs
            shape = (n_rows[idx_obs], n_cols[idx_obs])
            ranges_params_obs = ranges_params[idx_obs]
            jacobians_obs = [None] * (len(ranges_params_obs))
            for idx_param, idx_jac in enumerate(ranges_params_obs):
                jacobians_obs[idx_param] = g2.ImageD(jacobian[offset:end, idx_jac].view().reshape(shape))
            jacobians[idx_obs] = jacobians_obs
            residuals[idx_obs] = g2.ImageD(residual[offset:end].view().reshape(shape))
            offset = end

        model.setup_evaluators(
            evaluatormode=g2f.Model.EvaluatorMode.jacobian,
            outputs=jacobians,
            residuals=residuals,
            print=printout,
        )

        params = list({x: None for x in model.parameters(paramfilter=g2f.ParamFilter(fixed=False))})
        n_params = len(params)
        bounds = ([None] * n_params, [None] * n_params)
        params_init = [None] * n_params

        for idx, param in enumerate(params):
            limits = param.limits
            bounds[0][idx] = limits.min
            bounds[1][idx] = limits.max
            if not (limits.min <= param.value_transformed <= limits.max):
                raise RuntimeError(f'param={param}.value_transformed={param.value_transformed}'
                                   f' not within limits={limits}')
            params_init[idx] = param.value_transformed

        jacobian_full = jacobian
        jacobian = jacobian[:, 1:]
        time_init = time.process_time()
        result = spopt.least_squares(
            residual_func, params_init, jac=jacobian_func, bounds=bounds,
            args=(model, params, jacobian, residuals), x_scale='jac',
            **kwargs
        )
        time_run = time.process_time() - time_init
        return result, time_run, jacobian_full, jacobians, residual, residuals


    @staticmethod
    def make_components_linear(componentmixture: g2f.ComponentMixture, bands: List[g2f.Channel]) -> List[g2f.Component]:
        components = componentmixture.components
        if not len(components) > 0:
            raise ValueError(f"Can't get linear Source from {source=} with no components")
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

    def __init__(self):
        pass
