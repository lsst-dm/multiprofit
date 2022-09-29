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
import numpy as np
import scipy.optimize as spopt
import time


class Modeller:
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

    def __init__(self):
        pass
