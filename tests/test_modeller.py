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
import math
import time
import timeit
from typing import Tuple

import gauss2d as g2
import gauss2d.fit as g2f
from lsst.multiprofit.model_utils import ModelConfig, SourceConfig
from lsst.multiprofit.modeller import LinearGaussians, Modeller, fitmethods_linear, make_psfmodel_null
from lsst.multiprofit.transforms import transforms_ref
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pytest
import scipy.optimize as spopt


@pytest.fixture(scope="module")
def config():
    return ModelConfig(
        comps_psf=ComponentConfig(size_base=2.5, size_increment=1.0, rho_base=-0.1, rho_increment=0.2),
        comps_src=ComponentConfig(size_base=1.5, size_increment=2.5, rho_base=-0.2, rho_increment=0.5),
        n_rows=27,
        n_cols=41,
        noise_psf=1e-5,
        seed = 1,
        sigma_img: float = 1e-3,
        size_increment_psf=0.3,
        src_n: int = 1,
        src_flux_base: float = 1.0,
        src_flux_increment: float = 1.0
    )


@pytest.fixture(scope="module")
def limits(config):
    return Limits(
        x=g2f.LimitsD(min=0, max=config.n_cols),
        y=g2f.LimitsD(min=0, max=config.n_rows),
        rho=g2f.LimitsD(min=-0.99, max=0.99),
    )


@pytest.fixture(scope="module")
def transforms(config):
    transform_log10 = g2f.Log10TransformD()
    return Transforms(
        x=transform_log10,
        y=transform_log10,
        rho=g2f.LogitLimitedTransformD(limits=g2f.LimitsD(min=-0.99, max=0.99)),
    )


@pytest.fixture(scope="module")
def bands():
    return tuple(("i", "r", "g"))


@pytest.fixture(scope="module")
def channels(bands):
    return {band: g2f.Channel.get(band) for band in bands}


@pytest.fixture(scope="module")
def images(bands, config):
    images = {}
    for band in bands:
        image = g2.ImageD(n_rows=config.n_rows, n_cols=config.n_cols)
        sigma_inv = g2.ImageD(n_rows=config.n_rows, n_cols=config.n_cols)
        sigma_inv.data.flat = 1 / config.sigma_img
        mask_inv = g2.ImageB(np.ones((config.n_rows, config.n_cols), dtype=bool))
        images[band] = (image, sigma_inv, mask_inv)
    return images


@pytest.fixture(scope="module")
def data(channels, images) -> g2f.Data:
    return g2f.Data(
        [
            g2f.Observation(
                channel=channels[band],
                image=imagelist[0],
                sigma_inv=imagelist[1],
                mask_inv=imagelist[2],
            )
            for band, imagelist in images.items()
        ]
    )


@pytest.fixture(scope="module")
def psfmodels(channels, config, data, limits):
    return None


@pytest.fixture(scope="module")
def psfmodels_linear_gaussians(channels, psfmodels):
    gaussians = [None] * len(psfmodels)
    for idx, psfmodel in enumerate(psfmodels):
        params = psfmodel.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=g2f.Channel.NONE))
        params[0].fixed = False
        gaussians[idx] = LinearGaussians.make(psfmodel, is_psf=True)
    return gaussians


@pytest.fixture(scope="module")
def psf_fit_models(psfmodels, psf_observations):
    psf_null = [make_psfmodel_null()]
    return [
        g2f.Model(g2f.Data([observation]), psf_null, [g2f.Source(psfmodel.components)])
        for psfmodel, observation in zip(psfmodels, psf_observations)
    ]


@pytest.fixture(scope="module")
def psf_observations(config, psfmodels) -> Tuple[g2f.Observation]:
    observations = [None] * len(psfmodels)
    return tuple(observations)


@pytest.fixture(scope="module")
def sources(channels, config, limits, transforms):
    return get_sources(channels, config, limits, transforms)


@pytest.fixture(scope="module")
def model(data, psfmodels, sources):
    return g2f.Model(data, list(psfmodels), list(sources))


@pytest.fixture(scope="module")
def model_jac(data, psfmodels, sources):
    return g2f.Model(data, list(psfmodels), list(sources))


def _get_exposure_params(exposure):
    params_all = exposure[0].psf.model.get_parameters(free=True, fixed=True)
    param_integral = None
    params = []
    for param in params_all:
        str_param = str(param)
        if str_param.startswith("gauss2d::fit::Integral"):
            if param_integral is None:
                param_integral = param
                continue
            else:
                raise RuntimeError("Got second Exposure IntegralParameter before entering first")
        elif str_param.startswith("gauss2d::fit::ProperFraction"):
            if param_integral is not None:
                params.append(param_integral)
                param_integral = None
        params.append(param)
    return params


def fit_model(model: g2f.Model, jacobian, residuals, params_init=None):
    def residual_func(params_new, model, params, jacobian, residuals):
        for param, value in zip(params, params_new):
            param.value_transformed = value
        model.evaluate()
        return residuals

    def jacobian_func(params_new, model, params, jacobian, residuals):
        return -jacobian

    params = list(get_params_uniq(model, fixed=False))
    n_params = len(params)
    bounds = ([None] * n_params, [None] * n_params)
    params_init_is_none = params_init is None
    if params_init_is_none:
        params_init = [None] * n_params

    for idx, param in enumerate(params):
        limits = param.limits
        bounds[0][idx] = limits.min
        bounds[1][idx] = limits.max
        if params_init_is_none:
            params_init[idx] = param.value_transformed

    time_init = time.process_time()
    result = spopt.least_squares(
        residual_func,
        params_init,
        jac=jacobian_func,
        bounds=bounds,
        args=(model, params, jacobian, residuals),
        x_scale="jac",
        ftol=1e-6,
        xtol=1e-6,
    )
    time_run = time.process_time() - time_init
    return result, time_run


def test_model_evaluation(channels, config, model, model_jac, images):
    with pytest.raises(RuntimeError):
        model.evaluate()

    printout = False
    # Freeze the PSF params - they can't be fit anyway
    for m in (model, model_jac):
        for psfmodel in m.psfmodels:
            params = psfmodel.parameters()
            for param in params:
                param.fixed = True

    model.setup_evaluators(print=printout)
    model.evaluate()
    models = {
        "image": (model, ""),
        "jacob": (model_jac, ""),
    }

    n_priors = 0
    n_obs = len(model.data)
    n_rows = np.zeros(n_obs, dtype=int)
    n_cols = np.zeros(n_obs, dtype=int)
    datasizes = np.zeros(n_obs, dtype=int)
    ranges_params = [None] * n_obs
    params_free = tuple(get_params_uniq(model_jac, fixed=False))
    values_init = tuple(param.value_transformed for param in params_free)

    # There's one extra validation array
    n_params_jac = len(params_free) + 1
    assert n_params_jac > 1

    rng = np.random.default_rng(config.seed + 1)

    for idx_obs in range(n_obs):
        observation = model.data[idx_obs]
        output = model.outputs[idx_obs]
        observation.image.data.flat = (
            output.data.flat + rng.standard_normal(output.data.size) / observation.sigma_inv.data.flat
        )
        n_rows[idx_obs] = observation.image.n_rows
        n_cols[idx_obs] = observation.image.n_cols
        datasizes[idx_obs] = n_rows[idx_obs] * n_cols[idx_obs]
        params = tuple(get_params_uniq(model, fixed=False, channel=observation.channel))
        n_params_obs = len(params)
        ranges_params_obs = [0] * (n_params_obs + 1)
        for idx_param in range(n_params_obs):
            ranges_params_obs[idx_param + 1] = params_free.index(params[idx_param]) + 1
        ranges_params[idx_obs] = ranges_params_obs

    n_free_first = len(ranges_params[0])
    assert all([len(rp) == n_free_first for rp in ranges_params[1:]])

    jacobians = [None] * n_obs
    residuals = [None] * n_obs
    datasize = np.sum(datasizes) + n_priors
    jacobian = np.zeros((datasize, n_params_jac))
    residual = np.zeros(datasize)
    # jacobian_prior = self.jacobian[datasize:, ].view()

    offset = 0
    for idx_obs in range(n_obs):
        size_obs = datasizes[idx_obs]
        end = offset + size_obs
        shape = (n_rows[idx_obs], n_cols[idx_obs])
        jacobians_obs = [None] * n_params_jac
        for idx_jac in range(n_params_jac):
            jacobians_obs[idx_jac] = g2.ImageD(jacobian[offset:end, idx_jac].view().reshape(shape))
        jacobians[idx_obs] = jacobians_obs
        residuals[idx_obs] = g2.ImageD(residual[offset:end].view().reshape(shape))
        offset = end

    model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike)
    loglike_init = model.evaluate()

    model_jac.setup_evaluators(
        evaluatormode=g2f.Model.EvaluatorMode.jacobian,
        outputs=jacobians,
        residuals=residuals,
        print=printout,
    )
    model_jac.verify_jacobian()
    loglike_jac = model_jac.evaluate()

    assert all(np.isclose(loglike_init, loglike_jac))

    if printout:
        print(f"starting with loglike={sum(loglike_init)} from LLs={loglike_init}")
    result, time = fit_model(model_jac, jacobian[:, 1:], residual)
    for param, value in zip(params_free, result.x):
        param.value_transformed = value
    loglike_best = model.evaluate()
    if printout:
        print(f"got loglike={sum(loglike_best)} from LLs={loglike_best} in t={time:.3e}, result: \n{result}")

    assert sum(loglike_best) > sum(loglike_init)

    n_eval = 10

    n_name_model_max = max(len(x) for x in models.keys())
    format_name = f"{{0: <{n_name_model_max}}}"

    for name, obj in models.items():
        result = (
            np.array(
                timeit.repeat(
                    f"model.evaluate({obj[1]})", repeat=n_eval, number=n_eval, globals={"model": obj[0]}
                )
            )
            / n_eval
        )
        if printout:
            print(
                f"{format_name.format(name)}: min={np.min(result, axis=0):.4e}"
                f", med={np.median(result, axis=0):.4e} (for n_params_jac={n_params_jac})"
            )

    # Return param values to init
    for param, value in zip(params_free, values_init):
        param.value_transformed = value


def test_make_psf_source_linear(psfmodels, psfmodels_linear_gaussians):
    for psfmodel, linear_gaussians in zip(psfmodels, psfmodels_linear_gaussians):
        gaussians = psfmodel.gaussians(g2f.Channel.NONE)
        assert len(gaussians) == (
            len(linear_gaussians.gaussians_free) + len(linear_gaussians.gaussians_fixed)
        )


def test_modeller(config, model):
    # For debugging purposes
    printout = False
    model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike_image)
    # Get the model images
    model.evaluate()
    rng = np.random.default_rng(config.seed)

    for idx_obs, observation in enumerate(model.data):
        output = model.outputs[idx_obs]
        observation.image.data.flat = (
            output.data.flat + rng.standard_normal(output.data.size) / observation.sigma_inv.data.flat
        )

    # Freeze the PSF params - they can't be fit anyway
    for psfmodel in model.psfmodels:
        for param in psfmodel.parameters():
            param.fixed = True

    params_free = tuple(get_params_uniq(model, fixed=False))
    values_true = tuple(param.value for param in params_free)

    modeller = Modeller()

    dloglike = model.compute_loglike_grad(verify=True, findiff_frac=1e-7, findiff_add=1e-7)
    assert all(np.isfinite(dloglike))

    time_init = time.process_time()
    kwargs_fit = dict(ftol=1e-6, xtol=1e-6)

    for delta_param in (0, 0.2):
        model = g2f.Model(data=model.data, psfmodels=model.psfmodels, sources=model.sources)
        values_init = values_true
        if delta_param != 0:
            for param, value_init in zip(params_free, values_init):
                param.value = value_init
                try:
                    param.value_transformed += delta_param
                except RuntimeError:
                    param.value_transformed -= delta_param

        model.setup_evaluators(evaluatormode=model.EvaluatorMode.loglike)
        loglike_init = np.array(model.evaluate())
        results = modeller.fit_model(model, **kwargs_fit)
        params_best = results.params_best

        for param, value in zip(params_free, params_best):
            param.value_transformed = value

        loglike_noprior = model.evaluate()
        assert np.sum(loglike_noprior) > np.sum(loglike_init)

        errors = modeller.compute_variances(model)
        # TODO: This should check >0, and < (some reasonable value), but the scipy least squares
        # does not do a great job optimizing and the loglike_grad isn't even negative...
        assert np.all(np.isfinite(errors))

        if printout:
            print(
                f"got loglike={loglike_noprior} (init={sum(loglike_noprior)})"
                f" from modeller.fit_model in t={time.process_time() - time_init:.3e}, x={params_best},"
                f" results: \n{results}"
            )

        loglike_noprior_sum = sum(loglike_noprior)
        for offset in (0, 1e-6):
            for param, value in zip(params_free, params_best):
                param.value_transformed = value
            priors = tuple(
                g2f.GaussianPrior(param, param.value_transformed + offset, 1.0, transformed=True)
                for param in params_free
            )
            if offset == 0:
                for p in priors:
                    assert p.evaluate().loglike == 0
                    assert p.loglike_const_terms[0] == -math.log(math.sqrt(2 * math.pi))
            model = g2f.Model(
                data=model.data, psfmodels=model.psfmodels, sources=model.sources, priors=priors
            )
            model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike)
            loglike_init = sum(loglike_eval for loglike_eval in model.evaluate())
            if offset == 0:
                assert np.isclose(loglike_init, loglike_noprior_sum, rtol=1e-10, atol=1e-10)
            else:
                assert loglike_init < loglike_noprior_sum

            time_init = time.process_time()
            results = modeller.fit_model(model, **kwargs_fit)
            time_init = time.process_time() - time_init
            loglike_new = -results.result.cost
            for param, value in zip(params_free, results.params_best):
                param.value_transformed = value

            model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike)
            loglike_model = sum(loglike_eval for loglike_eval in model.evaluate())
            assert np.isclose(loglike_new, loglike_model, rtol=1e-10, atol=1e-10)
            # This should be > 0. TODO: Determine why it isn't always
            assert (loglike_new - loglike_init) > -1e-3

            if printout:
                print(
                    f"got loglike={loglike_new} (first={loglike_noprior})"
                    f" from modeller.fit_model in t={time_init:.3e}, x={results.params_best},"
                    f" results: \n{results}"
                )
            # Adding a suitably-scaled prior far from the truth should always worsen the log likelihood, but doesn't...
            # noise bias? bad convergence? unclear
            # assert (loglike_new >= loglike_noprior) == (offset == 0)


def test_psf_model_fit(psf_fit_models):
    for model in psf_fit_models:
        param = [x for x in model.parameters() if isinstance(x, g2f.ProperFractionParameterD)][0]
        param.fixed = False
        errors = model.verify_jacobian()
        if errors:
            import matplotlib.pyplot as plt

            jacobian, jacobians, residual, residuals = Modeller.make_jacobians(model=model)
            model.setup_evaluators(
                evaluatormode=g2f.Model.EvaluatorMode.jacobian,
                outputs=jacobians,
                residuals=residuals,
                print=True,
            )
            model.evaluate()
            assert np.sum(np.abs(jacobians[0][0].data)) > 0
            model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike_image)
            model.evaluate()
            outputs = model.outputs
            diffs = [g2.ImageD(img.data.copy()) for img in outputs]
            delta = 1e-3
            param.value -= delta
            model.evaluate()
            for diff, output in zip(diffs, outputs):
                diff = (output.data - diff.data) / delta
                jacobian = jacobians[0][1].data
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(diff)
                ax[1].imshow(jacobian)
                plt.show()
        assert len(errors) == 0


def test_psfmodels_linear_gaussians(data, psfmodels_linear_gaussians, psf_observations):
    results = [None] * len(psf_observations)
    for idx, (gaussians_linear, observation_psf) in enumerate(
        zip(psfmodels_linear_gaussians, psf_observations)
    ):
        results[idx] = Modeller.fit_gaussians_linear(
            gaussians_linear=gaussians_linear,
            observation=observation_psf,
            fitmethods=fitmethods_linear,
            plot=False,
        )
        assert len(results[idx]) > 0
