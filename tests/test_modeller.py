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

import math
import time

import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    CentroidConfig,
    FluxFractionParameterConfig,
    FluxParameterConfig,
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.model_utils import make_image_gaussians, make_psf_model_null
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.modeller import FitInputs, LinearGaussians, Modeller, fitmethods_linear
from lsst.multiprofit.observationconfig import CoordinateSystemConfig, ObservationConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pytest

sigma_inv = 1e4


@pytest.fixture(scope="module")
def channels() -> dict[str, g2f.Channel]:
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


@pytest.fixture(scope="module")
def data(channels) -> g2f.DataD:
    n_rows, n_cols = 25, 27
    x_min, y_min = 0, 0

    dn_rows, dn_cols = 2, -3
    dx_min, dy_min = -1, 1

    observations = []
    for idx, band in enumerate(channels):
        config = ObservationConfig(
            band=band,
            coordsys=CoordinateSystemConfig(
                x_min=x_min + idx*dx_min,
                y_min=y_min + idx*dy_min,
            ),
            n_rows=n_rows + idx*dn_rows,
            n_cols=n_cols + idx*dn_cols,
        )
        observation = config.make_observation()
        observation.image.fill(0)
        observation.sigma_inv.fill(sigma_inv)
        observation.mask_inv.fill(1)
        observations.append(observation)
    return g2f.DataD(observations)


@pytest.fixture(scope="module")
def psf_models(channels) -> list[g2f.PsfModel]:
    rho, size_x, size_y = 0.12, 1.6, 1.2
    drho, dsize_x, dsize_y = -0.3, 1.1, 1.9
    drho_chan, dsize_x_chan, dsize_y_chan = 0.03, 0.12, 0.14
    frac, dfrac = 0.62, -0.08

    n_components = 2
    psf_models = []

    for idx_chan, channel in enumerate(channels.values()):
        frac_chan = frac + idx_chan*dfrac
        config = SourceConfig(
            component_groups={
                'psf': ComponentGroupConfig(
                    components_gauss={
                        str(idx): GaussianComponentConfig(
                            rho=ParameterConfig(value_initial=rho + idx*drho + idx_chan*drho_chan),
                            size_x=ParameterConfig(
                                value_initial=size_x + idx*dsize_x + idx_chan*dsize_x_chan),
                            size_y=ParameterConfig(
                                value_initial=size_y + idx*dsize_y + idx_chan*dsize_y_chan),
                            **({
                                "flux": FluxParameterConfig(value_initial=1.0, fixed=True),
                                "fluxfrac": FluxFractionParameterConfig(value_initial=frac_chan, fixed=False),
                            } if (idx == 0) else {})
                        )
                        for idx in range(n_components)
                    },
                    is_fractional=True,
                )
            },
        )
        config.validate()
        psf_model, priors = config.make_psf_model([
            component_group.get_fluxes_default(
                channels=(g2f.Channel.NONE,),
                component_configs=component_group.get_component_configs(),
                is_fractional=component_group.is_fractional,
            )
            for component_group in config.component_groups.values()
        ])
        psf_models.append(psf_model)
    return psf_models


@pytest.fixture(scope="module")
def model(channels, data, psf_models) -> g2f.ModelD:
    rho, size_x, size_y, sersicn, flux = 0.4, 1.5, 1.9, 1.0, 4.7
    drho, dsize_x, dsize_y, dsersicn, dflux = -0.9, 2.5, 5.4, 3.0, 13.9

    components_sersic = {}
    fluxes_group = []

    # Linear interpolators fail to compute accurate likelihoods at knot values
    is_linear_interp = g2f.SersicMixComponentIndexParameterD(
        interpolator=SersicComponentConfig().get_interpolator(4)
    ).interptype == g2f.InterpType.linear

    for idx, name in enumerate(("exp", "dev")):
        components_sersic[name] = SersicComponentConfig(
            rho=ParameterConfig(value_initial=rho + idx*drho),
            size_x=ParameterConfig(value_initial=size_x + idx*dsize_x),
            size_y=ParameterConfig(value_initial=size_y + idx*dsize_y),
            sersic_index=SersicIndexParameterConfig(
                # Add a small offset since 1.0 and 4.0 are bound to be knots
                value_initial=sersicn + idx * dsersicn + 1e-4*is_linear_interp,
                fixed=idx == 0,
                prior_mean=None,
            ),
        )
        fluxes_comp = {
            channel: flux + idx_channel*dflux*idx
            for idx_channel, channel in enumerate(channels.values())
        }
        fluxes_group.append(fluxes_comp)

    modelconfig = ModelConfig(
        sources={
            "src": SourceConfig(
                component_groups={
                    "": ComponentGroupConfig(
                        components_sersic=components_sersic,
                        centroids={"default": CentroidConfig(
                            x=ParameterConfig(value_initial=12.14, fixed=True),
                            y=ParameterConfig(value_initial=13.78, fixed=True),
                        )},
                    ),
                }
            ),
        },
    )
    model = modelconfig.make_model([[fluxes_group]], data=data, psf_models=psf_models)
    return model


@pytest.fixture(scope="module")
def model_jac(model) -> g2f.ModelD:
    model_jac = g2f.ModelD(data=model.data, psfmodels=model.psfmodels, sources=model.sources)
    return model_jac


@pytest.fixture(scope="module")
def psf_observations(psf_models) -> list[g2f.ObservationD]:
    config = ObservationConfig(n_rows=17, n_cols=19)
    rng = np.random.default_rng(1)

    observations = []
    for psf_model in psf_models:
        observation = config.make_observation()
        # Have to make a duplicate image here because one can only call
        # make_image_gaussians with an owning pointer, whereas
        # observation.image is a reference
        image = g2.ImageD(observation.image.data)
        # Make the kernel centered
        gaussians_source = psf_model.gaussians(g2f.Channel.NONE)
        for idx in range(len(gaussians_source)):
            gaussian_idx = gaussians_source.at(idx)
            gaussian_idx.centroid.x = image.n_cols/2.
            gaussian_idx.centroid.y = image.n_rows/2.
        gaussians_kernel = g2.Gaussians([g2.Gaussian()])
        make_image_gaussians(
            gaussians_source=gaussians_source,
            gaussians_kernel=gaussians_kernel,
            output=image,
        )
        image.data.flat += 1e-4 * rng.standard_normal(image.data.size)
        observation.mask_inv.fill(1)
        observation.sigma_inv.fill(1e3)
        observations.append(observation)
    return observations


@pytest.fixture(scope="module")
def psf_fit_models(psf_models, psf_observations):
    psf_null = [make_psf_model_null()]
    return [
        g2f.ModelD(g2f.DataD([observation]), psf_null, [g2f.Source(psf_model.components)])
        for psf_model, observation in zip(psf_models, psf_observations)
    ]


def test_model_evaluation(channels, model, model_jac):
    with pytest.raises(RuntimeError):
        model.evaluate()

    printout = False
    # Freeze the PSF params - they can't be fit anyway
    for m in (model, model_jac):
        for psf_model in m.psfmodels:
            params = psf_model.parameters()
            for param in params:
                param.fixed = True

    model.setup_evaluators(print=printout)
    model.evaluate()

    n_priors = 0
    n_obs = len(model.data)
    n_rows = np.zeros(n_obs, dtype=int)
    n_cols = np.zeros(n_obs, dtype=int)
    datasizes = np.zeros(n_obs, dtype=int)
    ranges_params = [None] * n_obs
    params_free = tuple(get_params_uniq(model_jac, fixed=False))

    # There's one extra validation array
    n_params_jac = len(params_free) + 1
    assert n_params_jac > 1

    rng = np.random.default_rng(2)

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

    model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
    loglike_init = model.evaluate()

    model_jac.setup_evaluators(
        evaluatormode=g2f.EvaluatorMode.jacobian,
        outputs=jacobians,
        residuals=residuals,
        print=printout,
    )
    model_jac.verify_jacobian()
    loglike_jac = model_jac.evaluate()

    assert all(np.isclose(loglike_init, loglike_jac))


@pytest.fixture(scope="module")
def psf_models_linear_gaussians(channels, psf_models):
    gaussians = [None] * len(psf_models)
    for idx, psf_model in enumerate(psf_models):
        params = psf_model.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=g2f.Channel.NONE))
        params[0].fixed = False
        gaussians[idx] = LinearGaussians.make(psf_model, is_psf=True)
        # If this is not done, test_psf_model_fit will fail
        params[0].fixed = True
    return gaussians


def test_make_psf_source_linear(psf_models, psf_models_linear_gaussians):
    for psf_model, linear_gaussians in zip(psf_models, psf_models_linear_gaussians):
        gaussians = psf_model.gaussians(g2f.Channel.NONE)
        assert len(gaussians) == (
            len(linear_gaussians.gaussians_free) + len(linear_gaussians.gaussians_fixed)
        )


def test_modeller(model):
    # For debugging purposes
    printout = False
    model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike_image)
    # Get the model images
    model.evaluate()
    rng = np.random.default_rng(3)

    for idx_obs, observation in enumerate(model.data):
        output = model.outputs[idx_obs]
        observation.image.data.flat = (
            output.data.flat + rng.standard_normal(output.data.size) / observation.sigma_inv.data.flat
        )

    # Freeze the PSF params - they can't be fit anyway
    for psf_model in model.psfmodels:
        for param in psf_model.parameters():
            param.fixed = True

    params_free = tuple(get_params_uniq(model, fixed=False))
    values_true = tuple(param.value for param in params_free)

    modeller = Modeller()

    dloglike = model.compute_loglike_grad(verify=True, findiff_frac=1e-8, findiff_add=1e-8)
    assert all(np.isfinite(dloglike))

    time_init = time.process_time()
    kwargs_fit = dict(ftol=1e-6, xtol=1e-6)

    for delta_param in (0, 0.2):
        model = g2f.ModelD(data=model.data, psfmodels=model.psfmodels, sources=model.sources)
        values_init = values_true
        if delta_param != 0:
            for param, value_init in zip(params_free, values_init):
                param.value = value_init
                try:
                    param.value_transformed += delta_param
                except RuntimeError:
                    param.value_transformed -= delta_param

        model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
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
            model = g2f.ModelD(
                data=model.data, psfmodels=model.psfmodels, sources=model.sources, priors=priors
            )
            model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
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

            model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike)
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
            # Adding a suitably-scaled prior far from the truth should always
            # worsen loglikel, but doesn't - why? noise bias? bad convergence?
            # assert (loglike_new >= loglike_noprior) == (offset == 0)


def test_psf_model_fit(psf_fit_models):
    for model in psf_fit_models:
        params = get_params_uniq(model.sources[0])
        for param in params:
            # Fitting the total flux won't work in a fractional model (yet)
            if isinstance(param, g2f.IntegralParameterD):
                assert param.fixed
            else:
                param.fixed = False
        # Necessary whenever parameters are freed/fixed
        model.setup_evaluators(g2f.EvaluatorMode.jacobian, force=True)
        errors = model.verify_jacobian(rtol=5e-4, atol=5e-4, findiff_add=1e-6, findiff_frac=1e-6)
        if errors:
            import matplotlib.pyplot as plt
            print(model.parameters())

            fitinputs = FitInputs.from_model(model)
            model.setup_evaluators(
                evaluatormode=g2f.EvaluatorMode.jacobian,
                outputs=fitinputs.jacobians,
                residuals=fitinputs.residuals,
                print=True,
                force=True,
            )
            model.evaluate(print=True)
            assert (fitinputs.jacobians[0][0].data == 0).all()
            assert np.sum(np.abs(fitinputs.jacobians[0][1].data)) > 0
            model.setup_evaluators(evaluatormode=g2f.EvaluatorMode.loglike_image)
            model.evaluate()
            outputs = model.outputs
            diffs = [g2.ImageD(img.data.copy()) for img in outputs]
            delta = 1e-5
            param.value -= delta
            model.evaluate()
            for diff, output in zip(diffs, outputs):
                diff = (output.data - diff.data) / delta
                jacobian = fitinputs.jacobians[0][1].data
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(diff)
                ax[1].imshow(jacobian)
                plt.show()
        assert len(errors) == 0


def test_psf_models_linear_gaussians(data, psf_models_linear_gaussians, psf_observations):
    results = [None] * len(psf_observations)
    for idx, (gaussians_linear, observation_psf) in enumerate(
        zip(psf_models_linear_gaussians, psf_observations)
    ):
        results[idx] = Modeller.fit_gaussians_linear(
            gaussians_linear=gaussians_linear,
            observation=observation_psf,
            fitmethods=fitmethods_linear,
            plot=False,
        )
        assert len(results[idx]) > 0


def test_modeller_fit_linear(model):
    modeller = Modeller()
    results = modeller.fit_model_linear(model, validate=True)
    # TODO: add more here
    assert results is not None
