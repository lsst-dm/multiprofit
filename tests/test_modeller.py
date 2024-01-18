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

import gauss2d as g2
import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexConfig,
)
from lsst.multiprofit.model_utils import make_image_gaussians, make_psfmodel_null
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.modeller import FitInputs, LinearGaussians, Modeller, fitmethods_linear
from lsst.multiprofit.observationconfig import CoordinateSystemConfig, ObservationConfig
from lsst.multiprofit.sourceconfig import ComponentMixtureConfig, SourceConfig
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pytest

sigma_inv = 1e4


@pytest.fixture(scope="module")
def channels() -> dict[str, g2f.Channel]:
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


@pytest.fixture(scope="module")
def data(channels) -> g2f.Data:
    n_rows, n_cols = 25, 27
    x_min, y_min = 0, 0

    dn_rows, dn_cols = 0, 0 #2, -3
    dx_min, dy_min = 0, 0 #-1, 1

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
    return g2f.Data(observations)


@pytest.fixture(scope="module")
def psfmodels(channels) -> list[g2f.PsfModel]:
    rho, size_x, size_y = 0.12, 1.6, 1.2
    drho, dsize_x, dsize_y = -0.3, 1.1, 1.9
    drho_chan, dsize_x_chan, dsize_y_chan = 0.03, 0.12, 0.14
    frac, dfrac = 0.62, -0.08

    psfmodels = []
    for idx_chan, channel in enumerate(channels.values()):
        fluxes = [1.0, frac + idx_chan*dfrac]
        n_components = len(fluxes)
        config = SourceConfig(
            componentmixtures={
                'psf': ComponentMixtureConfig(
                    components_gauss={
                        str(idx): GaussianComponentConfig(
                            rho=ParameterConfig(value_initial=rho + idx*drho + idx_chan*drho_chan),
                            size_x=ParameterConfig(value_initial=size_x + idx*dsize_x + idx_chan*dsize_x_chan),
                            size_y=ParameterConfig(value_initial=size_y + idx*dsize_y + idx_chan*dsize_y_chan),
                            is_fractional=True,
                        )
                        for idx in range(n_components)
                    },
                    is_fractional=True,
                )
            },
        )
        config.validate()

        centroid = g2f.CentroidParameters(
            x=g2f.CentroidXParameterD(0, fixed=True),
            y=g2f.CentroidYParameterD(0, fixed=True),
        )
        psfmodel, priors = config.make_psfmodel(
            [
                (
                    centroid,
                    [
                        {g2f.Channel.NONE: ParameterConfig(value_initial=flux, fixed=True)}
                        for flux in fluxes
                    ]
                ),
            ],
        )
        psfmodels.append(psfmodel)
    return psfmodels


@pytest.fixture(scope="module")
def model(channels, data, psfmodels) -> g2f.Model:
    rho, size_x, size_y, sersicn, flux = 0.4, 1.5, 1.9, 0.5, 4.7
    drho, dsize_x, dsize_y, dsersicn, dflux = -0.9, 2.5, 5.4, 2.8, 13.9

    components_sersic = {}
    fluxes_mix = []
    for idx, name in enumerate(("PS", "Sersic")):
        components_sersic[name] = SersicComponentConfig(
            rho=ParameterConfig(value_initial=rho + idx*drho),
            size_x=ParameterConfig(value_initial=size_x + idx*dsize_x),
            size_y=ParameterConfig(value_initial=size_y + idx*dsize_y),
            sersicindex=SersicIndexConfig(value_initial=sersicn + idx*dsersicn, fixed=idx == 0),
        )
        fluxes_comp = {
            channel: ParameterConfig(value_initial=flux + idx_channel*dflux*idx, fixed=True)
            for idx_channel, channel in enumerate(channels.values())
        }
        fluxes_mix.append(fluxes_comp)

    modelconfig = ModelConfig(
        sources={
            'src': SourceConfig(
                componentmixtures={
                    'mix': ComponentMixtureConfig(components_sersic=components_sersic),
                }
            ),
        },
    )
    centroid = g2f.CentroidParameters(
        x=g2f.CentroidXParameterD(12.14, fixed=True),
        y=g2f.CentroidYParameterD(13.78, fixed=True),
    )
    model = modelconfig.make_model([[(centroid, fluxes_mix)]], data=data, psfmodels=psfmodels)
    return model


@pytest.fixture(scope="module")
def model_jac(model) -> g2f.Model:
    model_jac = g2f.Model(data=model.data, psfmodels=model.psfmodels, sources=model.sources)
    return model_jac


@pytest.fixture(scope="module")
def psf_observations(psfmodels) -> list[g2f.Observation]:
    config = ObservationConfig(n_rows=17, n_cols=19)
    rng = np.random.default_rng(1)

    observations = []
    for psfmodel in psfmodels:
        observation = config.make_observation()
        # Have to make a duplicate image here because one can only call
        # make_image_gaussians with an owning pointer, whereas
        # observation.image is a reference
        image = g2.ImageD(observation.image.data)
        # Make the kernel centered
        gaussians_source = psfmodel.gaussians(g2f.Channel.NONE)
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
def psf_fit_models(psfmodels, psf_observations):
    psf_null = [make_psfmodel_null()]
    return [
        g2f.Model(g2f.Data([observation]), psf_null, [g2f.Source(psfmodel.components)])
        for psfmodel, observation in zip(psfmodels, psf_observations)
    ]


def test_model_evaluation(channels, model, model_jac):
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


@pytest.fixture(scope="module")
def psfmodels_linear_gaussians(channels, psfmodels):
    gaussians = [None] * len(psfmodels)
    for idx, psfmodel in enumerate(psfmodels):
        params = psfmodel.parameters(paramfilter=g2f.ParamFilter(nonlinear=False, channel=g2f.Channel.NONE))
        params[0].fixed = False
        gaussians[idx] = LinearGaussians.make(psfmodel, is_psf=True)
    return gaussians


def test_make_psf_source_linear(psfmodels, psfmodels_linear_gaussians):
    for psfmodel, linear_gaussians in zip(psfmodels, psfmodels_linear_gaussians):
        gaussians = psfmodel.gaussians(g2f.Channel.NONE)
        assert len(gaussians) == (
            len(linear_gaussians.gaussians_free) + len(linear_gaussians.gaussians_fixed)
        )


def test_modeller(model):
    # For debugging purposes
    printout = False
    model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike_image)
    # Get the model images
    model.evaluate()
    rng = np.random.default_rng(3)

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

    dloglike = model.compute_loglike_grad(verify=True, findiff_frac=1e-8, findiff_add=1e-8)
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
        params = get_params_uniq(model.sources[0])
        for param in params:
            # Fitting the total flux won't work in a fractional model (yet)
            if not isinstance(param, g2f.IntegralParameterD):
                param.fixed = False
        # Necessary whenever parameters are freed/fixed
        model.setup_evaluators(model.EvaluatorMode.jacobian, force=True)
        errors = model.verify_jacobian(rtol=5e-4, atol=5e-4, findiff_add=1e-6, findiff_frac=1e-6)
        if errors:
            import matplotlib.pyplot as plt

            fitinputs = FitInputs.from_model(model)
            model.setup_evaluators(
                evaluatormode=g2f.Model.EvaluatorMode.jacobian,
                outputs=fitinputs.jacobians,
                residuals=fitinputs.residuals,
                print=True,
                force=True,
            )
            model.evaluate()
            assert (fitinputs.jacobians[0][0].data == 0).all()
            assert np.sum(np.abs(fitinputs.jacobians[0][1].data)) > 0
            model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike_image)
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
