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
# but WITHOUT ANY WARRANTY; without even the implied warrantyfluxes = u.ABmag.to(u.nanojansky, mags) of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import CentroidConfig, GaussianComponentConfig, ParameterConfig
from lsst.multiprofit.model_utils import make_psf_model_null
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.observationconfig import CoordinateSystemConfig, ObservationConfig
from lsst.multiprofit.plots import abs_mag_sol_lsst, bands_weights_lsst, plot_model_rgb
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
import numpy as np
import pytest

sigma_inv = 1e4


@pytest.fixture(scope="module")
def channels() -> dict[str, g2f.Channel]:
    return {band: g2f.Channel.get(band) for band in bands_weights_lsst}


@pytest.fixture(scope="module")
def data(channels) -> g2f.Data:
    n_rows, n_cols = 16, 21
    x_min, y_min = 0, 0

    dn_rows, dn_cols = 1, -2
    dx_min, dy_min = -2, 1

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
        observation.sigma_inv.fill(sigma_inv)
        observation.mask_inv.fill(1)
        observations.append(observation)
    return g2f.Data(observations)


@pytest.fixture(scope="module")
def psf_model():
    return make_psf_model_null()


@pytest.fixture(scope="module")
def psf_models(psf_model, channels) -> list[g2f.PsfModel]:
    return [psf_model]*len(channels)


@pytest.fixture(scope="module")
def model(channels, data, psf_models):
    fluxes_group = [{channels[band]: 10**(-0.4*(mag - 8.9)) for band, mag in abs_mag_sol_lsst.items()}]

    modelconfig = ModelConfig(
        sources={
            'src': SourceConfig(
                component_groups={
                    '': ComponentGroupConfig(
                        centroids={"default": CentroidConfig(
                            x=ParameterConfig(value_initial=6., fixed=True),
                            y=ParameterConfig(value_initial=11., fixed=True),
                        )},
                        components_gauss={
                            "": GaussianComponentConfig(
                                rho=ParameterConfig(value_initial=0.1),
                                size_x=ParameterConfig(value_initial=3.8),
                                size_y=ParameterConfig(value_initial=5.1),
                            )
                        },
                    )
                }
            ),
        },
    )
    model = modelconfig.make_model([[fluxes_group]], data=data, psf_models=psf_models)
    model.setup_evaluators(model.EvaluatorMode.image)
    model.evaluate()
    rng = np.random.default_rng(1)
    for output, obs in zip(model.outputs, model.data):
        img = obs.image.data
        img.flat = output.data.flat + rng.standard_normal(img.size) / sigma_inv
    return model


def test_plot_model_rgb(model):
    fig, ax, fig_gs, ax_gs, *_ = plot_model_rgb(
        model, minimum=0, stretch=0.15, Q=4, weights=bands_weights_lsst, plot_chi_hist=True,
    )
    assert fig is not None
    assert ax is not None
    assert fig_gs is not None
    assert ax_gs is not None


def test_plot_model_rgb_auto(model):
    fig, ax, *_ = plot_model_rgb(
        model, Q=6, weights=bands_weights_lsst, rgb_min_auto=True, rgb_stretch_auto=True,
        plot_singleband=False, plot_chi_hist=False,
    )
    assert fig is not None
    assert ax is not None
