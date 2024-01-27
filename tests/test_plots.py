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

import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import CentroidConfig, GaussianComponentConfig, ParameterConfig
from lsst.multiprofit.model_utils import make_psfmodel_null
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.observationconfig import CoordinateSystemConfig, ObservationConfig
from lsst.multiprofit.plots import plot_model_rgb
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
import numpy as np
import pytest

sigma_inv = 1e4

@pytest.fixture(scope="module")
def channels() -> dict[str, g2f.Channel]:
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


@pytest.fixture(scope="module")
def data(channels) -> g2f.Data:
    n_rows, n_cols = 15, 21
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
        observation.image.fill(0)
        observation.sigma_inv.fill(sigma_inv)
        observation.mask_inv.fill(1)
        observations.append(observation)
    return g2f.Data(observations)


@pytest.fixture(scope="module")
def psfmodel():
    return make_psfmodel_null()


@pytest.fixture(scope="module")
def psfmodels(psfmodel, channels) -> list[g2f.PsfModel]:
    return [psfmodel]*len(channels)


@pytest.fixture(scope="module")
def model(channels, data, psfmodels):
    fluxes_group = [{channel: 1.0 for channel in channels.values()}]

    modelconfig = ModelConfig(
        sources={
            'src': SourceConfig(
                componentgroups={
                    '': ComponentGroupConfig(
                        centroids={"default": CentroidConfig(
                            x=ParameterConfig(value_initial=8., fixed=True),
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
    model = modelconfig.make_model([[fluxes_group]], data=data, psfmodels=psfmodels)
    model.setup_evaluators(model.EvaluatorMode.image)
    model.evaluate()
    rng = np.random.default_rng(1)
    for output, obs in zip(model.outputs, model.data):
        img = obs.image.data
        img.flat = output.data.flat
        img.flat += rng.standard_normal(img.size) / sigma_inv
    return model


def test_plot_model_rgb(model):
    fig, ax, *_ = plot_model_rgb(model, stretch=10/sigma_inv, Q=3)
    assert fig is not None
    assert ax is not None
