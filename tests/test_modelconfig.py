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
from lsst.multiprofit.componentconfig import (
    CentroidConfig,
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.observationconfig import ObservationConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
import numpy as np
import pytest


@pytest.fixture(scope="module")
def channels() -> dict[str, g2f.Channel]:
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


@pytest.fixture(scope="module")
def data(channels) -> g2f.Data:
    config = ObservationConfig(n_rows=13, n_cols=19)
    observations = []
    for band in channels:
        config.band = band
        observations.append(config.make_observation())
    return g2f.Data(observations)


@pytest.fixture(scope="module")
def psfmodel():
    rho, size_x, size_y = 0.25, 1.6, 1.2
    drho, dsize_x, dsize_y = -0.4, 1.1, 1.9

    n_components = 3
    flux_total = 2.*(n_components + 1)
    fluxes = [x/flux_total for x in range(1, 1 + n_components)]

    config = SourceConfig(
        componentgroups={
            'src': ComponentGroupConfig(
                components_gauss={
                    str(idx): GaussianComponentConfig(
                        rho=ParameterConfig(value_initial=rho + idx*drho),
                        size_x=ParameterConfig(value_initial=size_x + idx*dsize_x),
                        size_y=ParameterConfig(value_initial=size_y + idx*dsize_y),
                    )
                    for idx in range(n_components)
                },
            )
        },
    )
    config.validate()
    channel = g2f.Channel.NONE
    psfmodel, priors = config.make_psfmodel(
        [
            [
                {channel: flux} for flux in fluxes
            ],
        ],
    )
    return psfmodel


@pytest.fixture(scope="module")
def psfmodels(psfmodel, channels) -> list[g2f.PsfModel]:
    return [psfmodel]*len(channels)


@pytest.fixture(scope="module")
def modelconfig_fluxes(channels):
    rho, size_x, size_y, sersicn, flux = 0.4, 1.5, 1.9, 0.5, 4.7
    drho, dsize_x, dsize_y, dsersicn, dflux = -0.9, 2.5, 5.4, 2.8, 13.9

    components_sersic = {}
    fluxes_mix = []
    for idx, name in enumerate(("PS", "Sersic")):
        components_sersic[name] = SersicComponentConfig(
            rho=ParameterConfig(value_initial=rho + idx*drho),
            size_x=ParameterConfig(value_initial=size_x + idx*dsize_x),
            size_y=ParameterConfig(value_initial=size_y + idx*dsize_y),
            sersicindex=SersicIndexParameterConfig(
                value_initial=sersicn + idx * dsersicn,
                fixed=idx == 0,
                prior_mean=None,
            ),
        )
        fluxes_comp = {
            channel: flux + idx_channel*dflux*idx
            for idx_channel, channel in enumerate(channels.values())
        }
        fluxes_mix.append(fluxes_comp)

    modelconfig = ModelConfig(
        sources={
            'src': SourceConfig(
                componentgroups={
                    'mix': ComponentGroupConfig(
                        centroids={
                            "default": CentroidConfig(
                                x=ParameterConfig(value_initial=15.8, fixed=True),
                                y=ParameterConfig(value_initial=14.3, fixed=False),
                            ),
                        },
                        components_sersic=components_sersic,
                    ),
                }
            ),
        },
    )
    return modelconfig, fluxes_mix


def test_ModelConfig(modelconfig_fluxes, data, psfmodels):
    modelconfig, fluxes = modelconfig_fluxes
    model = modelconfig.make_model([[fluxes]], data=data, psfmodels=psfmodels)
    assert model is not None
    assert model.data is data
    for observation in model.data:
        observation.sigma_inv.fill(1.)
        observation.mask_inv.fill(1)

    # Set the outputs to new images that refer to the existing data
    # because obs.image will not return a holding pointer
    outputs = [[g2.ImageD(obs.image.data)] for obs in model.data]
    model.setup_evaluators(model.EvaluatorMode.image, outputs=outputs)
    model.evaluate()
    model.setup_evaluators(model.EvaluatorMode.loglike)
    assert np.sum(model.evaluate()) == 0
