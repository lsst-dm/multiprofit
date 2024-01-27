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
from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pytest


@pytest.fixture(scope="module")
def centroid_limits():
    limits = g2f.LimitsD(min=-np.Inf, max=np.Inf)
    return limits


@pytest.fixture(scope="module")
def centroid(centroid_limits):
    cenx = g2f.CentroidXParameterD(0, limits=centroid_limits, fixed=True)
    ceny = g2f.CentroidYParameterD(0, limits=centroid_limits, fixed=True)
    centroid = g2f.CentroidParameters(cenx, ceny)
    return centroid


@pytest.fixture(scope="module")
def channels():
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


def test_ComponentGroupConfig(centroid):
    with pytest.raises(ValueError) as exc:
        config = ComponentGroupConfig(
            components_gauss={"x": GaussianComponentConfig()},
            components_sersic={"x": SersicComponentConfig()},
        )
        config.validate()


def test_SourceConfig_base():
    with pytest.raises(ValueError) as exc:
        config = SourceConfig()
        config.validate()

    with pytest.raises(ValueError) as exc:
        config = SourceConfig(componentgroups={})
        config.validate()


def test_SourceConfig_fractional(centroid):
    rho, size_x, size_y = -0.3, 1.4, 1.6
    drho, dsize_x, dsize_y = 0.5, 1.6, 1.3

    n_components = 2
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
                is_fractional=True,
            )
        },
    )
    config.validate()
    channel = g2f.Channel.NONE
    psfmodel, priors = config.make_psfmodel(
        [
            [
                {channel: 1.0},
                {channel: 0.5},
            ]
        ],
    )
    assert len(priors) == 0
    assert len(psfmodel.components) == n_components


def test_SourceConfig_linear(centroid, channels):
    rho, size_x, size_y, sersicn, flux = 0.4, 1.5, 1.9, 0.5, 4.7
    drho, dsize_x, dsize_y, dsersicn, dflux = -0.9, 2.5, 5.4, 2.8, 13.9

    names = ("PS", "Sersic")
    config = SourceConfig(
        componentgroups={
            'src': ComponentGroupConfig(
                components_sersic={
                    name: SersicComponentConfig(
                        rho=ParameterConfig(value_initial=rho + idx*drho),
                        size_x=ParameterConfig(value_initial=size_x + idx*dsize_x),
                        size_y=ParameterConfig(value_initial=size_y + idx*dsize_y),
                        sersicindex=SersicIndexParameterConfig(value_initial=sersicn + idx * dsersicn, fixed=idx == 0),
                    )
                    for idx, name in enumerate(names)
                }
            ),
        }
    )
    fluxes = [
        {
            channel: flux + idx_channel*dflux*idx_comp
            for idx_channel, channel in enumerate(channels.values())
        }
        for idx_comp in range(len(config.componentgroups["src"].components_sersic))
    ]
    source, priors = config.make_source([fluxes])
    assert len(priors) == 0
    for idx, component in enumerate(source.components):
        params = get_params_uniq(component)
        values_init = {
            g2f.RhoParameterD: rho + idx*drho,
            g2f.ReffXParameterD: size_x + idx*dsize_x,
            g2f.ReffYParameterD: size_y + idx*dsize_y,
            g2f.SersicIndexParameterD: sersicn + idx*dsersicn,
        }
        for name_group, componentgroup in config.componentgroups.items():
            fluxes_comp = fluxes[idx]
            name_comp = names[idx]
            config_comp = componentgroup.components_sersic[name_comp]
            fluxes_label = {
                config.format_label(
                    componentgroup.format_label(
                        label=config_comp.format_label(label=config.get_integral_label_default(),
                                                       name_channel=channel.name),
                        name_component=name_comp,
                    ),
                    name_group=name_group,
                ): fluxes_comp[channel]
                for channel in channels.values()
            }
            for param in params:
                if isinstance(param, g2f.IntegralParameterD):
                    assert fluxes_label[param.label] == param.value
                elif value_init := values_init.get(param.__class__):
                    assert param.value == value_init
