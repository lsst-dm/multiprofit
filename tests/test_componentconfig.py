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

import lsst.gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    EllipticalComponentConfig,
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.config import set_config_from_dict
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


def test_EllipticalComponentConfig():
    config = EllipticalComponentConfig()
    config2 = EllipticalComponentConfig()
    set_config_from_dict(config2, config.toDict())
    assert config == config2


def test_GaussianComponentConfig(centroid):
    config = GaussianComponentConfig(
        rho=ParameterConfig(value_initial=0),
        size_x=ParameterConfig(value_initial=1.4),
        size_y=ParameterConfig(value_initial=1.6),
    )
    channel = g2f.Channel.NONE
    component_data1 = config.make_component(
        centroid=centroid,
        integral_model=g2f.FractionalIntegralModel(
            [(channel, g2f.ProperFractionParameterD(0.5, fixed=False))],
            model=config.make_linear_integral_model({channel: 1.0}),
        ),
    )
    component_data2 = config.make_component(
        centroid=centroid,
        integral_model=g2f.FractionalIntegralModel(
            [(channel, g2f.ProperFractionParameterD(1.0, fixed=True))],
            model=component_data1.integral_model,
            is_final=True,
        ),
    )
    components = (component_data1, component_data2)
    n_components = len(components)
    for idx, component_data in enumerate(components):
        component = component_data.component
        assert component.centroid is centroid
        assert len(component_data.priors) == 0
        fluxes = list(get_params_uniq(component, nonlinear=False))
        assert len(fluxes) == 1
        assert isinstance(fluxes[0], g2f.IntegralParameterD)
        fracs = [param for param in get_params_uniq(component, linear=False)
                 if isinstance(param, g2f.ProperFractionParameterD)]
        assert len(fracs) == (idx + (idx == 0) - (idx == n_components))


def test_SersicConfig(centroid, channels):
    rho, size_x, size_y, sersic_index = -0.3, 1.4, 1.6, 3.2
    config = SersicComponentConfig(
        rho=ParameterConfig(value_initial=rho),
        size_x=ParameterConfig(value_initial=size_x),
        size_y=ParameterConfig(value_initial=size_y),
        sersic_index=SersicIndexParameterConfig(value_initial=sersic_index),
    )
    fluxes = {
        channel: 1.0 + idx
        for idx, channel in enumerate(channels.values())
    }
    integral_model = config.make_linear_integral_model(fluxes)
    component_data = config.make_component(
        centroid=centroid,
        integral_model=integral_model,
    )
    assert component_data.component is not None
    # As long as there's a default Sersic index prior
    assert len(component_data.priors) == 1
    params = get_params_uniq(component_data.component)
    values_init = {
        g2f.RhoParameterD: rho,
        g2f.ReffXParameterD: size_x,
        g2f.ReffYParameterD: size_y,
        g2f.SersicIndexParameterD: sersic_index,
    }
    fluxes_label = {
        config.format_label(config.get_integral_label_default(), name_channel=channel.name):
            fluxes[channel] for channel in fluxes.keys()
    }
    for param in params:
        if isinstance(param, g2f.IntegralParameterD):
            assert fluxes_label[param.label] == param.value
        elif value_init := values_init.get(param.__class__):
            assert param.value == value_init
