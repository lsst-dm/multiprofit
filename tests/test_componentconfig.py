import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    EllipticalComponentConfig,
    GaussianConfig,
    ParameterConfig,
    SersicConfig,
)
from lsst.multiprofit.config import set_config_from_dict
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


def test_GaussianComponentConfig_fractional(centroid):
    config1 = GaussianConfig(
        rho=ParameterConfig(value_initial=0),
        size_x=ParameterConfig(value_initial=1.4),
        size_y=ParameterConfig(value_initial=1.6),
        is_fractional=True,
    )
    channel = g2f.Channel.NONE
    component1, priors = config1.make_component(
        centroid=centroid,
        fluxes={channel: ParameterConfig(value_initial=1.0, fixed=True)},
        last=1.0,
        is_final=False,
    )
    assert component1 is not None
    assert len(priors) == 0


def test_GaussianComponentConfig_linear(centroid, channels):
    config = GaussianConfig(
        rho=ParameterConfig(value_initial=0),
        size_x=ParameterConfig(value_initial=1.4),
        size_y=ParameterConfig(value_initial=1.6),
    )
    component, priors = config.make_component(
        centroid=centroid,
        fluxes={
            channel: ParameterConfig(value_initial=float(idx))
            for idx, channel in enumerate(channels.values())
        },
    )
    assert component is not None
    assert len(priors) == 0


def test_SersicComponentConfig():
    pass
