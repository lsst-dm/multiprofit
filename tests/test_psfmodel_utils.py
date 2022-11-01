import gauss2d.fit as g2f
from multiprofit.psfmodel_utils import make_psf_source, make_psfmodel_null, make_psf_source_linear
import pytest


@pytest.fixture(scope='module')
def channel():
    return g2f.Channel.NONE


@pytest.fixture(scope='module')
def source_2comp_default():
    return make_psf_source(2)


def test_make_psfmodel_null(channel):
    psfmodel_null = make_psfmodel_null()
    gaussians = psfmodel_null.gaussians(channel)
    assert gaussians.size == 1
    gaussian = gaussians.at(0)
    assert gaussian.ellipse.sigma_x == 0
    assert gaussian.ellipse.sigma_y == 0
    assert gaussian.ellipse.rho == 0


def test_make_psf_source(channel):
    source = make_psf_source(3)
    assert source.gaussians(channel).size == 3


def test_make_psf_source_linear(channel, source_2comp_default):
    source = make_psf_source_linear(source_2comp_default)
    assert source.gaussians(channel).size == 2
