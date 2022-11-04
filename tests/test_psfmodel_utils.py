import gauss2d.fit as g2f
from multiprofit.psfmodel_utils import make_psf_source


def test_make_psf_source():
    source = make_psf_source(3)
    assert source.gaussians(g2f.Channel.NONE).size == 3
