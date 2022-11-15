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

import astropy
from dataclasses import dataclass
import gauss2d as g2
import gauss2d.fit as g2f
from multiprofit.fit_psf import CatalogExposurePsfABC, CatalogPsfFitter, CatalogPsfFitterConfig
from multiprofit.fit_source import (
    CatalogExposureSourcesABC, CatalogSourceFitter, CatalogSourceFitterConfig, SersicConfig,
    SersicIndexConfig
)
import numpy as np
import pytest

rng = np.random.default_rng(1)

channel = g2f.Channel.get("r")
shape_img = (23, 27)
sigma_x_init, sigma_y_init, rho_init = 2.5, 3.6, -0.25


def init_component(
    component: g2f.Component, sigma_x: float = None, sigma_y: float = None, rho: float = None,
    cenx: float = None, ceny: float = None,
):
    for parameter in component.parameters():
        if sigma_x is not None and isinstance(parameter, g2f.SigmaXParameterD):
            parameter.value = sigma_x
        elif sigma_y is not None and isinstance(parameter, g2f.SigmaYParameterD):
            parameter.value = sigma_y
        elif rho is not None and isinstance(parameter, g2f.RhoParameterD):
            parameter.value = rho
        elif cenx is not None and isinstance(parameter, g2f.CentroidXParameterD):
            parameter.value = cenx
        elif ceny is not None and isinstance(parameter, g2f.CentroidYParameterD):
            parameter.value = ceny


class CatalogExposurePsfTest(CatalogExposurePsfABC):
    def get_catalog(self) -> astropy.table.Table:
        return astropy.table.Table({'id': [0]})

    def get_psf_image(self, source):
        ellipse = g2.Ellipse(g2.EllipseValues(2.6, 3.4, 0.1))
        image = g2.make_gaussians_pixel_D(g2.ConvolvedGaussians([g2.ConvolvedGaussian(
            g2.Gaussian(centroid=g2.Centroid(x=shape_img[0]/2, y=shape_img[1]/2), ellipse=ellipse),
            g2.Gaussian(),
        )]), n_rows=shape_img[0], n_cols=shape_img[1]).data
        return image + 1e-4*rng.standard_normal(image.shape)


@dataclass(frozen=True)
class CatalogExposureSourcesTest(CatalogExposureSourcesABC):
    config_fit: CatalogSourceFitterConfig
    model_source: g2f.Source
    table_psf_fits: astropy.table.Table
    background: float = 1e2
    flux: float = 1e4

    @property
    def channel(self):
        return channel

    def get_catalog(self) -> astropy.table.Table:
        return astropy.table.Table({'id': [0]})

    def get_psfmodel(self, source):
        return self.config_fit_psf.rebuild_psfmodel(self.table_psf_fits[0])

    def get_source_observation(self, source) -> g2f.Observation:
        image = g2.ImageD(n_rows=shape_img[0], n_cols=shape_img[1])
        mask_inv = g2.ImageB(np.ones_like(image.data))
        obs = g2f.Observation(image=image, sigma_inv=image, mask_inv=mask_inv, channel=channel)
        model = g2f.Model(
            g2f.Data([obs]),
            psfmodels=[self.get_psfmodel(source)],
            sources=[self.model_source],
        )
        model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.image)
        model.evaluate()
        image = self.flux*model.outputs[0].data
        sigma = np.sqrt(image + self.background)
        image += sigma*rng.standard_normal(image.data.shape)
        sigma = 1/sigma
        obs = g2f.Observation(image=g2.ImageD(image), sigma_inv=g2.ImageD(sigma),
                              mask_inv=mask_inv, channel=channel)
        return obs

    def __post_init__(self):
        object.__setattr__(self, 'config_fit_psf',
                           CatalogPsfFitterConfig(**self.table_psf_fits.meta['config']))


class CatalogSourceFitterTest(CatalogSourceFitter):
    def initialize_model(self, model: g2f.Model, source: g2f.Source,
                         limits_x: g2f.LimitsD, limits_y: g2f.LimitsD):
        comp1, comp2 = model.sources[0].components
        observation = model.data[0]
        cenx = observation.image.n_cols/2.
        ceny = observation.image.n_rows/2.
        limits_x.max = float(observation.image.n_cols)
        limits_y.max = float(observation.image.n_rows)
        init_component(comp1, cenx=cenx, ceny=ceny)
        init_component(comp2, cenx=cenx, ceny=ceny, sigma_x=sigma_x_init, sigma_y=sigma_y_init, rho=rho_init)


@pytest.fixture(scope='module')
def config_psf():
    return CatalogPsfFitterConfig(sigmas=[3.0])


@pytest.fixture(scope='module')
def config_source():
    # TODO: Separately test n_pointsources=0 and sersics={}
    return CatalogSourceFitterConfig(
        n_pointsources=1,
        sersics={"comp1": SersicConfig(sersicindex=SersicIndexConfig(fixed=True))},
    )


@pytest.fixture(scope='module')
def table_psf_fits(config_psf):
    fitter = CatalogPsfFitter()
    return fitter.fit(CatalogExposurePsfTest(), config_psf)


def test_fit_psf(config_psf, table_psf_fits):
    assert len(table_psf_fits) == 1
    assert all(np.isfinite(list(table_psf_fits[0].values())))
    psfmodel = config_psf.rebuild_psfmodel(table_psf_fits[0])
    assert len(psfmodel.components) == len(config_psf.sigmas)


def test_fit_source(config_source, table_psf_fits):
    model_source = config_source.make_source(
        centroid=g2f.CentroidParameters(x=g2f.CentroidXParameterD(shape_img[1]/2),
                                        y=g2f.CentroidYParameterD(shape_img[0]/2),),
        channels=[channel],
    )
    # Have to do this here so that the model initializes its observation with
    # the extended component having the right size
    init_component(model_source.components[1], sigma_x=sigma_x_init, sigma_y=sigma_y_init, rho=rho_init)
    catexp = CatalogExposureSourcesTest(
        config_fit=config_source, model_source=model_source, table_psf_fits=table_psf_fits,
    )
    fitter = CatalogSourceFitterTest()
    results = fitter.fit(catalog_multi=catexp.get_catalog(), catexps=[catexp], config=config_source)
    assert len(results) == 1
    assert all(np.isfinite(list(results[0].values())))
