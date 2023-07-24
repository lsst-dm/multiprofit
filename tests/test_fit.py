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
from multiprofit.config import set_config_from_dict
from multiprofit.componentconfig import (
    GaussianConfig, init_component, ParameterConfig, SersicConfig, SersicIndexConfig,
)
from multiprofit.fit_psf import CatalogExposurePsfABC, CatalogPsfFitter, CatalogPsfFitterConfig
from multiprofit.fit_source import (
    CatalogExposureSourcesABC, CatalogSourceFitterABC, CatalogSourceFitterConfig,
)
from multiprofit.modeller import ModelFitConfig
import numpy as np
import pytest
from typing import Any, Mapping

rng = np.random.default_rng(1)

channel = g2f.Channel.get("r")
shape_img = (23, 27)
sigma_x_init, sigma_y_init, rho_init = 2.5, 3.6, -0.25


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

    @property
    def channel(self):
        return channel

    def get_catalog(self) -> astropy.table.Table:
        return astropy.table.Table({'id': [0]})

    def get_psfmodel(self, params: Mapping[str, Any]) -> g2f.PsfModel:
        return self.config_fit_psf.rebuild_psfmodel(self.table_psf_fits[0])

    def get_source_observation(self, source: Mapping[str, Any]) -> g2f.Observation:
        obs = g2f.Observation(
            image=g2.ImageD(n_rows=shape_img[0], n_cols=shape_img[1]),
            sigma_inv=g2.ImageD(n_rows=shape_img[0], n_cols=shape_img[1]),
            mask_inv=g2.ImageB(n_rows=shape_img[0], n_cols=shape_img[1]),
            channel=channel,
        )
        return obs

    def __post_init__(self):
        config_dict = self.table_psf_fits.meta['config']
        config = CatalogPsfFitterConfig()
        set_config_from_dict(config, config_dict)
        object.__setattr__(self, 'config_fit_psf', config)


class CatalogSourceFitterTest(CatalogSourceFitterABC):
    background: float = 1e2
    flux: float = 1e4

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float) -> tuple[float, float]:
        return float(cen_x), float(cen_y)

    def initialize_model(self, model: g2f.Model, source: g2f.Source,
                         limits_x: g2f.LimitsD=None, limits_y: g2f.LimitsD=None):
        comp1, comp2 = model.sources[0].components
        observation = model.data[0]
        cenx = observation.image.n_cols/2.
        ceny = observation.image.n_rows/2.
        if limits_x is not None:
            limits_x.max = float(observation.image.n_cols)
        if limits_y is not None:
            limits_y.max = float(observation.image.n_rows)
        init_component(comp1, cen_x=cenx, cen_y=ceny)
        init_component(comp2, cen_x=cenx, cen_y=ceny, sigma_x=sigma_x_init, sigma_y=sigma_y_init, rho=rho_init)

        # We should have done this in get_source_observation, but it gets called first
        model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.image)
        model.evaluate()
        observation = model.data[0]
        image, sigma_inv = observation.image, observation.sigma_inv
        image.data.flat = self.flux*model.outputs[0].data
        sigma_inv.data.flat = np.sqrt(image.data + self.background)
        image.data.flat = image.data + sigma_inv.data*rng.standard_normal(image.data.shape)
        sigma_inv.data.flat = 1/sigma_inv.data


@pytest.fixture(scope='module')
def config_psf():
    return CatalogPsfFitterConfig(gaussians={'comp1': GaussianConfig(size=ParameterConfig(value_initial=1.5))})


@pytest.fixture(scope='module')
def config_source_fit():
    # TODO: Separately test n_pointsources=0 and sersics={}
    return CatalogSourceFitterConfig(
        config_fit=ModelFitConfig(fit_linear_iter=3),
        n_pointsources=1,
        sersics={"comp1": SersicConfig(sersicindex=SersicIndexConfig(fixed=True))},
    )


@pytest.fixture(scope='module')
def table_psf_fits(config_psf):
    fitter = CatalogPsfFitter()
    return fitter.fit(CatalogExposurePsfTest(), config_psf)


def test_fit_psf(config_psf, table_psf_fits):
    assert len(table_psf_fits) == 1
    assert np.sum(table_psf_fits['mpf_psf_unknown_flag']) == 0
    assert all(np.isfinite(list(table_psf_fits[0].values())))
    psfmodel = config_psf.rebuild_psfmodel(table_psf_fits[0])
    assert len(psfmodel.components) == len(config_psf.gaussians)


def test_fit_source(config_source_fit, table_psf_fits):
    model_source, *_ = config_source_fit.make_source(channels=[channel])
    # Have to do this here so that the model initializes its observation with
    # the extended component having the right size
    init_component(model_source.components[1], sigma_x=sigma_x_init, sigma_y=sigma_y_init, rho=rho_init)
    catexp = CatalogExposureSourcesTest(
        config_fit=config_source_fit, model_source=model_source, table_psf_fits=table_psf_fits,
    )
    fitter = CatalogSourceFitterTest()
    catalog_multi = catexp.get_catalog()
    results = fitter.fit(catalog_multi=catalog_multi, catexps=[catexp], config=config_source_fit)
    assert len(results) == 1
    assert np.sum(results['mpf_unknown_flag']) == 0
    assert all(np.isfinite(list(results[0].values())))

    model = fitter.get_model(0, catalog_multi=catalog_multi, catexps=[catexp], config=config_source_fit,
                             results=results)
    variances = fitter.modeller.compute_variances(model)
    assert np.all(np.isfinite(variances))
