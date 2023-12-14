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
from functools import cached_property
import gauss2d as g2
import gauss2d.fit as g2f
import numpy

from .config import set_config_from_dict
from .componentconfig import init_component
from .fit_psf import CatalogExposurePsfABC, CatalogPsfFitterConfig
from .fit_source import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterABC,
    CatalogSourceFitterConfig,
)
from .utils import get_params_uniq

__all__ = ["SourceCatalogBootstrap", "CatalogExposurePsfBootstrap", "CatalogExposureSourcesBootstrap",
           "CatalogSourceFitterBootstrap"]


@dataclass(kw_only=True, frozen=True)
class SourceCatalogBootstrap:
    """Config class for a bootstrap source catalog fitter."""

    n_cols_img: int = 25
    n_rows_img: int = 25
    n_sources: int = 1

    @cached_property
    def catalog(self):
        catalog = astropy.table.Table({"id": np.arange(self.n_sources)})
        return catalog


@dataclass(kw_only=True, frozen=True)
class CatalogExposurePsfBootstrap(CatalogExposurePsfABC, SourceCatalogBootstrap):
    """Dataclass for a PSF-convolved bootstrap fitter."""

    noise: float = 1e-4
    sigma_x: float
    sigma_y: float
    rho: float
    nser: float

    @cached_property
    def centroid(self) -> g2.Centroid:
        centroid = g2.Centroid(x=self.n_cols_img / 2, y=self.n_rows_img / 2)
        return centroid

    @cached_property
    def ellipse(self) -> g2.Ellipse:
        g2.Centroid(x=self.n_cols_img / 2, y=self.n_rows_img / 2)
        ellipse = g2.Ellipse(g2.EllipseValues(self.sigma_x, self.sigma_y, self.rho))
        return ellipse

    @cached_property
    def image(self) -> numpy.ndarray:
        image = g2.make_gaussians_pixel_D(
            g2.ConvolvedGaussians(
                [
                    g2.ConvolvedGaussian(
                        g2.Gaussian(centroid=self.centroid, ellipse=self.ellipse),
                        g2.Gaussian(),
                    )
                ]
            ),
            n_rows=self.n_rows_img,
            n_cols=self.n_cols_img,
        ).data
        return image

    def get_catalog(self) -> astropy.table.Table:
        return self.catalog

    def get_psf_image(self, source) -> numpy.ndarray:
        rng = np.random.default_rng(source["id"])
        return self.image + 1e-4 * rng.standard_normal(self.image.shape)


@dataclass(kw_only=True, frozen=True)
class CatalogExposureSourcesBootstrap(CatalogExposureSourcesABC, SourceCatalogBootstrap):
    """A CatalogExposure for bootstrap fitting of source catalogs."""

    channel: g2f.Channel = g2f.Channel.NONE
    config_fit: CatalogSourceFitterConfig
    model_source: g2f.Source
    n_buffer_img: int = 10
    table_psf_fits: astropy.table.Table

    def get_catalog(self) -> astropy.table.Table:
        return self.catalog

    def get_psfmodel(self, params: Mapping[str, Any]) -> g2f.PsfModel:
        return self.config_fit_psf.rebuild_psfmodel(self.table_psf_fits[params["id"]])

    def get_source_observation(self, source: Mapping[str, Any]) -> g2f.Observation:
        n_rows = self.n_rows_img + self.n_buffer_img
        n_cols = self.n_cols_img + self.n_buffer_img
        obs = g2f.Observation(
            image=g2.ImageD(n_rows=n_rows, n_cols=n_cols),
            sigma_inv=g2.ImageD(n_rows=n_rows, n_cols=n_cols),
            mask_inv=g2.ImageB(n_rows=n_rows, n_cols=n_cols),
            channel=self.channel,
        )
        return obs

    def __post_init__(self):
        config_dict = self.table_psf_fits.meta["config"]
        config = CatalogPsfFitterConfig()
        set_config_from_dict(config, config_dict)
        object.__setattr__(self, "config_fit_psf", config)


@dataclass(kw_only=True)
class CatalogSourceFitterBootstrap(CatalogSourceFitterABC):
    """A catalog fitter that bootstraps a single model.

    This fitter generates a different noisy image of the specified model for
    each row. The resulting catalog can be used to examine performance and
    statistics of the best-fit parameters.
    """

    background: float = 1e2
    flux: float = 1e4
    reff_x: float
    reff_y: float
    rho: float
    nser: float

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float) -> tuple[float, float]:
        return float(cen_x), float(cen_y)

    def initialize_model(
        self, model: g2f.Model, source: g2f.Source, limits_x: g2f.LimitsD = None, limits_y: g2f.LimitsD = None
    ):
        comp1, comp2 = model.sources[0].components
        observation = model.data[0]
        cenx = observation.image.n_cols / 2.0
        ceny = observation.image.n_rows / 2.0
        if limits_x is not None:
            limits_x.max = float(observation.image.n_cols)
        if limits_y is not None:
            limits_y.max = float(observation.image.n_rows)
        init_component(comp1, cen_x=cenx, cen_y=ceny)
        init_component(
            comp2,
            cen_x=cenx,
            cen_y=ceny,
            reff_x=self.reff_x,
            reff_y=self.reff_y,
            rho=self.rho,
            nser=self.nser,
        )
        params_free = get_params_uniq(model, fixed=False)
        for param in params_free:
            if isinstance(param, g2f.IntegralParameterD):
                param.value = self.flux

        # Should be done in get_source_observation, but it gets called first
        # ... and therefore does not have the initialization above
        model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.image)
        model.evaluate()

        rng = np.random.default_rng(source["id"] + 100000)

        for idx_obs, observation in enumerate(model.data):
            image, sigma_inv = observation.image, observation.sigma_inv
            image.data.flat = model.outputs[0].data
            sigma_inv.data.flat = np.sqrt(image.data + self.background)
            image.data.flat = image.data + sigma_inv.data * rng.standard_normal(image.data.shape)
            sigma_inv.data.flat = 1 / sigma_inv.data
            # This is mandatory because C++ construction does no initialization
            # (could instead initialize in get_source_observation)
            # TODO: Do some timings to see which is more efficient
            observation.mask_inv.data.flat = 1

        self.params_values_init = tuple(param.value for param in params_free)
