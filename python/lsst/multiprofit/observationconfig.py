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
import lsst.pex.config as pexConfig
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass

from .model_utils import make_image_gaussians
from .sourceconfig import SourceConfig


class ObservationConfig(pexConfig.Config):
    band = pexConfig.Field[str](doc="The name of the band")
    model_source_psf = pexConfig.ConfigField[SourceConfig](doc="The PSF source configuration")
    n_rows = pexConfig.Field[int](doc="The number of rows in the image")
    n_cols = pexConfig.Field[int](doc="The number of columns in the image")
    noise_psf = pexConfig.Field[float](doc="The noise level in PSF images")
    seed = pexConfig.Field[int](doc="The random seed", default=1)
    sigma_img = pexConfig.Field[float](doc="The background noise per pixel")

    def make_psfmodel_observation(self) -> g2f.Observation:
        gaussians_kernel = g2.Gaussians([g2.Gaussian()])
        rng = np.random.default_rng(self.seed)
        psfmodel = self.model_source_psf.make_model()
        image = make_image_gaussians(
            gaussians_source=psfmodel.gaussians(g2f.Channel.NONE),
            gaussians_kernel=gaussians_kernel,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
        )
        data = image.data
        data += self.noise_psf * rng.standard_normal(image.data.shape)
        sigma_inv = g2.ImageD(np.full_like(image.data, self.noise_psf))
        mask = g2.ImageB(np.ones_like(image.data))
        return g2f.Observation(
            image=image,
            sigma_inv=sigma_inv,
            mask_inv=mask,
            channel=g2f.Channel.NONE,
        )

    def make_observation(self):
        observations = [None] * len(psfmodels)
        gaussians_kernel = g2.Gaussians([g2.Gaussian()])
        rng = np.random.default_rng(self.seed)
        for idx, psfmodel in enumerate(psfmodels):
            image = make_image_gaussians(
                gaussians_source=psfmodel.gaussians(g2f.Channel.NONE),
                gaussians_kernel=gaussians_kernel,
                n_rows=config.n_rows,
                n_cols=config.n_cols,
            )
            data = image.data
            data += config.noise_psf * rng.standard_normal(image.data.shape)
            sigma_inv = g2.ImageD(np.full_like(image.data, config.noise_psf))
            mask = g2.ImageB(np.ones_like(image.data))
            observations[idx] = g2f.Observation(
                image=image,
                sigma_inv=sigma_inv,
                mask_inv=mask,
                channel=g2f.Channel.NONE,
            )
        return tuple(observations)

    def make_psfmodel(self) -> g2f.PsfModel:
        n_comps = len(self.model_source_psf.components)
        n_last = n_comps - 1
        translog = transforms_ref["log10"]
        transrho = transforms_ref["logit_rho"]
        last = None
        components = [None] * n_comps
        centroid = g2f.CentroidParameters(
            g2f.CentroidXParameterD(config.n_cols / 2.0, limits=limits.x),
            g2f.CentroidYParameterD(config.n_rows / 2.0, limits=limits.y),
        )
        size_psf = config.size_increment_psf * i
        for c in range(n_comps):
            is_last = c == n_last
            last = g2f.FractionalIntegralModel(
                [
                    (
                        g2f.Channel.NONE,
                        g2f.ProperFractionParameterD(
                            (is_last == 1) or frac,
                            fixed=is_last,
                            transform=transforms_ref["logit"],
                        ),
                    )
                ]
            )
            g2f.LinearIntegralModel([(g2f.Channel.NONE, g2f.IntegralParameterD(1.0, fixed=True))])
            if (c == 0)
            else last,
            is_last,
            )
            components[c] = g2f.GaussianComponent(
                g2f.GaussianParametricEllipse(
                    g2f.SigmaXParameterD(
                        compconf.size_base + c * compconf.size_increment + size_psf, transform=translog
                    ),
                    g2f.SigmaYParameterD(
                        compconf.size_base + c * compconf.size_increment + size_psf, transform=translog
                    ),
                    g2f.RhoParameterD(
                        compconf.rho_base + c * compconf.rho_increment, limits=limits.rho,
                        transform=transrho
                    ),
                ),
                centroid,
                last,
            )
        return g2f.PsfModel(components)
