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
import lsst.pex.config as pexConfig

from .priors import ShapePriorConfig
from .transforms import transforms_ref


class ParameterConfig(pexConfig.Config):
    """Basic configuration for all parameters."""
    fixed = pexConfig.Field[bool](default=True, doc="Whether parameter is fixed or not (free)")
    value_initial = pexConfig.Field[float](doc="Initial value")


class GaussianConfig(ShapePriorConfig):
    """Configuration for a gauss2d.fit Gaussian component"""
    sigma_initial = pexConfig.Field[float](default=2.5, doc="Initial x- and y-axis sigma value")


class SersicIndexConfig(ParameterConfig):
    """Specific configuration for a Sersic index parameter."""
    def setDefaults(self):
        self.value_initial = 0.5


class SersicConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Sersic component."""
    sersicindex = pexConfig.ConfigField[SersicIndexConfig](doc="Sersic index config")

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        channels: list[g2f.Channel],
    ) -> g2f.Component:
        """Make a Component reflecting the current configuration.

        Parameters
        ----------
        centroid : `gauss2d.fit.CentroidParameters`
            Centroid parameters for the component.
        channels : list[`gauss2d.fit.Channel`]
            A list of gauss2d.fit.Channel to populate a
            `gauss2d.fit.LinearIntegralModel` with.

        Returns
        -------
        component: `gauss2d.fit.Component`
            A suitable `gauss2d.fit.GaussianComponent` if the Sersic index is
            fixed at 0.5, or a `gauss2d.fit.SersicMixComponent` otherwise.

        Notes
        -----
        The `gauss2d.fit.LinearIntegralModel` will be populated with normalized
        `gauss2d.fit.IntegralParameterD` instances, in preparation for linear
        least squares fitting.
        """
        transform_flux = transforms_ref['log10']
        transform_size = transforms_ref['log10']
        transform_rho = transforms_ref['logit_rho']
        integral = g2f.LinearIntegralModel(
            {channel: g2f.IntegralParameterD(1.0, transform=transform_flux) for channel in channels}
        )
        if self.sersicindex.value_initial == 0.5 and self.sersicindex.fixed:
            return g2f.GaussianComponent(
                centroid=centroid,
                ellipse=g2f.GaussianParametricEllipse(
                    sigma_x=g2f.SigmaXParameterD(0, transform=transform_size),
                    sigma_y=g2f.SigmaYParameterD(0, transform=transform_size),
                    rho=g2f.RhoParameterD(0, transform=transform_rho),
                ),
                integral=integral,
            )
        return g2f.SersicMixComponent(
            centroid=centroid,
            ellipse=g2f.SersicParametricEllipse(
                size_x=g2f.ReffXParameterD(0, transform=transform_size),
                size_y=g2f.ReffYParameterD(0, transform=transform_size),
                rho=g2f.RhoParameterD(0, transform=transform_rho),
            ),
            integral=integral,
            sersicindex=g2f.SersicMixComponentIndexParameterD(
                value=self.sersicindex.value_initial,
                fixed=self.sersicindex.fixed,
                transform=transforms_ref['logit_sersic'] if not self.sersicindex.fixed else None,
            ),
        )