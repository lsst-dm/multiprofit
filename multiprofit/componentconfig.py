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

from abc import abstractmethod
import gauss2d.fit as g2f
import lsst.pex.config as pexConfig

from .priors import ShapePriorConfig
from .transforms import transforms_ref

parameter_names = {
    g2f.ReffXParameterD: 'reff_x',
    g2f.ReffYParameterD: 'reff_y',
    g2f.SigmaXParameterD: 'sigma_x',
    g2f.SigmaYParameterD: 'sigma_y',
}


def init_component(component: g2f.Component, **kwargs):
    for parameter in set(component.parameters()):
        if kwarg := parameter_names.get(parameter.__class__):
            if value := kwargs.get(kwarg):
                parameter.value = value


class ParameterConfig(pexConfig.Config):
    """Basic configuration for all parameters."""
    fixed = pexConfig.Field[bool](default=False, doc="Whether parameter is fixed or not (free)")
    value_initial = pexConfig.Field[float](default=0, doc="Initial value")


class SersicIndexConfig(ParameterConfig):
    """Specific configuration for a Sersic index parameter."""
    def setDefaults(self):
        self.value_initial = 0.5


class EllipticalComponentConfig(ShapePriorConfig):
    """Config for an elliptically-symmetric component"""
    rho = pexConfig.ConfigField[ParameterConfig](doc="Rho parameter config")
    size = pexConfig.ConfigField[ParameterConfig](doc="Size parameter config")

    @abstractmethod
    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        channels: list[g2f.Channel],
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
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
            A suitable `gauss2d.fit.GaussianComponent`.
        priors:
            A list of priors.

        Notes
        -----
        The `gauss2d.fit.LinearIntegralModel` will be populated with normalized
        `gauss2d.fit.IntegralParameterD` instances, in preparation for linear
        least squares fitting.
        """


class GaussianConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Gaussian component"""
    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        channels: list[g2f.Channel],
    ) -> g2f.Component:
        transform_flux = transforms_ref['log10']
        transform_size = transforms_ref['log10']
        transform_rho = transforms_ref['logit_rho']
        ellipse = g2f.GaussianParametricEllipse(
            sigma_x=g2f.SigmaXParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
            sigma_y=g2f.SigmaYParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
            rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
        )
        prior = self.get_shape_prior(ellipse)
        return g2f.GaussianComponent(
            centroid=centroid,
            ellipse=ellipse,
            integral=g2f.LinearIntegralModel(
                {channel: g2f.IntegralParameterD(1.0, transform=transform_flux)
                 for channel in channels}
            ),
        ), ([] if prior is None else [prior])


class SersicIndexConfig(ParameterConfig):
    """Specific configuration for a Sersic index parameter."""
    def setDefaults(self):
        self.value_initial = 0.5


class SersicConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Sersic component.

    Notes
    -----
    make_component will return a `gauss2d.fit.GaussianComponent` if the Sersic
    index is fixed at 0.5, or a `gauss2d.fit.SersicMixComponent` otherwise.
    """
    sersicindex = pexConfig.ConfigField[SersicIndexConfig](doc="Sersic index config")

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        channels: list[g2f.Channel],
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
        transform_flux = transforms_ref['log10']
        transform_size = transforms_ref['log10']
        transform_rho = transforms_ref['logit_rho']
        integral = g2f.LinearIntegralModel(
            {channel: g2f.IntegralParameterD(1.0, transform=transform_flux) for channel in channels}
        )
        if self.sersicindex.value_initial == 0.5 and self.sersicindex.fixed:
            ellipse = g2f.GaussianParametricEllipse(
                sigma_x=g2f.SigmaXParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
                sigma_y=g2f.SigmaYParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
                rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
            )
            component = g2f.GaussianComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integral,
            )
        else:
            ellipse = g2f.SersicParametricEllipse(
                size_x=g2f.ReffXParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
                size_y=g2f.ReffYParameterD(self.size.value_initial, transform=transform_size, fixed=self.size.fixed),
                rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
            )
            component = g2f.SersicMixComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integral,
                sersicindex=g2f.SersicMixComponentIndexParameterD(
                    value=self.sersicindex.value_initial,
                    fixed=self.sersicindex.fixed,
                    transform=transforms_ref['logit_sersic'] if not self.sersicindex.fixed else None,
                ),
            )
        prior = self.get_shape_prior(ellipse)
        return component, ([] if prior is None else [prior])
