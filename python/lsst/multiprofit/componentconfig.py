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
from typing import Any

import gauss2d.fit as g2f
import lsst.pex.config as pexConfig

from .priors import ShapePriorConfig
from .transforms import transforms_ref

parameter_names = {
    g2f.CentroidXParameterD: "cen_x",
    g2f.CentroidYParameterD: "cen_y",
    g2f.ReffXParameterD: "reff_x",
    g2f.ReffYParameterD: "reff_y",
    g2f.RhoParameterD: "rho",
    g2f.SigmaXParameterD: "sigma_x",
    g2f.SigmaYParameterD: "sigma_y",
}

__all__ = [
    "init_component",
    "ParameterConfig",
    "EllipticalComponentConfig",
    "GaussianConfig",
    "SersicIndexConfig",
    "SersicConfig",
]


def init_component(component: g2f.Component, **kwargs: Any):
    """Initialize a component with parameter name-value pairs.

    Parameters
    ----------
    component
        The component to initialize.
    **kwargs
        Additional keyword arguments.

    Notes
    -----
    kwargs keywords should be a value in parameter_names and values should be
    valid for initializing the parameter of that type.
    """
    for parameter in set(component.parameters()):
        if kwarg := parameter_names.get(parameter.__class__):
            if value := kwargs.get(kwarg):
                parameter.value = value


class ParameterConfig(pexConfig.Config):
    """Basic configuration for all parameters."""

    fixed = pexConfig.Field[bool](default=False, doc="Whether parameter is fixed or not (free)")
    value_initial = pexConfig.Field[float](default=0, doc="Initial value")


Fluxes = dict[g2f.Channel, ParameterConfig]


class EllipticalComponentConfig(ShapePriorConfig):
    """Config for an elliptically-symmetric component.

    This class can be initialized but cannot implement make_component.
    """

    rho = pexConfig.ConfigField[ParameterConfig](doc="Rho parameter config")
    size_x = pexConfig.ConfigField[ParameterConfig](doc="x-axis size parameter config")
    size_y = pexConfig.ConfigField[ParameterConfig](doc="y-axis size parameter config")
    transform_flux_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux parameters",
        default="log10",
        optional=True,
    )
    transform_rho_name = pexConfig.Field[str](
        doc="The name of the reference transform for rho parameters",
        default="logit_rho",
        optional=True,
    )
    transform_size_name = pexConfig.Field[str](
        doc="The name of the reference transform for size parameters",
        default="log10",
        optional=True,
    )

    @abstractmethod
    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes,
        label_integral: str | None = None,
        **kwargs
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
        """Make a Component reflecting the current configuration.

        Parameters
        ----------
        centroid
            Centroid parameters for the component.
        fluxes
            A dictionary of initial fluxes by gauss2d.fit.Channel to populate
            a `gauss2d.fit.IntegralModel` with.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant channel with e.g. {{channel.name}}.
        kwargs
            Additional optional keyword arguments for subclasses.

        Returns
        -------
        component
            An appropriate `gauss2d.fit.GaussianComponent`.
        priors
            A list of priors.

        Notes
        -----
        The default `gauss2d.fit.LinearIntegralModel` can be populated with
        unit fluxes (`gauss2d.fit.IntegralParameterD` instances) to prepare
        for linear least squares fitting.
        """
        raise RuntimeError("EllipticalComponent cannot not implement make_component")

    def get_transform_flux(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_flux_name] if self.transform_flux_name else None

    def get_transform_rho(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_rho_name] if self.transform_rho_name else None

    def get_transform_size(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_size_name] if self.transform_size_name else None

    def make_gaussianparametricellipse(self) -> g2f.GaussianParametricEllipse:
        transform_size = self.get_transform_size()
        transform_rho = self.get_transform_rho()
        ellipse = g2f.GaussianParametricEllipse(
            sigma_x=g2f.SigmaXParameterD(
                self.size_x.value_initial, transform=transform_size, fixed=self.size_x.fixed
            ),
            sigma_y=g2f.SigmaYParameterD(
                self.size_y.value_initial, transform=transform_size, fixed=self.size_y.fixed
            ),
            rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
        )
        return ellipse

    def make_linearintegralmodel(
        self,
        fluxes: Fluxes,
        label_integral: str | None = None,
    ) -> g2f.IntegralModel:
        if label_integral is None:
            label_integral = "{{channel.name}}-band"
        transform_flux = self.get_transform_flux()
        integralmodel = g2f.LinearIntegralModel(
            [
                (
                    channel,
                    g2f.IntegralParameterD(
                        config_flux.value_initial,
                        transform=transform_flux,
                        fixed=config_flux.fixed,
                        label=label_integral.format(channel=channel),
                    ),
                )
                for channel, config_flux in fluxes.items()
            ]
        )
        return integralmodel


class GaussianConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Gaussian component."""
    is_fractional = pexConfig.Field[bool](doc="Whether the integralmodel is fractional", default=False)
    transform_frac_name = pexConfig.Field[str](
        doc="The name of the reference transform for size parameters",
        default="log10",
        optional=True,
    )

    def get_transform_frac(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_frac_name] if self.transform_frac_name else None

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes,
        label_integral: str | None = None,
        last: g2f.IntegralModel | float | None = None,
        is_final: bool | None = None,
        **kwargs,
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
        """Make a Component reflecting the current configuration.

        Parameters
        ----------
        centroid
            Centroid parameters for the component.
        fluxes
            A dictionary of initial fluxes by gauss2d.fit.Channel to populate
            a `gauss2d.fit.IntegralModel` with.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant channel with e.g. {{channel.name}}.
        last
            The previous IntegralModel, or the value of the total flux if
            this is the first component. Required if self.is_fractional and
            must be None otherwise.
        is_final
            Whether this is the final component in a fractional model.
            Required if self.is_fractional and must be None otherwise.

        Returns
        -------
        component
            An appropriate `gauss2d.fit.GaussianComponent`.
        priors
            A list of priors.

        Notes
        -----
        The default `gauss2d.fit.LinearIntegralModel` can be populated with
        unit fluxes (`gauss2d.fit.IntegralParameterD` instances) to prepare
        for linear least squares fitting.
        """
        if kwargs:
            raise ValueError(f"GaussianConfig.make_component got unrecognized kwargs: {list(kwargs.keys())=}")
        if label_integral is None:
            label_integral = "Gaussian {{channel.name}}-band"
        if self.is_fractional:
            if is_final is None:
                raise ValueError(f"is_final must be specified since {self.is_fractional=} is True")
            is_first = not isinstance(last, g2f.IntegralModel)
            channel = g2f.Channel.NONE
            config_flux = fluxes[channel]
            param_frac = g2f.ProperFractionParameterD(
                config_flux.value_initial,
                fixed=is_final | config_flux.fixed,
                transform=self.get_transform_frac(),
            )
            integralmodel = g2f.LinearIntegralModel(
                [(channel, g2f.IntegralParameterD(last if last is not None else 1.0))]
            ) if is_first else last
            integral = g2f.FractionalIntegralModel(
                [(channel, param_frac)],
                integralmodel,
                is_final,
            )
        else:
            if last is not None or is_final is not None:
                raise ValueError(
                    f"Cannot specify {last=} or {is_final=} since {self.is_fractional=} is not True"
                )
            integral = self.make_linearintegralmodel(fluxes, label_integral=label_integral)
        ellipse = self.make_gaussianparametricellipse()
        prior = self.get_shape_prior(ellipse)
        return g2f.GaussianComponent(
            centroid=centroid,
            ellipse=ellipse,
            integral=integral,
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

    _interpolators: dict[int, g2f.SersicMixInterpolator] = {}

    order = pexConfig.ChoiceField[int](doc="Sersic mix order", allowed={4: "Four", 8: "Eight"}, default=4)
    sersicindex = pexConfig.ConfigField[SersicIndexConfig](doc="Sersic index config")

    def get_interpolator(self, order: int):
        return self._interpolators.get(
            order,
            (
                g2f.GSLSersicMixInterpolator
                if hasattr(g2f, "GSLSersicMixInterpolator")
                else g2f.LinearSersicMixInterpolator
            )(order=order),
        )

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes,
        label_integral: str | None = None,
        **kwargs,
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
        if kwargs:
            raise ValueError("SersicConfig.make_component does not take kwargs")
        is_gaussian = self.sersicindex.value_initial == 0.5 and self.sersicindex.fixed
        if label_integral is None:
            label_integral = f"{'Gaussian' if is_gaussian else 'Sersic'} {{channel.name}}-band"
        integral = self.make_linearintegralmodel(fluxes, label_integral=label_integral)
        transform_size = self.get_transform_size()
        transform_rho = self.get_transform_rho()
        if is_gaussian:
            ellipse = self.make_gaussianparametricellipse()
            component = g2f.GaussianComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integral,
            )
        else:
            ellipse = g2f.SersicParametricEllipse(
                size_x=g2f.ReffXParameterD(
                    self.size_x.value_initial, transform=transform_size, fixed=self.size_y.fixed
                ),
                size_y=g2f.ReffYParameterD(
                    self.size_y.value_initial, transform=transform_size, fixed=self.size_y.fixed
                ),
                rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
            )
            component = g2f.SersicMixComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integral,
                sersicindex=g2f.SersicMixComponentIndexParameterD(
                    value=self.sersicindex.value_initial,
                    fixed=self.sersicindex.fixed,
                    transform=transforms_ref["logit_sersic"] if not self.sersicindex.fixed else None,
                    interpolator=self.get_interpolator(order=self.order),
                ),
            )
        prior = self.get_shape_prior(ellipse)
        return component, ([] if prior is None else [prior])
