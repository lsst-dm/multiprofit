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
import string
from typing import Any

import gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import pydantic
from pydantic.dataclasses import dataclass

from .priors import ShapePriorConfig
from .transforms import transforms_ref
from .utils import ArbitraryAllowedConfig

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
    "GaussianComponentConfig",
    "SersicIndexConfig",
    "SersicComponentConfig",
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


@dataclass(kw_only=True, frozen=True, config=ArbitraryAllowedConfig)
class ComponentData:
    component: g2f.Component = pydantic.Field(title="The component instance")
    integralmodel: g2f.IntegralModel = pydantic.Field(title="The component's integralmodel")
    priors: list[g2f.Prior] = pydantic.Field(title="The priors associated with the component")


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

    def format_label(self, label: str, name_channel: str) -> str:
        return string.Template(label).safe_substitute(
            type_component=self.get_type_name(), name_channel=name_channel,
        )

    @staticmethod
    def get_integral_label_default() -> str:
        return "${type_component} ${name_channel}-band"

    @abstractmethod
    def get_type_name(self) -> str:
        """Return a descriptive component name."""
        raise NotImplementedError("EllipticalComponent does not implement get_type_name")

    def get_transform_flux(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_flux_name] if self.transform_flux_name else None

    def get_transform_rho(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_rho_name] if self.transform_rho_name else None

    def get_transform_size(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_size_name] if self.transform_size_name else None

    @abstractmethod
    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes,
        label_integral: str | None = None,
        **kwargs
    ) -> ComponentData:
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
            relevant channel with e.g. {channel.name}.
        **kwargs
            Additional optional keyword arguments for subclasses.

        Returns
        -------
        componentdata
            An appropriate ComponentData including the initialized component.

        Notes
        -----
        The default `gauss2d.fit.LinearIntegralModel` can be populated with
        unit fluxes (`gauss2d.fit.IntegralParameterD` instances) to prepare
        for linear least squares fitting.
        """
        raise NotImplementedError("EllipticalComponent cannot not implement make_component")

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
            label_integral = self.get_integral_label_default()
        transform_flux = self.get_transform_flux()
        integralmodel = g2f.LinearIntegralModel(
            [
                (
                    channel,
                    g2f.IntegralParameterD(
                        config_flux.value_initial,
                        transform=transform_flux,
                        fixed=config_flux.fixed,
                        label=self.format_label(label_integral, name_channel=channel.name),
                    ),
                )
                for channel, config_flux in fluxes.items()
            ]
        )
        return integralmodel


class GaussianComponentConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Gaussian component."""

    is_fractional = pexConfig.Field[bool](
        doc="Whether the integralmodel is fractional",
        default=False,
        optional=False,
    )
    transform_frac_name = pexConfig.Field[str](
        doc="The name of the reference transform for size parameters",
        default="log10",
        optional=True,
    )

    def get_type_name(self) -> str:
        return "Gaussian"

    def get_transform_frac(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_frac_name] if self.transform_frac_name else None

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes | None,
        label_integral: str | None = None,
        last: g2f.IntegralModel | dict[g2f.Channel, float] | None = None,
        is_final: bool | None = None,
        **kwargs,
    ) -> ComponentData:
        """Make a Component reflecting the current configuration.

        Parameters
        ----------
        centroid
            Centroid parameters for the component.
        fluxes
            A dictionary of initial fluxes (or fractions if is_fractional) by
            gauss2d.fit.Channel to populate an appropriate
            `gauss2d.fit.IntegralModel` with.
        label_integral
            A label to apply to integral parameters. See format_label for
            valid templates for substitution via string formatting.
        last
            The previous IntegralModel, or dict of total flux values by channel
            if this is the first component. Required if self.is_fractional and
            must be None otherwise.
        is_final
            Whether this is the final component in a fractional model.
            Required if self.is_fractional and must be None otherwise.
        **kwargs
            Any additional keyword arguments are invalid and will raise a
            ValueError.

        Returns
        -------
        componentdata
            An appropriate ComponentData including the initialized component.

        Notes
        -----
        The default `gauss2d.fit.LinearIntegralModel` can be populated with
        unit fluxes (`gauss2d.fit.IntegralParameterD` instances) to prepare
        for linear least squares fitting.
        """
        if kwargs:
            raise ValueError(f"GaussianConfig.make_component got unrecognized kwargs: {list(kwargs.keys())=}")
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        if self.is_fractional:
            if is_final is None:
                raise ValueError(f"is_final must be specified since {self.is_fractional=} is True")
            is_first = not isinstance(last, g2f.IntegralModel)
            channel = g2f.Channel.NONE
            if is_final:
                if fluxes is not None:
                    raise ValueError(f"fluxes must not be specified if {is_final=}")
                value_initial = 1.0
                fixed = True
            else:
                config_flux = fluxes[channel]
                fixed = config_flux.fixed
                value_initial = config_flux.value_initial

            param_frac = g2f.ProperFractionParameterD(
                value_initial,
                fixed=fixed,
                transform=self.get_transform_frac(),
            )
            if is_first:
                if last is not None:
                    config_flux = last[channel]
                    value_initial, fixed = config_flux.value_initial, config_flux.fixed
                else:
                    value_initial, fixed = 1.0, False
                integralmodel = g2f.LinearIntegralModel(
                    [(channel, g2f.IntegralParameterD(value=value_initial, fixed=fixed))]
                )
            else:
                integralmodel = last
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
        return ComponentData(
            component=g2f.GaussianComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integral,
            ),
            integralmodel=integral,
            priors=[] if prior is None else [prior],
        )


class SersicIndexConfig(ParameterConfig):
    """Specific configuration for a Sersic index parameter."""

    def setDefaults(self):
        self.value_initial = 0.5


class SersicComponentConfig(EllipticalComponentConfig):
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

    def get_type_name(self) -> str:
        is_gaussian_fixed = self.is_gaussian_fixed()
        return f"{'Gaussian (fixed Sersic)' if is_gaussian_fixed else 'Sersic'}"

    def is_gaussian_fixed(self):
        return self.sersicindex.value_initial == 0.5 and self.sersicindex.fixed

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: Fluxes,
        label_integral: str | None = None,
        **kwargs,
    ) -> tuple[g2f.Component, list[g2f.Prior]]:
        if kwargs:
            raise ValueError("SersicConfig.make_component does not take kwargs")
        is_gaussian_fixed = self.is_gaussian_fixed()
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        integral = self.make_linearintegralmodel(fluxes, label_integral=label_integral)
        transform_size = self.get_transform_size()
        transform_rho = self.get_transform_rho()
        if is_gaussian_fixed:
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
        return ComponentData(
            component=component,
            integralmodel=integral,
            priors=[] if prior is None else [prior],
        )
