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

import string

import gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import pydantic
from pydantic.dataclasses import dataclass

from .limits import limits_ref
from .priors import ShapePriorConfig
from .transforms import transforms_ref
from .utils import FrozenArbitraryAllowedConfig

__all__ = [
    "ParameterConfig",
    "FluxFractionParameterConfig",
    "FluxParameterConfig",
    "CentroidConfig",
    "ComponentData",
    "Fluxes",
    "EllipticalComponentConfig",
    "GaussianComponentConfig",
    "SersicIndexParameterConfig",
    "SersicComponentConfig",
]


class ParameterConfig(pexConfig.Config):
    """Configuration for a parameter."""

    fixed = pexConfig.Field[bool](default=False, doc="Whether parameter is fixed or not (free)")
    value_initial = pexConfig.Field[float](default=0, doc="Initial value")


class FluxParameterConfig(ParameterConfig):
    """Configuration for flux parameters (IntegralParameterD).

    The safest initial value for a flux is 1.0, because if it's set to zero,
    linear fitting will not work correctly initially.
    """

    def setDefaults(self):
        super().setDefaults()
        self.value_initial = 1.0


class FluxFractionParameterConfig(ParameterConfig):
    """Configuration for flux fraction parameters (ProperFractionParameterD).

    The safest initial value for a flux fraction is 0.5, because if it's set
    to one, downstream fractions will be zero, while if it's set to zero,
    linear fitting will not work correctly initially.
    """

    def setDefaults(self):
        super().setDefaults()
        self.value_initial = 1.0


class CentroidConfig(pexConfig.Config):
    """Configuration for a component centroid."""

    x = pexConfig.ConfigField[ParameterConfig](doc="The x-axis centroid configuration")
    y = pexConfig.ConfigField[ParameterConfig](doc="The y-axis centroid configuration")

    def make_centroid(self) -> g2f.CentroidParameters:
        cen_x, cen_y = (
            type_param(config.value_initial, fixed=config.fixed, limits=g2f.LimitsD())
            for (config, type_param) in (
                (self.x, g2f.CentroidXParameterD), (self.y, g2f.CentroidYParameterD)
            )
        )
        centroid = g2f.CentroidParameters(x=cen_x, y=cen_y)
        return centroid


@dataclass(kw_only=True, frozen=True, config=FrozenArbitraryAllowedConfig)
class ComponentData:
    """Dataclass for a Component config."""

    component: g2f.Component = pydantic.Field(title="The component instance")
    integralmodel: g2f.IntegralModel = pydantic.Field(title="The component's integralmodel")
    priors: list[g2f.Prior] = pydantic.Field(title="The priors associated with the component")


Fluxes = dict[g2f.Channel, float]


class EllipticalComponentConfig(ShapePriorConfig):
    """Configuration for an elliptically-symmetric component.

    This class can be initialized but cannot implement make_component.
    """

    fluxfrac = pexConfig.ConfigField[FluxFractionParameterConfig](
        doc="Fractional flux parameter(s) config",
        default=None,
    )
    flux = pexConfig.ConfigField[FluxParameterConfig](
        doc="Flux parameter(s) config",
        default=FluxParameterConfig,
    )

    rho = pexConfig.ConfigField[ParameterConfig](doc="Rho parameter config")
    size_x = pexConfig.ConfigField[ParameterConfig](doc="x-axis size parameter config")
    size_y = pexConfig.ConfigField[ParameterConfig](doc="y-axis size parameter config")
    transform_flux_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux parameters",
        default="log10",
        optional=True,
    )
    transform_fluxfrac_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux fraction parameters",
        default="logit_fluxfrac",
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
        """Format a label for a band-dependent parameter.

        Parameters
        ----------
        label
            The label to format.
        name_channel
            The name of the channel to format with.

        Returns
        -------
        label_formmated
            The formatted label.
        """
        label_formatted = string.Template(label).safe_substitute(
            type_component=self.get_type_name(), name_channel=name_channel,
        )
        return label_formatted

    @staticmethod
    def get_integral_label_default() -> str:
        """Return the default integral label."""
        return "${type_component} ${name_channel}-band"

    def get_size_label(self) -> str:
        """Return the label for the component's size parameters."""
        raise NotImplementedError("EllipticalComponent does not implement get_size_label")

    def get_type_name(self) -> str:
        """Return a descriptive component name."""
        raise NotImplementedError("EllipticalComponent does not implement get_type_name")

    def get_transform_fluxfrac(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_fluxfrac_name] if self.transform_fluxfrac_name else None

    def get_transform_flux(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_flux_name] if self.transform_flux_name else None

    def get_transform_rho(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_rho_name] if self.transform_rho_name else None

    def get_transform_size(self) -> g2f.TransformD | None:
        return transforms_ref[self.transform_size_name] if self.transform_size_name else None

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        integralmodel: g2f.IntegralModel,
    ) -> ComponentData:
        """Make a Component reflecting the current configuration.

        Parameters
        ----------
        centroid
            Centroid parameters for the component.
        integralmodel
            The integralmodel for this component.

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

    def make_fluxfrac_parameter(
        self,
        value: float | None,
        label: str | None = None,
        **kwargs
    ) -> g2f.ProperFractionParameterD:
        parameter = g2f.ProperFractionParameterD(
            value if value is None else self.fluxfrac.value_initial,
            fixed=self.fluxfrac.fixed,
            transform=self.get_transform_fluxfrac(),
            label=label if label is not None else "",
            **kwargs
        )
        return parameter

    def make_flux_parameter(
        self,
        value: float | None,
        label: str | None = None,
        **kwargs
    ) -> g2f.IntegralParameterD:
        parameter = g2f.IntegralParameterD(
            value if value is not None else self.flux.value_initial,
            fixed=self.flux.fixed,
            transform=self.get_transform_flux(),
            label=label if label is not None else "",
            **kwargs
        )
        return parameter

    def make_linearintegralmodel(
        self,
        fluxes: Fluxes,
        label_integral: str | None = None,
        **kwargs
    ) -> g2f.IntegralModel:
        """Make a gauss2d.fit LinearIntegralModel for this component.

        Parameters
        ----------
        fluxes
            Configurations, including initial values, for the flux
            parameters by channel.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant channel with e.g. {channel.name}.
        **kwargs
            Additional keyword arguments to pass to make_flux_parameter.
            Some parameters cannot be overriden from their configs.

        Returns
        -------
        integralmodel
            The requested integralmodel.
        """
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        integralmodel = g2f.LinearIntegralModel(
            [
                (
                    channel,
                    self.make_flux_parameter(
                        flux,
                        label=self.format_label(label_integral, name_channel=channel.name),
                        **kwargs,
                    ),
                )
                for channel, flux in fluxes.items()
            ]
        )
        return integralmodel

    @staticmethod
    def set_size_x(component: g2f.EllipticalComponent, size_x: float) -> None:
        component.ellipse.sigma_x = size_x

    @staticmethod
    def set_size_y(component: g2f.EllipticalComponent, size_y: float) -> None:
        component.ellipse.sigma_y = size_y

    @staticmethod
    def set_rho(component: g2f.EllipticalComponent, rho: float) -> None:
        component.ellipse.rho = rho


class GaussianComponentConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Gaussian component."""

    transform_frac_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux fraction parameters",
        default="log10",
        optional=True,
    )

    def get_size_label(self) -> str:
        return "sigma"

    def get_type_name(self) -> str:
        return "Gaussian"

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        integralmodel: g2f.IntegralModel,
    ) -> ComponentData:
        ellipse = self.make_gaussianparametricellipse()
        prior = self.get_shape_prior(ellipse)
        componentdata = ComponentData(
            component=g2f.GaussianComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integralmodel,
            ),
            integralmodel=integralmodel,
            priors=[] if prior is None else [prior],
        )
        return componentdata


class SersicIndexParameterConfig(ParameterConfig):
    """Configuration for a gauss2d.fit Sersic index parameter."""

    prior_mean = pexConfig.Field[float](doc="Mean for the prior (untransformed)", default=1.0, optional=True)
    prior_stddev = pexConfig.Field[float](doc="Std. dev. for the prior", default=0.5, optional=True)
    prior_transformed = pexConfig.Field[float](
        doc="Whether the prior should be in transformed values", default=True,
    )

    def get_prior(self, param: g2f.SersicIndexParameterD) -> g2f.Prior | None:
        if self.prior_mean is not None:
            mean = param.transform.forward(self.prior_mean) if self.prior_transformed else self.prior_mean
            stddev = (
                param.transform.forward(self.prior_mean + self.prior_stddev/2.) -
                param.transform.forward(self.prior_mean - self.prior_stddev/2.)
            ) if self.prior_transformed else self.prior_stddev
            return g2f.GaussianPrior(
                param=param, mean=mean, stddev=stddev, transformed=self.prior_transformed,
            )
        return None

    def setDefaults(self):
        self.value_initial = 0.5

    def validate(self):
        super().validate()
        if self.prior_mean is not None:
            if not self.prior_mean > 0.:
                raise ValueError("Sersic index prior mean must be > 0")
            if not self.prior_stddev > 0.:
                raise ValueError("Sersic index prior std. dev. must be > 0")


class SersicComponentConfig(EllipticalComponentConfig):
    """Configuration for a gauss2d.fit Sersic component.

    Notes
    -----
    make_component will return a `gauss2d.fit.GaussianComponent` if the Sersic
    index is fixed at 0.5, or a `gauss2d.fit.SersicMixComponent` otherwise.
    """

    _interpolators: dict[int, g2f.SersicMixInterpolator] = {}

    order = pexConfig.ChoiceField[int](doc="Sersic mix order", allowed={4: "Four", 8: "Eight"}, default=4)
    sersicindex = pexConfig.ConfigField[SersicIndexParameterConfig](doc="Sersic index config")

    def get_interpolator(self, order: int):
        return self._interpolators.get(
            order,
            (
                g2f.GSLSersicMixInterpolator
                if hasattr(g2f, "GSLSersicMixInterpolator")
                else g2f.LinearSersicMixInterpolator
            )(order=order),
        )

    def get_size_label(self) -> str:
        return "reff"

    def get_type_name(self) -> str:
        is_gaussian_fixed = self.is_gaussian_fixed()
        return f"{'Gaussian (fixed Sersic)' if is_gaussian_fixed else 'Sersic'}"

    def is_gaussian_fixed(self):
        return self.sersicindex.value_initial == 0.5 and self.sersicindex.fixed

    def make_component(
        self,
        centroid: g2f.CentroidParameters,
        integralmodel: g2f.IntegralModel,
    ) -> ComponentData:
        is_gaussian_fixed = self.is_gaussian_fixed()
        transform_size = self.get_transform_size()
        transform_rho = self.get_transform_rho()
        if is_gaussian_fixed:
            ellipse = self.make_gaussianparametricellipse()
            component = g2f.GaussianComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integralmodel,
            )
            priors = []
        else:
            ellipse = g2f.SersicParametricEllipse(
                size_x=g2f.ReffXParameterD(
                    self.size_x.value_initial, transform=transform_size, fixed=self.size_x.fixed
                ),
                size_y=g2f.ReffYParameterD(
                    self.size_y.value_initial, transform=transform_size, fixed=self.size_y.fixed
                ),
                rho=g2f.RhoParameterD(self.rho.value_initial, transform=transform_rho, fixed=self.rho.fixed),
            )
            sersicindex = g2f.SersicMixComponentIndexParameterD(
                value=self.sersicindex.value_initial,
                fixed=self.sersicindex.fixed,
                transform=transforms_ref["logit_sersic"] if not self.sersicindex.fixed else None,
                interpolator=self.get_interpolator(order=self.order),
                limits=limits_ref["n_ser_multigauss"],
            )
            component = g2f.SersicMixComponent(
                centroid=centroid,
                ellipse=ellipse,
                integral=integralmodel,
                sersicindex=sersicindex,
            )
            prior = self.sersicindex.get_prior(sersicindex) if not sersicindex.fixed else None
            priors = [prior] if prior else []
        prior = self.get_shape_prior(ellipse)
        if prior:
            priors.append(prior)
        return ComponentData(
            component=component,
            integralmodel=integralmodel,
            priors=priors,
        )

    def validate(self):
        super().validate()
