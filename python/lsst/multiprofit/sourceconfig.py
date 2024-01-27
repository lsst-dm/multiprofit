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

from .componentconfig import (
    CentroidConfig,
    EllipticalComponentConfig,
    Fluxes,
    GaussianComponentConfig,
    SersicComponentConfig,
)

__all__ = [
    "ComponentConfigs", "CentroidConfig", "ComponentGroupConfig", "SourceConfig",
]

ComponentConfigs = dict[str, EllipticalComponentConfig]


class ComponentGroupConfig(pexConfig.Config):
    """Configuration for a group of gauss2d.fit Components.

    ComponentGroups may have linked CentroidParameters
    and IntegralModels, e.g. if is_fractional is True.

    Notes
    -----
    Gaussian components are generated first, then Sersic.

    This config class has no equivalent in gauss2dfit, because gauss2dfit
    model parameter dependencies implicitly. This class implements only a
    subset of typical use cases, i.e. PSFs sharing a fractional integral
    model with fixed unit flux, and galaxies/PSF components sharing a single
    common centroid.
    If greater flexibility in linking parameter values is needed,
    users must assemble their own gauss2dfit models directly.
    """

    centroids = pexConfig.ConfigDictField[str, CentroidConfig](
        doc="Centroids by key, which can be a component name or 'default'."
            "The 'default' key-value pair must be specified if it is needed.",
        default={"default": CentroidConfig()},
    )
    # TODO: Change this to just one EllipticalComponentConfig field
    # when pex_config supports derived types in ConfigDictField
    # (possibly DM-41049)
    components_gauss = pexConfig.ConfigDictField[str, GaussianComponentConfig](
        doc="Gaussian Components in the source",
        optional=False,
        default={},
    )
    components_sersic = pexConfig.ConfigDictField[str, SersicComponentConfig](
        doc="Sersic Components in the component mixture",
        optional=False,
        default={},
    )
    is_fractional = pexConfig.Field[bool](doc="Whether the integralmodel is fractional", default=False)
    transform_fluxfrac_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux parameters",
        default="logit_fluxfrac",
        optional=True,
    )
    transform_flux_name = pexConfig.Field[str](
        doc="The name of the reference transform for flux parameters",
        default="log10",
        optional=True,
    )

    @staticmethod
    def format_label(label: str, name_component: str) -> str:
        return string.Template(label).safe_substitute(name_component=name_component)

    @staticmethod
    def get_integral_label_default() -> str:
        return "comp: ${name_component} " + EllipticalComponentConfig.get_integral_label_default()

    def get_componentconfigs(self) -> ComponentConfigs:
        componentconfigs: ComponentConfigs = dict(self.components_gauss)
        for name, component in self.components_sersic.items():
            componentconfigs[name] = component
        return componentconfigs

    @staticmethod
    def get_fluxes_default(
        channels: tuple[g2f.Channel], componentconfigs: ComponentConfigs, is_fractional: bool = False,
    ) -> list[Fluxes]:
        if len(componentconfigs) == 0:
            raise ValueError("Must provide at least one ComponentConfig")
        fluxes = []
        componentconfigs_iter = tuple(componentconfigs.values())[:len(componentconfigs) - is_fractional]
        for idx, componentconfig in enumerate(componentconfigs_iter):
            if is_fractional:
                if idx == 0:
                    value = componentconfig.flux.value_initial
                    fluxes.append({channel: value for channel in channels})
                value = componentconfig.fluxfrac.value_initial
                fluxes.append({channel: value for channel in channels})
            else:
                value = componentconfig.flux.value_initial
                fluxes.append({channel: value for channel in channels})
        return fluxes

    def make_components(
        self,
        componentfluxes: list[Fluxes],
        label_integral: str | None = None,
    ) -> tuple[list[g2f.Component], list[g2f.Prior]]:
        """Make a list of gauss2d.fit.Component from this configuration.

        Parameters
        ----------
        componentfluxes
            A list of Fluxes to populate an appropriate
            `gauss2d.fit.IntegralModel` with.
            If self.is_fractional, the first item in the list must be
            total fluxes while the remainder are fractions (the final
            fraction is always fixed at 1.0 and must not be provided).
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant component name with ${name_component}}.

        Returns
        -------
        componentdata
            An appropriate ComponentData including the initialized component.
        """
        componentconfigs = self.get_componentconfigs()
        fluxes_first = componentfluxes[0]
        channels = fluxes_first.keys()
        fluxes_all = (componentfluxes[1:] + [None]) if self.is_fractional else componentfluxes
        if len(fluxes_all) != len(componentconfigs):
            raise ValueError(f"{len(fluxes_all)=} != {len(componentconfigs)=}")
        priors = []
        idx_final = len(componentconfigs) - 1
        components = []
        last = None

        centroid_default = None
        for idx, (fluxes_component, (name_component, config_comp)) in enumerate(
            zip(fluxes_all, componentconfigs.items())
        ):
            label_integral_comp = self.format_label(
                label_integral if label_integral is not None else (
                    config_comp.get_integral_label_default()
                ),
                name_component=name_component,
            )

            if self.is_fractional:
                if idx == 0:
                    last = config_comp.make_linearintegralmodel(
                        fluxes=fluxes_first,
                        label_integral=label_integral_comp,
                    )

                is_final = idx == idx_final
                if is_final:
                    params_frac = [
                        (channel, g2f.ProperFractionParameterD(1.0, fixed=True))
                        for channel in channels
                    ]
                else:
                    if fluxes_component.keys() != channels:
                        raise ValueError(f"{name_component=} {fluxes_component=}")
                    params_frac = [
                        (
                            channel,
                            config_comp.make_fluxfrac_parameter(value=fluxfrac),
                        ) for channel, fluxfrac in fluxes_component.items()
                    ]

                integralmodel = g2f.FractionalIntegralModel(
                    params_frac,
                    model=last,
                    is_final=is_final,
                )
                # TODO: Omitting this crucial step should raise but doesn't
                # There shouldn't be two integralmodels with the same last
                # especially not one is_final and one not
                last = integralmodel
            else:
                integralmodel = config_comp.make_linearintegralmodel(
                    fluxes_component,
                    label_integral=label_integral_comp,
                )

            centroid = self.centroids.get(name_component)
            if not centroid:
                if centroid_default is None:
                    centroid_default = self.centroids["default"].make_centroid()
                centroid = centroid_default
            componentdata = config_comp.make_component(
                centroid=centroid,
                integralmodel=integralmodel,
            )
            components.append(componentdata.component)
            priors.extend(componentdata.priors)
        return components, priors

    def validate(self):
        super().validate()
        errors = []
        components: ComponentConfigs = dict(self.components_gauss)

        for name, component in self.components_sersic.items():
            if name in components:
                errors.append(
                    f"key={name} cannot be used in both self.components_gauss and self.components_sersic"
                )
            components[name] = component

        keys = set(self.centroids.keys())
        has_default = "default" in keys
        for name in components.keys():
            if name in keys:
                keys.remove(name)
            elif not has_default:
                errors.append(f"component {name=} has no entry in self.centroids and default not specified")
        if errors:
            newline = "\n"
            raise ValueError(f"ComponentMixtureConfig has validation errors:\n{newline.join(errors)}")


class SourceConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Source.

    Sources may contain components with distinct centroids that may be linked
    by a prior (e.g. a galaxy + AGN + star clusters),
    although such priors are not yet implemented.
    """

    componentgroups = pexConfig.ConfigDictField[str, ComponentGroupConfig](
        doc="Components in the source",
        optional=False,
    )

    def _make_components_priors(
        self,
        componentgroup_fluxes: list[list[Fluxes]],
        label_integral: str,
        validate_psf: bool = False,
    ) -> [list[g2f.Component], list[g2f.Prior]]:
        if len(componentgroup_fluxes) != len(self.componentgroups):
            raise ValueError(f"{len(componentgroup_fluxes)=} != {len(self.componentgroups)=}")
        components = []
        priors = []
        if validate_psf:
            keys_expected = tuple((g2f.Channel.NONE,))
        for componentfluxes, (name_group, componentgroup) in zip(
            componentgroup_fluxes, self.componentgroups.items()
        ):
            if validate_psf:
                for idx, fluxes_comp in enumerate(componentfluxes):
                    keys = tuple(fluxes_comp.keys())
                    if keys != keys_expected:
                        raise ValueError(
                            f"{name_group=} comp[{idx}] {keys=} != {keys_expected=} with {validate_psf=}"
                        )

            components_i, priors_i = componentgroup.make_components(
                componentfluxes=componentfluxes,
                label_integral=self.format_label(label=label_integral, name_group=name_group),
            )
            components.extend(components_i)
            priors.extend(priors_i)

        return components, priors

    @staticmethod
    def format_label(label: str, name_group: str) -> str:
        return string.Template(label).safe_substitute(name_group=name_group)

    def get_componentconfigs(self) -> ComponentConfigs:
        has_prefix_group = self.has_prefix_group()
        componentconfigs = {}
        for name_group, config_group in self.componentgroups.items():
            prefix_group = f"{name_group}_" if has_prefix_group else ""
            for name_comp, componentconfig in config_group.get_componentconfigs().items():
                componentconfigs[f"{prefix_group}{name_comp}"] = componentconfig
        return componentconfigs

    def get_integral_label_default(self) -> str:
        prefix = "mix: ${name_group} " if self.has_prefix_group() else ""
        return f"{prefix}{ComponentGroupConfig.get_integral_label_default()}"

    def has_prefix_group(self) -> bool:
        return (len(self.componentgroups) > 1) or next(iter(self.componentgroups.keys()))

    def make_source(
        self,
        componentgroup_fluxes: list[list[Fluxes]],
        label_integral: str | None = None,
    ) -> [g2f.Source, list[g2f.Prior]]:
        """Make a gauss2d.fit.Source from this configuration.

        Parameters
        ----------
        componentgroup_fluxes
            A list of Fluxes for each of the self.componentgroups to use
            when calling make_components.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant component mixture name with ${name_group}.

        Returns
        -------
        source
            An appropriate gauss2d.fit.Source.
        priors
            A list of priors from all constituent components.
        """
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        components, priors = self._make_components_priors(
            componentgroup_fluxes=componentgroup_fluxes,
            label_integral=label_integral,
        )
        source = g2f.Source(components)
        return source, priors

    def make_psfmodel(
        self,
        componentgroup_fluxes: list[list[Fluxes]],
        label_integral: str | None = None,
    ) -> [g2f.PsfModel, list[g2f.Prior]]:
        """Make a gauss2d.fit.PsfModel from this configuration.

        This method will validate that the arguments make a valid PSF model,
        i.e. with a unity total flux, and only one config for the none band.

        Parameters
        ----------
        componentgroup_fluxes
            A list of CentroidFluxes for each of the self.componentgroups
            when calling make_components.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant component mixture name with ${name_group}.

        Returns
        -------
        psfmodel
            An appropriate gauss2d.fit.PSfModel.
        priors
            A list of priors from all constituent components.
        """
        if label_integral is None:
            label_integral = f"PSF {self.get_integral_label_default()}"
        components, priors = self._make_components_priors(
            componentgroup_fluxes=componentgroup_fluxes,
            label_integral=label_integral,
            validate_psf=True,
        )
        model = g2f.PsfModel(components=components)

        return model, priors

    def validate(self):
        super().validate()
        if not self.componentgroups:
            raise ValueError("Must have at least one componentgroup")
