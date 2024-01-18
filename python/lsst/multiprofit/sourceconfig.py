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

from .componentconfig import EllipticalComponentConfig, Fluxes, GaussianComponentConfig, SersicComponentConfig

CentroidFluxes = list[tuple[g2f.CentroidParameters, list[Fluxes]]]


class ComponentMixtureConfig(pexConfig.Config):
    """Configuration for a group of gauss2d.fit Components sharing a centroid.

    ComponentMixtures may also have linked IntegralModels, e.g. if
    is_fractional is True.
    """

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

    def format_label(self, label: str, name_component: str) -> str:
        return string.Template(label).safe_substitute(name_component=name_component)

    @staticmethod
    def get_integral_label_default() -> str:
        return "comp: ${name_component} " + EllipticalComponentConfig.get_integral_label_default()

    def make_components(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: list[Fluxes],
        label_integral: str | None = None,
    ) -> tuple[list[g2f.Component], list[g2f.Prior]]:
        """Make a list of gauss2d.fit.Component from this configuration.

        Parameters
        ----------
        centroid
            Centroid parameters for all components.
        fluxes
            A list of dictionary of initial fluxes by gauss2d.fit.Channel to
            populate an appropriate `gauss2d.fit.IntegralModel` with.
            If self.is_fractional, the first item in the list must be
            total fluxes while the remainder are fractions (the final
            fraction is always fixed at 1.0 and must not be provided).
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant component name with {{name_component}}.

        Returns
        -------
        componentdata
            An appropriate ComponentData including the initialized component.
        """
        if label_integral is None:
            label_integral = self.get_integral_label_default()

        componentconfigs: dict[str, EllipticalComponentConfig] = dict(self.components_gauss)
        for name, component in self.components_sersic.items():
            componentconfigs[name] = component
        fluxes_iter = (fluxes[1:] + [None]) if self.is_fractional else fluxes
        if len(fluxes_iter) != len(componentconfigs):
            raise ValueError(f"{len(fluxes_iter)=} != {len(componentconfigs)=}")
        priors = []
        idx_last = len(componentconfigs) - 1
        kwargs = {"last": fluxes[0]} if self.is_fractional else {}
        components = []

        for idx, (fluxes_component, (name_component, config)) in enumerate(
            zip(fluxes_iter, componentconfigs.items())
        ):
            if self.is_fractional:
                kwargs["is_final"] = idx == idx_last

            componentdata = config.make_component(
                centroid=centroid,
                fluxes=fluxes_component,
                label_integral=self.format_label(label_integral, name_component=name_component),
                **kwargs
            )
            if self.is_fractional:
                kwargs["last"] = componentdata.integralmodel
            components.append(componentdata.component)
            priors.extend(componentdata.priors)
        return components, priors

    def validate(self):
        errors = []
        components: dict[str, EllipticalComponentConfig] = dict(self.components_gauss)
        for name, component in self.components_sersic.items():
            if name in components:
                errors.append(
                    f"key={name} cannot be used in both self.components_gauss and self.components_sersic"
                )
            components[name] = component

        n_components = len(components)
        n_components_min = 1 + self.is_fractional
        if not (n_components >= n_components_min):
            errors.append(f"Must have at least 1 + {self.is_fractional=} = {n_components_min} components")
        for name, component in components:
            if hasattr(component, "is_fractional"):
                is_fractional = component.is_fractional
                if is_fractional != self.is_fractional:
                    errors.append(
                        f"components[{name}].is_fractional={is_fractional} != {self.is_fractional=}"
                    )
            elif self.is_fractional:
                errors.append(f"components[{name}] cannot be fractional but {self.is_fractional=}")
        if errors:
            newline = "\n"
            raise ValueError(f"ComponentMixtureConfig has validation errors:\n{newline.join(errors)}")


class SourceConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Source.

    Sources may contain components with distinct centroids that may be linked
    by a prior (e.g. a galaxy + AGN + star clusters),
    although such priors are not yet implemented.
    """

    componentmixtures = pexConfig.ConfigDictField[str, ComponentMixtureConfig](
        doc="Components in the source",
        optional=False,
    )

    def _make_components_priors(
        self,
        centroid_fluxes: CentroidFluxes,
        label_integral: str,
        validate_psf: bool = False,
    ) -> [list[g2f.Component], list[g2f.Prior]]:
        if len(centroid_fluxes) != len(self.componentmixtures):
            raise ValueError(f"{len(centroid_fluxes)=} != {len(self.componentmixtures)=}")
        components = []
        priors = []
        if validate_psf:
            keys_expected = tuple((g2f.Channel.NONE,))
        for (centroid, fluxes), (name_mix, componentmixture) in zip(
            centroid_fluxes, self.componentmixtures.items()
        ):
            if validate_psf:
                for idx, fluxes_bands in enumerate(fluxes):
                    keys = tuple(fluxes_bands.keys())
                    if keys != keys_expected:
                        raise ValueError(
                            f"{name_mix=} comp[{idx}] {keys=} != {keys_expected=} with {validate_psf=}"
                        )

            components_i, priors_i = componentmixture.make_components(
                centroid=centroid,
                fluxes=fluxes,
                label_integral=self.format_label(label=label_integral, name_mixture=name_mix),
            )
            components.extend(components_i)
            priors.extend(priors_i)
        # TODO: Do more thorough PSF model validation
        # Consider asserting that is_fractional is True...
        # ...but PSF models don't have to be fractional.
        # Perhaps only check for unity total flux - but only fractional
        # models can actually guarantee this (to what threshold otherwise?)
        return components, priors

    def format_label(self, label: str, name_mixture: str) -> str:
        return string.Template(label).safe_substitute(name_mixture=name_mixture)

    @staticmethod
    def get_integral_label_default() -> str:
        return "mix: ${name_mixture} " + ComponentMixtureConfig.get_integral_label_default()

    def make_source(
        self,
        centroid_fluxes: CentroidFluxes,
        label_integral: str | None = None,
    ) -> [g2f.Source, list[g2f.Prior]]:
        """Make a gauss2d.fit.Source from this configuration.

        Parameters
        ----------
        centroid_fluxes
            A pair of Centroid parameters and a list of Fluxes to pass to each
            of the self.componentmixtures when calling make_components.
        label_integral
            A label to apply to integral parameters. Can reference the
            relevant component mixture name with {{name_mixture}}.

        Returns
        -------
        componentdata
            An appropriate ComponentData including the initialized component.
        """
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        components, priors = self._make_components_priors(
            centroid_fluxes=centroid_fluxes,
            label_integral=label_integral,
        )
        source = g2f.Source(components)
        return source, priors

    def make_psfmodel(
        self,
        centroid_fluxes: CentroidFluxes,
        label_integral: str | None = None,
    ) -> [g2f.PsfModel, list[g2f.Prior]]:
        if label_integral is None:
            label_integral = f"PSF {self.get_integral_label_default()}"
        components, priors = self._make_components_priors(
            centroid_fluxes=centroid_fluxes,
            label_integral=label_integral,
            validate_psf=True,
        )
        model = g2f.PsfModel(components=components)

        return model, priors

    def validate(self):
        errors = []
        if self.componentmixtures is None:
            errors.append("components is not optional")
        else:
            n_components = len(self.componentmixtures)
            n_components_min = 1
            if not (n_components >= n_components_min):
                errors.append(f"Must have at least {n_components_min=} components")
        if errors:
            newline = "\n"
            raise ValueError(f"SourceConfig has validation errors:\n{newline.join(errors)}")
