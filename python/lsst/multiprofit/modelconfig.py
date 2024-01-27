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
from typing import Iterable

import gauss2d.fit as g2f
import lsst.pex.config as pexConfig

from .componentconfig import Fluxes
from .sourceconfig import SourceConfig


class ModelConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Model."""

    sources = pexConfig.ConfigDictField[str, SourceConfig](doc="The configuration for sources")

    @staticmethod
    def format_label(label: str, name_source: str) -> str:
        return string.Template(label).safe_substitute(name_source=name_source)

    def get_integral_label_default(self, sourceconfig: SourceConfig) -> str:
        prefix = "src: {name_source} " if self.has_prefix_source() else ""
        return f"{prefix}{sourceconfig.get_integral_label_default()}"

    def has_prefix_source(self) -> bool:
        return (len(self.sources) > 1) or next(iter(self.sources.keys()))

    def make_sources(
        self,
        componentgroup_fluxes_srcs: Iterable[list[list[Fluxes]]],
        label_integral: str | None = None,
    ) -> tuple[list[g2f.Source], list[g2f.Prior]]:
        n_src = len(self.sources)
        if componentgroup_fluxes_srcs is None or len(componentgroup_fluxes_srcs) != n_src:
            raise ValueError(f"{len(componentgroup_fluxes_srcs)=} != {n_src=}")

        sources = []
        priors = []
        for componentgroup_fluxes, (name_src, config_src) in zip(
                componentgroup_fluxes_srcs, self.sources.items()
        ):
            label_integral_src = label_integral if label_integral is not None else (
                self.get_integral_label_default(config_src))

            source, priors_src = config_src.make_source(
                componentgroup_fluxes=componentgroup_fluxes,
                label_integral=self.format_label(label=label_integral_src, name_source=name_src)
            )
            sources.append(source)
            priors.extend(priors_src)

        return sources, priors

    def make_model(
        self,
        componentgroup_fluxes_srcs: Iterable[list[list[Fluxes]]],
        data: g2f.Data,
        psfmodels: list[g2f.PsfModel],
        label_integral: str | None = None,
    ) -> g2f.Model:
        sources, priors = self.make_sources(
            componentgroup_fluxes_srcs=componentgroup_fluxes_srcs,
            label_integral=label_integral,
        )

        model = g2f.Model(data=data, psfmodels=psfmodels, sources=sources, priors=priors)

        return model
