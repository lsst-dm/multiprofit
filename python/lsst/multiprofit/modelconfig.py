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

from .sourceconfig import CentroidFluxes, SourceConfig


class ModelConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Model."""

    sources = pexConfig.ConfigDictField[str, SourceConfig](doc="The configuration for sources")

    def format_label(self, label: str, name_source: str) -> str:
        return string.Template(label).safe_substitute(name_source=name_source)

    @staticmethod
    def get_integral_label_default() -> str:
        return "src: {name_source} " + SourceConfig.get_integral_label_default()

    def make_model(
        self,
        centroid_fluxes_srcs: Iterable[CentroidFluxes],
        data: g2f.Data,
        psfmodels: list[g2f.PsfModel],
        label_integral: str | None = None,
    ) -> g2f.Model:
        if label_integral is None:
            label_integral = self.get_integral_label_default()
        n_src = len(self.sources)
        if centroid_fluxes_srcs is None or len(centroid_fluxes_srcs) != n_src:
            raise ValueError(f"{len(centroid_fluxes_srcs)=} != {n_src=}")

        sources = []
        priors = []
        for centroid_fluxes, (name_src, config_src) in zip(centroid_fluxes_srcs, self.sources.items()):
            source, priors = config_src.make_source(
                centroid_fluxes=centroid_fluxes,
                label_integral=self.format_label(label=label_integral, name_source=name_src)
            )
            sources.append(source)
            priors.extend(priors)

        model = g2f.Model(data=data, psfmodels=psfmodels, sources=sources, priors=priors)

        return model
