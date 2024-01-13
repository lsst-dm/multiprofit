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

from .componentconfig import EllipticalComponentConfig


class SourceConfig(pexConfig.Config):
    components = pexConfig.ConfigDictField[str, EllipticalComponentConfig](
        doc="Components in the source",
        optional=False,
    )

    def make_source(
        self,
        centroid: g2f.CentroidParameters,
        fluxes: list[dict[g2f.Channel, float]],
        label_integral: str | None = None,
    ) -> g2f.Source:
        if len(fluxes) != len(self.components):
            raise ValueError(f"{len(fluxes)=} != {len(self.components)=}")
        components = []
        priors = []
        for fluxes_component, (name_component, config) in zip(fluxes, self.components):
            component, priors_comp = config.make_component(
                centroid=centroid, fluxes=fluxes_component,
                label_integral=f"Comp {name_component} {label_integral}"
            )
            components.append(component)
            priors.extend(priors)
        source = g2f.Source(components)
        return source
