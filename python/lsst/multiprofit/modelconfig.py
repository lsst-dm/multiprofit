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
import pydantic
from pydantic.dataclasses import dataclass

from .sourceconfig import SourceConfig
from .utils import ArbitraryAllowedConfig


@dataclass(frozen=True, kw_only=True, config=ArbitraryAllowedConfig)
class Limits:
    x = pydantic.Field[g2f.LimitsD](doc="x centroid parameter limits")
    y = pydantic.Field[g2f.LimitsD](doc="y centroid parameter limits")
    rho = pydantic.Field[g2f.LimitsD](doc="rho parameter limits")


@dataclass(frozen=True, kw_only=True, config=ArbitraryAllowedConfig)
class Transforms:
    x = pydantic.Field[g2f.TransformD](doc="x centroid parameter limits")
    y = pydantic.Field[g2f.TransformD](doc="y centroid parameter limits")
    rho = pydantic.Field[g2f.TransformD](doc="rho parameter limits")


class ModelConfig(pexConfig.Config):
    bands = pexConfig.ListField[str](doc="The bands for this observation")
    comps_src = pexConfig.ConfigDictField[str, SourceConfig](doc="The configuration for objects")

    def make_model(self, channels, limits: Limits, transforms: Transforms):
        compconf = self.comps_src
        n_components = len(compconf)
        sources = [None] * n_components

        for idx, (name, config) in compconf:
            components = [None] * n_components
            position_ratio = (1 + idx) / (1 + config.src_n)
            centroid = g2f.CentroidParameters(
                g2f.CentroidXParameterD(config.n_cols * position_ratio, limits=limits.x),
                g2f.CentroidYParameterD(config.n_rows * position_ratio, limits=limits.y),
            )
            sources[i] = g2f.Source(components)
            gaussians = sources[i].gaussians(list(channels.values())[0])

        return sources
