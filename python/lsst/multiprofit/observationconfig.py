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

import gauss2d as g2
import gauss2d.fit as g2f
import lsst.pex.config as pexConfig


class CoordinateSystemConfig(pexConfig.Config):
    """Configuration for a gauss2d CoordinateSystem."""

    dx1 = pexConfig.Field[float](doc="The x-axis pixel scale", optional=False, default=1.0)
    dy2 = pexConfig.Field[float](doc="The y-axis pixel scale", optional=False, default=1.0)
    x_min = pexConfig.Field[float](
        doc="The x-axis coordinate of the bottom left corner",
        optional=False,
        default=0.0,
    )
    y_min = pexConfig.Field[float](
        doc="The y-axis coordinate of the bottom left corner",
        optional=False,
        default=0.0,
    )

    def make_coordinate_system(self) -> g2.CoordinateSystem:
        return g2.CoordinateSystem(dx1=self.dx1, dy2=self.dy2, x_min=self.x_min, y_min=self.y_min)


class ObservationConfig(pexConfig.Config):
    """Configuration for a gauss2d.fit Observation."""

    band = pexConfig.Field[str](doc="The name of the band", optional=False, default="None")
    coordsys = pexConfig.ConfigField[CoordinateSystemConfig](doc="The coordinate system config")
    n_rows = pexConfig.Field[int](doc="The number of rows in the image")
    n_cols = pexConfig.Field[int](doc="The number of columns in the image")

    def make_observation(self) -> g2f.Observation:
        coordsys = self.coordsys.make_coordinate_system() if self.coordsys else None
        image = g2.ImageD(n_rows=self.n_rows, n_cols=self.n_cols, coordsys=coordsys)
        sigma_inv = g2.ImageD(n_rows=self.n_rows, n_cols=self.n_cols, coordsys=coordsys)
        mask = g2.ImageB(n_rows=self.n_rows, n_cols=self.n_cols, coordsys=coordsys)
        observation = g2f.Observation(
            image=image,
            sigma_inv=sigma_inv,
            mask_inv=mask,
            channel=g2f.Channel.get(self.band),
        )
        return observation
