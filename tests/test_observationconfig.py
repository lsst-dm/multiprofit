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
from lsst.multiprofit.observationconfig import CoordinateSystemConfig, ObservationConfig
import pytest


@pytest.fixture(scope="module")
def kwargs_coordsys():
    return {
        'dx1': 0.4,
        'dy2': 1.6,
        'x_min': -51.3,
        'y_min': 1684.5,
    }


@pytest.fixture(scope="module")
def config_coordsys(kwargs_coordsys) -> CoordinateSystemConfig:
    return CoordinateSystemConfig(**kwargs_coordsys)


@pytest.fixture(scope="module")
def coordsys(config_coordsys) -> g2.CoordinateSystem:
    return config_coordsys.make_coordinate_system()


def test_CoordinateSystemConfig(kwargs_coordsys, coordsys):
    for kwarg, value in kwargs_coordsys.items():
        assert getattr(coordsys, kwarg) == value


def test_ObservationConfig():
    n_cols, n_rows = 15, 17
    shape = [n_rows, n_cols]
    config = ObservationConfig(n_cols=n_cols, n_rows=n_rows)
    observation = config.make_observation()
    assert observation.channel == g2f.Channel.NONE
    planes = ("image", "mask_inv", "sigma_inv")
    for plane in planes:
        attr = getattr(observation, plane)
        assert attr.shape == shape
    config.band = "red"
    observation2 = config.make_observation()
    assert observation2.channel == g2f.Channel.get("red")
    for plane in planes:
        attr1, attr2 = (getattr(obs, plane) for obs in (observation, observation2))
        assert attr1 is not attr2
        # Initialize both images; comparison checks equality
        attr1.fill(0)
        attr2.fill(0)
        assert attr1 == attr2
