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

from lsst.multiprofit.fit_psf import CatalogPsfFitterConfig, CatalogPsfFitterConfigData
import pytest


@pytest.fixture(scope="module")
def fitter_config() -> CatalogPsfFitterConfig:
    config = CatalogPsfFitterConfig()
    return config

@pytest.fixture(scope="module")
def fitter_config_data(fitter_config) -> CatalogPsfFitterConfigData:
    configdata = CatalogPsfFitterConfigData(config=fitter_config)
    return configdata


def test_fitter_config_data(fitter_config_data):
    parameters = fitter_config_data.parameters
    psfmodel = fitter_config_data.psfmodel
