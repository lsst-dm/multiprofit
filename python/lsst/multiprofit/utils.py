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
import numpy
import numpy as np

__all__ = ["ArbitraryAllowedConfig", "get_params_uniq", "normalize"]


class ArbitraryAllowedConfig:
    """Pydantic config to allow arbitrary typed Fields.

    Also disallows unused init kwargs.
    """

    arbitrary_types_allowed = True
    extra = "forbid"


def get_params_uniq(parametric: g2f.Parametric, **kwargs):
    """Get a sorted set of parameters matching a filter"""
    return {p: None for p in parametric.parameters(paramfilter=g2f.ParamFilter(**kwargs))}.keys()


def normalize(ndarray: numpy.ndarray, return_sum: bool = False):
    """Normalize a numpy array."""
    total = np.sum(ndarray)
    ndarray /= total
    if return_sum:
        return ndarray, total
    return ndarray
