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

from typing import Any

import gauss2d.fit as g2f
import numpy
import numpy as np

__all__ = ["ArbitraryAllowedConfig", "get_params_uniq", "normalize"]


class ArbitraryAllowedConfig:
    """Pydantic config to allow arbitrary typed Fields."""

    arbitrary_types_allowed = True
    # Disallow any extra kwargs
    extra = "forbid"


class FrozenArbitraryAllowedConfig(ArbitraryAllowedConfig):
    """Pydantic config to allow arbitrary typed Fields for frozen classes."""


def get_params_uniq(parametric: g2f.Parametric, **kwargs: Any):
    """Get a sorted set of parameters matching a filter.

    Parameters
    ----------
    parametric
        The parametric object to get parameters from.
    **kwargs
        Keyword arguments to pass to g2f.ParamFilter.

    Returns
    -------
    params
        The unique parameters from the parametric object matching the filter.
    """
    params = parametric.parameters(paramfilter=g2f.ParamFilter(**kwargs))
    # This should always return the same list as:
    # list({p: None for p in }.keys())
    return g2f.params_unique(params)


def normalize(ndarray: numpy.ndarray, return_sum: bool = False):
    """Normalize a numpy array.

    Parameters
    ----------
    ndarray
        The array to normalize.
    return_sum
        Whether to return the sum.

    Returns
    -------
    ndarray
        The input ndarray.
    total
        The sum of the array.
    """
    total = np.sum(ndarray)
    ndarray /= total
    if return_sum:
        return ndarray, total
    return ndarray
