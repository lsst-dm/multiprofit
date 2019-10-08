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

import functools
import math
import numpy as np
from scipy import special


class Transform:
    @classmethod
    def null(cls, value):
        return value

    # TODO: There must be a better way to implement this without multiplication
    @classmethod
    def unit(cls, value):
        return np.ones_like(value)

    @classmethod
    def zero(cls, value):
        return np.zeros_like(value)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __str__(self):
        attrs = ', '.join([
            f'{var}={value}' for var, value in dict(
                transform=self.transform,
                reverse=self.reverse,
                derivative=self.derivative
            ).items()
        ])
        return f'Transform(name={self.name})({attrs})'

    def __init__(self, transform=None, reverse=None, derivative=None, name=None):
        if transform is None or reverse is None:
            if transform is not reverse:
                raise ValueError(
                    "One of transform (type={:s}) and reverse (type={:s}) is {:s} but "
                    "both or neither must be".format(type(transform), type(reverse), type(None))
                )
            else:
                transform = self.null
                reverse = self.null
                derivative = self.unit
        self.transform = transform
        self.reverse = reverse
        self.derivative = derivative
        self.name = name
        # TODO: Verify if forward(reverse(x)) == reverse(forward(x)) +/- error for x in ???


def dlog10dx(x):
    return 0.434294481903251827651128918916605082294397005803666566/x


def dlogitdx(x):
    return 1/x + 1/(1-x)


def negativeinversesquare(x):
    return -1/x**2


def logstretch(x, lower, factor=1.0):
    return np.log10(x-lower)*factor


def powstretch(x, lower, factor=1.0):
    return 10**(x*factor) + lower


def get_stretch_log(lower, factor=1.0):
    return Transform(
        transform=functools.partial(logstretch, lower=lower, factor=factor),
        reverse=functools.partial(powstretch, lower=lower, factor=1./factor),
        name=f"stretch_log_lower={lower}_factor={factor}",
    )


def logitlimited(x, lower, extent, factor=1.0, validate=False):
    y = (x-lower)/extent
    if y > 1:
        if validate:
            raise ValueError(f"logitlimited passed x={x}, lower={lower}, extent={extent}, factor={factor}"
                             f" yielding y={y}>1, which will return nan")
        return np.nan
    elif y == 1:
        return np.Inf
    elif y > 0:
        return math.log(y/(1-y))*factor
    elif y == 0:
        return -np.Inf
    else:
        if validate:
            raise ValueError(f"logitlimited passed x={x}, lower={lower}, extent={extent}, factor={factor}"
                             f" yielding y={y}!>=0, which will return nan")
        return np.nan


def logitlimiteddx(x, lower, extent, factor=1.0):
    y = (x - lower)/extent
    if y == 1 or y == 0:
        return np.Inf
    return (1/y + 1/(1-y))*factor/extent


def expitlimited(x, lower, extent, factor=1.0):
    y = -x*factor
    # math.log(np.finfo(np.float64) = 709.782712893384
    # things will go badly well before then
    if y > 709.7827:
        return lower
    y = 1+math.exp(y)
    if y == 0:
        return np.Inf
    return extent/y + lower


def get_logit_limited(lower, upper, factor=1.0, name=None):
    return Transform(
        transform=functools.partial(logitlimited, lower=lower, extent=upper-lower, factor=factor),
        reverse=functools.partial(expitlimited, lower=lower, extent=upper-lower, factor=1./factor),
        derivative=functools.partial(logitlimiteddx, lower=lower, extent=upper-lower, factor=factor),
        name=f"logit_limited_[{lower},{upper}]_factor={factor}" if name is None else None,
    )


transforms_ref = {
    "none": Transform(name="ref_None"),
    "log": Transform(transform=np.log, reverse=np.exp, derivative=np.reciprocal, name="ref_log"),
    "log10": Transform(transform=np.log10, reverse=functools.partial(np.power, 10.), derivative=dlog10dx,
                       name="ref_log10"),
    "inverse": Transform(transform=np.reciprocal, reverse=np.reciprocal, derivative=negativeinversesquare,
                         name="ref_inverse"),
    "logit": Transform(transform=special.logit, reverse=special.expit, derivative=dlogitdx, name="ref_logit"),
    "logitrho": get_logit_limited(-1, 1, name="ref_logitrho[-1,1]"),
    "logitsigned": get_logit_limited(-1, 1, name="ref_logitsigned[-1,1]"),
    "logitaxrat": get_logit_limited(1e-4, 1, name="ref_logitaxrat[1e-4,1]"),
    "logitsersic": get_logit_limited(0.3, 6.0, name="ref_logitsersic[0.3, 6]"),
    "logitmultigausssersic": get_logit_limited(0.5, 6.0, name="ref_logitmultigausssersic[0.5,6]"),
}
