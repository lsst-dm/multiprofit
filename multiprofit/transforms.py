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

    def __init__(self, transform=None, reverse=None, derivative=None):
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
        # TODO: Verify if forward(reverse(x)) == reverse(forward(x)) +/- error for x in ???


def dlog10dx(x):
    return 0.434294481903251827651128918916605082294397005803666566*np.reciprocal(x)


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
        reverse=functools.partial(powstretch, lower=lower, factor=1./factor))


def logitlimited(x, lower, extent, factor=1.0):
    return special.logit((x-lower)/extent)*factor


def logitlimiteddx(x, lower, extent, factor=1.0):
    y = (x - lower)/extent
    return (1/y + 1/(1-y))*factor/extent


def expitlimited(x, lower, extent, factor=1.0):
    return special.expit(x*factor)*extent + lower


def get_logit_limited(lower, upper, factor=1.0):
    return Transform(
        transform=functools.partial(logitlimited, lower=lower, extent=upper-lower, factor=factor),
        reverse=functools.partial(expitlimited, lower=lower, extent=upper-lower, factor=1./factor),
        derivative=functools.partial(logitlimiteddx, lower=lower, extent=upper-lower, factor=factor)
    )


transforms_ref = {
    "none": Transform(),
    "log": Transform(transform=np.log, reverse=np.exp, derivative=np.reciprocal),
    "log10": Transform(transform=np.log10, reverse=functools.partial(np.power, 10.), derivative=dlog10dx),
    "inverse": Transform(transform=np.reciprocal, reverse=np.reciprocal, derivative=negativeinversesquare),
    "logit": Transform(transform=special.logit, reverse=special.expit, derivative=dlogitdx),
    "logitrho": get_logit_limited(-1, 1),
    "logitsigned": get_logit_limited(-1, 1),
    "logitaxrat": get_logit_limited(1e-4, 1),
    "logitsersic": get_logit_limited(np.nextafter(0.3, 0), np.nextafter(6.0, np.Inf)),
    "logitmultigausssersic": get_logit_limited(np.nextafter(0.5, 0), np.nextafter(6.0, np.Inf)),
}
