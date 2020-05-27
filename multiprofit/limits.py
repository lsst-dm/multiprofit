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

import numpy as np
from multiprofit.transforms import transforms_ref


class Limits:
    """
        Limits for a Parameter.
    """
    def clip(self, value):
        if self.within(value):
            return value
        else:
            return np.clip(value, self.lower, self.upper)

    def within(self, value):
        return self.lower <= value <= self.upper

    def __str__(self):
        attrs = ', '.join([
            f'{var}={value}' for var, value in dict(
                lower=self.lower,
                upper=self.upper,
                transformed=self.transformed,
            ).items()
        ])
        return f'Limits({attrs})'

    def __init__(self, lower=-np.inf, upper=np.inf, is_lower_inclusive=True, is_upper_inclusive=True,
                 transformed=True):
        is_nan_lower = np.isnan(lower)
        is_nan_upper = np.isnan(upper)
        if is_nan_lower or is_nan_upper:
            raise ValueError("Limits lower,upper={},{} finite check={},{}".format(
                lower, upper, is_nan_lower, is_nan_upper))
        if not upper >= lower:
            raise ValueError("Limits upper={} !>= lower{}".format(lower, upper))
        # TODO: Should pass in the transform and check if lower
        if not is_lower_inclusive:
            lower = np.nextafter(lower, lower+1.)
        if not is_upper_inclusive:
            upper = np.nextafter(upper, upper-1.)
        self.lower = lower
        self.upper = upper
        self.transformed = transformed


# TODO: Replace with a parameter factory and/or profile factory
limits_ref = {
    "none": Limits(),
    "none_untransformed": Limits(transformed=False),
    "fraction": Limits(lower=0., upper=1., transformed=True),
    "fractionlog10": Limits(upper=0., transformed=True),
    "axratlog10": Limits(lower=-2., upper=0., transformed=True),
    "coninverse": Limits(lower=0.1, upper=0.9090909, transformed=True),
    "nser": Limits(lower=0.3, upper=6.0),
    "nsermultigauss": Limits(lower=transforms_ref["logitmultigausssersic"](0.5),
                             upper=transforms_ref["logitmultigausssersic"](6.0),
                             transformed=True),
    "nserlog10": Limits(lower=np.log10(0.3), upper=np.log10(6.0), is_lower_inclusive=False,
                        is_upper_inclusive=False, transformed=True),
    "rho": Limits(lower=-1+1e-8, upper=1-1e-8),
    "logitrho": Limits(lower=transforms_ref["logitrho"](-1+1e-8),
                       upper=transforms_ref["logitrho"](1-1e-8),
                       transformed=True),
}
