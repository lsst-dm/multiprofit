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


def Limits(*args, **kwargs):
    return g2f.LimitsD(*args, **kwargs)


# TODO: Replace with a parameter factory and/or profile factory
limits_ref = {
    "none": Limits(),
    "fraction": Limits(min=0., max=1.,),
    "axrat": Limits(min=1e-2, max=1),
    "con": Limits(min=1, max=10),
    "nser": Limits(min=0.3, max=6.0),
    "nsermultigauss": Limits(min=0.5, max=6.0),
    "rho": Limits(min=-1+1e-8, max=1-1e-8),
}
