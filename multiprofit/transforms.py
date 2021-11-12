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


def get_logit_limited(lower, upper, factor=1.0, name=None):
    return g2f.LogitLimitedTransformD(
        limits=g2f.LimitsD(
            min=lower, max=upper, name=name if name is not None else
            f"LogitLimitedTransformD(min={lower}, max={upper}, factor={factor})",
        ),
        factor=factor,
    )


transforms_ref = {
    "none": g2f.UnitTransformD(),  # Transform(name="ref_None"),
    "log": g2f.LogTransformD(),
    "log10": g2f.Log10TransformD(),
    "inverse": g2f.InverseTransformD(),
    "logit": g2f.LogitTransformD(),
    "logitrho": get_logit_limited(-1, 1, name="ref_logit_rho[-1, 1]"),
    "logitsigned": get_logit_limited(-1, 1, name="ref_logit_signed[-1, 1]"),
    "logitaxrat": get_logit_limited(1e-4, 1, name="ref_logit_axrat[1e-4, 1]"),
    "logitsersic": get_logit_limited(0.3, 6.0, name="ref_logit_sersic[0.3, 6]"),
    "logitmultigausssersic": get_logit_limited(0.5, 6.0, name="ref_logit_multigausssersic[0.5, 6]"),
}
