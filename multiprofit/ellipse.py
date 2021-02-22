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


def covar_terms_as(sigma_x_sq, sigma_y_sq, offdiag, matrix=True, params=False):
    if matrix:
        return np.array([[sigma_x_sq, offdiag], [offdiag, sigma_y_sq]])
    else:
        if params:
            sigma_x = np.sqrt(sigma_x_sq)
            sigma_y = np.sqrt(sigma_y_sq)
            denom = (sigma_x*sigma_y)
            return sigma_x, sigma_y, offdiag/denom if denom > 0 else 0
        else:
            return sigma_x_sq, sigma_y_sq, offdiag


def covar_matrix_as(covar, params=False):
    sigma_x_sq, sigma_y_sq, offdiag = covar[0, 0], covar[1, 1], covar[0, 1]
    return covar_terms_as(sigma_x_sq, sigma_y_sq, offdiag, matrix=False, params=params)
