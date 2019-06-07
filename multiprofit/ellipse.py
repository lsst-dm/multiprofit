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


class Ellipse:
    @staticmethod
    def covar_terms_as(sigma_x_sq, sigma_y_sq, offdiag, matrix=True, params=False):
        if matrix:
            return np.array([[sigma_x_sq, offdiag], [offdiag, sigma_y_sq]])
        else:
            if params:
                sigma_x = np.sqrt(sigma_x_sq)
                sigma_y = np.sqrt(sigma_y_sq)
                return sigma_x, sigma_y, offdiag/(sigma_x*sigma_y)
            else:
                return sigma_x_sq, sigma_y_sq, offdiag

    @staticmethod
    def covar_matrix_as(covar, params=False):
        sigma_x_sq, sigma_y_sq, offdiag = covar[0, 0], covar[1, 1], covar[0, 1]
        return Ellipse.covar_terms_as(sigma_x_sq, sigma_y_sq, offdiag, matrix=False, params=params)

    @staticmethod
    def _check_all(sigma_x, sigma_y, rho):
        if not 0 <= sigma_x < np.Inf:
            raise ValueError("!(0 <= sigma_x,y={},{} < np.Inf) and/or !(-1 < rho ={} < 1)".format(
                sigma_x, sigma_y, rho))

    @staticmethod
    def _check_sigma(x, name=''):
        if not 0 <= x < np.Inf:
            raise ValueError("!(0 <= {}={} < np.Inf)".format(name, x))

    @staticmethod
    def _check_rho(x):
        if not -1 < x < 1:
            raise ValueError("!(-1 < rho ={} < 1)".format(x))

    def _check(self):
        sigma_x, sigma_y, rho = self.get()
        self._check_all(sigma_x, sigma_y, rho)

    def convolve(self, ellipse, new=False):
        if new:
            return Ellipse(*self.get()).convolve(ellipse)
        sigma_x = np.sqrt(self.get_sigma_x()**2 + ellipse.get_sigma_x()**2)
        sigma_y = np.sqrt(self.get_sigma_y()**2 + ellipse.get_sigma_y()**2)
        self.set_rho((self.get_covar_offdiag() + ellipse.get_covar_offdiag())/(sigma_x*sigma_y))
        self.set_sigma_x(sigma_x)
        self.set_sigma_y(sigma_y)
        return self

    def get_covariance(self, matrix=True, params=False):
        sigma_x, sigma_y, rho = self.get_sigma_x(), self.get_sigma_y(), self.get_rho()
        if matrix or not params:
            return self.covar_terms_as(sigma_x * sigma_x, sigma_y * sigma_y, sigma_x * sigma_y * rho,
                                       matrix=matrix, params=params)
        return np.array([sigma_x, sigma_y, rho])

    def get(self):
        return self.sigma_x, self.sigma_y, self.rho

    def get_sigma_x(self):
        return self.sigma_x

    def get_sigma_y(self):
        return self.sigma_y

    def get_rho(self):
        return self.rho

    def get_covar_offdiag(self):
        return self.get_sigma_x()*self.get_sigma_y()*self.get_rho()

    def _set_sigma_x(self, x):
        self.sigma_x = x

    def _set_sigma_y(self, x):
        self.sigma_y = x

    def _set_rho(self, x):
        self.rho = x

    def set_sigma_x(self, x, check=True):
        if check:
            self._check_sigma(x, "x")
        self._set_sigma_x(x)
        return self

    def set_sigma_y(self, x, check=True):
        if check:
            self._check_sigma(x, "y")
        self._set_sigma_y(x)
        return self

    def set_rho(self, x, check=True):
        if check:
            self._check_rho(x)
        self._set_rho(x)
        return self

    def set(self, sigma_x=0, sigma_y=0, rho=0, check=True):
        if check:
            self._check_all(sigma_x, sigma_y, rho)
        self._set_rho(rho)
        self._set_sigma_x(sigma_x)
        self._set_sigma_y(sigma_y)
        return self

    def set_from_covariance(self, covariance):
        if covariance[0, 1] != covariance[1, 0] or not (0 <= covariance[0, 0]) < np.Inf or not (
                0 <= covariance[1, 1] < np.Inf):
            raise ValueError("Covariance {} diagonal elements not both 0 <= x < Inf and/or "
                             "off-diagonal elements not identical".format(covariance))
        try:
            sigma_x = np.sqrt(covariance[0, 0])
            sigma_y = np.sqrt(covariance[1, 1])
            rho = covariance[1, 0]/(sigma_x*sigma_y)
            self._check_all(sigma_x, sigma_y, rho)
        except Exception as e:
            raise ValueError("Failed to set Ellipse from covariance {}".format(covariance))
        return self.set(sigma_x, sigma_y, rho, check=False)

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self):
        return "{} sigma_x, sigma_y, rho = {:6e}, {:6e}, {:6e}".format(self.__class__, *self.get())

    def __init__(self, sigma_x=0, sigma_y=0, rho=0):
        self.set(sigma_x, sigma_y, rho)
