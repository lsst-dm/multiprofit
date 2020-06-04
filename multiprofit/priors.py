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

from abc import ABCMeta, abstractmethod
from multiprofit.ellipse import Ellipse
import multiprofit.gaussutils as mpfgauss
import numpy as np
import scipy.special as spspec
import scipy.stats as spstats


class LsqPrior(metaclass=ABCMeta):
    @abstractmethod
    def calc_residual(self, calc_jacobian=False, delta_jacobian=1e-5):
        ...

    @abstractmethod
    def __len__(self):
        ...


class GaussianLsqPrior(LsqPrior):
    def calc_residual(self, calc_jacobian=False, delta_jacobian=1e-5):
        value = self.param.get_value(transformed=self.transformed)
        residual = (value - self.mean)/self.std
        if not np.isfinite(residual):
            raise RuntimeError(f'Infinite axis ratio prior residual from y={value},'
                               f' mean={self.mean} stddev={self.std}')
        prior = spstats.norm.logpdf(residual)
        jacobians = {}
        # PDF = k exp(-y^2/2) where k = 1/(s*sqrt(2*pi)); y = (x-m)/s [m=mean, s=sigma)
        # logPDF = log(k) - y^2/2
        # dlogPDF/dx = -x/s^2
        if calc_jacobian:
            jacobians[self.param] = value/(self.std*self.std)

        return prior, (residual,), jacobians

    def __len__(self):
        return 1

    def __init__(self, param, mean, std, transformed=False):
        self.param = param
        self.mean = mean
        self.std = std
        self.transformed = transformed


class ShapeLsqPrior(LsqPrior):
    def calc_residual(self, calc_jacobian=False, delta_jacobian=1e-5):
        prior = 0
        residuals = []
        jacobians = {}
        if self.size_mean_std or self.axrat_params:
            size_x = self.size_x.get_value(transformed=False)
            size_y = self.size_y.get_value(transformed=False)
            rho = self.rho.get_value(transformed=False)
            size_maj, axrat, _ = mpfgauss.covar_to_ellipse(Ellipse(size_x, size_y, rho))
            if self.size_mean_std:
                if self.size_log10:
                    size_maj = np.log10(size_maj)
                residual = (size_maj - self.size_mean_std[0])/self.size_mean_std[1]
                residuals.append(residual)
                prior += spstats.norm.logpdf(residual)
            if self.axrat_params:
                residual = ((spspec.logit(axrat/self.axrat_params[2]) - self.axrat_params[0])
                            / self.axrat_params[1])
                if not np.isfinite(residual):
                    raise RuntimeError(f'Infinite axis ratio prior residual from q={axrat} and mean, std, '
                                       f'logit stretch divisor = {self.axrat_params}')
                residuals.append(residual)
                prior += spstats.norm.logpdf(residual)
            if calc_jacobian:
                dsize_x = delta_jacobian*np.max((size_x, 1e-3))
                dsize_y = delta_jacobian*np.max((size_y, 1e-3))
                drho = -delta_jacobian*np.sign(rho)
                values = {x: x.get_value(transformed=x.transformed)
                          for x in (self.size_x, self.size_y, self.rho)}
                for param, delta in ((self.size_x, dsize_x), (self.size_y, dsize_y), (self.rho, drho)):
                    good = False
                    for sign in (1, -1):
                        try:
                            eps = sign*delta
                            param.set_value(param.get_value(transformed=False) + eps, transformed=False)
                            good = True
                            delta = eps
                            break
                        except:
                            pass
                    if not good:
                        raise RuntimeError(f"Couldn't set param {param} with eps=+/-{delta}")
                    _, residuals_new, _ = self.calc_residual(calc_jacobian=False)
                    jacobians[param] = (residuals_new[0] - residuals[0])/eps, (
                            residuals_new[1] - residuals[1])/eps
                    # Reset to original value
                    param.set_value(values[param], transformed=param.transformed)

        return prior, residuals, jacobians

    def __len__(self):
        return bool(self.size_mean_std) + bool(self.axrat_params)

    def __init__(self, size_x, size_y, rho, size_mean_std=None, size_log10=True, axrat_params=None):
        self.size_x = size_x
        self.size_y = size_y
        self.rho = rho
        self.size_log10 = size_log10
        self.size_mean_std = size_mean_std
        self.axrat_params = axrat_params
