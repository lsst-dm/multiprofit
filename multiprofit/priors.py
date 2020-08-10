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
        """Compute the prior log likehood, residuals and optional jacobians.

        :param calc_jacobian: `bool`; whether to compute the jacobian of the prior probability or not.
        :param delta_jacobian: `float`; fractional change in paramater value for finite difference
            approximation of the jacobian. Should be ignored if analytic jacobian calculation is implemented.
        :return: result: `tuple` consisting of:
            prior: `float`; the prior log-likelihood.
            residual: `tuple` [`float`]; A tuple of len equal to __len__(), containing values of chi
                chi=(value_prior_i - mean_prior_i)/std_dev_i for each of the distinct prior functions.
            jacobians: `dict` [`multiprofit.objects.Parameter`]; a dict with an entry for each parameter that
                the prior log likelihood depends one. Each entry has the same format as `residual`.

        Notes
        -----
        This method is intended to provide return values required for least squares fitting using
        `scipy.optimize.least_squares`.
        """
        ...

    @abstractmethod
    def __len__(self):
        """ Get the number of distinct prior functions.

        :return: `int`; the number of distinct prior functions.
        """
        ...


class GaussianLsqPrior(LsqPrior):
    def calc_residual(self, calc_jacobian=False, delta_jacobian=None):
        value = self.param.get_value(transformed=self.transformed)
        residual = (value - self.mean)/self.stddev
        if not np.isfinite(residual):
            raise RuntimeError(f'Infinite axis ratio prior residual from y={value},'
                               f' mean={self.mean} stddev={self.stddev}')
        prior = spstats.norm.logpdf(residual)
        jacobians = {}
        # PDF = k exp(-y^2/2) where k = 1/(s*sqrt(2*pi)); y = (x-m)/s [m=mean, s=sigma); dy/dx = 1/s
        # logPDF = log(k) - y^2/2
        # dlogPDF dy = -y
        # dlogPDF dx = -y * dy/dx = - y/s
        # d -(x-m)^2/(2s^2) dx = -(x-m)/s^2
        # dlogPDF/dx = -y/s
        if calc_jacobian:
            jacobians[self.param] = -residual/self.stddev

        return prior, (residual,), jacobians

    def __len__(self):
        return 1

    def __init__(self, param, mean, stddev, transformed=False):
        """

        :param param: `multiprofit.objects.Parameter`; param to apply the prior to.
        :param mean: float; prior mean.
        :param stddev: float; prior standard deviation.
        :param transformed: bool; whether the prior applies to the transformed parameter value.
        """
        self.param = param
        self.mean = np.copy(mean)
        self.stddev = np.copy(stddev)
        self.transformed = transformed is True


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
            if not axrat > 0:
                raise RuntimeError(f'r_eff={size_maj}, axrat={axrat} from x={size_x}, y={size_y}, rho={rho}')
            if self.size_mean_std:
                if self.size_log10:
                    size_maj = np.log10(size_maj)
                residual = (size_maj - self.size_mean_std[0])/self.size_mean_std[1]
                if not np.isfinite(residual):
                    raise RuntimeError(f'Infinite size ratio prior residual from r={size_maj} and mean, std'
                                       f' = {self.size_mean_std}')
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
            drho = -delta_jacobian*(np.sign(rho) + (rho == 0))
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
                    except RuntimeError:
                        pass
                if not good or not np.abs(eps) > 0:
                    raise RuntimeError(f"Couldn't set param {param} with eps=+/-{delta}")
                _, residuals_new, _ = self.calc_residual(calc_jacobian=False)
                jacobian = (residuals_new[0] - residuals[0])/eps, (residuals_new[1] - residuals[1])/eps
                if not all(np.isfinite(jacobian)):
                    raise RuntimeError(f'Non-finite ShapeLsqPrior jacobian={jacobian} for param {param}')
                jacobians[param] = jacobian
                # Reset to original value
                value_reset = values[param]
                param.set_value(value_reset, transformed=param.transformed)
                if param.get_value(transformed=param.transformed) != value_reset:
                    raise RuntimeError(f'Failed to reset param {param} to value={value_reset}; check limits')

        return prior, residuals, jacobians

    def __len__(self):
        return bool(self.size_mean_std) + bool(self.axrat_params)

    def __init__(self, size_x, size_y, rho, size_mean_std=None, size_log10=True, axrat_params=None):
        """Initialize a size/shape prior.

        :param size_x: `multiprofit.objects.Parameter`; x size param to apply the prior to.
        :param size_y: `multiprofit.objects.Parameter`; y size param to apply the prior to.
        :param rho:  `multiprofit.objects.Parameter`; correlation param to apply the prior to.
        :param size_mean_std: `float`, `float`; mean and std.dev. of the size prior. Ignored if None.
        :param size_log10: `bool`; whether the prior is on the logarithm of the size rather than linear size.
        :param axrat_params:
        """
        if (size_mean_std is not None and (not all(np.isfinite(size_mean_std)))) or (
                axrat_params is not None and (not all(np.isfinite(axrat_params)))):
            raise ValueError(f'Non-finite {self.__class__} init '
                             f'size_mean_std={size_mean_std} and/or axrat_params={axrat_params}')
        self.size_x = size_x
        self.size_y = size_y
        self.rho = rho
        self.size_log10 = size_log10
        self.size_mean_std = size_mean_std
        self.axrat_params = axrat_params


def get_hst_size_prior(mag_psf_i):
    """ Return the mean and stddev for a reasonable HST-based size (major axis half-light radius) prior.

    Notes
    -----
    Return values are log10 scaled in units of arcseconds. The input should be a PSF mag because other
    magnitudes - even Gaussian - can be unreliable for low S/N (non-)detections.
    """
    return 0.75*(19 - np.clip(mag_psf_i, 10, 30))/6.5, 0.2
