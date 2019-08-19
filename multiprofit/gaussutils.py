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
import scipy.optimize as spopt
from multiprofit.ellipse import Ellipse
from _multiprofit import loglike_gaussians_pixel as loglike_gaussians_pixel_pb


def sigma_to_reff(sigma):
    """
    Convert a Gaussian dispersion sigma to a Sersic profile effective (half-light) radius.

    The constant is np.sqrt(2*np.log(2)) - the conversion from sigma to HWHM - since Gaussian r_50 = HWHM

    :param sigma: Float; Gaussian dispersion sigma.
    :return: r_eff: Float; the Sersic effective (half-light) radius.
    """
    return sigma*1.1774100225154746635070068805362097918987


def reff_to_sigma(r_eff):
    """
    Convert a Gaussian dispersion sigma to a Sersic profile effective (half-light) radius.

    :param r_eff: Float; the Sersic effective (half-light) radius.
    :return: sigma: Float; Gaussian dispersion sigma.
    """
    return r_eff/1.1774100225154746635070068805362097918987


def covar_to_ellipse(x, use_method_eigen=True):
    """
    Convert a covariance matrix or parameter array to ellipse parameters sigma_maj, axrat, ang.

    :param x: A 2x2 covariance matrix or a length[3] array of sigma_x_sq, sigma_y_sq, covariance.
    :param use_method_eigen: Use an eigenvalue method to get the principal axes.
        This is slower but usually more accurate.
    :return: sigma_maj, axrat, ang: Floats of the major axis sigma, axis ratio, and position angle in degrees
        following the usual convention of countner-clockwise from the +x axis.
    """
    if isinstance(x, Ellipse):
        is_matrix = True
        covar = x.get_covariance(matrix=True)
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be ndarray or Ellipse, not {}".format(type(x)))
        is_matrix = x.shape != (2, 2)
        if is_matrix:
            covar = x
        else:
            if not x.ndim == 1 and x.size == 3:
                raise ValueError("x.shape={} must be (2, 2) or (3,)".format(x.shape))
            if use_method_eigen:
                covar = Ellipse.covar_terms_as(*x)
    sigma_x_sq, sigma_y_sq, offdiag = (covar[0, 0], covar[1, 1], covar[0, 1]) if is_matrix else x
    # TODO: Determine if this preserves the right quadrant in all cases
    ang = np.arctan2(2*offdiag, sigma_x_sq - sigma_y_sq)/2
    use_method_eigen = use_method_eigen and np.abs(np.linalg.cond(covar)) < 1e8
    if not use_method_eigen:
        if np.pi/4 < (np.abs(ang) % np.pi) < 3*np.pi/4:
            sin_ang_sq = np.sin(ang)**2
            cos_ang_sq = 1-sin_ang_sq
        else:
            cos_ang_sq = np.cos(ang)**2
            sin_ang_sq = 1-cos_ang_sq
        #  == cos^2 - sin^2 == cos(2*theta)
        denom = 2.*cos_ang_sq - 1.
        if np.abs(denom) < 1e-4 or (1 - np.abs(denom)) < 1e-4:
            use_method_eigen = True
            if not is_matrix:
                covar = Ellipse.covar_terms_as(*x)
        else:
            sigma_u = np.sqrt((cos_ang_sq*sigma_x_sq - sin_ang_sq*sigma_y_sq)/denom)
            sigma_v = np.sqrt((cos_ang_sq*sigma_y_sq - sin_ang_sq*sigma_x_sq)/denom)
            sigma_maj = np.max([sigma_u, sigma_v])
            axrat = sigma_u/sigma_v if sigma_u < sigma_v else sigma_v/sigma_u
            if not 0 <= axrat <= 1:
                raise RuntimeError("Got unreasonable axis ratio {} from input={} and "
                                   "sigma_u={} sigma_v={}".format(axrat, x, sigma_u, sigma_v))
    if use_method_eigen:
        eigenvalues, eigenvecs = np.linalg.eigh(covar)
        index_maj = np.argmax(eigenvalues)
        sigma_maj = np.sqrt(eigenvalues[index_maj])
        axrat = 1 if (sigma_maj == 0 and eigenvalues[1-index_maj] == 0) else \
            np.sqrt(eigenvalues[1-index_maj])/sigma_maj
        if not 0 <= axrat <= 1:
            raise RuntimeError("Got unreasonable axis ratio {} from input={} and "
                               "eigenvalues={} eigenvecs={}".format(axrat, x, eigenvalues, eigenvecs))
    return sigma_maj, axrat, np.degrees(ang)


def ellipse_to_covar(sigma, axrat, ang, return_as_matrix=True, return_as_params=False):
    """
    Convert ellipse parameters to a covariance matrix/array or correlation array.
    :param sigma: Float; major-axis sigma of the normal distribution.
    :param axrat: Float; minor-to-major axis ratio.
    :param ang: Float; position angle in degrees, measured as standard (counter-clockwise from +x).
    :param return_as_matrix: Bool; return a covariance matrix?
    :param return_as_params: Bool; return correlation parameter array instead of covariance?
    :return: Covariance matrix/array or correlation array, as requested.
    """
    ang = np.radians(ang)
    sin_ang = np.sin(ang)
    cos_ang = np.cos(ang)
    maj_sq = sigma**2
    min_sq = (sigma*axrat)**2
    sin_ang_sq = sin_ang**2
    cos_ang_sq = cos_ang**2
    sigma_x_sq = maj_sq*cos_ang_sq + min_sq*sin_ang_sq
    sigma_y_sq = maj_sq*sin_ang_sq + min_sq*cos_ang_sq
    covar = (maj_sq-min_sq)*cos_ang*sin_ang
    return Ellipse.covar_terms_as(sigma_x_sq, sigma_y_sq, covar, matrix=return_as_matrix, params=return_as_params)


# https://www.wolframalpha.com/input/?i=Integrate+2*pi*x*exp(-x%5E2%2F(2*s%5E2))%2F(s*sqrt(2*pi))+dx+from+0+to+r
# The fraction of the total flux of a 2D Sersic profile contained within r
# For efficiency, we can replace r with r/sigma.
# Note the trivial renormalization allows dropping annoying sigma and pi constants
# It returns (0, 1) for inputs of (0, inf)
def gauss2dint(xdivsigma):
    return 1 - np.exp(-xdivsigma**2/2.)


# Compute the fraction of the integrated flux within x for a sum of Gaussians
# x is a length in arbitrary units
# Weightsizes is a list of tuples of the weight (flux) and size (r_eff in the units of x) of each gaussian
# Note that gauss2dint expects x/sigma, but size is re, so we pass x/re*re/sigma = x/sigma
# 0 > quant > 1 turns it into a function that returns zero
#   at the value of x containing a fraction quant of the total flux
# This is so you can use root finding algorithms to find x for a given quant (see below)
def multigauss2dint(x, weightsizes, quant=0):
    re_to_sigma = np.sqrt(2.*np.log(2.))
    weight_sum_to_x = 0
    weight_sum = 0
    for weight, size in weightsizes:
        weight_sum_to_x += weight*(gauss2dint(x/size*re_to_sigma) if size > 0 else 1)
        weight_sum += weight
    return weight_sum_to_x/weight_sum - quant


# Compute x_quant for a sum of Gaussians, where 0<quant<1
# There's probably an analytic solution to this if you care to work it out
# Weightsizes and quant are as above
# Choose xmin, xmax so that xmin < x_quant < xmax.
# Ideally we'd just set xmax=np.inf but brentq doesn't work then; a very large number like 1e5 suffices.
def multigauss2drquant(weightsizes, quant=0.5, xmin=0, xmax=1e5):
    if not 0 <= quant <= 1:
        raise ValueError('Quant {} not >=0 & <=1'.format(quant, quant))
    weight_sum_zero_size = 0
    weight_sum = 0
    for weight, size in weightsizes:
        if not (size > 0):
            weight_sum_zero_size += weight
        weight_sum += weight
    if weight_sum_zero_size/weight_sum >= quant:
        return 0
    return spopt.brentq(multigauss2dint, a=xmin, b=xmax, args=(weightsizes, quant))


zeros_double = np.zeros((0, 0))
zeros_uint64 = np.zeros((0, 0), dtype=np.uint64)


def loglike_gaussians_pixel(
        data, variance_inv, gaussians, x_min=None, x_max=None, y_min=None, y_max=None,
        to_add=False, output=None, grad=None, grad_param_map=None, grad_param_factor=None,
        sersic_param_map=None, sersic_param_factor=None,
):
    """
    A wrapper for the pybind11 loglike_gaussians_pixel function with reasonable default values for all of the
    arguments, since most of them don't accept None (nullptr)/
    :param data: np.array, ndim=2; image data if computing likelihood.
    :param variance_inv: np.array, shape==data.shape; inverse variance image if computing likelihood.
    :param gaussians: np.array, shape=(6|9, N): Parameters for N Gaussians:
        cenx, ceny, flux, sigma_x, sigma_y, rho.
        Optionally include PSF sigma_x, sigma_y, rho.
    :param x_min: Float; the x-coordinate of the left edge of the image.
    :param x_max: Float; the x-coordinate of the right edge of the image.
    :param y_min: Float; the y-coordinate of the bottom edge of the image.
    :param y_max: Float; the y-coordinate of the top edge of the image.
    :param to_add: Bool; if outputting the model, should it add to the output image or overwrite it?
    :param output: np.array, shape==data.shape; output image for model or residual (if computing Jacobian).
    :param grad: np.array, shape~gaussians.shape to output likelihood gradient;
        shape==(data.shape[0],data.shape[1], N~gaussians.size) to output Jacobian
    :param grad_param_map: np.array[int], shape==gaussian.shape; optional indices of where each Gaussian
        parameter gradient/Jacobian should enter into grad.
    :param grad_param_factor: np.array[float], shape==gaussian.shape; optional multiplicative factor
        as grad_param_map.
    :param sersic_param_map: np.array[int], shape==gaussian.shape[1]; indices as grad_param_map but for the
        Sersic index of multi-Gaussian profiles.
    :param sersic_param_factor: np.array[float], shape==gaussian.shape[1]; as grad_param_factor but for the
        Sersic index of multi-Gaussian profiles.
    :return: the log-likelihood of the model, or zero for Jacobian (the residuals are output instead).
    """
    if output is None:
        output = zeros_double
    if grad is None:
        grad = zeros_double
    if grad_param_map is None:
        grad_param_map = zeros_uint64
    if sersic_param_map is None:
        sersic_param_map = zeros_uint64
    if grad_param_factor is None:
        grad_param_factor = zeros_double
    if sersic_param_factor is None:
        sersic_param_factor = zeros_double
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = data.shape[1]
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = data.shape[0]
    return loglike_gaussians_pixel_pb(
        data=data, variance_inv=variance_inv, gaussians=gaussians, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, to_add=to_add, output=output, grad=grad,
        grad_param_map=grad_param_map, grad_param_factor=grad_param_factor,
        sersic_param_map=sersic_param_map, sersic_param_factor=sersic_param_factor)
