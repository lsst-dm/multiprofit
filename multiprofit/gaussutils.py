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


def sigma2reff(sigma):
    return sigma*1.1774100225154746635070068805362097918987


def reff2sigma(reff):
    return reff/1.1774100225154746635070068805362097918987


def covartoellipseeig(covar):
    eigenvals, eigenvecs = np.linalg.eigh(covar)
    indexmaj = np.argmax(eigenvals)
    sigmamaj = np.sqrt(eigenvals[indexmaj])
    axrat = np.sqrt(eigenvals[1-indexmaj])/sigmamaj
    ang = np.degrees(np.arctan2(2*covar[0, 1], covar[0, 0] - covar[1, 1])/2)
    return sigmamaj, axrat, ang


def covartoellipse(sigxsq, sigysq, covdiag):
    theta = np.arctan2(2*covdiag, sigxsq - sigysq)/2
    cossqth = np.cos(theta)**2
    sinsqth = 1-cossqth
    #  == cos^2 - sin^2 == cos(2*theta)
    denom = 2.*cossqth - 1.
    if denom < 1e-4:
        return covartoellipseeig(np.array([[sigxsq, covdiag], [covdiag, sigysq]]))
    else:
        sigu = np.sqrt((cossqth*sigxsq - sinsqth*sigysq)/denom)
        sigv = np.sqrt((cossqth*sigysq - sinsqth*sigxsq)/denom)
        sigmamaj = np.max([sigu, sigv])
        axrat = sigu/sigv if sigu < sigv else sigv/sigu
    return sigmamaj, axrat, np.degrees(theta)


def ellipsetocovar(sigma, axrat, ang):
    ang = np.radians(ang)
    sinang = np.sin(ang)
    cosang = np.cos(ang)
    majsq = sigma**2
    minsq = (sigma*axrat)**2
    sinangsq = sinang**2
    cosangsq = cosang**2
    sigxsq = majsq*cosangsq + minsq*sinangsq
    sigysq = majsq*sinangsq + minsq*cosangsq
    covxy = (majsq-minsq)*cosang*sinang
    covar = np.array([[sigxsq, covxy], [covxy, sigysq]])
    return covar


def get_covar_elements(covar):
    sigma_x = np.sqrt(covar[0, 0])
    sigma_y = np.sqrt(covar[1, 1])
    rho = covar[0, 1]/sigma_x/sigma_y
    return sigma_x, sigma_y, rho


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
    retosigma = np.sqrt(2.*np.log(2.))
    weightsumtox = 0
    weightsum = 0
    for weight, size in weightsizes:
        weightsumtox += weight*(gauss2dint(x/size*retosigma) if size > 0 else 1)
        weightsum += weight
    return weightsumtox/weightsum - quant


# Compute x_quant for a sum of Gaussians, where 0<quant<1
# There's probably an analytic solution to this if you care to work it out
# Weightsizes and quant are as above
# Choose xmin, xmax so that xmin < x_quant < xmax.
# Ideally we'd just set xmax=np.inf but brentq doesn't work then; a very large number like 1e5 suffices.
def multigauss2drquant(weightsizes, quant=0.5, xmin=0, xmax=1e5):
    if not 0 <= quant <= 1:
        raise ValueError('Quant {} not >=0 & <=1'.format(quant, quant))
    weightsumzerosize = 0
    weightsum = 0
    for weight, size in weightsizes:
        if not (size > 0):
            weightsumzerosize += weight
        weightsum += weight
    if weightsumzerosize/weightsum >= quant:
        return 0
    return spopt.brentq(multigauss2dint, a=xmin, b=xmax, args=(weightsizes, quant))
