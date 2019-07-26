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


# Remove negative elements in an array and preserve the sum by zeroing the smallest positive elements
# This is a crude method of removing noise and returning a strictly positive image with the same total flux
def absconservetotal(ndarray):
    shape = ndarray.shape
    ndarray.shape = np.prod(shape)
    if any(ndarray < 0):
        indices = np.argsort(ndarray)
        # Not sure if this is any faster than cumsum - probably if most pixels are positive
        idx_indices = 0
        sum_neg = 0
        while ndarray[indices[idx_indices]] < 0:
            sum_neg += ndarray[indices[idx_indices]]
            ndarray[indices[idx_indices]] = 0
            idx_indices += 1
        while sum_neg < 0 and idx_indices < ndarray.shape[0]:
            sum_neg += ndarray[indices[idx_indices]]
            ndarray[indices[idx_indices]] = 0
            idx_indices += 1
        ndarray[indices[idx_indices-1]] = sum_neg
        if idx_indices == ndarray.shape[0]:
            raise RuntimeError("absconservetotal failed for array with sum {}".format(np.sum(ndarray)))
    ndarray.shape = shape
    return ndarray


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def get_chisqred(chis):
    chisum = 0
    num_chi = 0
    for chivalues in chis:
        chisum += np.sum(chivalues*chivalues)
        num_chi += chivalues.size
    return chisum/num_chi


def flux_to_mag(ndarray):
    return -2.5*np.log10(ndarray)


def mag_to_flux(ndarray):
    return 10**(-0.4*ndarray)


# Fairly standard moment of inertia estimate of ellipse orientation and size
# TODO: compare with galsim's convenient calculateHLR/FWHM
# TODO: replace with the stack's method (in meas_?)
def estimate_ellipse(
        img, cenx=None, ceny=None, denoise=True, deconvolution_matrix=None, sigma_sq_min=0, do_recenter=True,
        return_cens=False):
    sum_img = np.sum(img)
    if not sum_img > 0:
        raise RuntimeError(f"Tried to estimate ellipse for img={img} with sum={sum_img}")
    imgmeas = absconservetotal(np.copy(img)) if denoise else img
    if cenx is None:
        cenx = imgmeas.shape[1]/2.
    if ceny is None:
        ceny = imgmeas.shape[0]/2.
    y_0, x_0 = np.nonzero(imgmeas)
    flux = imgmeas[y_0, x_0]
    fluxsum = np.sum(flux)
    y = y_0 + 0.5 - ceny
    x = x_0 + 0.5 - cenx
    inertia = np.zeros((2, 2))
    finished = False
    while not finished:
        x_sq = x**2
        y_sq = y**2
        xy = x*y
        inertia[0, 0] = np.sum(flux*x_sq)/fluxsum
        inertia[0, 1] = np.sum(flux*xy)/fluxsum
        inertia[1, 1] = np.sum(flux*y_sq)/fluxsum
        if do_recenter:
            x_shift = np.sum(flux*x)/fluxsum
            y_shift = np.sum(flux*y)/fluxsum
            finished = np.abs(x_shift) < 0.1 and np.abs(y_shift) < 0.1
            if not finished:
                cenx += x_shift
                ceny += y_shift
                y = y_0 + 0.5 - ceny
                x = x_0 + 0.5 - cenx
        else:
            finished = True

    if deconvolution_matrix is not None:
        inertia -= deconvolution_matrix
        inertia[0, 0] = np.clip(inertia[0, 0], sigma_sq_min, np.Inf)
        inertia[1, 1] = np.clip(inertia[1, 1], sigma_sq_min, np.Inf)
    inertia[1, 0] = inertia[0, 1]
    if return_cens:
        return inertia, cenx, ceny
    return inertia


def normalize(ndarray, return_sum=False):
    total = np.sum(ndarray)
    ndarray /= total
    if return_sum:
        return ndarray, total
    return ndarray


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot parse {} as boolean.'.format(v))
