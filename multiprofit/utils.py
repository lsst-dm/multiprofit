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
from scipy import ndimage


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
        if idx_indices == ndarray.shape[0] and sum_neg < 0:
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
        return_cens=False, validate=True, contiguous=False, pixel_min=None, num_pix_min=2,
        raise_on_fail=False):
    """

    :param img: ndarray; an image of a source to estimate the moments of.
    :param cenx, ceny: float; initial estimate of the centroid of the source.
    :param denoise: bool; whether to attempt to naively de-noise the image by zeroing all negative pixels and
        the faintest positive pixels while conserving the total flux.
    :param deconvolution_matrix: ndarray; a covariance matrix to subtract from the moments to deconvolve them,
        e.g. for PSF estimation.
    :param sigma_sq_min: float; the minimum variance to return, especially if deconvolving.
    :param do_recenter: bool; whether to iteratively re-estimate the centroid.
    :param return_cens: bool; whether to return the centroid.
    :param validate: bool; whether to check if the image has positive net flux before processing.
    :param contiguous: bool; whether to keep only the contiguous above-threshold pixels around the centroid.
    :param pixel_min: float; the minimum threshold pixel value. Default 0.
    :param num_pix_min: int; the minimum number of positive pixels required for processing.
    :param raise_on_fail: bool; whether to raise an Exception on any failure instead of attempting to
        continue by not strictly obeying the inputs.
    :return: inertia: ndarray; the moment of inertia/covariance matrix.
        cenx, ceny: The centroids, if return_cens is True.
    """
    if validate or denoise:
        sum_img = np.sum(img)
    if validate and raise_on_fail:
        if not sum_img > 0:
            raise RuntimeError(f"Tried to estimate ellipse for img(shape={img.shape}, sum={sum_img})")
    if cenx is None:
        cenx = img.shape[1]/2.
    if ceny is None:
        ceny = img.shape[0]/2.
    pixel_min = 0 if pixel_min is None else np.max([0, pixel_min])
    if denoise and sum_img > 0:
        img_meas = absconservetotal(np.copy(img))
        mask = img_meas
    else:
        img_meas = img
        mask = img > pixel_min
    if contiguous:
        pix_cenx, pix_ceny = round(cenx), round(ceny)
        if img_meas[pix_ceny, pix_cenx] > pixel_min:
            labels, _ = ndimage.label(mask)
            mask = labels == labels[pix_ceny, pix_cenx]
    y_0, x_0 = np.nonzero(mask)
    num_pix = len(y_0)
    flux = img_meas[y_0, x_0]
    flux_sum = np.sum(flux)
    if not num_pix >= num_pix_min:
        if raise_on_fail:
            raise RuntimeError(f'estimate_ellipse failed finding {num_pix}!>={num_pix_min}')
        else:
            finished = True
    else:
        finished = False
    inertia = np.zeros((2, 2))
    while not finished:
        y = y_0 + 0.5 - ceny
        x = x_0 + 0.5 - cenx
        x_sq = x**2
        y_sq = y**2
        xy = x*y
        inertia[0, 0] = np.sum(flux*x_sq)/flux_sum
        inertia[0, 1] = np.sum(flux*xy)/flux_sum
        inertia[1, 1] = np.sum(flux*y_sq)/flux_sum
        if do_recenter:
            x_shift = np.sum(flux*x)/flux_sum
            y_shift = np.sum(flux*y)/flux_sum
            finished = np.abs(x_shift) < 0.1 and np.abs(y_shift) < 0.1
            if not finished:
                cenx += x_shift
                ceny += y_shift
        else:
            finished = True
    if deconvolution_matrix is not None:
        inertia -= deconvolution_matrix
    inertia[0, 0] = np.clip(inertia[0, 0], sigma_sq_min, np.Inf)
    inertia[1, 1] = np.clip(inertia[1, 1], sigma_sq_min, np.Inf)
    inertia[1, 0] = inertia[0, 1]
    if not np.all(np.isfinite(inertia)):
        raise RuntimeError(f'Inertia {inertia} not all finite')
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
