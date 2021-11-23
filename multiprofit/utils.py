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
        img, cen_x=None, cen_y=None, denoise=True, deconvolution_params=None, sigma_sq_min=1e-8,
        do_recenter=True, return_cens=False, validate=True, contiguous=False, pixel_min=None,
        pixel_sn_min=None, sigma_inverse=None, num_pix_min=2, raise_on_fail=False, return_as_params=False):
    """

    :param img: ndarray; an image of a source to estimate the moments of.
    :param cen_x, cen_y: float; initial estimate of the centroid of the source.
    :param denoise: bool; whether to attempt to naively de-noise the image by zeroing all negative pixels and
        the faintest positive pixels while conserving the total flux.
    :param deconvolution_params: array-like; xx, yy and xy moments to subtract from measurements for
        deconvolution
    :param sigma_sq_min: float; the minimum variance to return, especially if deconvolving.
    :param do_recenter: bool; whether to iteratively re-estimate the centroid.
    :param return_cens: bool; whether to return the centroid.
    :param validate: bool; whether to check if the image has positive net flux before processing.
    :param contiguous: bool; whether to keep only the contiguous above-threshold pixels around the centroid.
    :param pixel_min: float; the minimum threshold pixel value. Default 0. Clipped to positive.
    :param pixel_sn_min: float; the minimum threshold pixel S/N. Default 0. Clipped to positive. Ignored if
        sigma_inverse is not supplied.
    :param sigma_inverse: ndarray; an inverse error map for applying S/N cuts. Must be same size/units as img.
    :param num_pix_min: int; the minimum number of positive pixels required for processing.
    :param raise_on_fail: bool; whether to raise an Exception on any failure instead of attempting to
        continue by not strictly obeying the inputs.
    :param return_as_params: bool; whether to return a tuple of sigma_x^2, sigma_y^2, covar instead of the
        full matrix (which is symmetric)
    :return: inertia: ndarray; the moment of inertia/covariance matrix or parameters.
        cen_x, cen_y: The centroids, if return_cens is True.
    """
    if not (sigma_sq_min >= 0):
        raise ValueError(f'sigma_sq_min={sigma_sq_min} !>= 0')
    if validate or denoise:
        sum_img = np.sum(img)
    if validate and raise_on_fail:
        if not sum_img > 0:
            raise RuntimeError(f"Tried to estimate ellipse for img(shape={img.shape}, sum={sum_img})")
    if cen_x is None:
        cen_x = img.shape[1]/2.
    if cen_y is None:
        cen_y = img.shape[0]/2.
    pixel_min = 0 if pixel_min is None else np.max([0, pixel_min])
    pixel_sn_min = 0 if pixel_sn_min is None else np.max([0, pixel_sn_min])
    if denoise and sum_img > 0:
        img = absconservetotal(np.copy(img))
    mask = img > pixel_min
    if sigma_inverse is not None:
        mask &= (img > pixel_sn_min)
    if contiguous:
        pix_cen_x, pix_cen_y = round(cen_x), round(cen_y)
        if img[pix_cen_y, pix_cen_x] > pixel_min:
            labels, _ = ndimage.label(mask)
            mask = labels == labels[pix_cen_y, pix_cen_x]
    y_0, x_0 = np.nonzero(mask)
    num_pix = len(y_0)
    flux = img[y_0, x_0]
    flux_sum = np.sum(flux)
    if not ((num_pix >= num_pix_min) and (flux_sum > 0)):
        if raise_on_fail:
            raise RuntimeError(f'estimate_ellipse failed with n_pix={num_pix} !>= min={num_pix_min}'
                               f' and/or flux_sum={flux_sum} !>0')
        finished = True
        i_xx, i_yy, i_xy = sigma_sq_min, sigma_sq_min, 0.
    else:
        finished = False
        i_xx, i_yy, i_xy = 0., 0., 0.

    while not finished:
        y = y_0 + 0.5 - cen_y
        x = x_0 + 0.5 - cen_x
        x_sq = x**2
        y_sq = y**2
        xy = x*y
        i_xx = np.sum(flux*x_sq)/flux_sum
        i_yy = np.sum(flux*y_sq)/flux_sum
        i_xy = np.sum(flux*xy)/flux_sum

        if do_recenter:
            x_shift = np.sum(flux*x)/flux_sum
            y_shift = np.sum(flux*y)/flux_sum
            finished = np.abs(x_shift) < 0.1 and np.abs(y_shift) < 0.1
            if not finished:
                cen_x_new = cen_x + x_shift
                cen_y_new = cen_y + y_shift
                if not ((0 < cen_x_new < img.shape[1]) and (0 < cen_y_new < img.shape[0])):
                    if raise_on_fail:
                        raise RuntimeError(f'cen_y,cen_x={cen_y},{cen_x} outside img.shape={img.shape}')
                    finished = True
                cen_x = cen_x_new
                cen_y = cen_y_new
        else:
            finished = True

    deconvolve = deconvolution_params is not None
    if deconvolve:
        d_xx, d_yy, d_xy = deconvolution_params
        if not ((i_xx > d_xx) and (i_yy > d_yy)):
            if raise_on_fail:
                raise RuntimeError(f'Moments {i_xx},{i_yy} not > deconvolution {d_xx},{d_yy}')
            cor = i_xy/np.sqrt(i_xx*i_yy) if (i_xx > 0 and i_yy > 0) else 0
            i_xx /= 2
            i_yy /= 2
            i_xy = 0 if (cor == 0) else i_xy/2
        else:
            cor = i_xy/np.sqrt(i_xx*i_yy)
            i_xx -= d_xx
            i_yy -= d_yy
            i_xy -= d_xy

    # Need to clip here before finishing deconvolving or else could get !(-1 < cor_new < 1)
    i_xx = np.clip(i_xx, sigma_sq_min, np.Inf)
    i_yy = np.clip(i_yy, sigma_sq_min, np.Inf)

    if deconvolve:
        cor_new = i_xy/np.sqrt(i_xx*i_yy) if (i_xx > 0 and i_yy > 0) else np.nan
        if not (-1 < cor_new < 1):
            if raise_on_fail:
                raise RuntimeError(f'Deconvolved moments {i_xx},{i_yy},{i_xy} give !(-1<rho={cor_new}<1)')
            if cor is not None and -1 < cor < 1:
                i_xy = cor*np.sqrt(i_xx*i_yy)
            else:
                i_xy = 0

    inertia = np.array(((i_xx, i_xy), (i_xy, i_yy))) if not return_as_params else (i_xx, i_yy, i_xy)
    if not np.all(np.isfinite(inertia)):
        raise RuntimeError(f'Inertia {inertia} not all finite')
    if return_cens:
        return inertia, cen_x, cen_y
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
