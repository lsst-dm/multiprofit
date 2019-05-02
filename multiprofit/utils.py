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
        indexarr = 0
        sumneg = 0
        while ndarray[indices[indexarr]] < 0:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        while sumneg < 0 and indexarr < ndarray.shape[0]:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        ndarray[indices[indexarr-1]] = sumneg
        if indexarr == ndarray.shape[0]:
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


def getchisqred(chis):
    chisum = 0
    chicount = 0
    for chivals in chis:
        chisum += np.sum(chivals*chivals)
        chicount += chivals.size
    return chisum/chicount


def fluxtomag(ndarray):
    return -2.5*np.log10(ndarray)


def magtoflux(ndarray):
    return 10**(-0.4*ndarray)


# Fairly standard moment of inertia estimate of ellipse orientation and size
# TODO: compare with galsim's convenient calculateHLR/FWHM
# TODO: replace with the stack's method (in meas_?)
def estimateellipse(img, denoise=True):
    imgmeas = absconservetotal(np.copy(img)) if denoise else img
    y, x = np.nonzero(imgmeas)
    flux = imgmeas[y, x]
    y = y - imgmeas.shape[0]/2.
    x = x - imgmeas.shape[1]/2.
    inertia = np.zeros((2, 2))
    inertia[0, 0] = np.sum(flux*x**2)
    inertia[0, 1] = np.sum(flux*x*y)
    inertia[1, 0] = inertia[0, 1]
    inertia[1, 1] = np.sum(flux*y**2)
    evals, evecs = np.linalg.eig(inertia)
    idxevalmax = np.argmax(evals)
    axrat = evals[1-idxevalmax]/evals[idxevalmax]
    ang = np.degrees(np.arctan2(evecs[1, idxevalmax], evecs[0, idxevalmax])) - 90
    if ang < 0:
        ang += 360
    return axrat, ang, np.sqrt(evals[idxevalmax]/np.sum(flux))


def normalize(ndarray):
    ndarray /= np.sum(ndarray)
    return ndarray


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot parse {} as boolean.'.format(v))

