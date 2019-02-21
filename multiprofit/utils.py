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
        chisum += np.sum(chivals**2)
        chicount += len(chivals)**2
    return chisum/chicount


def fluxtomag(x):
    return -2.5*np.log10(x)


def magtoflux(x):
    return 10**(-0.4*x)


def normalize(array):
    array /= np.sum(array)
    return array


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot parse {} as boolean.'.format(v))

