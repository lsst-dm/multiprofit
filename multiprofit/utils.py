import numpy as np


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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Cannot parse {} as boolean.'.format(v))


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
