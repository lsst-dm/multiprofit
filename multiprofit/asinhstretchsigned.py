import astropy.visualization as apvis
import numpy as np

# This is all hacked from astropy's AsinhStretch


def _prepare(values, clip=True, out=None):
    """
    Prepare the data by optionally clipping and copying, and return the
    array that should be subsequently used for in-place calculations.
    """

    if clip:
        return np.clip(values, 0., 1., out=out)
    else:
        if out is None:
            return np.array(values, copy=True)
        else:
            out[:] = np.asarray(values)
            return out


class AsinhStretchSigned(apvis.BaseStretch):
    r"""
    A signed asinh stretch.

    The stretch is given by:

    .. math::
        y = 0.5(1 + sign(x - 0.5)\frac{{\rm asinh}(2(x - 0.5) / a)}{{\rm asinh}(1 / a)}).

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula.  The value of
        this parameter is where the asinh curve transitions from linear
        to logarithmic behavior, expressed as a fraction of the
        normalized image.  Must be in the range between 0 and 1.
        Default is 0.1
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = a

    #    [docs]
    def __call__(self, values, clip=True, out=None):
        values = _prepare(values, clip=clip, out=out)
        values *= 2
        values -= 1
        signs = np.sign(values)
        np.abs(values, out=values)
        np.true_divide(values, self.a, out=values)
        np.arcsinh(values, out=values)
        np.true_divide(values, np.arcsinh(1. / self.a), out=values)
        np.true_divide(1. + signs*values, 2., out=values)
        return values

    @property
    def inverse(self):
        """A stretch object that performs the inverse operation."""
        return SinhStretchSigned(a=1. / np.arcsinh(1. / self.a))


class SinhStretchSigned(apvis.BaseStretch):
    r"""
    A sinh stretch.

    The stretch is given by:

    .. math::
        y = \frac{{\rm sinh}(x / a)}{{\rm sinh}(1 / a)}

    Parameters
    ----------
    a : float, optional
        The ``a`` parameter used in the above formula.  Default is 1/3.
    """

    def __init__(self, a=1. / 3.):
        super().__init__()
        self.a = a

#    [docs]

    def __call__(self, values, clip=True, out=None):
        values = _prepare(values, clip=clip, out=out)
        values *= 2.
        values -= 1.
        np.true_divide(values, self.a, out=values)
        np.sinh(values, out=values)
        np.true_divide(values, np.sinh(1. / self.a), out=values)
        values += 1.
        values /= 2.
        return values

    @property
    def inverse(self):
        """A stretch object that performs the inverse operation."""
        return AsinhStretchSigned(a=1. / np.sinh(1. / self.a))
