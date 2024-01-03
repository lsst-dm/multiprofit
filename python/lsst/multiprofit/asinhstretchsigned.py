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


import astropy.visualization as apvis
import numpy as np

# This is all hacked from astropy's AsinhStretch

__all__ = ["AsinhStretchSigned", "SinhStretchSigned"]


def _prepare(values: np.ndarray, clip: bool = True, out: np.ndarray | None = None):
    """Return clipped and/or copied values from input.

    Parameters
    ----------
    values
        The values to copy/clip from.
    clip
        Whether to clip values to between 0 and 1 (inclusive).
    out
        An existing array to assign to.

    Returns
    -------
    prepared
        The prepared values.
    """
    if clip:
        return np.clip(values, 0.0, 1.0, out=out)
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
        Default is 0.1.
    """  # noqa: W505

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
        np.true_divide(values, np.arcsinh(1.0 / self.a), out=values)
        np.true_divide(1.0 + signs * values, 2.0, out=values)
        return values

    @property
    def inverse(self):
        """A stretch object that performs the inverse operation.

        Returns
        -------
        inverse
            The inverse stretch.
        """
        return SinhStretchSigned(a=1.0 / np.arcsinh(1.0 / self.a))


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

    def __init__(self, a=1.0 / 3.0):
        super().__init__()
        self.a = a

    #    [docs]

    def __call__(self, values, clip=True, out=None):
        values = _prepare(values, clip=clip, out=out)
        values *= 2.0
        values -= 1.0
        np.true_divide(values, self.a, out=values)
        np.sinh(values, out=values)
        np.true_divide(values, np.sinh(1.0 / self.a), out=values)
        values += 1.0
        values /= 2.0
        return values

    @property
    def inverse(self):
        """A stretch object that performs the inverse operation.

        Returns
        -------
        inverse
            The inverse stretch.
        """
        return AsinhStretchSigned(a=1.0 / np.sinh(1.0 / self.a))
