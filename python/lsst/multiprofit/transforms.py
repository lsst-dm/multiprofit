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

from typing import Any

import gauss2d.fit as g2f
import numpy as np

from .limits import limits_ref

__all__ = ["get_logit_limited", "verify_transform_derivative", "transforms_ref"]


def get_logit_limited(lower: float, upper: float, factor: float = 1.0, name: str | None = None):
    """Get a logit transform stretched to span a different range than [0,1].

    Parameters
    ----------
    lower
        The lower limit of the range to span.
    upper
        The upper limit of the range to span.
    factor
        A multiplicative factor to apply to the transformed result.
    name
        A descriptive name for the transform.

    Returns
    -------
    transform
        A modified logit transform as specified.
    """
    return g2f.LogitLimitedTransformD(
        limits=g2f.LimitsD(
            min=lower,
            max=upper,
            name=name
            if name is not None
            else f"LogitLimitedTransformD(min={lower}, max={upper}, factor={factor})",
        ),
        factor=factor,
    )


def verify_transform_derivative(
    transform: g2f.TransformD,
    value_transformed: float,
    derivative: float | None = None,
    abs_max: float = 1e6,
    dx_ratios=None,
    **kwargs: Any,
):
    """Verify that the derivative of a transform class is correct.

    Parameters
    ----------
    transform
        The transform to verify.
    value_transformed
        The un-transformed value at which to verify the transform.
    derivative
        The nominal derivative at value_transformed.
        Must equal transform.derivative(value_transformed).
    abs_max
        The x value to skip verification if np.abs(derivative) > x.
    dx_ratios
        Iterable of signed ratios to set dx for finite differencing.
        dx = value*ratio (untransformed). Only used if dx is None.
    **kwargs
        Keyword arguments to pass to np.isclose when comparing derivatives to
        finite differences.

    Raises
    ------
    RuntimeError
        Raised if the transform derivative doesn't match finite differences
        within the specified tolerances.

    Notes
    -----
    derivative should only be specified if it has previously been computed for
    the exact value_transformed, to avoid re-computing it unnecessarily.

    Default dx_ratios are [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14].
    Verification will test all ratios until at least one passes.
    """
    # Skip testing finite differencing if the derivative is very large
    # This might happen e.g. near the limits of the transformation
    # TODO: Check if better finite differencing is possible for large values
    if abs_max is None:
        abs_max = 1e8
    value = transform.reverse(value_transformed)
    if derivative is None:
        derivative = transform.derivative(value)
    is_close = np.abs(derivative) > abs_max
    if not is_close:
        if dx_ratios is None:
            dx_ratios = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
        for ratio in dx_ratios:
            dx = value * ratio
            fin_diff = (transform.forward(value + dx) - value_transformed) / dx
            if not np.isfinite(fin_diff):
                fin_diff = -(transform.forward(value - dx) - value_transformed) / dx
            is_close = np.isclose(derivative, fin_diff, **kwargs)
            if is_close:
                break
    if not is_close:
        raise RuntimeError(
            f"{transform} derivative={derivative:.8e} != last "
            f"finite diff.={fin_diff:8e} with dx={dx} dx_abs_max={abs_max}"
        )


transforms_ref = {
    "none": g2f.UnitTransformD(),
    "log": g2f.LogTransformD(),
    "log10": g2f.Log10TransformD(),
    "inverse": g2f.InverseTransformD(),
    "logit": g2f.LogitTransformD(),
    "logit_rho": get_logit_limited(
        limits_ref["rho"].min,
        limits_ref["rho"].max,
        name=f"ref_logit_rho[{limits_ref['rho'].min}, {limits_ref['rho'].max}]",
    ),
    "logit_axrat": get_logit_limited(1e-4, 1, name="ref_logit_axrat[1e-4, 1]"),
    "logit_axrat_prior": get_logit_limited(-0.001, 1.001, name="ref_logit_axrat_prior[-0.001, 1.001]"),
    "logit_sersic": get_logit_limited(0.5, 6.0, name="ref_logit_sersic[0.5, 6.0]"),
}
