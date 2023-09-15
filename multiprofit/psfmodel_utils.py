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

import gauss2d.fit as g2f

import numpy as np

from multiprofit.limits import limits_ref


def make_psf_source(
    sigma_xs: list[float] | None = None,
    sigma_ys: list[float] | None = None,
    rhos: list[float] | None = None,
    fracs: list[float] | None = None,
    transforms: dict[str, g2f.TransformD] = None,
    limits_rho: g2f.LimitsD = None,
) -> g2f.Source:
    """Make a Gaussian mixture PSF source from parameter values.

    Parameters
    ----------
    sigma_xs : list[float]
        Gaussian sigma_x values.
    sigma_ys : list[float]
        Gaussian sigma_y values.
    rhos : list[float]
        Gaussian rho values.
    fracs : list[float]
        Gaussian sigma_x values.
    transforms : dict[str, gauss2d.fit.TransformD]
        Dict of transforms by variable name (frac/rho/sigma). If not set,
        will default to Logit/LogitLimited/Log10, respectively.
    limits_rho : gauss2d.fit.LimitsD
        Limits for rho parameters. Defaults to limits_ref['rho'].

    Returns
    -------
    source : gauss2d.fit.Source
        A source model with Gaussians initialized as specified.

    Notes
    -----
    Parameter lists must all be the same length.
    """
    if limits_rho is None:
        limits_rho = limits_ref['rho']
    if sigma_xs is None:
        sigma_xs = [1.5, 3.0] if sigma_ys is not None else sigma_ys
    if sigma_ys is None:
        sigma_ys = sigma_xs
    n_gaussians = len(sigma_xs)
    if n_gaussians == 0:
        raise ValueError(f"{n_gaussians=}!>0")
    if rhos is None:
        rhos = [0.]*n_gaussians
    if fracs is None:
        fracs = np.arange(1, n_gaussians + 1)/n_gaussians
    if transforms is None:
        transforms = {}
    transforms_default = {
        'frac': transforms.get('frac', g2f.LogitTransformD()),
        'rho': transforms.get('rho', g2f.LogitLimitedTransformD(limits=limits_rho)),
        'sigma': transforms.get('sigma', g2f.Log10TransformD()),
    }
    for key, value in transforms_default.items():
        if key not in transforms:
            transforms[key] = value

    if (len(sigma_ys) != n_gaussians) or (len(rhos) != n_gaussians) or (len(fracs) != n_gaussians):
        raise ValueError(f"{len(sigma_ys)=} and/or {len(rhos)=} and/or {len(fracs)=} != {n_gaussians=}")

    errors = []
    for idx, (sigma_x, sigma_y, rho, frac) in enumerate(zip(sigma_xs, sigma_ys, rhos, fracs)):
        if not ((sigma_x >= 0) and (sigma_y >= 0)):
            errors.append(f"sigma_xs[{idx}]={sigma_x} and/or sigma_ys[{idx}]={sigma_y} !>=0")
        if not (limits_rho.check(rho)):
            errors.append(f"rhos[{idx}]={rho} !within({limits_rho=})")
        if not (frac >= 0):
            errors.append(f"fluxes[{idx}]={frac} !>0")
    if errors:
        raise ValueError("; ".join(errors))
    fracs[-1] = 1.

    components = [None] * n_gaussians
    cenx = g2f.CentroidXParameterD(0, limits=g2f.LimitsD(min=0, max=100))
    ceny = g2f.CentroidYParameterD(0, limits=g2f.LimitsD(min=0, max=100))
    centroid = g2f.CentroidParameters(cenx, ceny)

    n_last = n_gaussians - 1
    last = None

    for c in range(n_gaussians):
        is_last = c == n_last
        last = g2f.FractionalIntegralModel(
            [
                (g2f.Channel.NONE, g2f.ProperFractionParameterD(
                    fracs[c], fixed=is_last, transform=transforms['frac']
                ))
            ],
            g2f.LinearIntegralModel([
                (g2f.Channel.NONE, g2f.IntegralParameterD(1.0, fixed=True))
            ]) if (c == 0) else last,
            is_last,
        )
        components[c] = g2f.GaussianComponent(
            g2f.GaussianParametricEllipse(
                g2f.SigmaXParameterD(sigma_xs[c], transform=transforms['sigma']),
                g2f.SigmaYParameterD(sigma_ys[c], transform=transforms['sigma']),
                g2f.RhoParameterD(rhos[c], transform=transforms['rho']),
            ),
            centroid,
            last,
        )
    return g2f.Source(components)
