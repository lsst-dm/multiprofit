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


def make_psf_source(
    n_gaussians: int = None,
    transforms: dict[str, g2f.TransformD] = None,
) -> g2f.Source:
    if n_gaussians is None:
        n_gaussians = 2
    if not n_gaussians > 0:
        raise ValueError(f"{n_gaussians=}!>0")
    if transforms is None:
        transforms = {}
    components = [None] * n_gaussians
    cenx = g2f.CentroidXParameterD(0, limits=g2f.LimitsD(min=0, max=100))
    ceny = g2f.CentroidYParameterD(0, limits=g2f.LimitsD(min=0, max=100))
    centroid = g2f.CentroidParameters(cenx, ceny)

    n_last = n_gaussians - 1
    last = None

    transforms = {
        'frac': transforms.get('frac', g2f.LogitTransformD()),
        'rho': transforms.get('rho', g2f.LogitLimitedTransformD(limits=g2f.LimitsD(min=-0.99, max=0.99))),
        'sigma': transforms.get('sigma', g2f.LogTransformD()),
    }

    for c in range(n_gaussians):
        is_last = c == n_last
        last = g2f.FractionalIntegralModel(
            {
                g2f.Channel.NONE: g2f.ProperFractionParameterD(
                    (is_last == 1) or (0.5 + 0.5 * (c > 0)), fixed=is_last, transform=transforms['frac']
                )
            },
            g2f.LinearIntegralModel({
                g2f.Channel.NONE: g2f.IntegralParameterD(1.0, fixed=True)
            }) if (c == 0) else last,
            is_last,
        )
        components[c] = g2f.GaussianComponent(
            g2f.GaussianParametricEllipse(
                g2f.SigmaXParameterD(1 + c, transform=transforms['sigma']),
                g2f.SigmaYParameterD(1 + c, transform=transforms['sigma']),
                g2f.RhoParameterD(0, transform=transforms['rho']),
            ),
            centroid,
            last,
        )
    return g2f.Source(components)
