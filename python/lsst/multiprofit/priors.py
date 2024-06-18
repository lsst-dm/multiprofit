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

import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import numpy as np

from .transforms import transforms_ref

__all__ = ["ShapePriorConfig", "get_hst_size_prior"]


class ShapePriorConfig(pexConfig.Config):
    """Configuration for a shape prior."""

    prior_axrat_mean = pexConfig.Field[float](
        default=0.7,
        doc="Prior mean on axis ratio (prior ignored if not >0)",
    )
    prior_axrat_stddev = pexConfig.Field[float](
        default=0,
        doc="Prior std. dev. on axis ratio",
    )
    prior_size_mean = pexConfig.Field[float](
        default=1,
        doc="Prior std. dev. on size_major",
    )
    prior_size_stddev = pexConfig.Field[float](
        default=0,
        doc="Prior std. dev. on size_major (prior ignored if not >0)",
    )

    def get_shape_prior(self, ellipse: g2f.ParametricEllipse) -> g2f.ShapePrior | None:
        use_prior_axrat = (self.prior_axrat_stddev > 0) and np.isfinite(self.prior_axrat_stddev)
        use_prior_size = (self.prior_size_stddev > 0) and np.isfinite(self.prior_size_stddev)

        if use_prior_axrat or use_prior_size:
            prior_size = (
                g2f.ParametricGaussian1D(
                    g2f.MeanParameterD(self.prior_size_mean, transform=transforms_ref["log10"]),
                    g2f.StdDevParameterD(self.prior_size_stddev),
                )
                if use_prior_size
                else None
            )
            prior_axrat = (
                g2f.ParametricGaussian1D(
                    g2f.MeanParameterD(self.prior_axrat_mean, transform=transforms_ref["logit_axrat_prior"]),
                    g2f.StdDevParameterD(self.prior_axrat_stddev),
                )
                if use_prior_axrat
                else None
            )
            return g2f.ShapePrior(ellipse, prior_size, prior_axrat)
        return None


def get_hst_size_prior(mag_psf_i):
    """Return the mean and stddev for an HST-based size prior.

    The size is major axis half-light radius.

    Parameters
    ----------
    mag_psf_i
        The i-band PSF magnitudes of the source(s).

    Notes
    -----
    Return values are log10 scaled in units of arcseconds.
    The input should be a PSF mag because other magnitudes - even Gaussian -
    can be unreliable for low S/N (non-)detections.
    """
    return 0.75 * (19 - np.clip(mag_psf_i, 10, 30)) / 6.5, 0.2
