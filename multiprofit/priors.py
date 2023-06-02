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
import lsst.pex.config as pexConfig


class ShapePriorConfig(pexConfig.Config):
    prior_axrat_stddev = pexConfig.Field[float](
        default=0,
        doc="Prior std. dev. on axis ratio (ignored if not >0)",
    )
    prior_size_stddev = pexConfig.Field[float](
        default=0,
        doc="Prior std. dev. on size_major (ignored if not >0)",
    )

    def get_shape_prior(self, ellipse: g2f.ParametricEllipse) -> g2f.ShapePrior | None:
        use_prior_axrat = self.prior_axrat_stddev > 0 and np.isfinite(self.prior_axrat_stddev)
        use_prior_size = self.prior_size_stddev > 0 and np.isfinite(self.prior_size_stddev)

        if use_prior_axrat or use_prior_size:
            return g2f.ShapePrior(
                ellipse,
                (g2f.ParametricGaussian1D(None, g2f.StdDevParameterD(self.prior_axrat_stddev))
                 if use_prior_size
                 else None),
                (g2f.ParametricGaussian1D(None, g2f.StdDevParameterD(self.use_prior_axrat))
                 if use_prior_axrat
                 else None),
            )
        return None


def get_hst_size_prior(mag_psf_i):
    """ Return the mean and stddev for a reasonable HST-based size (major axis half-light radius) prior.

    Notes
    -----
    Return values are log10 scaled in units of arcseconds. The input should be a PSF mag because other
    magnitudes - even Gaussian - can be unreliable for low S/N (non-)detections.
    """
    return 0.75*(19 - np.clip(mag_psf_i, 10, 30))/6.5, 0.2
