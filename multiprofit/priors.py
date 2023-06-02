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


def get_hst_size_prior(mag_psf_i):
    """ Return the mean and stddev for a reasonable HST-based size (major axis half-light radius) prior.

    Notes
    -----
    Return values are log10 scaled in units of arcseconds. The input should be a PSF mag because other
    magnitudes - even Gaussian - can be unreliable for low S/N (non-)detections.
    """
    return 0.75*(19 - np.clip(mag_psf_i, 10, 30))/6.5, 0.2
