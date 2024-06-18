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

import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f

__all__ = ["make_image_gaussians", "make_psf_model_null"]


def make_image_gaussians(
    gaussians_source: g2.Gaussians,
    gaussians_kernel: g2.Gaussians | None = None,
    **kwargs: Any,
) -> g2.ImageD:
    """Make an image array from a set of Gaussians.

    Parameters
    ----------
    gaussians_source
        Gaussians representing components of sources.
    gaussians_kernel
        Gaussians representing the smoothing kernel.
    **kwargs
        Additional keyword arguments to pass to gauss2d.make_gaussians_pixel_D
        (i.e. image size, etc.).

    Returns
    -------
    image
        The rendered image of the given Gaussians.
    """
    if gaussians_kernel is None:
        gaussians_kernel = g2.Gaussians([g2.Gaussian()])
    gaussians = g2.ConvolvedGaussians(
        [
            g2.ConvolvedGaussian(source=source, kernel=kernel)
            for source in gaussians_source for kernel in gaussians_kernel
        ]
    )
    return g2.make_gaussians_pixel_D(gaussians=gaussians, **kwargs)


def make_psf_model_null() -> g2f.PsfModel:
    """Make a default (null) PSF model.

    Returns
    -------
    model
        A null PSF model consisting of a single, normalized, zero-size
        Gaussian.
    """
    return g2f.PsfModel(g2f.GaussianComponent.make_uniq_default_gaussians([0], True))
