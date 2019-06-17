/*
 * This file is part of multiprofit.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __MULTIPROFIT_GAUSSIAN_INTEGRATOR_H
#define __MULTIPROFIT_GAUSSIAN_INTEGRATOR_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<double> ndarray;

namespace multiprofit {
/*
    Numerically integrate a 2D Gaussian centered on (cen_x, cen_x) with total flux L, half-light radius r_eff,
    position angle ang (annoying GALFIT convention of up=0), axis ratio axrat, at the centers of pixels on a
    grid defined by the corners (x_min, y_min) and (x_max, y_max) with dim_x x dim_y pixels to a
    tolerance acc.
    TODO: Review and revise acc parameter
*/
    ndarray make_gaussian(
        const double cen_x, const double cen_y, const double mag,
        const double r_eff, const double axrat, const double ang,
        const double x_min, const double x_max, const double y_min, const double y_max,
        const unsigned int dim_x, const unsigned int dim_y, const double acc
    );
}
#endif //MULTIPROFIT_GAUSSIAN_INTEGRATOR_H
