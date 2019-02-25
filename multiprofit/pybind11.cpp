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

#ifndef __MULTIPROFIT_GAUSSIAN_H_
#include "../src/gaussian.h"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

using namespace pybind11::literals;

PYBIND11_MODULE(_multiprofit, m)
{
    //py::module m("multiprofit");
    m.doc() = "MultiProFit Pybind11 functions"; // optional module docstring

    m.def(
        "make_gaussian", &multiprofit::make_gaussian,
        "Integrate a 2D Gaussian over a rectangular grid.",
        "xcen"_a, "ycen"_a, "mag"_a, "re"_a, "ang"_a, "axrat"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a, "acc"_a
    );

    m.def(
        "make_gaussian_pixel", &multiprofit::make_gaussian_pixel,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF.",
        "xcen"_a, "ycen"_a, "l"_a, "r"_a, "ang"_a, "axrat"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a
    );

   
    m.def(
        "make_gaussian_pixel_sersic", &multiprofit::make_gaussian_pixel_sersic,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the 2D Sersic PDF.",
        "xcen"_a, "ycen"_a, "l"_a, "r"_a, "ang"_a, "axrat"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a
    );
 

    m.def(
        "make_gaussian_pixel_covar", &multiprofit::make_gaussian_pixel_covar,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF and given a covariance matrix.",
        "xcen"_a, "ycen"_a, "l"_a, "sigx"_a, "sigy"_a, "rho"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a
    );

    m.def(
        "loglike_gaussian_pixel", &multiprofit::loglike_gaussian_pixel,
        "Evaluate the log likelihood of a 2D Gaussian mixture model at the centers of pixels on a rectangular"
        "grid using the standard bivariate Gaussian PDF.",
        "data"_a, "varinverse"_a, "gaussians"_a, "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a
    );

    m.def(
        "make_gaussian_mix_8_pixel", &multiprofit::make_gaussian_mix_8_pixel,
        "Evaluate eight 2D Gaussians at the centers of pixels on a rectangular grid using the 2D Sersic PDF.",
        "xcen"_a, "ycen"_a,
        "l1"_a, "l2"_a, "l3"_a, "l4"_a, "l5"_a, "l6"_a, "l7"_a, "l8"_a,
        "r1"_a, "r2"_a, "r3"_a, "r4"_a, "r5"_a, "r6"_a, "r7"_a, "r8"_a,
        "ang1"_a, "ang2"_a, "ang3"_a, "ang4"_a, "ang5"_a, "ang6"_a, "ang7"_a, "ang8"_a,
        "q1"_a, "q2"_a, "q3"_a, "q4"_a, "q5"_a, "q6"_a, "q7"_a, "q8"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a,  "ydim"_a
    );
}
