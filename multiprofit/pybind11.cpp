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

#ifndef MULTIPROFIT_GAUSSIAN_INTEGRATOR_H
#include "../src/gaussian_integrator.h"
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
        "xcen"_a, "ycen"_a, "mag"_a, "re"_a, "axrat"_a, "ang"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a, "acc"_a
    );

    m.def(
        "make_gaussian_pixel", &multiprofit::make_gaussian_pixel,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF.",
        "xcen"_a, "ycen"_a, "l"_a, "r"_a, "axrat"_a, "ang"_a,
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a
    );

    m.def(
        "make_gaussian_pixel_sersic", &multiprofit::make_gaussian_pixel_sersic,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the 2D Sersic PDF.",
        "xcen"_a, "ycen"_a, "l"_a, "r"_a,  "axrat"_a, "ang"_a,
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
        "make_gaussians_pixel", &multiprofit::make_gaussians_pixel,
        "Evaluate 2D Gaussians at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF.",
        "gaussians"_a.noconvert(), "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "xdim"_a, "ydim"_a
    );

    m.def(
        "add_gaussians_pixel", &multiprofit::add_gaussians_pixel,
        "Evaluate 2D Gaussians at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF, adding to an existing matrix.",
        "gaussians"_a.noconvert(), "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "output"_a.noconvert()
    );

    m.def(
        "loglike_gaussians_pixel", &multiprofit::loglike_gaussians_pixel,
        "Evaluate the log likelihood of a 2D Gaussian mixture model at the centers of pixels on a rectangular"
        "grid using the standard bivariate Gaussian PDF.",
        "data"_a.noconvert(), "varinverse"_a.noconvert(), "gaussians"_a.noconvert(),
        "xmin"_a, "xmax"_a, "ymin"_a, "ymax"_a, "to_add"_a, "output"_a.noconvert(), "grad"_a.noconvert(),
        "grad_param_map"_a.noconvert(), "grad_param_factor"_a.noconvert()
    );
}
