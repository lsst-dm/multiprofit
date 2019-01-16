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

namespace py = pybind11;

PYBIND11_MODULE(_multiprofit, m)
{
    //py::module m("multiprofit");
    m.doc() = "MultiProFit Pybind11 functions"; // optional module docstring

    m.def(
        "make_gaussian", &multiprofit::make_gaussian,
        "Integrate a 2D Gaussian over a rectangular grid.",
        py::arg("xcen"), py::arg("ycen"), py::arg("mag"), py::arg("re"), py::arg("ang"), py::arg("axrat"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
        py::arg("xdim"), py::arg("ydim"), py::arg("acc")
    );

    m.def(
        "make_gaussian_pixel", &multiprofit::make_gaussian_pixel,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF.",
        py::arg("xcen"), py::arg("ycen"), py::arg("l"), py::arg("r"), py::arg("ang"), py::arg("axrat"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
        py::arg("xdim"), py::arg("ydim")
    );

   
    m.def(
        "make_gaussian_pixel_sersic", &multiprofit::make_gaussian_pixel_sersic,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the 2D Sersic PDF.",
        py::arg("xcen"), py::arg("ycen"), py::arg("l"), py::arg("r"), py::arg("ang"), py::arg("axrat"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
        py::arg("xdim"), py::arg("ydim")
    );
 

    m.def(
        "make_gaussian_pixel_covar", &multiprofit::make_gaussian_pixel_covar,
        "Evaluate a 2D Gaussian at the centers of pixels on a rectangular grid using the standard bivariate"
        "Gaussian PDF and given a covariance matrix.",
        py::arg("xcen"), py::arg("ycen"), py::arg("l"), py::arg("sigx"), py::arg("sigy"), py::arg("rho"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
        py::arg("xdim"), py::arg("ydim")
    );

    m.def(
        "loglike_gaussian_pixel", &multiprofit::loglike_gaussian_pixel,
        "Evaluate the log likelihood of a 2D Gaussian mixture model at the centers of pixels on a rectangular"
        "grid using the standard bivariate Gaussian PDF.",
        py::arg("data"), py::arg("varinverse"), py::arg("gaussians"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax")
    );

    m.def(
        "make_gaussian_mix_8_pixel", &multiprofit::make_gaussian_mix_8_pixel,
        "Evaluate eight 2D Gaussians at the centers of pixels on a rectangular grid using the 2D Sersic PDF.",
        py::arg("xcen"), py::arg("ycen"),
        py::arg("l1"), py::arg("l2"), py::arg("l3"), py::arg("l4"),
        py::arg("l5"), py::arg("l6"), py::arg("l7"), py::arg("l8"),
        py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"),
        py::arg("r5"), py::arg("r6"), py::arg("r7"), py::arg("r8"),
        py::arg("ang1"), py::arg("ang2"), py::arg("ang3"), py::arg("ang4"),
        py::arg("ang5"), py::arg("ang6"), py::arg("ang7"), py::arg("ang8"),
        py::arg("q1"), py::arg("q2"), py::arg("q3"), py::arg("q4"),
        py::arg("q5"), py::arg("q6"), py::arg("q7"), py::arg("q8"),
        py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"),
        py::arg("xdim"), py::arg( "ydim")
    );
}
