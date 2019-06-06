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
#define __MULTIPROFIT_GAUSSIAN_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<double> ndarray;
typedef py::array_t<size_t> ndarray_s;
typedef py::array_t<double> paramsgauss;

namespace multiprofit {

/*
    Numerically integrate a 2D Gaussian centered on (XCEN, YCEN) with magnitude MAG, half-light radius RE,
    position angle ANG (annoying GALFIT convention of up=0), axis ratio AXRAT, over a grid of defined by the
    corners (XMIN, YMIN) and (XMAX, YMAX) with XDIM x YDIM pixels, to a relative tolerance ACC.
*/
ndarray make_gaussian(
    const double XCEN, const double YCEN, const double MAG,
    const double RE, const double AXRAT, const double ANG,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM,
    const double ACC);

/*
    Efficiently evaluate a 2D Gaussian centered on (XCEN, YCEN) with total flux L, half-light radius R,
    position angle ANG (annoying GALFIT convention of up=0), axis ratio AXRAT, at the centers of pixels on a
    grid defined by the corners (XMIN, YMIN) and (XMAX, YMAX) with XDIM x YDIM pixels.
*/
ndarray make_gaussian_pixel(
    const double XCEN, const double YCEN, const double L,
    const double R, const double AXRAT, const double ANG,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    An alternative implementation of make_gaussian_pixel based on the Sersic PDF. Should be the same +/-
    machine eps.
*/
ndarray make_gaussian_pixel_sersic(
    const double XCEN, const double YCEN, const double L,
    const double R, const double AXRAT, const double ANG,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    As make_gaussian_pixel but taking (modified) elements of a covariance matrix - sig_x, sig_y, and rho
    Where rho is covar[0,1]/sigx/sigy
*/
ndarray make_gaussian_pixel_covar(const double XCEN, const double YCEN, const double L,
    const double SIGX, const double SIGY, const double RHO,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    As make_gaussian_pixel but for multiple Gaussians.
    GAUSSIANS is an ndarray with rows of Gaussian parameters in the same order as for make_gaussian_pixel:
    [XCEN, YCEN, L, R, ANG, AXRAT]
*/
ndarray make_gaussians_pixel(const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX,
    const double YMIN, const double YMAX, const unsigned int XDIM, const unsigned int YDIM);

/*
    As make_gaussians_pixel but outputs to an existing matrix. Can skip output for benchmarking purposes.
*/
void add_gaussians_pixel(const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX,
    const double YMIN, const double YMAX, ndarray & output);

/*
    Compute the log likelihood of a Gaussian mixture model given some data and an inverse variance map.
    GAUSSIANS is as in make_gaussians_pixel()
    output is an optional matrix to output the model to. Must be the same size as DATA if provided.
*/
double loglike_gaussians_pixel(const ndarray & DATA, const ndarray & VARINVERSE, const paramsgauss& GAUSSIANS,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX, bool to_add, ndarray & output,
    ndarray & grad, ndarray_s & grad_param_map, ndarray & grad_param_factor);
}
#endif
