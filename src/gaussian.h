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
typedef py::array_t<double> params_gauss;

namespace multiprofit {

/*
    Efficiently evaluate a 2D Gaussian centered on (cen_x, cen_x) with total flux L, half-light radius r_eff,
    position angle ang (annoying GALFIT convention of up=0), axis ratio axrat, at the centers of pixels on a
    grid defined by the corners (x_min, y_min) and (x_max, y_max) with dim_x x dim_y pixels.
*/
ndarray make_gaussian_pixel(
    double cen_x, double cen_y, double L,
    double r_eff, double axrat, double ang,
    double x_min, double x_max, double y_min, double y_max,
    unsigned int dim_x, unsigned int dim_y);

/*
    An alternative implementation of make_gaussian_pixel based on the Sersic PDF. Should be the same +/-
    machine eps.
*/
ndarray make_gaussian_pixel_sersic(
    double cen_x, double cen_y, double L,
    double r_eff, double axrat, double ang,
    double x_min, double x_max, double y_min, double y_max,
    unsigned int dim_x, unsigned int dim_y);

/*
    As make_gaussian_pixel but taking (modified) elements of a covariance matrix - sig_x, sig_y, and rho
    Where rho is covar[0,1]/sigx/sigy
*/
ndarray make_gaussian_pixel_covar(
    double cen_x, double cen_y, double L,
    double sig_x, double sig_y, double rho,
    double x_min, double x_max, double y_min, double y_max,
    unsigned int dim_x, unsigned int dim_y);

/*
    As make_gaussian_pixel but for multiple Gaussians.
    GAUSSIANS is an ndarray with rows of Gaussian parameters in the same order as for make_gaussian_pixel:
    [XCEN, YCEN, L, R, ANG, AXRAT]
*/
ndarray make_gaussians_pixel(
    const params_gauss& gaussians, double x_min, double x_max, double y_min, double y_max,
    unsigned int dim_x, unsigned int dim_y);

/*
    As make_gaussians_pixel but outputs to an existing matrix. Can skip output for benchmarking purposes.
*/
void add_gaussians_pixel(
    const params_gauss& gaussians, double x_min, double x_max, double y_min, double y_max,
    ndarray & output);

/*
    Compute the log likelihood of a Gaussian mixture model given some data and an inverse variance map.
    GAUSSIANS is as in make_gaussians_pixel()
    output is an optional matrix to output the model to. Must be the same size as DATA if provided.
*/
double loglike_gaussians_pixel(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    double x_min, double x_max, double y_min, double y_max, bool to_add,
    ndarray & output, ndarray & residual, ndarray & grad,
    ndarray_s & grad_param_map, ndarray & grad_param_factor,
    ndarray_s & sersic_param_map, ndarray & sersic_param_factor);
}
#endif
