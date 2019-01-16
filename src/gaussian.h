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

#include <Eigen/Core>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 6> paramsgauss;

namespace multiprofit {

/*
    Numerically integrate a 2D Gaussian centered on (XCEN, YCEN) with magnitude MAG, half-light radius RE,
    position angle ANG (annoying GALFIT convention of up=0), axis ratio AXRAT, over a grid of defined by the
    corners (XMIN, YMIN) and (XMAX, YMAX) with XDIM x YDIM pixels, to a relative tolerance ACC.
*/
Matrix make_gaussian(
    const double XCEN, const double YCEN, const double MAG, const double RE,
    const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM,
    const double ACC);

/*
    Efficiently evaluate a 2D Gaussian centered on (XCEN, YCEN) with total flux L, half-light radius R,
    position angle ANG (annoying GALFIT convention of up=0), axis ratio AXRAT, at the centers of pixels on a
    grid defined by the corners (XMIN, YMIN) and (XMAX, YMAX) with XDIM x YDIM pixels.
*/
Matrix make_gaussian_pixel(
    const double XCEN, const double YCEN, const double L,
    const double R, const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    An alternative implementation of make_gaussian_pixel based on the Sersic PDF. Should be the same +/-
    machine eps.
*/
Matrix make_gaussian_pixel_sersic(
    const double XCEN, const double YCEN, const double L,
    const double R, const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    As make_gaussian_pixel but taking (modified) elements of a covariance matrix - sig_x, sig_y, and rho
    Where rho is covar[0,1]/sigx/sigy
*/
Matrix make_gaussian_pixel_covar(const double XCEN, const double YCEN, const double L,
    const double SIGX, const double SIGY, const double RHO,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);

/*
    TODO: Describe
*/
double loglike_gaussian_pixel(const Matrix & DATA, const Matrix & VARINVERSE,
    const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX);

/*
    As make_gaussian_pixel but for eight different Gaussians.

    TODO: Template this
*/
Matrix make_gaussian_mix_8_pixel(
    const double XCEN, const double YCEN,
    const double L1, const double L2, const double L3, const double L4,
    const double L5, const double L6, const double L7, const double L8,
    const double RE1, const double RE2, const double RE3, const double RE4,
    const double RE5, const double RE6, const double RE7, const double RE8,
    const double ANG1, const double ANG2, const double ANG3, const double ANG4,
    const double ANG5, const double ANG6, const double ANG7, const double ANG8,
    const double Q1, const double Q2, const double Q3, const double Q4,
    const double Q5, const double Q6, const double Q7, const double Q8,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM);
}

#endif
