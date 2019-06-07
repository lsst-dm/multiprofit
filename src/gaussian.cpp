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
#include "gaussian.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

typedef pybind11::detail::unchecked_reference<double, 2l> MatrixUnchecked;
typedef pybind11::detail::unchecked_reference<size_t, 2l> MatrixSUnchecked;
typedef pybind11::detail::unchecked_mutable_reference<double, 1l> ArrayUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<size_t, 2l> MatrixSUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<double, 2l> MatrixUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<double, 3l> Array3UncheckedMutable;

namespace multiprofit {

typedef std::vector<double> vecd;
typedef std::unique_ptr<vecd> vecdptr;
typedef std::vector<vecdptr> vecdptrvec;


inline void check_is_matrix(const ndarray * mat, std::string name = "matrix")
{
    if(mat == nullptr) throw std::invalid_argument("Passed null " + name + " to check_is_matrix");
    if(mat->ndim() != 2)
    {
        throw std::invalid_argument("Passed " + name + " with ndim=" + std::to_string(mat->ndim()) + " !=2");
    }
}

inline void check_is_jacobian(const ndarray * mat, std::string name = "matrix")
{
    if(mat == nullptr) throw std::invalid_argument("Passed null " + name + " to check_is_jacobian");
    if(mat->ndim() != 3)
    {
        throw std::invalid_argument("Passed " + name + " with ndim=" + std::to_string(mat->ndim()) + " !=3");
    }
}

const size_t N_PARAMS = 6;
const size_t N_PARAMS_CONV = 9;
inline void check_is_gaussians(const ndarray & mat, bool isconv=true)
{
    check_is_matrix(&mat, "Gaussian parameter matrix");
    const size_t LENGTH = isconv ? N_PARAMS_CONV : N_PARAMS;
    if(mat.shape(1) != LENGTH)
    {
        throw std::invalid_argument("Passed Gaussian parameter matrix with shape=[" +
            std::to_string(mat.shape(0)) + ", " + std::to_string(mat.shape(1)) + "!=" +
            std::to_string(LENGTH) + "]");
    }
}

struct TermsMoments
{
    vecdptr x;
    vecdptr x_norm;
    vecdptr xx;
};

inline TermsMoments
gaussian_pixel_x_xx(const double cen_x, const double x_min, const double bin_x, const unsigned int dim_x,
    const double xx_weight, const double xy_weight)
{
    const double x_init = x_min - cen_x + bin_x/2.;
    vecdptr x = std::make_unique<vecd>(dim_x);
    vecdptr xx = std::make_unique<vecd>(dim_x);
    vecdptr x_norm = std::make_unique<vecd>(dim_x);
    for(unsigned int i = 0; i < dim_x; i++)
    {
        double dist = x_init + i*bin_x;
        (*x)[i] = dist;
        (*xx)[i] = dist*dist*xx_weight;
        (*x_norm)[i] = dist*xy_weight;
    }
    return TermsMoments({std::move(x), std::move(x_norm), std::move(xx)});
}

inline void gaussian_pixel(ndarray & mat, const double weight, const double cen_x, const double cen_y,
    const double x_min, const double y_min, const double bin_x, const double bin_y,
    const double xx_weight, const double yy_weight, const double xy_weight)
{
    check_is_matrix(&mat);
    // don't ask me why these are reversed
    const unsigned int dim_x = mat.shape(1);
    const unsigned int dim_y = mat.shape(0);

    const auto moment_terms_y = gaussian_pixel_x_xx(cen_y, y_min, bin_y, dim_y, yy_weight, xy_weight);
    const vecd & y_weighted = *(moment_terms_y.x_norm);
    const vecd & yy_weighted = *(moment_terms_y.xx);

    auto matref = mat.mutable_unchecked<2>();
    double x = x_min-cen_x+bin_x/2.;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < dim_x; i++)
    {
        const double xx_weighted = x*x*xx_weight;
        for(unsigned int j = 0; j < dim_y; j++)
        {
           matref(j, i) = weight*exp(-(xx_weighted + yy_weighted[j] - x*y_weighted[j]));
        }
        x += bin_x;
    }
}

/*
This is some largely unnecessary algebra to derive conversions between the ellipse parameterization and the
covariance matrix parameterization of a bivariate Gaussian.

See e.g. http://mathworld.wolfram.com/BivariateNormalDistribution.html
... and https://www.unige.ch/sciences/astro/files/5413/8971/4090/2_Segransan_StatClassUnige.pdf

tan 2th = 2*rho*sig_x*sig_y/(sig_x^2 - sig_y^2)

sigma_maj^2 = (cos2t*sig_x^2 - sin2t*sig_y^2)/(cos2t-sin2t)
sigma_min^2 = (cos2t*sig_y^2 - sin2t*sig_x^2)/(cos2t-sin2t)

(sigma_maj^2*(cos2t-sin2t) + sin2t*sig_y^2)/cos2t = sig_x^2
-(sigma_min^2*(cos2t-sin2t) - cos2t*sig_y^2)/sin2t = sig_x^2

(sigma_maj^2*(cos2t-sin2t) + sin2t*sig_y^2)/cos2t = -(sigma_min^2(cos2t-sin2t) - cos2t*sig_y^2)/sin2t
sin2t*(sigma_maj^2*(cos2t-sin2t) + sin2t*sig_y^2) + cos2t*(sigma_min^2*(cos2t-sin2t) - cos2t*sig_y^2) = 0
cos4t*sig_y^2 - sin^4th*sig_y^2 = sin2t*(sigma_maj^2*(cos2t-sin2t)) + cos2t*(sigma_min^2*(cos2t-sin2t))

cos^4x - sin^4x = (cos^2 + sin^2)*(cos^2-sin^2) = cos^2 - sin^2

sig_y^2 = (sin2t*(sigma_maj^2*(cos2t-sin2t)) + cos2t*(sigma_min^2*(cos2t-sin2t)))/(cos2t - sin2t)
       = (sin2t*sigma_maj^2 + cos2t*sigma_min^2)

sigma_maj^2*(cos2t-sin2t) + sin2t*sig_y^2 = cos2t*sig_x^2
sig_x^2 = (sigma_maj^2*(cos2t-sin2t) + sin2t*sig_y^2)/cos2t
       = (sigma_maj^2*(cos2t-sin2t) + sin2t*(sin2t*sigma_maj^2 + cos2t*sigma_min^2))/cos2t
       = (sigma_maj^2*(cos2t-sin2t+sin4t) + sin2tcos2t*sigma_min^2)/cos2t
       = (sigma_maj^2*cos4t + sin2t*cos2t*sigma_min^2)/cos2t
       = (sigma_maj^2*cos2t + sigma_min^2*sin2t)

sig_x^2 - sig_y^2 = sigma_maj^2*cos2t + sigma_min^2*sin2t - sigma_maj^2*sin2t - sigma_min^2*cos2t
                = (sigma_maj^2 - sigma_min^2*)*(cos2t-sin2t)
                = (sigma_maj^2 - sigma_min^2*)*(1-tan2t)*cos2t

rho = tan2th/2/(sig_x*sig_y)*(sig_x^2 - sig_y^2)
    = tanth/(1-tan2t)/(sig_x*sig_y)*(sig_x^2 - sig_y^2)
    = tanth/(1-tan2t)/(sig_x*sig_y)*(sigma_maj^2 - sigma_min^2*)*(1-tan2t)*cos2t
    = tanth/(sig_x*sig_y)*(sigma_maj^2 - sigma_min^2)*cos2t
    = sint*cost/(sig_x*sig_y)*(sigma_maj^2 - sigma_min^2)
*/

// Conversion constant of ln(2); gaussian FWHM = 2*R_eff
inline double reff_to_sigma_gauss(double reff)
{
    return reff/1.1774100225154746635070068805362097918987;
}

inline double degrees_to_radians(double degrees)
{
    return degrees*M_PI/180.;
}

struct Covar
{
    double sig_x;
    double sig_y;
    double rho;
};

Covar ellipse_to_covar(const double sigma_maj, const double axrat, const double ANGINRAD)
{
    const double sigma_min = sigma_maj*axrat;
    // TODO: Check optimal order for precision
    const double sigma_maj_sq = sigma_maj * sigma_maj;
    const double SIGMAMINSQ = sigma_min * sigma_min;

    const double SINT = sin(ANGINRAD);
    const double COST = cos(ANGINRAD);
    // TODO: Remember if this is actually preferable to sin/cos
    const double sin_th_sq = std::pow(SINT, 2.0);
    const double cos_th_sq = std::pow(COST, 2.0);
    const bool ISCIRCLE = axrat == 1;
    const double sig_x = ISCIRCLE ? sigma_maj : sqrt(cos_th_sq*sigma_maj_sq + sin_th_sq*SIGMAMINSQ);
    const double sig_y = ISCIRCLE ? sigma_maj : sqrt(sin_th_sq*sigma_maj_sq + cos_th_sq*SIGMAMINSQ);

    Covar rval = {
        .sig_x = sig_x,
        .sig_y = sig_y,
        .rho = ISCIRCLE ? 0 : SINT*COST/sig_x/sig_y*(sigma_maj_sq-SIGMAMINSQ)
    };
    return rval;
}

Covar convolution(const Covar & C, const Covar & K)
{
    const double sig_x = sqrt(C.sig_x*C.sig_x + K.sig_x*K.sig_x);
    const double sig_y = sqrt(C.sig_y*C.sig_y + K.sig_y*K.sig_y);
    return Covar{sig_x, sig_y, (C.rho*C.sig_x*C.sig_y + K.rho*K.sig_x*K.sig_y)/(sig_x*sig_y)};
}

/*
    xmc = x - cen_x 
 
    m = L*bin_x*bin_y/(2.*M_PI*cov.sig_x*cov.sig_y)/sqrt(1-cov.rho*cov.rho) * exp(-(
        + xmc[g]^2/cov.sig_x^2/(2*(1-cov.rho*cov.rho))
        + ymc[g][j]^2/cov.sig_y^2/(2*(1-cov.rho*cov.rho))
        - cov.rho*xmc[g]*ymc[g][j]/(1-cov.rho*cov.rho)/cov.sig_x/cov.sig_y
    ))

    norm_exp = 1./(2*(1-cov.rho*cov.rho))
    weight = L*bin_x*bin_y
    T = Terms
    T.weight = weight/(2.*M_PI*cov.sig_x*cov.sig_y)*sqrt(2.*norm_exp),
    T.xx = norm_exp/cov.sig_x/cov.sig_x,
    T.yy = norm_exp/cov.sig_y/cov.sig_y,
    T.xy = 2.*cov.rho*norm_exp/cov.sig_x/cov.sig_y

    m = T.weight * exp(-(xmc[g]^2*T.xx + ymc[g][j]^2*T.yy - cov.rho*xmc[g]*ymc[g][j]*T.xy))
 
    cov.rho -> r; cov.sig_x/y -> s;

    weight_xx[g] = T.xx
    dxmc^2/dcen_x = -2*xmc[g] - r*ymc[g][j]*T.xy

    ymc_weighted = r*ymc*T.xy
 
    dm/dL = m/L
    dm/dcen_x = (2*xmc[g]*weight_xx[g] - ymc_weighted[g][j])*m
    dm/ds = -m/s + m*(2/s*[xy]sqs[g] - xy/s) = m/s*(2*[xy]sqs[g] - (1+xy))
    dm/dr = -m*(r/(1-r^2) + 4*r*(1-r^2)*(xmc_sq_norm[g] + yy_weighted[g][j]) -
        (1/r + 2*r/(1+r)*xmc[g]*ymc[g][j])

    To verify (all on one line):

    https://www.wolframalpha.com/input/?i=differentiate+F*s*t%2F(2*pi*sqrt(1-r%5E2))*
    exp(-((x-m)%5E2*s%5E2+%2B+(y-n)%5E2*t%5E2+-+2*r*(x-m)*(y-n)*s*t)%2F(2*(1-r%5E2)))+wrt+s
*/

// Various multiplicative terms that appread in a Gaussian PDF
struct Terms
{
    double weight;
    double xx;
    double yy;
    double xy;
};

Terms terms_from_covar(const double weight, const Covar & cov)
{
    const double norm_exp = 1./(2*(1-cov.rho*cov.rho));
    Terms rval = {
        .weight = weight/(2.*M_PI*cov.sig_x*cov.sig_y)*sqrt(2.*norm_exp),
        .xx = norm_exp/cov.sig_x/cov.sig_x,
        .yy = norm_exp/cov.sig_y/cov.sig_y,
        .xy = 2.*cov.rho*norm_exp/cov.sig_x/cov.sig_y
    };
    return rval;
}

class TermsPixel
{
public:
    vecdptr weight = nullptr;
    vecdptr xmc = nullptr;
    vecdptr xmc_weighted = nullptr;
    vecdptrvec ymc_weighted;
    vecdptr weight_xx = nullptr;
    vecdptr xmc_sq_norm = nullptr;
    vecdptrvec yy_weighted;
    // TODO: Give these variables more compelling names

    TermsPixel(size_t N_GAUSS)
    {
        this->weight = std::make_unique<vecd>(N_GAUSS);
        this->xmc = std::make_unique<vecd>(N_GAUSS);
        this->xmc_weighted = std::make_unique<vecd>(N_GAUSS);
        ymc_weighted.resize(N_GAUSS);
        this->weight_xx = std::make_unique<vecd>(N_GAUSS);
        this->xmc_sq_norm = std::make_unique<vecd>(N_GAUSS);
        yy_weighted.resize(N_GAUSS);
    }

    void set(size_t g, double weight, double xmc, double xx, vecdptr ymc_weighted, vecdptr yy)
    {
        (*(this->weight))[g] = weight;
        (*(this->xmc))[g] = xmc;
        (*(this->weight_xx))[g] = xx;
        this->ymc_weighted[g] = std::move(ymc_weighted);
        this->yy_weighted[g] = std::move(yy);
    }
};

class TermsGradient
{
public:
    vecdptrvec ymc;
    vecdptr xx_weight = nullptr;
    vecdptr xy_weight = nullptr;
    vecdptr yy_weight = nullptr;
    vecdptr rho_factor = nullptr;
    vecdptr sig_x_inv = nullptr;
    vecdptr sig_y_inv = nullptr;
    vecdptr rho_xy_factor = nullptr;
    vecdptr sig_x_src_div_conv = nullptr;
    vecdptr sig_y_src_div_conv = nullptr;
    vecdptr drho_c_dsig_x_src = nullptr;
    vecdptr drho_c_dsig_y_src = nullptr;
    vecdptr drho_c_drho_s = nullptr;
    vecdptr xmc_t_xy_weight = nullptr;

    TermsGradient(size_t N_GAUSS)
    {
        if(N_GAUSS > 0)
        {
            this->ymc.resize(N_GAUSS);
            this->xx_weight = std::make_unique<vecd>(N_GAUSS);
            this->xy_weight = std::make_unique<vecd>(N_GAUSS);
            this->yy_weight = std::make_unique<vecd>(N_GAUSS);
            this->rho_factor = std::make_unique<vecd>(N_GAUSS);
            this->sig_x_inv = std::make_unique<vecd>(N_GAUSS);
            this->sig_y_inv = std::make_unique<vecd>(N_GAUSS);
            this->rho_xy_factor = std::make_unique<vecd>(N_GAUSS);
            this->sig_x_src_div_conv = std::make_unique<vecd>(N_GAUSS);
            this->sig_y_src_div_conv = std::make_unique<vecd>(N_GAUSS);
            this->drho_c_dsig_x_src = std::make_unique<vecd>(N_GAUSS);
            this->drho_c_dsig_y_src = std::make_unique<vecd>(N_GAUSS);
            this->drho_c_drho_s = std::make_unique<vecd>(N_GAUSS);
            this->xmc_t_xy_weight = std::make_unique<vecd>(N_GAUSS);
        }
    }

    void set(size_t g, double L, const Terms & terms, const Covar & cov_src, const Covar & cov_psf,
        const Covar & cov, vecdptr ymc)
    {
        this->ymc[g] = std::move(ymc);
        const double sig_xy = cov.sig_x*cov.sig_y;
        (*(this->xx_weight))[g] = terms.xx;
        (*(this->xy_weight))[g] = terms.xy;
        (*(this->yy_weight))[g] = terms.yy;
        (*(this->rho_factor))[g] = cov.rho/(1. - cov.rho*cov.rho);
        (*(this->sig_x_inv))[g] = 1/cov.sig_x;
        (*(this->sig_y_inv))[g] = 1/cov.sig_y;
        (*(this->rho_xy_factor))[g] = 1./(1. - cov.rho*cov.rho)/sig_xy;
        /*

    sigma_conv = sqrt(sigma_src^2 + sigma_psf^2)
    https://www.wolframalpha.com/input/?i=d(sqrt(x%5E2%2By%5E2))%2Fdx
    dsigma_conv/dsigma_src = sigma_src/sigma_conv
    df/dsigma_src = df/d_sigma_conv * dsigma_conv/dsigma_src

    rho_conv*sigmaxy_conv = rho_src*sigmaxy_src + rho_psf*sigmaxy_psf
    rho_conv = (rho_src*sigmaxy_src + rho_psf*sigmaxy_psf)/sigmaxy_conv
    drho_conv/drho_src = sigmaxy_src/sigmaxy_conv

    drho_conv/dsigmax_src = rho_src*sigmay_src/sigmay_conv*sigmax_psf^2/sigmax_conv^3 (*sigmax_src/sigmax_src)
                          = rho_conv/sigmax_src*(sigmax_psf/sigmax_conv)^2
        + rho_psf*sigmax_psf*sigmay_psf*sigmax_src/sigmay_conv/sigmax_conv^3

        */
        double sig_x_src_div_conv = cov_src.sig_x/cov.sig_x;
        double sig_y_src_div_conv = cov_src.sig_y/cov.sig_y;
        (*(this->sig_x_src_div_conv))[g] = sig_x_src_div_conv;
        (*(this->sig_y_src_div_conv))[g] = sig_y_src_div_conv;
        double offdiagpsf_dsig_xy = cov_psf.rho*cov_psf.sig_x*cov_psf.sig_y/sig_xy;
        if(cov_psf.sig_x > 0)
        {
            double sig_p_ratio = cov_psf.sig_x/cov.sig_x;
            (*(this->drho_c_dsig_x_src))[g] = cov_src.rho*sig_y_src_div_conv*sig_p_ratio*sig_p_ratio/cov.sig_x
                - offdiagpsf_dsig_xy*(cov_src.sig_x/cov.sig_x)/cov.sig_x;
        }
        if(cov_psf.sig_y > 0)
        {
            double sig_p_ratio = cov_psf.sig_y/cov.sig_y;
            (*(this->drho_c_dsig_y_src))[g] = cov_src.rho*sig_x_src_div_conv*sig_p_ratio*sig_p_ratio/cov.sig_y
                - offdiagpsf_dsig_xy*(cov_src.sig_y/cov.sig_y)/cov.sig_y;
        }
        (*(this->drho_c_drho_s))[g] = cov_src.sig_x*cov_src.sig_y/sig_xy;
    }
};

// Evaluate a Gaussian on a grid given the three elements of the symmetric covariance matrix
// Actually, rho is scaled by sig_x and sig_y (i.e. the covariance is rho*sig_x*sig_y)
ndarray make_gaussian_pixel_covar(const double cen_x, const double cen_y, const double L,
    const double sig_x, const double sig_y, const double rho,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const unsigned int dim_x, const unsigned int dim_y)
{
    const double bin_x=(x_max-x_min)/dim_x;
    const double bin_y=(y_max-y_min)/dim_y;
    const Covar cov {.sig_x = sig_x, .sig_y=sig_y, .rho=rho};
    const Terms terms = terms_from_covar(L*bin_x*bin_y, cov);

    ndarray mat({dim_y, dim_x});
    gaussian_pixel(mat, terms.weight, cen_x, cen_y, x_min, y_min, bin_x, bin_y, terms.xx, terms.yy, terms.xy);

    return mat;
}

// Evaluate a Gaussian on a grid given R_e, the axis ratio and position angle
ndarray make_gaussian_pixel(
    const double cen_x, const double cen_y, const double L,
    const double r_eff, const double axrat, const double ang,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const unsigned int dim_x, const unsigned int dim_y)
{
    const Covar cov = ellipse_to_covar(reff_to_sigma_gauss(r_eff), axrat, degrees_to_radians(ang));

// Verify transformations
// TODO: Move this to a test somewhere and fix inverse transforms to work over the whole domain
/*
    std::cout << sigma_maj << "," << sigma_min << "," << ANGRAD << std::endl;
    std::cout << sig_x << "," << sig_y << "," << rho << std::endl;
    std::cout << sqrt((cos_th_sq*sig_x_sq - sin_th_sq*sig_y_sq)/(cos_th_sq-sin_th_sq)) << "," <<
        sqrt((cos_th_sq*sig_y_sq - sin_th_sq*sig_x_sq)/(cos_th_sq-sin_th_sq)) << "," <<
        atan(2*rho*sig_x*sig_y/(sig_x_sq-sig_y_sq))/2. << std::endl;
*/
    return make_gaussian_pixel_covar(cen_x, cen_y, L, cov.sig_x, cov.sig_y, cov.rho,
        x_min, x_max, y_min, y_max, dim_x, dim_y);
}

ndarray make_gaussian_pixel_sersic(
    const double cen_x, const double cen_y, const double L,
    const double r_eff, const double axrat, const double ang,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const unsigned int dim_x, const unsigned int dim_y)
{
    // I don't remember why this isn't just 1/(2*ln(2)) but anyway it isn't
    const double weight_sersic = 0.69314718055994528622676398299518;
    const double bin_x=(x_max-x_min)/dim_x;
    const double bin_y=(y_max-y_min)/dim_y;
    const double weight = L*weight_sersic/(M_PI*axrat)/(r_eff*r_eff)*bin_x*bin_y;
    const double axrat_sq_inv = 1.0/axrat/axrat;

    ndarray mat({dim_y, dim_x});
    double x,y;
    const double bin_x_half=bin_x/2.;
    const double bin_yHALF=bin_y/2.;

    const double reff_y_inv = sin(ang*M_PI/180.)*sqrt(weight_sersic)/r_eff;
    const double reff_x_inv = cos(ang*M_PI/180.)*sqrt(weight_sersic)/r_eff;

    unsigned int i=0,j=0;
    auto matref = mat.mutable_unchecked<2>();
    x = x_min-cen_x+bin_x_half;
    for(i = 0; i < dim_x; i++)
    {
        y = y_min - cen_y + bin_yHALF;
        for(j = 0; j < dim_y; j++)
        {
           const double dist_1 = (x*reff_x_inv + y*reff_y_inv);
           const double dist_2 = (x*reff_y_inv - y*reff_x_inv);
           // mat(j,i) = ... is slower, but perhaps allows for images with dim_x*dim_y > INT_MAX ?
           matref(j, i) = weight*exp(-(dist_1*dist_1 + dist_2*dist_2*axrat_sq_inv));
           y += bin_y;
        }
        x += bin_x;
    }

    return mat;
}

typedef std::array<double, N_PARAMS> Weights;
enum class OutputType : unsigned char
{
    none      = 0,
    overwrite = 1,
    add       = 2,
    residual  = 3,
};
enum class GradientType : unsigned char
{
    none      = 0,
    loglike   = 1,
    jacobian  = 2,
};

template <OutputType output_type>
inline void gaussians_pixel_output(MatrixUncheckedMutable & output, double value, unsigned int dim1,
    unsigned int dim2) {};

template <>
inline void gaussians_pixel_output<OutputType::overwrite>(MatrixUncheckedMutable & output, const double value,
    unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) = value;
}

template <>
inline void gaussians_pixel_output<OutputType::add>(MatrixUncheckedMutable & output, const double value,
    unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) += value;
}

template <OutputType output_type>
inline void gaussians_pixel_residual(MatrixUncheckedMutable & output, const MatrixUnchecked & data,
    double model, unsigned int dim1, unsigned int dim2) {};

template <>
inline void gaussians_pixel_residual<OutputType::residual>(MatrixUncheckedMutable & output,
    const MatrixUnchecked & data, const double model, unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) = data(dim1, dim2) - model;
}

template <bool getlikelihood, GradientType gradient_type>
inline void gaussians_pixel_add_like(double & loglike, const double model, const MatrixUnchecked & data,
    const MatrixUnchecked & variance_inv, unsigned int dim1, unsigned int dim2, const bool is_variance_matrix)
{};

template <>
inline void gaussians_pixel_add_like<true, GradientType::none>(double & loglike, const double model,
    const MatrixUnchecked & data, const MatrixUnchecked & variance_inv, unsigned int dim1, unsigned int dim2,
    const bool is_variance_matrix)
{
    double diff = data(dim1, dim2) - model;
    // TODO: Check what the performance penalty for using IDXVAR is and rewrite if it's worth it
    loglike -= (diff*(diff*variance_inv(dim1*is_variance_matrix, dim2*is_variance_matrix)))/2.;
}

inline void gaussian_pixel_add_jacobian(
    double & cen_x, double & cen_y, double & L, double & sig_x, double & sig_y, double & rho,
    const double cenxweight, const double cenyweight, const double lweight,
    const double sig_x_weight, const double sig_y_weight, const double rhoweight,
    const double m, const double m_unweight,
    const double xmc_norm, const double ymc_weighted, const double xmc, const double ymc,
    const double norms_yy, const double xmc_t_xy_factor, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_weighted,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src = 1, const double dsig_y_conv_src = 1,
    const double drho_c_dsig_x_src = 0, const double drho_c_dsig_y_src = 0,
    const double drho_c_drho_s = 1)
{
    cen_x += cenxweight*m*(2*xmc_norm - ymc_weighted);
    cen_y += cenyweight*m*(2*ymc*norms_yy - xmc_t_xy_factor);
    L += lweight*m_unweight;
    const double one_p_xy = 1. + xy_norm;
    //df/dsigma_src = df/d_sigma_conv * dsigma_conv/dsigma_src
    const double dfdrho_c = m*(
            rho_div_one_m_rhosq*(1 - 2*(xx_norm + yy_weighted - xy_norm)) + xmc*ymc*norms_xy_div_rho);
    sig_x += sig_x_weight*(dfdrho_c*drho_c_dsig_x_src + dsig_x_conv_src*(m*sig_x_inv*(2*xx_norm - one_p_xy)));
    sig_y += sig_y_weight*(dfdrho_c*drho_c_dsig_y_src + dsig_y_conv_src*(m*sig_y_inv*(2*yy_weighted -
        one_p_xy)));
    //drho_conv/drho_src = sigmaxy_src/sigmaxy_conv
    rho += rhoweight*dfdrho_c*drho_c_drho_s;
}

template <GradientType gradient_type>
inline void gaussian_pixel_add_jacobian_type(Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor, const size_t g,
    unsigned int dim1, unsigned int dim2, const double m, const double m_unweight,
    const double xmc_norm, const double ymc_weighted, const double xmc, const double ymc,
    const double norms_yy, const double xmc_t_xy_factor, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_weighted,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src, const double dsig_y_conv_src,
    const double drho_dsig_x_src, const double drho_dsig_y_src,
    const double drho_c_drho_s) {};

template <>
inline void gaussian_pixel_add_jacobian_type<GradientType::jacobian>(Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor, const size_t g,
    unsigned int dim1, unsigned int dim2, const double m, const double m_unweight,
    const double xmc_norm, const double ymc_weighted, const double xmc, const double ymc,
    const double norms_yy, const double xmc_t_xy_factor, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_weighted,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src, const double dsig_y_conv_src,
    const double drho_dsig_x_src, const double drho_dsig_y_src,
    const double drho_c_drho_s)
{
    gaussian_pixel_add_jacobian(
        output(dim1, dim2, grad_param_map(g, 0)), output(dim1, dim2, grad_param_map(g, 1)),
        output(dim1, dim2, grad_param_map(g, 2)), output(dim1, dim2, grad_param_map(g, 3)),
        output(dim1, dim2, grad_param_map(g, 4)), output(dim1, dim2, grad_param_map(g, 5)),
        grad_param_factor(g, 0), grad_param_factor(g, 1), grad_param_factor(g, 2),
        grad_param_factor(g, 3), grad_param_factor(g, 4), grad_param_factor(g, 5),
        m, m_unweight, xmc_norm, ymc_weighted, xmc, ymc, norms_yy, xmc_t_xy_factor, sig_x_inv, sig_y_inv,
        xy_norm, xx_norm, yy_weighted, rho_div_one_m_rhosq, norms_xy_div_rho,
        dsig_x_conv_src, dsig_y_conv_src, drho_dsig_x_src, drho_dsig_y_src, drho_c_drho_s
    );
}


// Computes dmodel/dx for x in [cen_x, cen_y, flux, sigma_x, sigma_y, rho]
template <GradientType gradient_type>
inline void gaussian_pixel_set_weights(Weights & output, const double m,
    const double m_unweight, const double xy) {};

template <>
inline void gaussian_pixel_set_weights<GradientType::loglike>(Weights & output, const double m,
    const double m_unweight, const double xy)
{
    output[0] = m;
    output[1] = m_unweight;
    output[2] = xy;
}

// Computes and stores LL along with dll/dx for all components
template <bool getlikelihood, GradientType gradient_type>
inline void gaussians_pixel_add_like_grad(ArrayUncheckedMutable & output,
    MatrixSUnchecked & grad_param_map, MatrixUnchecked & grad_param_factor, const size_t N_GAUSS,
    const std::vector<Weights> & gaussweights, double & loglike, const double model,
    const MatrixUnchecked & data, const MatrixUnchecked & variance_inv, unsigned int dim1, unsigned int dim2,
    const bool is_variance_matrix, const TermsPixel & terms_pixel, const TermsGradient & gradterms) {};

template <>
inline void gaussians_pixel_add_like_grad<true, GradientType::loglike>(ArrayUncheckedMutable & output,
    MatrixSUnchecked & grad_param_map, MatrixUnchecked & grad_param_factor, const size_t N_GAUSS,
    const std::vector<Weights> & gaussweights, double & loglike, const double model,
    const MatrixUnchecked & data, const MatrixUnchecked & variance_inv, unsigned int dim1, unsigned int dim2,
    const bool is_variance_matrix, const TermsPixel & terms_pixel, const TermsGradient & gradterms)
{
    double diff = data(dim1, dim2) - model;
    double diffvar = diff*variance_inv(dim1*is_variance_matrix, dim2*is_variance_matrix);
    /*
     * Derivation:
     *
        ll = sum(-(data-model)^2*varinv/2)
        dll/dx = --2*dmodel/dx*(data-model)*varinv/2
        dmodelsum/dx = d(.. + model[g])/dx = dmodel[g]/dx
        dll/dx = dmodel[g]/dx*diffvar
    */
    loglike -= diff*diffvar/2.;
    for(size_t g = 0; g < N_GAUSS; ++g)
    {
        const Weights & weights = gaussweights[g];
        gaussian_pixel_add_jacobian(
            output(grad_param_map(g, 0)), output(grad_param_map(g, 1)), output(grad_param_map(g, 2)),
            output(grad_param_map(g, 3)), output(grad_param_map(g, 4)), output(grad_param_map(g, 5)),
            grad_param_factor(g, 0), grad_param_factor(g, 1), grad_param_factor(g, 2),
            grad_param_factor(g, 3), grad_param_factor(g, 4), grad_param_factor(g, 5),
            weights[0]*diffvar, weights[1]*diffvar,
            (*(terms_pixel.xmc_weighted))[g], (*(terms_pixel.ymc_weighted[g]))[dim1],
            (*(terms_pixel.xmc))[g], (*(gradterms.ymc[g]))[dim1],
            (*(gradterms.yy_weight))[g], (*(gradterms.xmc_t_xy_weight))[g],
            (*(gradterms.sig_x_inv))[g], (*(gradterms.sig_y_inv))[g], weights[2],
            (*(terms_pixel.xmc_sq_norm))[g], (*(terms_pixel.yy_weighted[g]))[dim1],
            (*(gradterms.rho_factor))[g], (*(gradterms.rho_xy_factor))[g],
            (*(gradterms.sig_x_src_div_conv))[g], (*(gradterms.sig_y_src_div_conv))[g],
            (*(gradterms.drho_c_dsig_x_src))[g], (*(gradterms.drho_c_dsig_y_src))[g],
            (*(gradterms.drho_c_drho_s))[g]
        );
    }
}

template <GradientType gradient_type>
double gaussian_pixel_add_all(size_t g, size_t j, size_t i, double weight, const TermsPixel & terms_pixel,
    Array3UncheckedMutable & output_jac_ref, const MatrixSUnchecked & grad_param_map_ref,
    const MatrixUnchecked & grad_param_factor_ref, std::vector<Weights> & gradweights,
    const TermsGradient & gradterms)
{
    double xy = (*(terms_pixel.xmc))[g]*(*(terms_pixel.ymc_weighted[g]))[j];
    double value_unweight = (*(terms_pixel.weight))[g]*exp(
        -((*(terms_pixel.xmc_sq_norm))[g] + (*(terms_pixel.yy_weighted[g]))[j] - xy));
    double value = weight*value_unweight;
    gaussian_pixel_set_weights<gradient_type>(gradweights[g], value, value_unweight, xy);
    gaussian_pixel_add_jacobian_type<gradient_type>(output_jac_ref, grad_param_map_ref,
        grad_param_factor_ref, g, j, i, value, value_unweight,
        (*(terms_pixel.xmc_weighted))[g], (*(terms_pixel.ymc_weighted[g]))[j], (*(terms_pixel.xmc))[g],
        (*(gradterms.ymc[g]))[j], (*(gradterms.yy_weight))[g], (*(gradterms.xmc_t_xy_weight))[g],
        (*(gradterms.sig_x_inv))[g], (*(gradterms.sig_y_inv))[g], xy, (*(terms_pixel.xmc_sq_norm))[g],
        (*(terms_pixel.yy_weighted[g]))[j], (*(gradterms.rho_factor))[g], (*(gradterms.rho_xy_factor))[g],
        (*(gradterms.sig_x_src_div_conv))[g], (*(gradterms.sig_y_src_div_conv))[g],
        (*(gradterms.drho_c_dsig_x_src))[g], (*(gradterms.drho_c_dsig_y_src))[g],
        (*(gradterms.drho_c_drho_s))[g]);
    return value;
}

// Compute Gaussian mixtures with the option to write output and/or evaluate the log likehood
// TODO: Reconsider whether there's a better way to do this
// The template arguments ensure that there is no performance penalty to any of the versions of this function.
// However, some messy validation is required as a result.
template <OutputType output_type, bool getlikelihood, GradientType gradient_type>
double gaussians_pixel_template(const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const ndarray * const data, const ndarray * const variance_inv, ndarray * output = nullptr,
    ndarray * grads = nullptr, ndarray_s * grad_param_map = nullptr, ndarray * grad_param_factor = nullptr)
{
    check_is_gaussians(gaussians);
    const size_t DATASIZE = data == nullptr ? 0 : data->size();
    std::unique_ptr<ndarray> matrix_null;
    std::unique_ptr<ndarray> array_null;
    std::unique_ptr<ndarray> array3_null;
    std::unique_ptr<ndarray_s> matrix_s_null;
    const bool writeoutput = output_type != OutputType::none;
    const bool do_gradient = gradient_type != GradientType::none;
    if(writeoutput)
    {
        check_is_matrix(output, "output image");
        if(output->size() == 0) throw std::runtime_error("gaussians_pixel_template can't write to empty "
            "matrix");
    }
    if(!writeoutput or !do_gradient)
    {
        matrix_null = std::make_unique<ndarray>(pybind11::array::ShapeContainer({0, 0}));
    }
    if(gradient_type == GradientType::none)
    {
        matrix_s_null = std::make_unique<ndarray_s>(pybind11::array::ShapeContainer({0, 0}));
    }
    if(gradient_type != GradientType::loglike)
    {
        array_null = std::make_unique<ndarray>(pybind11::array::ShapeContainer({0}));
    }
    if(gradient_type != GradientType::jacobian)
    {
        array3_null = std::make_unique<ndarray>(pybind11::array::ShapeContainer({0, 0, 0}));
    }
    const size_t n_gaussians = gaussians.shape(0);
    std::unique_ptr<ndarray_s> grad_param_map_default;
    std::unique_ptr<ndarray> grad_param_factor_default;
    ArrayUncheckedMutable outputgradref = gradient_type == GradientType::loglike ?
        (*grads).mutable_unchecked<1>() : (*array_null).mutable_unchecked<1>();
    Array3UncheckedMutable output_jac_ref = gradient_type == GradientType::jacobian ?
        (*grads).mutable_unchecked<3>() : (*array3_null).mutable_unchecked<3>();
    const size_t n_params_type = gradient_type == GradientType::loglike ? outputgradref.shape(0) :
        (gradient_type == GradientType::jacobian ? output_jac_ref.shape(2) : 0);
    if(do_gradient)
    {
        const size_t n_params_grad = n_gaussians*N_PARAMS;
        const bool has_grad_param_map = grad_param_map != nullptr and (*grad_param_map).size() != 0;
        const bool has_grad_param_factor = grad_param_factor != nullptr and (*grad_param_factor).size() != 0;
        if(!has_grad_param_map)
        {
            if(n_params_type != n_params_grad)
            {
                throw std::runtime_error("Passed gradient vector of size=" + std::to_string(n_params_type) +
                   "!= default mapping size of ngaussiansx6=" + std::to_string(n_params_grad));
            }
            grad_param_map_default = std::make_unique<ndarray_s>(
                pybind11::array::ShapeContainer({n_gaussians, N_PARAMS}));
            grad_param_map = grad_param_map_default.get();
            size_t index = 0;
            MatrixSUncheckedMutable grad_param_map_ref = (*grad_param_map).mutable_unchecked<2>();
            for(size_t g = 0; g < n_gaussians; ++g)
            {
                for(size_t p = 0; p < N_PARAMS; ++p)
                {
                    grad_param_map_ref(g, p) = index++;
                }
            }
        }
        if(!has_grad_param_factor)
        {
            grad_param_factor_default = std::make_unique<ndarray>(
                pybind11::array::ShapeContainer({n_gaussians, N_PARAMS}));
            grad_param_factor = grad_param_factor_default.get();
            MatrixUncheckedMutable grad_param_factor_ref = (*grad_param_factor).mutable_unchecked<2>();
            for(size_t g = 0; g < n_gaussians; ++g)
            {
                for(size_t p = 0; p < N_PARAMS; ++p)
                {
                    grad_param_factor_ref(g, p) = 1.;
                }
            }
        }
        else
        {
            /*
            MatrixUnchecked grad_param_factor_ref = (*grad_param_factor).unchecked<2>();
            for(size_t g = 0; g < n_gaussians; ++g)
            {
                for(size_t p = 0; p < N_PARAMS; ++p)
                {
                    std::cout << grad_param_factor_ref(g, p) << ",";
                }
                std::cout << std::endl;
            }
            */
        }
        if(has_grad_param_map or has_grad_param_factor)
        {
            auto & grad_param_map_ref = *grad_param_map;
            auto & grad_param_factor_ref = *grad_param_factor;
            const size_t ndim_map = grad_param_map_ref.ndim();
            const size_t ndim_fac = grad_param_factor_ref.ndim();
            if(ndim_map != 2 or ndim_fac != 2) throw std::runtime_error(
                "grad_param_map ndim=" + std::to_string(ndim_map) + " and/or grad_param_factor ndim=" +
                std::to_string(ndim_fac) + " != 2");
            const size_t rows_map = grad_param_map_ref.shape(0);
            const size_t cols_map = grad_param_map_ref.shape(1);
            const size_t rows_fac = grad_param_factor_ref.shape(0);
            const size_t cols_fac = grad_param_factor_ref.shape(1);
            if(rows_map != n_gaussians or rows_map != rows_fac or cols_map != N_PARAMS or
                cols_map != cols_fac)
            {
                throw std::runtime_error("grad_param_map shape (" + std::to_string(rows_map) + "," +
                    std::to_string(cols_map) + ") and/or grad_param_factor shape (" +
                    std::to_string(rows_fac) + "," + std::to_string(cols_fac) + ") != (" +
                    std::to_string(n_gaussians) + "," + std::to_string(N_PARAMS) + ")");
            }
        }
    }
    if(getlikelihood)
    {
        check_is_matrix(data);
        check_is_matrix(variance_inv);
        if(DATASIZE == 0 || variance_inv->size() == 0) throw std::runtime_error("gaussians_pixel_template "
            "can't compute loglikelihood with empty data or variance_inv");
    }
    if(!writeoutput && !getlikelihood)
    {
        if(output->size() == 0 && DATASIZE == 0 && variance_inv->size() == 0) throw std::runtime_error(
            "gaussians_pixel_template can't infer size of matrix without one of data or output.");
    }

    const ndarray & matcomparesize = DATASIZE ? *data : *output;
    const unsigned int dim_x = matcomparesize.shape(1);
    const unsigned int dim_y = matcomparesize.shape(0);
    const bool is_variance_matrix = getlikelihood ? variance_inv->size() > 1 : false;

    if(gradient_type == GradientType::jacobian &&
        !(output_jac_ref.shape(0) == dim_y && output_jac_ref.shape(1) == dim_x))
    {
        throw std::runtime_error("Data/output matrix dimensions [" + std::to_string(dim_x) + ',' +
            std::to_string(dim_y) + "] don't match Jacobian matrix dimensions [" +
            std::to_string(output_jac_ref.shape(1)) + ',' + std::to_string(output_jac_ref.shape(0)) + ']');
    }

    if(getlikelihood)
    {
        // The case of constant variance per pixel
        if(is_variance_matrix == 1 && (dim_x != variance_inv->shape(1) || dim_y != variance_inv->shape(0)))
        {
            throw std::runtime_error("Data matrix dimensions [" + std::to_string(dim_x) + ',' +
                std::to_string(dim_y) + "] don't match inverse variance dimensions [" +
                std::to_string(variance_inv->shape(1)) + ',' + std::to_string(variance_inv->shape(0)) + ']');
        }
    }
    if(writeoutput)
    {
        if(dim_x != output->shape(1) || dim_y != output->shape(0))
        {
            throw std::runtime_error("Data matrix dimensions [" + std::to_string(dim_x) + ',' +
                std::to_string(dim_y) + "] don't match output matrix dimensions [" +
                std::to_string(output->shape(1)) + ',' + std::to_string(output->shape(0)) + ']');
        }
    }
    const double bin_x = (x_max-x_min)/dim_x;
    const double bin_y = (y_max-y_min)/dim_y;
    const double bin_x_half = bin_x/2.;

    TermsPixel terms_pixel(n_gaussians);
    // These are to store pre-computed values for gradients and are unused otherwise
    // TODO: Move these into a simple class/struct
    const size_t ngaussgrad = n_gaussians*(do_gradient);
    TermsGradient terms_grad(ngaussgrad);
    std::vector<Weights> weights_grad(n_gaussians*(gradient_type == GradientType::loglike),
        Weights());
    const MatrixUnchecked gaussians_ref = gaussians.unchecked<2>();
    for(size_t g = 0; g < n_gaussians; ++g)
    {
        const double cen_x = gaussians_ref(g, 0);
        const double cen_y = gaussians_ref(g, 1);
        const double L = gaussians_ref(g, 2);
        const Covar cov_psf = Covar{gaussians_ref(g, 6), gaussians_ref(g, 7), gaussians_ref(g, 8)};
        const Covar cov_src = Covar{gaussians_ref(g, 3), gaussians_ref(g, 4), gaussians_ref(g, 5)};
        const Covar cov = convolution(cov_src, cov_psf);
        // Deliberately omit luminosity for now
        const Terms terms = terms_from_covar(bin_x*bin_y, cov);

        auto yvals = gaussian_pixel_x_xx(cen_y, y_min, bin_y, dim_y, terms.yy, terms.xy);

        terms_pixel.set(g, terms.weight, x_min - cen_x + bin_x_half, terms.xx,
            std::move(yvals.x_norm), std::move(yvals.xx));
        if(do_gradient)
        {
            terms_grad.set(g, L, terms, cov_src, cov_psf, cov, std::move(yvals.x));
        }
    }
    /*
        Somewhat ugly hack here to set refs to point at null if we know they won't be used
        I would love to replace these by unique_ptrs or something and get rid of the nulls but I can't figure
         out how to construct an unchecked reference even after looking at the pybind11 source.
    */
    MatrixUncheckedMutable outputref = writeoutput ? (*output).mutable_unchecked<2>() :
        (*matrix_null).mutable_unchecked<2>();
    MatrixSUnchecked grad_param_map_ref = do_gradient ? (*grad_param_map).unchecked<2>() :
        (*matrix_s_null).unchecked<2>();
    MatrixUnchecked grad_param_factor_ref = do_gradient ? (*grad_param_factor).unchecked<2>() :
        (*matrix_null).unchecked<2>();
    const MatrixUnchecked data_ref = getlikelihood ? (*data).unchecked<2>() : gaussians_ref;
    const MatrixUnchecked variance_inv_ref = getlikelihood ? (*variance_inv).unchecked<2>() : gaussians_ref;
    double loglike = 0;
    double model = 0;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < dim_x; i++)
    {
        for(size_t g = 0; g < n_gaussians; ++g)
        {
            (*(terms_pixel.xmc_weighted))[g] = (*(terms_pixel.xmc))[g]*(*(terms_pixel.weight_xx))[g];
            (*(terms_pixel.xmc_sq_norm))[g] = (*(terms_pixel.xmc_weighted))[g]*(*(terms_pixel.xmc))[g];
            if(do_gradient) (*(terms_grad.xmc_t_xy_weight))[g] =
                (*(terms_pixel.xmc))[g]*(*(terms_grad.xy_weight))[g];
        }
        for(unsigned int j = 0; j < dim_y; j++)
        {
            model = 0;
            for(size_t g = 0; g < n_gaussians; ++g)
            {
                model += gaussian_pixel_add_all<gradient_type>(g, j, i, gaussians_ref(g, 2), terms_pixel,
                    output_jac_ref, grad_param_map_ref, grad_param_factor_ref, weights_grad, terms_grad);
            }
            gaussians_pixel_output<output_type>(outputref, model, j, i);
            gaussians_pixel_residual<output_type>(outputref, data_ref, model, j, i);
            gaussians_pixel_add_like<getlikelihood, gradient_type>(loglike, model, data_ref,
                variance_inv_ref, j, i, is_variance_matrix);
            gaussians_pixel_add_like_grad<getlikelihood, gradient_type>(outputgradref, grad_param_map_ref,
                grad_param_factor_ref, n_gaussians, weights_grad, loglike, model, data_ref, variance_inv_ref,
                j, i, is_variance_matrix, terms_pixel, terms_grad);
        }
        for(size_t g = 0; g < n_gaussians; ++g) (*(terms_pixel.xmc))[g] += bin_x;
    }
    return loglike;
}

GradientType get_gradient_type(const ndarray & grad)
{
    const auto ndim = grad.ndim() * (grad.size() > 0);
    return ndim == 1 ? GradientType::loglike : (ndim == 3 ? GradientType::jacobian : GradientType::none);
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type, bool get_likelihood>
double loglike_gaussians_pixel_getlike(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output, ndarray & grad, ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const GradientType gradient_type = get_gradient_type(grad);
    if(gradient_type == GradientType::loglike)
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::loglike>(
            gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
    else if (gradient_type == GradientType::jacobian)
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::jacobian>(
            gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
    else
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::none>(
            gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type>
double loglike_gaussians_pixel_output(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output, ndarray & grad, ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const bool get_likelihood = data.size() > 0;
    if(get_likelihood)
    {
        return loglike_gaussians_pixel_getlike<output_type, true>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
    else
    {
        return loglike_gaussians_pixel_getlike<output_type, false>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
}

/**
 * Compute the model and/or log-likehood and/or gradient (d(log-likehood)/dx) and/or Jacobian (dmodel/dx)
 * for a Gaussian mixture model.
 *
 * TODO: Consider override to compute LL and Jacobian, even if it's only useful for debug purposes.
 *
 * @param data 2D input image matrix.
 * @param variance_inv 2D inverse variance map of the same size as image.
 * @param gaussians N x 6 matrix of Gaussian parameters [cen_x, cen_y, flux, sigma_x, sigma_y, rho]
 * @param x_min x coordinate of the left edge of the box.
 * @param x_max x coordinate of the right edge of the box.
 * @param y_min y coordinate of the bottom edge of the box.
 * @param y_max y coordinate of the top edge of the box.
 * @param output 2D output matrix of the same size as image.
 * @param grad Output for gradients. Can either an M x 1 vector or M x image 3D Jacobian matrix,
 *    where M <= N x 6 to allow for condensing gradients based on grad_param_map.
 * @param grad_param_map Nx6 matrix of indices of grad to add each gradient to. For example, if four gaussians
 *    share the same cen_x, one could set grad_param_map[0:4,0] = 0. All values must be < grad.size().
 * @param grad_param_factor Nx6 matrix of multiplicative factors for each gradient term. For example, if a
 *    Gaussian is a sub-component of a multi-Gaussian component with a total flux parameter but fixed
 *    ratios, as in multi-Gaussian Sersic models.
 * @return The log likelihood.
 */
double loglike_gaussians_pixel(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max, bool to_add,
    ndarray & output, ndarray & grad,
    ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const GradientType gradient_type = get_gradient_type(grad);
    const OutputType output_type = output.size() > 0 ? (
            gradient_type == GradientType::jacobian ? OutputType::residual : (
            to_add ? OutputType::add : OutputType::overwrite)) :
        OutputType::none;
    if(output_type == OutputType::overwrite)
    {
        return loglike_gaussians_pixel_output<OutputType::overwrite>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
    else if(output_type == OutputType::add)
    {
        return loglike_gaussians_pixel_output<OutputType::add>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
    else if(output_type == OutputType::residual)
    {
        return loglike_gaussians_pixel_output<OutputType::residual>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
    else
    {
        return loglike_gaussians_pixel_output<OutputType::none>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, grad, grad_param_map, grad_param_factor);
    }
}

ndarray make_gaussians_pixel(
    const params_gauss& gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const unsigned int dim_x, const unsigned int dim_y)
{
    ndarray mat({dim_y, dim_x});
    gaussians_pixel_template<OutputType::overwrite, false, GradientType::none>(
        gaussians, x_min, x_max, y_min, y_max, nullptr, nullptr, &mat);
    return mat;
}

void add_gaussians_pixel(
    const params_gauss& gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output)
{
    gaussians_pixel_template<OutputType::add, true, GradientType::none>(
        gaussians, x_min, x_max, y_min, y_max, nullptr, nullptr, &output);
}
}
#endif
