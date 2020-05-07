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

#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

typedef pybind11::detail::unchecked_reference<double, 1l> ArrayUnchecked;
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

inline void check_type(const ndarray * data, std::string name, size_t n_dim)
{
    if(data == nullptr) throw std::invalid_argument("Passed null " + name + " to check_type");
    if(data->ndim() != n_dim)
    {
        throw std::invalid_argument(
            "Passed " + name + " with ndim=" + std::to_string(data->ndim()) + " !=" + std::to_string(n_dim));
    }
}

inline void check_is_array(const ndarray * data, std::string name = "array")
{
    check_type(data, name, 1);
}

inline void check_is_matrix(const ndarray * data, std::string name = "matrix")
{
    check_type(data, name, 2);
}

inline void check_is_jacobian(const ndarray * data, std::string name = "jacobian")
{
    check_type(data, name, 3);
}

inline void check_len(const ndarray & data, size_t min, size_t max, std::string name)
{
    const size_t size_data = data.size();
    if(!(size_data >= min & size_data <= max))
    {
        throw std::invalid_argument("ndarray " + name + " not " + std::to_string(min) + "<=" +
            std::to_string(size_data) + "<=" + std::to_string(max));
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

struct ValuesGauss
{
    double cen_x;
    double cen_y;
    double L;
    double sigma_x;
    double sigma_y;
    double rho;
};

typedef std::array<double, N_PARAMS> Weights;
enum class BackgroundType : unsigned char
{
    none      = 0,
    constant  = 1,
};
enum class OutputType : unsigned char
{
    none      = 0,
    overwrite = 1,
    add       = 2,
};
enum class GradientType : unsigned char
{
    none      = 0,
    loglike   = 1,
    jacobian  = 2,
};

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
    double weight = 0;
    double xmc = 0;
    double xmc_weighted = 0;
    vecdptr ymc_weighted = nullptr;
    double weight_xx = 0;
    double xmc_sq_norm = 0;
    vecdptr yy_weighted = nullptr;
    // TODO: Give these variables more compelling names

    TermsPixel(double weight_i=0, double xmc_i=0, double xmc_weighted_i=0, vecdptr ymc_weighted_i=nullptr,
        double weight_xx_i=0, double xmc_sq_norm_i=0, vecdptr yy_weighted_i=nullptr) :
        weight(weight_i), xmc(xmc_i), xmc_weighted(xmc_weighted_i), ymc_weighted(std::move(ymc_weighted_i)),
        weight_xx(weight_xx_i), xmc_sq_norm(xmc_sq_norm_i), yy_weighted(std::move(yy_weighted_i)) {}

    void set(double weight, double xmc, double xx, vecdptr ymc_weighted, vecdptr yy_weighted)
    {
        this->weight = weight;
        this->xmc = xmc;
        this->weight_xx = xx;
        this->ymc_weighted = std::move(ymc_weighted);
        this->yy_weighted = std::move(yy_weighted);
    }
};

typedef std::vector<TermsPixel> TermsPixelVec;

class TermsGradient
{
public:
    vecdptr ymc = nullptr;
    double xx_weight = 0;
    double xy_weight = 0;
    double yy_weight = 0;
    double rho_factor = 0;
    double sig_x_inv = 0;
    double sig_y_inv = 0;
    double rho_xy_factor = 0;
    double sig_x_src_div_conv = 0;
    double sig_y_src_div_conv = 0;
    double drho_c_dsig_x_src = 0;
    double drho_c_dsig_y_src = 0;
    double drho_c_drho_s = 0;
    double xmc_t_xy_weight = 0;

    TermsGradient(vecdptr ymc_i=nullptr, double xx_weight_i=0, double xy_weight_i=0, double yy_weight_i=0,
        double rho_factor_i=0, double sig_x_inv_i=0, double sig_y_inv_i=0, double rho_xy_factor_i=0,
        double sig_x_src_div_conv_i=0, double sig_y_src_div_conv_i=0, double drho_c_dsig_x_src_i=0,
        double drho_c_dsig_y_src_i=0, double drho_c_drho_s_i=0, double xmc_t_xy_weight_i=0) :
        ymc(std::move(ymc_i)), xx_weight(xx_weight_i), xy_weight(xy_weight_i), yy_weight(yy_weight_i),
        rho_factor(rho_factor_i), sig_x_inv(sig_x_inv_i), sig_y_inv(sig_y_inv_i),
        rho_xy_factor(rho_xy_factor_i), sig_x_src_div_conv(sig_x_src_div_conv_i),
        sig_y_src_div_conv(sig_y_src_div_conv_i), drho_c_dsig_x_src(drho_c_dsig_x_src_i),
        drho_c_dsig_y_src(drho_c_dsig_y_src_i), drho_c_drho_s(drho_c_drho_s_i),
        xmc_t_xy_weight(xmc_t_xy_weight_i) {}

    void set(const Terms & terms, const Covar & cov_src, const Covar & cov_psf,
        const Covar & cov, vecdptr ymc)
    {
        this->ymc = std::move(ymc);
        const double sig_xy = cov.sig_x*cov.sig_y;
        this->xx_weight = terms.xx;
        this->xy_weight = terms.xy;
        this->yy_weight = terms.yy;
        this->rho_factor = cov.rho/(1. - cov.rho*cov.rho);
        this->sig_x_inv = 1/cov.sig_x;
        this->sig_y_inv = 1/cov.sig_y;
        this->rho_xy_factor = 1./(1. - cov.rho*cov.rho)/sig_xy;
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
        this->sig_x_src_div_conv = sig_x_src_div_conv;
        this->sig_y_src_div_conv = sig_y_src_div_conv;
        double covar_psf_dsig_xy = cov_psf.rho*cov_psf.sig_x*cov_psf.sig_y/sig_xy;
        if(cov_psf.sig_x > 0)
        {
            double sig_p_ratio = cov_psf.sig_x/cov.sig_x;
            this->drho_c_dsig_x_src = cov_src.rho*sig_y_src_div_conv*sig_p_ratio*sig_p_ratio/cov.sig_x
                - covar_psf_dsig_xy*(cov_src.sig_x/cov.sig_x)/cov.sig_x;
        }
        if(cov_psf.sig_y > 0)
        {
            double sig_p_ratio = cov_psf.sig_y/cov.sig_y;
            this->drho_c_dsig_y_src = cov_src.rho*sig_x_src_div_conv*sig_p_ratio*sig_p_ratio/cov.sig_y
                - covar_psf_dsig_xy*(cov_src.sig_y/cov.sig_y)/cov.sig_y;
        }
        this->drho_c_drho_s = cov_src.sig_x*cov_src.sig_y/sig_xy;
    }
};

void validate_param_map_factor(const MatrixSUnchecked & param_map, const MatrixUnchecked & param_factor,
    const size_t n_rows, const size_t n_cols_map, const size_t n_cols_fac, const std::string name)
{
    const size_t ndim_map = param_map.ndim();
    const size_t ndim_fac = param_factor.ndim();
    if(ndim_map != 2 or ndim_fac != 2) throw std::runtime_error(
        name + " param_map ndim=" + std::to_string(ndim_map) + " and/or param_factor ndim=" +
        std::to_string(ndim_fac) + " != 2");
    const size_t rows_map = param_map.shape(0);
    const size_t cols_map = param_map.shape(1);
    const size_t rows_fac = param_factor.shape(0);
    const size_t cols_fac = param_factor.shape(1);
    if(rows_map != n_rows or rows_map != rows_fac or cols_map != n_cols_map or cols_fac != n_cols_fac)
    {
        throw std::runtime_error(name + " param_map/factor rows " + std::to_string(rows_map) + "," +
                                 std::to_string(rows_fac) + "!=" + std::to_string(n_rows) + " and/or cols " +
                                 std::to_string(cols_map) + "," + std::to_string(cols_fac) + "!=" +
                                 std::to_string(n_cols_map) + "," + std::to_string(n_cols_fac));
    }
}

typedef std::vector<TermsGradient> TermsGradientVec;

class GradientsSersic
{
private:
    MatrixSUnchecked param_map;
    MatrixUnchecked param_factor;
    size_t g_check = 0;

public:
    GradientsSersic(const ndarray_s & param_map_in, const ndarray & param_factor_in) :
        param_map(std::move(param_map_in.unchecked<2>())),
        param_factor(std::move(param_factor_in.unchecked<2>()))
    {
        validate_param_map_factor(param_map, param_factor, param_map.shape(0), 2, 3, "Sersic");
    }

    GradientsSersic() = delete;

    const inline void add_index_to_set(size_t g, std::set<size_t> & set)
    {
        set.insert(param_map(g, 1));
    }

    const inline void add(Array3UncheckedMutable & output, size_t g, size_t dim1, size_t dim2,
        const ValuesGauss & gradients)
    {
        // Reset g_check to zero once g rolls back to zero itself
        g_check *= (g != 0);
        const bool to_add = g == param_map(this->g_check, 0);
        const auto idx = param_map(g, 1);
        double value = gradients.L*param_factor(g, 0) + gradients.sigma_x*param_factor(g, 1) +
            gradients.sigma_y*param_factor(g, 2);
        output(dim1, dim2, idx) += value;
        this->g_check += to_add;
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
    const double bin_y_half=bin_y/2.;

    const double reff_y_inv = sin(ang*M_PI/180.)*sqrt(weight_sersic)/r_eff;
    const double reff_x_inv = cos(ang*M_PI/180.)*sqrt(weight_sersic)/r_eff;

    unsigned int i=0,j=0;
    auto matref = mat.mutable_unchecked<2>();
    x = x_min-cen_x+bin_x_half;
    for(i = 0; i < dim_x; i++)
    {
        y = y_min - cen_y + bin_y_half;
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

template <OutputType output_type>
inline void gaussians_pixel_output(MatrixUncheckedMutable & output, const double value, unsigned int dim1,
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

template <bool do_residual>
inline void gaussians_pixel_residual(MatrixUncheckedMutable & output, const MatrixUnchecked & data,
    double model, unsigned int dim1, unsigned int dim2) {};

template <>
inline void gaussians_pixel_residual<true>(MatrixUncheckedMutable & output,
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

inline void gaussian_pixel_get_jacobian(
    ValuesGauss & out,
    const double m, const double m_unweight,
    const double xmc_norm, const double ymc_weighted, const double xmc, const double ymc,
    const double norms_yy, const double xmc_t_xy_factor, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_weighted,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src = 1, const double dsig_y_conv_src = 1,
    const double drho_c_dsig_x_src = 0, const double drho_c_dsig_y_src = 0,
    const double drho_c_drho_s = 1)
{
    out.cen_x = m*(2*xmc_norm - ymc_weighted);
    out.cen_y = m*(2*ymc*norms_yy - xmc_t_xy_factor);
    out.L = m_unweight;
    const double one_p_xy = 1. + xy_norm;
    //df/dsigma_src = df/d_sigma_conv * dsigma_conv/dsigma_src
    const double dfdrho_c = m*(
            rho_div_one_m_rhosq*(1 - 2*(xx_norm + yy_weighted - xy_norm)) + xmc*ymc*norms_xy_div_rho);
    out.sigma_x = dfdrho_c*drho_c_dsig_x_src + dsig_x_conv_src*(m*sig_x_inv*(2*xx_norm - one_p_xy));
    out.sigma_y = dfdrho_c*drho_c_dsig_y_src + dsig_y_conv_src*(m*sig_y_inv*(2*yy_weighted - one_p_xy));
    //drho_conv/drho_src = sigmaxy_src/sigmaxy_conv
    out.rho = dfdrho_c*drho_c_drho_s;
}

inline void gaussian_pixel_get_jacobian_from_terms(
    ValuesGauss & out, const size_t dim1, const TermsPixel & terms_pixel,
    const TermsGradient & terms_grad, const double m, const double m_unweighted, const double xy_norm)
{
    gaussian_pixel_get_jacobian(out,
        m, m_unweighted,
        terms_pixel.xmc_weighted, (*terms_pixel.ymc_weighted)[dim1],
        terms_pixel.xmc, (*terms_grad.ymc)[dim1],
        terms_grad.yy_weight, terms_grad.xmc_t_xy_weight,
        terms_grad.sig_x_inv, terms_grad.sig_y_inv, xy_norm,
        terms_pixel.xmc_sq_norm, (*terms_pixel.yy_weighted)[dim1],
        terms_grad.rho_factor, terms_grad.rho_xy_factor,
        terms_grad.sig_x_src_div_conv, terms_grad.sig_y_src_div_conv,
        terms_grad.drho_c_dsig_x_src, terms_grad.drho_c_dsig_y_src,
        terms_grad.drho_c_drho_s
    );
}

inline void gaussian_pixel_add_values(
    double & cen_x, double & cen_y, double & L, double & sig_x, double & sig_y, double & rho,
    const ValuesGauss & values,
    const double weight_cen_x=1, const double weight_cen_y=1, const double weight_L=1,
    const double weight_sig_x=1, const double weight_sig_y=1, const double weight_rho=1)
{
    cen_x += weight_cen_x*values.cen_x;
    cen_y += weight_cen_y*values.cen_y;
    L += weight_L*values.L;
    sig_x += weight_sig_x*values.sigma_x;
    sig_y += weight_sig_y*values.sigma_y;
    rho += weight_rho*values.rho;
}

template <GradientType gradient_type, bool do_sersic>
inline void gaussian_pixel_add_jacobian_type(
    Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor,
    ValuesGauss & gradients, const size_t g, unsigned int dim1, unsigned int dim2,
    double value, double value_unweighted, double xy_norm,
    const TermsPixelVec & terms_pixel, const TermsGradientVec & terms_grad,
    GradientsSersic * grad_sersic) {};

template <>
inline void gaussian_pixel_add_jacobian_type<GradientType::jacobian, false>(
    Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor,
    ValuesGauss & gradients, const size_t g, unsigned int dim1, unsigned int dim2,
    double value, double value_unweighted, double xy_norm,
    const TermsPixelVec & terms_pixel, const TermsGradientVec & terms_grad,
    GradientsSersic * grad_sersic)
{
    gaussian_pixel_get_jacobian_from_terms(gradients, dim1, terms_pixel[g], terms_grad[g],
        value, value_unweighted, xy_norm);
    gaussian_pixel_add_values(
        output(dim1, dim2, grad_param_map(g, 0)), output(dim1, dim2, grad_param_map(g, 1)),
        output(dim1, dim2, grad_param_map(g, 2)), output(dim1, dim2, grad_param_map(g, 3)),
        output(dim1, dim2, grad_param_map(g, 4)), output(dim1, dim2, grad_param_map(g, 5)),
        gradients,
        grad_param_factor(g, 0), grad_param_factor(g, 1), grad_param_factor(g, 2),
        grad_param_factor(g, 3), grad_param_factor(g, 4), grad_param_factor(g, 5)
    );
}

template <>
inline void gaussian_pixel_add_jacobian_type<GradientType::jacobian, true>(
    Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor,
    ValuesGauss & gradients, const size_t g, unsigned int dim1, unsigned int dim2,
    double value, double value_unweighted, double xy_norm,
    const TermsPixelVec & terms_pixel, const TermsGradientVec & terms_grad,
    GradientsSersic * grad_sersic)
{
    gaussian_pixel_add_jacobian_type<GradientType::jacobian, false>(
        output, grad_param_map, grad_param_factor, gradients, g, dim1, dim2, value, value_unweighted, xy_norm,
        terms_pixel, terms_grad, grad_sersic);
    grad_sersic->add(output, g, dim1, dim2, gradients);
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
    const bool is_variance_matrix, const TermsPixelVec & terms_pixel, const TermsGradientVec & gradterms,
    ValuesGauss & gradients) {};

template <>
inline void gaussians_pixel_add_like_grad<true, GradientType::loglike>(ArrayUncheckedMutable & output,
    MatrixSUnchecked & grad_param_map, MatrixUnchecked & grad_param_factor, const size_t N_GAUSS,
    const std::vector<Weights> & gaussweights, double & loglike, const double model,
    const MatrixUnchecked & data, const MatrixUnchecked & variance_inv, unsigned int dim1, unsigned int dim2,
    const bool is_variance_matrix, const TermsPixelVec & terms_pixel, const TermsGradientVec & terms_grad,
    ValuesGauss & gradients)
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
        gaussian_pixel_get_jacobian_from_terms(gradients, dim1, terms_pixel[g], terms_grad[g],
            weights[0]*diffvar, weights[1]*diffvar, weights[2]);
        gaussian_pixel_add_values(
            output(grad_param_map(g, 0)), output(grad_param_map(g, 1)), output(grad_param_map(g, 2)),
            output(grad_param_map(g, 3)), output(grad_param_map(g, 4)), output(grad_param_map(g, 5)),
            gradients,
            grad_param_factor(g, 0), grad_param_factor(g, 1), grad_param_factor(g, 2),
            grad_param_factor(g, 3), grad_param_factor(g, 4), grad_param_factor(g, 5)
        );
    }
}

template <GradientType gradient_type, bool do_sersic>
inline double gaussian_pixel_add_all(size_t g, size_t j, size_t i, double weight,
    const TermsPixelVec & terms_pixel_vec, Array3UncheckedMutable & output_jac_ref,
    const MatrixSUnchecked & grad_param_map_ref, const MatrixUnchecked & grad_param_factor_ref,
    std::vector<Weights> & gradweights, const TermsGradientVec & terms_grad_vec, ValuesGauss & gradients,
    GradientsSersic * grad_sersic = nullptr)
{
    const TermsPixel & terms_pixel = terms_pixel_vec[g];
    const double xy_norm = terms_pixel.xmc*(*terms_pixel.ymc_weighted)[j];
    const double value_unweight = terms_pixel.weight*exp(
        -(terms_pixel.xmc_sq_norm + (*terms_pixel.yy_weighted)[j] - xy_norm));
    const double value = weight*value_unweight;
    gaussian_pixel_set_weights<gradient_type>(gradweights[g], value, value_unweight, xy_norm);
    gaussian_pixel_add_jacobian_type<gradient_type, do_sersic>(output_jac_ref, grad_param_map_ref,
        grad_param_factor_ref, gradients, g, j, i, value, value_unweight, xy_norm,
        terms_pixel_vec, terms_grad_vec, grad_sersic);
    return value;
}

template <GradientType gradient_type>
void reset_pixel(Array3UncheckedMutable & output_jac_ref, size_t dim1, size_t dim2,
    const std::vector<size_t> & grad_param_idx, size_t grad_param_idx_size) {};

template <>
void reset_pixel<GradientType::jacobian>(Array3UncheckedMutable & output_jac_ref, size_t dim1, size_t dim2,
    const std::vector<size_t> & grad_param_idx, size_t grad_param_idx_size)
{
    for(size_t k = 0; k < grad_param_idx_size; ++k)
    {
        output_jac_ref(dim1, dim2, grad_param_idx[k]) = 0;
    }
}

// Compute Gaussian mixtures with the option to write output and/or evaluate the log likehood
// TODO: Reconsider whether there's a better way to do this
// The template arguments ensure that there is no performance penalty to any of the versions of this function.
// However, some messy validation is required as a result.
template <
    OutputType output_type, bool getlikelihood, BackgroundType background_type, bool do_residual,
    GradientType gradient_type, bool do_sersic
>
double gaussians_pixel_template(const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const ndarray * const data, const ndarray * const variance_inv, ndarray * output = nullptr,
    ndarray * residual = nullptr, ndarray * grads = nullptr,
    ndarray_s * grad_param_map = nullptr, ndarray * grad_param_factor = nullptr,
    std::unique_ptr<GradientsSersic> grad_sersic = nullptr, ndarray * background = nullptr)
{
    check_is_gaussians(gaussians);
    const size_t DATASIZE = data == nullptr ? 0 : data->size();
    std::unique_ptr<ndarray> matrix_null;
    std::unique_ptr<ndarray> array_null;
    std::unique_ptr<ndarray> array3_null;
    std::unique_ptr<ndarray_s> matrix_s_null;
    const bool writeoutput = output_type != OutputType::none;
    const bool do_gradient = gradient_type != GradientType::none;
    double background_flat = 0;
    if(writeoutput)
    {
        check_is_matrix(output, "output image");
        if(output->size() == 0) throw std::runtime_error(
            "gaussians_pixel_template can't write model to empty matrix");
    }
    if(do_residual)
    {
        check_is_matrix(residual, "residual image");
        if(residual->size() == 0) throw std::runtime_error(
            "gaussians_pixel_template can't write residuals to empty matrix");
    }
    if(!writeoutput or !do_gradient)
    {
        matrix_null = std::make_unique<ndarray>(pybind11::array::ShapeContainer({0, 0}));
    }
    if(background_type == BackgroundType::constant)
    {
        check_is_array(background, "background");
        check_len(*background, 1, 1, "background");
        background_flat += (*background).at(0);
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
    std::vector<size_t> grad_param_idx;
    size_t grad_param_idx_size = 0;
    if(do_gradient)
    {
        const size_t n_params_grad = n_gaussians*(N_PARAMS) + (background_type == BackgroundType::constant);
        const bool has_grad_param_map = grad_param_map != nullptr and (*grad_param_map).size() != 0;
        const bool has_grad_param_factor = grad_param_factor != nullptr and (*grad_param_factor).size() != 0;
        if(!has_grad_param_map)
        {
            if(n_params_type != n_params_grad)
            {
                throw std::runtime_error("Passed gradient vector of size=" + std::to_string(n_params_type) +
                   "!= default mapping size of ngaussiansx6 + has_bg=" + std::to_string(n_params_grad));
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

    /*
    Somewhat ugly hack here to set refs to point at null if we know they won't be used
    I would love to replace these by unique_ptrs or something and get rid of the nulls but I can't figure
     out how to construct an unchecked reference even after looking at the pybind11 source.
    */
    MatrixUncheckedMutable outputref = writeoutput ? (*output).mutable_unchecked<2>() :
                                       (*matrix_null).mutable_unchecked<2>();
    MatrixUncheckedMutable residual_ref = do_residual ? (*residual).mutable_unchecked<2>() :
                                       (*matrix_null).mutable_unchecked<2>();
    MatrixSUnchecked grad_param_map_ref = do_gradient ? (*grad_param_map).unchecked<2>() :
                                          (*matrix_s_null).unchecked<2>();
    MatrixUnchecked grad_param_factor_ref = do_gradient ? (*grad_param_factor).unchecked<2>() :
                                            (*matrix_null).unchecked<2>();
    // I don't know if there's much overhead in calling get but there's no need to do it N times
    GradientsSersic * grad_sersic_ptr = grad_sersic.get();

    if(gradient_type == GradientType::jacobian)
    {
        if(!(output_jac_ref.shape(0) == dim_y && output_jac_ref.shape(1) == dim_x))
        {
            throw std::runtime_error("Data/output matrix dimensions [" + std::to_string(dim_x) + ',' +
                std::to_string(dim_y) + "] don't match Jacobian matrix dimensions [" +
                std::to_string(output_jac_ref.shape(1)) + ',' +
                std::to_string(output_jac_ref.shape(0)) + ']');
        }
        std::set<size_t> grad_param_idx_uniq;
        for (size_t g = 0; g < n_gaussians; ++g) {
            for (size_t p = 0; p < N_PARAMS; ++p) {
                grad_param_idx_uniq.insert(grad_param_map_ref(g, p));
            }
            if(grad_sersic_ptr != nullptr) grad_sersic_ptr->add_index_to_set(g, grad_param_idx_uniq);
        }
        std::copy(grad_param_idx_uniq.begin(), grad_param_idx_uniq.end(),
                  std::back_inserter(grad_param_idx));
        grad_param_idx_size = grad_param_idx.size();
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

    TermsPixelVec terms_pixel(n_gaussians);
    // These are to store pre-computed values for gradients and are unused otherwise
    // TODO: Move these into a simple class/struct
    const size_t ngaussgrad = n_gaussians*(do_gradient);
    TermsGradientVec terms_grad(ngaussgrad);
    std::vector<Weights> weights_grad(n_gaussians*(gradient_type == GradientType::loglike),
        Weights());
    const MatrixUnchecked gaussians_ref = gaussians.unchecked<2>();
    for(size_t g = 0; g < n_gaussians; ++g)
    {
        const double cen_x = gaussians_ref(g, 0);
        const double cen_y = gaussians_ref(g, 1);
        const Covar cov_psf = Covar{gaussians_ref(g, 6), gaussians_ref(g, 7), gaussians_ref(g, 8)};
        const Covar cov_src = Covar{gaussians_ref(g, 3), gaussians_ref(g, 4), gaussians_ref(g, 5)};
        const Covar cov = convolution(cov_src, cov_psf);
        // Deliberately omit luminosity for now
        const Terms terms = terms_from_covar(bin_x*bin_y, cov);

        auto yvals = gaussian_pixel_x_xx(cen_y, y_min, bin_y, dim_y, terms.yy, terms.xy);

        terms_pixel[g].set(terms.weight, x_min - cen_x + bin_x_half, terms.xx,
            std::move(yvals.x_norm), std::move(yvals.xx));
        if(do_gradient)
        {
            terms_grad[g].set(terms, cov_src, cov_psf, cov, std::move(yvals.x));
        }
    }
    if(do_gradient) validate_param_map_factor(grad_param_map_ref, grad_param_factor_ref,
        n_gaussians, N_PARAMS, N_PARAMS, "gradient");
    const MatrixUnchecked data_ref = getlikelihood ? (*data).unchecked<2>() : gaussians_ref;
    const MatrixUnchecked variance_inv_ref = getlikelihood ? (*variance_inv).unchecked<2>() : gaussians_ref;
    double loglike = 0;
    double model = 0;
    ValuesGauss gradients = {0, 0, 0, 0, 0, 0};
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < dim_x; i++)
    {
        for(size_t g = 0; g < n_gaussians; ++g)
        {
            terms_pixel[g].xmc_weighted = terms_pixel[g].xmc*terms_pixel[g].weight_xx;
            terms_pixel[g].xmc_sq_norm = terms_pixel[g].xmc_weighted*terms_pixel[g].xmc;
            if(do_gradient) terms_grad[g].xmc_t_xy_weight = terms_pixel[g].xmc*terms_grad[g].xy_weight;
        }
        for(unsigned int j = 0; j < dim_y; j++)
        {
            model = background_flat;
            reset_pixel<gradient_type>(output_jac_ref, j, i, grad_param_idx, grad_param_idx_size);
            for(size_t g = 0; g < n_gaussians; ++g)
            {
                model += gaussian_pixel_add_all<gradient_type, do_sersic>(g, j, i, gaussians_ref(g, 2),
                    terms_pixel, output_jac_ref, grad_param_map_ref, grad_param_factor_ref, weights_grad,
                    terms_grad, gradients, grad_sersic_ptr);
                //if(i == 57 && j == 88) std::cout << std::setprecision(18) << model << std::endl;
            }
            gaussians_pixel_output<output_type>(outputref, model, j, i);
            gaussians_pixel_residual<do_residual>(residual_ref, data_ref, model, j, i);
            gaussians_pixel_add_like<getlikelihood, gradient_type>(loglike, model, data_ref,
                variance_inv_ref, j, i, is_variance_matrix);
            gaussians_pixel_add_like_grad<getlikelihood, gradient_type>(outputgradref, grad_param_map_ref,
                grad_param_factor_ref, n_gaussians, weights_grad, loglike, model, data_ref, variance_inv_ref,
                j, i, is_variance_matrix, terms_pixel, terms_grad, gradients);
        }
        for(size_t g = 0; g < n_gaussians; ++g) terms_pixel[g].xmc += bin_x;
    }
    return loglike;
}

GradientType get_gradient_type(const ndarray & grad)
{
    const auto ndim = grad.ndim() * (grad.size() > 0);
    return ndim == 1 ? GradientType::loglike : (ndim == 3 ? GradientType::jacobian : GradientType::none);
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type, bool get_likelihood, BackgroundType background_type,
    bool do_residual, bool do_sersic>
double loglike_gaussians_pixel_getlike(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output, ndarray & residual, ndarray & grad, ndarray_s & grad_param_map,
    ndarray & grad_param_factor, std::unique_ptr<GradientsSersic> grad_sersic, ndarray & background)
{
    const GradientType gradient_type = get_gradient_type(grad);
    if(gradient_type == GradientType::loglike)
    {
        return gaussians_pixel_template<
            output_type, get_likelihood, background_type, do_residual, GradientType::loglike, do_sersic>(
                gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &residual, &grad,
                &grad_param_map, &grad_param_factor, std::move(grad_sersic), &background);
    }
    else if (gradient_type == GradientType::jacobian)
    {
        return gaussians_pixel_template<
            output_type, get_likelihood, background_type, do_residual, GradientType::jacobian, do_sersic>(
                gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &residual, &grad,
                &grad_param_map, &grad_param_factor, std::move(grad_sersic), &background);
    }
    else
    {
        return gaussians_pixel_template<
            output_type, get_likelihood, background_type, do_residual, GradientType::none, do_sersic>(
                gaussians, x_min, x_max, y_min, y_max, &data, &variance_inv, &output, &residual, &grad,
                &grad_param_map, &grad_param_factor, std::move(grad_sersic), &background);
    }
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type, bool get_likelihood, BackgroundType background_type>
double loglike_gaussians_pixel_sersic(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output, ndarray & residual, ndarray & grad, ndarray_s & grad_param_map,
    ndarray & grad_param_factor, std::unique_ptr<GradientsSersic> grad_sersic, ndarray & background)
{
    const bool do_sersic = grad_sersic != nullptr;
    const bool do_residual = residual.size() > 0;
    if(do_residual)
    {
        if(do_sersic)
        {
            return loglike_gaussians_pixel_getlike<
                output_type, get_likelihood, background_type, true, true>(
                    data, variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual, grad,
                    grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
        else
        {
            return loglike_gaussians_pixel_getlike<
                output_type, get_likelihood, background_type, true, false>(
                    data, variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual, grad,
                    grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
    } else {
        if(do_sersic)
        {
            return loglike_gaussians_pixel_getlike<
                output_type, get_likelihood, background_type, false, true>(
                    data, variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual, grad,
                    grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
        else
        {
            return loglike_gaussians_pixel_getlike<
                output_type, get_likelihood, background_type, false, false>(
                    data, variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual, grad,
                    grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
    }
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type>
double loglike_gaussians_pixel_output(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output, ndarray & residual, ndarray & grad, ndarray_s & grad_param_map,
    ndarray & grad_param_factor, std::unique_ptr<GradientsSersic> grad_sersic,
    ndarray & background)
{
    const bool has_background = background.size() > 0;
    const bool get_likelihood = data.size() > 0;
    if(get_likelihood)
    {
        if(has_background)
        {
            return loglike_gaussians_pixel_sersic<output_type, true, BackgroundType::constant>(data,
                variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual,
                grad, grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
        else
        {
            return loglike_gaussians_pixel_sersic<output_type, true, BackgroundType::none>(data,
                variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual,
                grad, grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
    }
    else
    {
        if(has_background)
        {
            return loglike_gaussians_pixel_sersic<output_type, false, BackgroundType::constant>(data,
                variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual,
                grad, grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
        else
        {
            return loglike_gaussians_pixel_sersic<output_type, false, BackgroundType::none>(data,
                variance_inv, gaussians, x_min, x_max, y_min, y_max, output, residual,
                grad, grad_param_map, grad_param_factor, std::move(grad_sersic), background);
        }
    }
}

/**
 * Compute the model and/or log-likehood and/or gradient (d(log-likehood)/dx) and/or Jacobian (dmodel/dx)
 * for a Gaussian mixture model.
 *
 * This function calls a series of templated functions with explicit instantiations. This is solely for the
 * purpose of avoiding having to manually write a series of nested conditionals. My hope is that
 * the templating will insert no-op functions wherever there's nothing to do instead of a needless branch
 * inside each pixel's loop, and that the compiler will actually inline everything for maximum performance.
 * Whether that actually happens or not is anyone's guess.
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
 * @param output 2D output matrix of the same size as image.
 * @return The log likelihood.
 */
double loglike_gaussians_pixel(
    const ndarray & data, const ndarray & variance_inv, const params_gauss & gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max, bool to_add,
    ndarray & output, ndarray & residual, ndarray & grad,
    ndarray_s & grad_param_map, ndarray & grad_param_factor,
    ndarray_s & sersic_param_map, ndarray & sersic_param_factor,
    ndarray & background)
{
    const OutputType output_type = output.size() > 0 ? (
            to_add ? OutputType::add : OutputType::overwrite) :
        OutputType::none;
    const bool do_sersic = sersic_param_map.size() > 0;
    std::unique_ptr<GradientsSersic> grad_sersic = do_sersic ?
        std::make_unique<GradientsSersic>(sersic_param_map, sersic_param_factor) : nullptr;
    if(output_type == OutputType::overwrite)
    {
        return loglike_gaussians_pixel_output<OutputType::overwrite>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, residual, grad, grad_param_map, grad_param_factor,
            std::move(grad_sersic), background);
    }
    else if(output_type == OutputType::add)
    {
        return loglike_gaussians_pixel_output<OutputType::add>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, residual, grad, grad_param_map, grad_param_factor,
            std::move(grad_sersic), background);
    }
    else
    {
        return loglike_gaussians_pixel_output<OutputType::none>(data, variance_inv, gaussians,
            x_min, x_max, y_min, y_max, output, residual, grad, grad_param_map, grad_param_factor,
            std::move(grad_sersic), background);
    }
}

ndarray make_gaussians_pixel(
    const params_gauss& gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    const unsigned int dim_x, const unsigned int dim_y)
{
    ndarray mat({dim_y, dim_x});
    gaussians_pixel_template<
        OutputType::overwrite, false, BackgroundType::none, false, GradientType::none, false>(
            gaussians, x_min, x_max, y_min, y_max, nullptr, nullptr, &mat);
    return mat;
}

void add_gaussians_pixel(
    const params_gauss& gaussians,
    const double x_min, const double x_max, const double y_min, const double y_max,
    ndarray & output)
{
    gaussians_pixel_template<
        OutputType::add, true, BackgroundType::none, false, GradientType::none, false>(
            gaussians, x_min, x_max, y_min, y_max, nullptr, nullptr, &output);
}
}
#endif
