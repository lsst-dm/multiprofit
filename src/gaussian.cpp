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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <type_traits>

template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

typedef pybind11::detail::unchecked_reference<double, 2l> MatrixUnchecked;
typedef pybind11::detail::unchecked_reference<size_t, 2l> MatrixSUnchecked;
typedef pybind11::detail::unchecked_mutable_reference<double, 1l> ArrayUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<size_t, 2l> MatrixSUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<double, 2l> MatrixUncheckedMutable;
typedef pybind11::detail::unchecked_mutable_reference<double, 3l> Array3UncheckedMutable;

namespace multiprofit {

const double inf = std::numeric_limits<double>::infinity();

typedef std::vector<double> vecd;
typedef std::vector<vecd> vecvecd;
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

const size_t NGAUSSPARAMS = 6;
const size_t NGAUSSPARAMSCONV = 9;
inline void check_is_gaussians(const ndarray & mat, bool isconv=true)
{
    check_is_matrix(&mat, "Gaussian parameter matrix");
    const size_t LENGTH = isconv ? NGAUSSPARAMSCONV : NGAUSSPARAMS;
    if(mat.shape(1) != LENGTH)
    {
        throw std::invalid_argument("Passed Gaussian parameter matrix with shape=[" +
            std::to_string(mat.shape(0)) + ", " + std::to_string(mat.shape(1)) + "!=" +
            std::to_string(LENGTH) + "]");
    }
}

struct GaussianMoments
{
    vecdptr x;
    vecdptr x_norm;
    vecdptr xx;
};

inline GaussianMoments
gaussian_pixel_x_xx(const double XCEN, const double XMIN, const double XBIN, const unsigned int XDIM,
    const double XXNORMINV, const double XYNORMINV)
{
    const double XINIT = XMIN - XCEN + XBIN/2.;
    vecdptr x = std::make_unique<vecd>(XDIM);
    vecdptr xx = std::make_unique<vecd>(XDIM);
    vecdptr x_norm = std::make_unique<vecd>(XDIM);
    for(unsigned int i = 0; i < XDIM; i++)
    {
        double dist = XINIT + i*XBIN;
        (*x)[i] = dist;
        (*xx)[i] = dist*dist*XXNORMINV;
        (*x_norm)[i] = dist*XYNORMINV;
    }
    return GaussianMoments({std::move(x), std::move(x_norm), std::move(xx)});
}

inline void gaussian_pixel(ndarray & mat, const double NORM, const double XCEN, const double YCEN,
    const double XMIN, const double YMIN, const double XBIN, const double YBIN,
    const double XXNORMINV, const double YYNORMINV, const double XYNORMINV)
{
    check_is_matrix(&mat);
    // don't ask me why these are reversed
    const unsigned int XDIM = mat.shape(1);
    const unsigned int YDIM = mat.shape(0);

    const auto YVALS = gaussian_pixel_x_xx(YCEN, YMIN, YBIN, YDIM, YYNORMINV, XYNORMINV);
    const vecd & YN = *(YVALS.x_norm);
    const vecd & YY = *(YVALS.xx);

    auto matref = mat.mutable_unchecked<2>();
    double x = XMIN-XCEN+XBIN/2.;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < XDIM; i++)
    {
        const double XSQ = x*x*XXNORMINV;
        for(unsigned int j = 0; j < YDIM; j++)
        {
           matref(j, i) = NORM*exp(-(XSQ + YY[j] - x*YN[j]));
        }
        x += XBIN;
    }
}

/*
This is some largely unnecessary algebra to derive conversions between the ellipse parameterization and the
covariance matrix parameterization of a bivariate Gaussian.

See e.g. http://mathworld.wolfram.com/BivariateNormalDistribution.html
... and https://www.unige.ch/sciences/astro/files/5413/8971/4090/2_Segransan_StatClassUnige.pdf

tan 2th = 2*rho*sigx*sigy/(sigx^2 - sigy^2)

sigma_maj^2 = (cos2t*sigx^2 - sin2t*sigy^2)/(cos2t-sin2t)
sigma_min^2 = (cos2t*sigy^2 - sin2t*sigx^2)/(cos2t-sin2t)

(sigma_maj^2*(cos2t-sin2t) + sin2t*sigy^2)/cos2t = sigx^2
-(sigma_min^2*(cos2t-sin2t) - cos2t*sigy^2)/sin2t = sigx^2

(sigma_maj^2*(cos2t-sin2t) + sin2t*sigy^2)/cos2t = -(sigma_min^2(cos2t-sin2t) - cos2t*sigy^2)/sin2t
sin2t*(sigma_maj^2*(cos2t-sin2t) + sin2t*sigy^2) + cos2t*(sigma_min^2*(cos2t-sin2t) - cos2t*sigy^2) = 0
cos4t*sigy^2 - sin^4th*sigy^2 = sin2t*(sigma_maj^2*(cos2t-sin2t)) + cos2t*(sigma_min^2*(cos2t-sin2t))

cos^4x - sin^4x = (cos^2 + sin^2)*(cos^2-sin^2) = cos^2 - sin^2

sigy^2 = (sin2t*(sigma_maj^2*(cos2t-sin2t)) + cos2t*(sigma_min^2*(cos2t-sin2t)))/(cos2t - sin2t)
       = (sin2t*sigma_maj^2 + cos2t*sigma_min^2)

sigma_maj^2*(cos2t-sin2t) + sin2t*sigy^2 = cos2t*sigx^2
sigx^2 = (sigma_maj^2*(cos2t-sin2t) + sin2t*sigy^2)/cos2t
       = (sigma_maj^2*(cos2t-sin2t) + sin2t*(sin2t*sigma_maj^2 + cos2t*sigma_min^2))/cos2t
       = (sigma_maj^2*(cos2t-sin2t+sin4t) + sin2tcos2t*sigma_min^2)/cos2t
       = (sigma_maj^2*cos4t + sin2t*cos2t*sigma_min^2)/cos2t
       = (sigma_maj^2*cos2t + sigma_min^2*sin2t)

sigx^2 - sigy^2 = sigma_maj^2*cos2t + sigma_min^2*sin2t - sigma_maj^2*sin2t - sigma_min^2*cos2t
                = (sigma_maj^2 - sigma_min^2*)*(cos2t-sin2t)
                = (sigma_maj^2 - sigma_min^2*)*(1-tan2t)*cos2t

rho = tan2th/2/(sigx*sigy)*(sigx^2 - sigy^2)
    = tanth/(1-tan2t)/(sigx*sigy)*(sigx^2 - sigy^2)
    = tanth/(1-tan2t)/(sigx*sigy)*(sigma_maj^2 - sigma_min^2*)*(1-tan2t)*cos2t
    = tanth/(sigx*sigy)*(sigma_maj^2 - sigma_min^2)*cos2t
    = sint*cost/(sigx*sigy)*(sigma_maj^2 - sigma_min^2)
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
    double sigx;
    double sigy;
    double rho;
};

Covar ellipse_to_covar(const double SIGMAMAJ, const double AXRAT, const double ANGINRAD)
{
    const double SIGMAMIN = SIGMAMAJ*AXRAT;
    // TODO: Check optimal order for precision
    const double SIGMAMAJSQ = SIGMAMAJ * SIGMAMAJ;
    const double SIGMAMINSQ = SIGMAMIN * SIGMAMIN;

    const double SINT = sin(ANGINRAD);
    const double COST = cos(ANGINRAD);
    // TODO: Remember if this is actually preferable to sin/cos
    const double SINSQT = std::pow(SINT, 2.0);
    const double COSSQT = std::pow(COST, 2.0);
    const bool ISCIRCLE = AXRAT == 1;
    const double SIGX = ISCIRCLE ? SIGMAMAJ : sqrt(COSSQT*SIGMAMAJSQ + SINSQT*SIGMAMINSQ);
    const double SIGY = ISCIRCLE ? SIGMAMAJ : sqrt(SINSQT*SIGMAMAJSQ + COSSQT*SIGMAMINSQ);

    Covar rval = {
        .sigx = SIGX,
        .sigy = SIGY,
        .rho = ISCIRCLE ? 0 : SINT*COST/SIGX/SIGY*(SIGMAMAJSQ-SIGMAMINSQ)
    };
    return rval;
}

Covar convolution(const Covar & C, const Covar & K)
{
    const double sigx = sqrt(C.sigx*C.sigx + K.sigx*K.sigx);
    const double sigy = sqrt(C.sigy*C.sigy + K.sigy*K.sigy);
    return Covar{sigx, sigy, (C.rho*C.sigx*C.sigy + K.rho*K.sigx*K.sigy)/(sigx*sigy)};
}


// Various multiplicative terms that appread in a Gaussian PDF
struct TermsGaussPDF
{
    double norm;
    double xx;
    double yy;
    double xy;
};

TermsGaussPDF terms_from_covar(const double NORM, const Covar & COV)
{
    const double EXPNORM = 1./(2*(1-COV.rho*COV.rho));
    TermsGaussPDF rval = {
        .norm = NORM/(2.*M_PI*COV.sigx*COV.sigy)*sqrt(2.*EXPNORM),
        .xx = EXPNORM/COV.sigx/COV.sigx,
        .yy = EXPNORM/COV.sigy/COV.sigy,
        .xy = 2.*COV.rho*EXPNORM/COV.sigx/COV.sigy
    };
    return rval;
}

class GaussianTerms
{
public:
    vecdptr norms = nullptr;
    vecdptr xmc = nullptr;
    vecdptr xmc_norm = nullptr;
    vecdptrvec ymc_norm;
    vecdptr norms_xx = nullptr;
    vecdptr xsqs = nullptr;
    vecdptrvec yy_norm;
    // TODO: Give these variables more compelling names

    GaussianTerms(size_t ngauss)
    {
        this->norms = std::make_unique<vecd>(ngauss);
        this->xmc = std::make_unique<vecd>(ngauss);
        this->xmc_norm = std::make_unique<vecd>(ngauss);
        ymc_norm.resize(ngauss);
        this->norms_xx = std::make_unique<vecd>(ngauss);
        this->xsqs = std::make_unique<vecd>(ngauss);
        yy_norm.resize(ngauss);
    }

    void set(size_t g, double norm, double xmc, double xx, vecdptr y_norm, vecdptr yy)
    {
        (*(this->norms))[g] = norm;
        (*(this->xmc))[g] = xmc;
        (*(this->norms_xx))[g] = xx;
        this->ymc_norm[g] = std::move(y_norm);
        this->yy_norm[g] = std::move(yy);
    }
};

class GradientTerms
{
public:
    vecdptrvec ymc;
    vecdptr xx_factor = nullptr;
    vecdptr xy_factor = nullptr;
    vecdptr yy_factor = nullptr;
    vecdptr rho_factor = nullptr;
    vecdptr sig_x_inv = nullptr;
    vecdptr sig_y_inv = nullptr;
    vecdptr rho_xy_factor = nullptr;
    vecdptr sig_x_src_div_conv = nullptr;
    vecdptr sig_y_src_div_conv = nullptr;
    vecdptr drho_c_dsig_x_src = nullptr;
    vecdptr drho_c_dsig_y_src = nullptr;
    vecdptr drho_c_drho_s = nullptr;
    vecdptr xsnormxy = nullptr;

    GradientTerms(size_t ngauss = 0)
    {
        if(ngauss > 0)
        {
            this->ymc.resize(ngauss);
            this->xx_factor = std::make_unique<vecd>(ngauss);
            this->xy_factor = std::make_unique<vecd>(ngauss);
            this->yy_factor = std::make_unique<vecd>(ngauss);
            this->rho_factor = std::make_unique<vecd>(ngauss);
            this->sig_x_inv = std::make_unique<vecd>(ngauss);
            this->sig_y_inv = std::make_unique<vecd>(ngauss);
            this->rho_xy_factor = std::make_unique<vecd>(ngauss);
            this->sig_x_src_div_conv = std::make_unique<vecd>(ngauss);
            this->sig_y_src_div_conv = std::make_unique<vecd>(ngauss);
            this->drho_c_dsig_x_src = std::make_unique<vecd>(ngauss);
            this->drho_c_dsig_y_src = std::make_unique<vecd>(ngauss);
            this->drho_c_drho_s = std::make_unique<vecd>(ngauss);
            this->xsnormxy = std::make_unique<vecd>(ngauss);
        }
    }

    void set(size_t g, double l, const TermsGaussPDF & TERMS, const Covar & COVSRC, const Covar & COVPSF,
        const Covar & COV, vecdptr ymc)
    {
        this->ymc[g] = std::move(ymc);
        const double SIGXY = COV.sigx*COV.sigy;
        (*(this->xx_factor))[g] = TERMS.xx;
        (*(this->xy_factor))[g] = TERMS.xy;
        (*(this->yy_factor))[g] = TERMS.yy;
        (*(this->rho_factor))[g] = COV.rho/(1. - COV.rho*COV.rho);
        (*(this->sig_x_inv))[g] = 1/COV.sigx;
        (*(this->sig_y_inv))[g] = 1/COV.sigy;
        (*(this->rho_xy_factor))[g] = 1./(1. - COV.rho*COV.rho)/SIGXY;
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
        double sig_x_src_div_conv = COVSRC.sigx/COV.sigx;
        double sig_y_src_div_conv = COVSRC.sigy/COV.sigy;
        (*(this->sig_x_src_div_conv))[g] = sig_x_src_div_conv;
        (*(this->sig_y_src_div_conv))[g] = sig_y_src_div_conv;
        double offdiagpsf_dsigxy = COVPSF.rho*COVPSF.sigx*COVPSF.sigy/SIGXY;
        if(COVPSF.sigx > 0)
        {
            double sig_p_ratio = COVPSF.sigx/COV.sigx;
            (*(this->drho_c_dsig_x_src))[g] = COVSRC.rho*sig_y_src_div_conv*sig_p_ratio*sig_p_ratio/COV.sigx -
                offdiagpsf_dsigxy*(COVSRC.sigx/COV.sigx)/COV.sigx;
        }
        if(COVPSF.sigy > 0)
        {
            double sig_p_ratio = COVPSF.sigy/COV.sigy;
            (*(this->drho_c_dsig_y_src))[g] = COVSRC.rho*sig_x_src_div_conv*sig_p_ratio*sig_p_ratio/COV.sigy -
                offdiagpsf_dsigxy*(COVSRC.sigy/COV.sigy)/COV.sigy;
        }
        (*(this->drho_c_drho_s))[g] = COVSRC.sigx*COVSRC.sigy/SIGXY;
    }
};

// Evaluate a Gaussian on a grid given the three elements of the symmetric covariance matrix
// Actually, rho is scaled by sigx and sigy (i.e. the covariance is RHO*SIGX*SIGY)
ndarray make_gaussian_pixel_covar(const double XCEN, const double YCEN, const double L,
    const double SIGX, const double SIGY, const double RHO,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;
    const Covar COV {.sigx = SIGX, .sigy=SIGY, .rho=RHO};
    const TermsGaussPDF TERMS = terms_from_covar(L*XBIN*YBIN, COV);

    ndarray mat({YDIM, XDIM});
    gaussian_pixel(mat, TERMS.norm, XCEN, YCEN, XMIN, YMIN, XBIN, YBIN, TERMS.xx, TERMS.yy, TERMS.xy);

    return mat;
}

// Evaluate a Gaussian on a grid given R_e, the axis ratio and position angle
ndarray make_gaussian_pixel(
    const double XCEN, const double YCEN, const double L,
    const double R, const double AXRAT, const double ANG,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    const Covar COV = ellipse_to_covar(reff_to_sigma_gauss(R), AXRAT, degrees_to_radians(ANG));

// Verify transformations
// TODO: Move this to a test somewhere and fix inverse transforms to work over the whole domain
/*
    std::cout << SIGMAMAJ << "," << SIGMAMIN << "," << ANGRAD << std::endl;
    std::cout << SIGX << "," << SIGY << "," << RHO << std::endl;
    std::cout << sqrt((COSSQT*SIGXSQ - SINSQT*SIGYSQ)/(COSSQT-SINSQT)) << "," <<
        sqrt((COSSQT*SIGYSQ - SINSQT*SIGXSQ)/(COSSQT-SINSQT)) << "," <<
        atan(2*RHO*SIGX*SIGY/(SIGXSQ-SIGYSQ))/2. << std::endl;
*/
    return make_gaussian_pixel_covar(XCEN, YCEN, L, COV.sigx, COV.sigy, COV.rho, XMIN, XMAX, YMIN, YMAX,
        XDIM, YDIM);
}

ndarray make_gaussian_pixel_sersic(
    const double XCEN, const double YCEN, const double L,
    const double R, const double AXRAT, const double ANG,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    // I don't remember why this isn't just 1/(2*ln(2)) but anyway it isn't
    const double NORMRFAC = 0.69314718055994528622676398299518;
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;
    const double NORM = L*NORMRFAC/(M_PI*AXRAT)/R/R*XBIN*YBIN;
    const double INVAXRATSQ = 1.0/AXRAT/AXRAT;

    ndarray mat({YDIM, XDIM});
    double x,y;
    const double XBINHALF=XBIN/2.;
    const double YBINHALF=YBIN/2.;

    const double INVREY = sin(ANG*M_PI/180.)*sqrt(NORMRFAC)/R;
    const double INVREX = cos(ANG*M_PI/180.)*sqrt(NORMRFAC)/R;

    unsigned int i=0,j=0;
    auto matref = mat.mutable_unchecked<2>();
    x = XMIN-XCEN+XBINHALF;
    for(i = 0; i < XDIM; i++)
    {
        y = YMIN - YCEN + YBINHALF;
        for(j = 0; j < YDIM; j++)
        {
           const double DISTONE = (x*INVREX + y*INVREY);
           const double DISTTWO = (x*INVREY - y*INVREX);
           // mat(j,i) = ... is slower, but perhaps allows for images with XDIM*YDIM > INT_MAX ?
           matref(j, i) = NORM*exp(-(DISTONE*DISTONE + DISTTWO*DISTTWO*INVAXRATSQ));
           y += YBIN;
        }
        x += XBIN;
    }

    return mat;
}

typedef std::array<double, NGAUSSPARAMS> GaussWeights;
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
inline void gaussians_pixel_residual(MatrixUncheckedMutable & output, const MatrixUnchecked & DATA,
    double model, unsigned int dim1, unsigned int dim2) {};

template <>
inline void gaussians_pixel_residual<OutputType::residual>(MatrixUncheckedMutable & output,
    const MatrixUnchecked & DATA, const double model, unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) = DATA(dim1, dim2) - model;
}

template <bool getlikelihood, GradientType gradient_type>
inline void gaussians_pixel_add_like(double & loglike, const double model, const MatrixUnchecked & DATA,
    const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2, const bool VARISMAT) {};

template <>
inline void gaussians_pixel_add_like<true, GradientType::none>(double & loglike, const double model,
    const MatrixUnchecked & DATA, const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2,
    const bool VARISMAT)
{
    double diff = DATA(dim1, dim2) - model;
    // TODO: Check what the performance penalty for using IDXVAR is and rewrite if it's worth it
    loglike -= (diff*(diff*VARINVERSE(dim1*VARISMAT, dim2*VARISMAT)))/2.;
}

/*
    m = L*XBIN*YBIN/(2.*M_PI*COV.sigx*COV.sigy)/sqrt(1-COV.rho*COV.rho) * exp(-(
        + xmc[g]^2/COV.sigx^2/(2*(1-COV.rho*COV.rho))
        + ymc[g][j]^2/COV.sigy^2/(2*(1-COV.rho*COV.rho))
        - COV.rho*xmc[g]*ymc[g][j]/(1-COV.rho*COV.rho)/COV.sigx/COV.sigy
    ))

        rho -> r; X/YBIN -> B_1/2, sigx/y -> s_1/2, X/YCEN -> c_1,2

    m = L*B_1*B_2/(2*pi*s_1*s_2)/sqrt(1-r^2)*exp(-(
        + (X-C_1)^2/s_1^2/2/(1-r^2)
        + (Y-C_2)^2/s_2^2/2/(1-r^2)
        - r*(X-C_1)*(Y-C_2)/(1-r^2)/s_1/s_2
    ))

    dm/dL = m/L
    dm/dXCEN = (2*xmc[g]*norms_xx[g] - ymc[g][j])*m
    dm/ds_[12] = -m/s + m*(2/s*[xy]sqs[g] - xy/s) = m/s*(2*[xy]sqs[g] - (1+xy))
    dm/dr = -m*(r/(1-r^2) + 4*r*(1-r^2)*(xsqs[g] + yy_norm[g][j]) - (1/r + 2*r/(1+r)*xmc[g]*ymc[g][j])

    To verify (all on one line):

    https://www.wolframalpha.com/input/?i=differentiate+F*s*t%2F(2*pi*sqrt(1-r%5E2))*
    exp(-((x-m)%5E2*s%5E2+%2B+(y-n)%5E2*t%5E2+-+2*r*(x-m)*(y-n)*s*t)%2F(2*(1-r%5E2)))+wrt+s

*/
inline void gaussian_pixel_add_jacobian(
    double & cenx, double & ceny, double & l, double & sigx, double & sigy, double & rho,
    const double cenxweight, const double cenyweight, const double lweight,
    const double sigxweight, const double sigyweight, const double rhoweight,
    const double m, const double m_unweight,
    const double xmc_norm, const double ymc_norm, const double xmc, const double ymc,
    const double norms_yy, const double xsnormxy, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_norm,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src = 1, const double dsig_y_conv_src = 1,
    const double drho_c_dsig_x_src = 0, const double drho_c_dsig_y_src = 0,
    const double drho_c_drho_s = 1)
{
    cenx += cenxweight*m*(2*xmc_norm - ymc_norm);
    ceny += cenyweight*m*(2*ymc*norms_yy - xsnormxy);
    l += lweight*m_unweight;
    const double onepxy = 1. + xy_norm;
    //df/dsigma_src = df/d_sigma_conv * dsigma_conv/dsigma_src
    const double dfdrho_c = m*(
            rho_div_one_m_rhosq*(1 - 2*(xx_norm + yy_norm - xy_norm)) + xmc*ymc*norms_xy_div_rho);
    sigx += sigxweight*(dfdrho_c*drho_c_dsig_x_src + dsig_x_conv_src*(m*sig_x_inv*(2*xx_norm - onepxy)));
    sigy += sigyweight*(dfdrho_c*drho_c_dsig_y_src + dsig_y_conv_src*(m*sig_y_inv*(2*yy_norm - onepxy)));
    //drho_conv/drho_src = sigmaxy_src/sigmaxy_conv
    rho += rhoweight*dfdrho_c*drho_c_drho_s;
}

template <GradientType gradient_type>
inline void gaussian_pixel_add_jacobian_type(Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor, const size_t g,
    unsigned int dim1, unsigned int dim2, const double m, const double m_unweight,
    const double xmc_norm, const double ymc_norm, const double xmc, const double ymc,
    const double norms_yy, const double xsnormxy, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_norm,
    const double rho_div_one_m_rhosq, const double norms_xy_div_rho,
    const double dsig_x_conv_src, const double dsig_y_conv_src,
    const double drho_dsig_x_src, const double drho_dsig_y_src,
    const double drho_c_drho_s) {};

template <>
inline void gaussian_pixel_add_jacobian_type<GradientType::jacobian>(Array3UncheckedMutable & output,
    const MatrixSUnchecked & grad_param_map, const MatrixUnchecked & grad_param_factor, const size_t g,
    unsigned int dim1, unsigned int dim2, const double m, const double m_unweight,
    const double xmc_norm, const double ymc_norm, const double xmc, const double ymc,
    const double norms_yy, const double xsnormxy, const double sig_x_inv, const double sig_y_inv,
    const double xy_norm, const double xx_norm, const double yy_norm,
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
        m, m_unweight, xmc_norm, ymc_norm, xmc, ymc, norms_yy, xsnormxy, sig_x_inv, sig_y_inv,
        xy_norm, xx_norm, yy_norm, rho_div_one_m_rhosq, norms_xy_div_rho,
        dsig_x_conv_src, dsig_y_conv_src, drho_dsig_x_src, drho_dsig_y_src, drho_c_drho_s
    );
}


// Computes dmodel/dx for x in [cenx, ceny, flux, sigma_x, sigma_y, rho]
template <GradientType gradient_type>
inline void gaussian_pixel_set_weights(GaussWeights & output, const double m,
    const double m_unweight, const double xy) {};

template <>
inline void gaussian_pixel_set_weights<GradientType::loglike>(GaussWeights & output, const double m,
    const double m_unweight, const double xy)
{
    output[0] = m;
    output[1] = m_unweight;
    output[2] = xy;
}

// Computes and stores LL along with dll/dx for all components
template <bool getlikelihood, GradientType gradient_type>
inline void gaussians_pixel_add_like_grad(ArrayUncheckedMutable & output,
    MatrixSUnchecked & grad_param_map, MatrixUnchecked & grad_param_factor, const size_t ngauss,
    const std::vector<GaussWeights> & gaussweights, double & loglike, const double model,
    const MatrixUnchecked & DATA, const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2,
    const bool VARISMAT, const GaussianTerms & gaussterms, const GradientTerms & gradterms) {};

template <>
inline void gaussians_pixel_add_like_grad<true, GradientType::loglike>(ArrayUncheckedMutable & output,
    MatrixSUnchecked & grad_param_map, MatrixUnchecked & grad_param_factor, const size_t ngauss,
    const std::vector<GaussWeights> & gaussweights, double & loglike, const double model,
    const MatrixUnchecked & DATA, const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2,
    const bool VARISMAT, const GaussianTerms & gaussterms, const GradientTerms & gradterms)
{
    double diff = DATA(dim1, dim2) - model;
    double diffvar = diff*VARINVERSE(dim1*VARISMAT, dim2*VARISMAT);
    /*
     * Derivation:
     *
        ll = sum(-(data-model)^2*varinv/2)
        dll/dx = --2*dmodel/dx*(data-model)*varinv/2
        dmodelsum/dx = d(.. + model[g])/dx = dmodel[g]/dx
        dll/dx = dmodel[g]/dx*diffvar
    */
    loglike -= diff*diffvar/2.;
    for(size_t g = 0; g < ngauss; ++g)
    {
        const GaussWeights & weights = gaussweights[g];
        gaussian_pixel_add_jacobian(
            output(grad_param_map(g, 0)), output(grad_param_map(g, 1)), output(grad_param_map(g, 2)),
            output(grad_param_map(g, 3)), output(grad_param_map(g, 4)), output(grad_param_map(g, 5)),
            grad_param_factor(g, 0), grad_param_factor(g, 1), grad_param_factor(g, 2),
            grad_param_factor(g, 3), grad_param_factor(g, 4), grad_param_factor(g, 5),
            weights[0]*diffvar, weights[1]*diffvar,
            (*(gaussterms.xmc_norm))[g], (*(gaussterms.ymc_norm[g]))[dim1],
            (*(gaussterms.xmc))[g], (*(gradterms.ymc[g]))[dim1],
            (*(gradterms.yy_factor))[g], (*(gradterms.xsnormxy))[g],
            (*(gradterms.sig_x_inv))[g], (*(gradterms.sig_y_inv))[g], weights[2],
            (*(gaussterms.xsqs))[g], (*(gaussterms.yy_norm[g]))[dim1],
            (*(gradterms.rho_factor))[g], (*(gradterms.rho_xy_factor))[g],
            (*(gradterms.sig_x_src_div_conv))[g], (*(gradterms.sig_y_src_div_conv))[g],
            (*(gradterms.drho_c_dsig_x_src))[g], (*(gradterms.drho_c_dsig_y_src))[g],
            (*(gradterms.drho_c_drho_s))[g]
        );
    }
}

template <GradientType gradient_type>
double gaussian_pixel_add_all(size_t g, size_t j, size_t i, double weight, const GaussianTerms & gaussterms,
    Array3UncheckedMutable & output_jac_ref, const MatrixSUnchecked & grad_param_map_ref,
    const MatrixUnchecked & grad_param_factor_ref, std::vector<GaussWeights> & gradweights, 
    const GradientTerms & gradterms)
{
    double xy = (*(gaussterms.xmc))[g]*(*(gaussterms.ymc_norm[g]))[j];
    double value_unweight = (*(gaussterms.norms))[g]*exp(
            -((*(gaussterms.xsqs))[g] + (*(gaussterms.yy_norm[g]))[j] - xy));
    double value = weight*value_unweight;
    gaussian_pixel_set_weights<gradient_type>(gradweights[g], value, value_unweight, xy);
    gaussian_pixel_add_jacobian_type<gradient_type>(output_jac_ref, grad_param_map_ref,
        grad_param_factor_ref, g, j, i, value, value_unweight,
        (*(gaussterms.xmc_norm))[g], (*(gaussterms.ymc_norm[g]))[j], (*(gaussterms.xmc))[g],
        (*(gradterms.ymc[g]))[j], (*(gradterms.yy_factor))[g], (*(gradterms.xsnormxy))[g],
        (*(gradterms.sig_x_inv))[g], (*(gradterms.sig_y_inv))[g], xy, (*(gaussterms.xsqs))[g],
        (*(gaussterms.yy_norm[g]))[j], (*(gradterms.rho_factor))[g], (*(gradterms.rho_xy_factor))[g],
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
double gaussians_pixel_template(const paramsgauss & GAUSSIANS,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const ndarray * const DATA, const ndarray * const VARINVERSE, ndarray * output = nullptr,
    ndarray * grads = nullptr, ndarray_s * grad_param_map = nullptr, ndarray * grad_param_factor = nullptr)
{
    check_is_gaussians(GAUSSIANS);
    const size_t DATASIZE = DATA == nullptr ? 0 : DATA->size();
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
    const size_t NGAUSSIANS = GAUSSIANS.shape(0);
    std::unique_ptr<ndarray_s> grad_param_map_default;
    std::unique_ptr<ndarray> grad_param_factor_default;
    ArrayUncheckedMutable outputgradref = gradient_type == GradientType::loglike ?
        (*grads).mutable_unchecked<1>() : (*array_null).mutable_unchecked<1>();
    Array3UncheckedMutable output_jac_ref = gradient_type == GradientType::jacobian ?
        (*grads).mutable_unchecked<3>() : (*array3_null).mutable_unchecked<3>();
    const size_t NPARAMS = gradient_type == GradientType::loglike ? outputgradref.shape(0) :
        (gradient_type == GradientType::jacobian ? output_jac_ref.shape(2) : 0);
    if(do_gradient)
    {
        const size_t NPARAMS_GRAD = NGAUSSIANS*NGAUSSPARAMS;
        const bool has_grad_param_map = grad_param_map != nullptr and (*grad_param_map).size() != 0;
        const bool has_grad_param_factor = grad_param_factor != nullptr and (*grad_param_factor).size() != 0;
        if(!has_grad_param_map)
        {
            if(NPARAMS != NPARAMS_GRAD)
            {
                throw std::runtime_error("Passed gradient vector of size=" + std::to_string(NPARAMS) +
                   "!= default mapping size of ngaussiansx6=" + std::to_string(NPARAMS_GRAD));
            }
            grad_param_map_default = std::make_unique<ndarray_s>(
                pybind11::array::ShapeContainer({NGAUSSIANS, NGAUSSPARAMS}));
            grad_param_map = grad_param_map_default.get();
            size_t index = 0;
            MatrixSUncheckedMutable grad_param_map_ref = (*grad_param_map).mutable_unchecked<2>();
            for(size_t g = 0; g < NGAUSSIANS; ++g)
            {
                for(size_t p = 0; p < NGAUSSPARAMS; ++p)
                {
                    grad_param_map_ref(g, p) = index++;
                }
            }
        }
        if(!has_grad_param_factor)
        {
            grad_param_factor_default = std::make_unique<ndarray>(
                pybind11::array::ShapeContainer({NGAUSSIANS, NGAUSSPARAMS}));
            grad_param_factor = grad_param_factor_default.get();
            MatrixUncheckedMutable grad_param_factor_ref = (*grad_param_factor).mutable_unchecked<2>();
            for(size_t g = 0; g < NGAUSSIANS; ++g)
            {
                for(size_t p = 0; p < NGAUSSPARAMS; ++p)
                {
                    grad_param_factor_ref(g, p) = 1.;
                }
            }
        }
        else
        {
            /*
            MatrixUnchecked grad_param_factor_ref = (*grad_param_factor).unchecked<2>();
            for(size_t g = 0; g < NGAUSSIANS; ++g)
            {
                for(size_t p = 0; p < NGAUSSPARAMS; ++p)
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
            if(rows_map != NGAUSSIANS or rows_map != rows_fac or cols_map != NGAUSSPARAMS or
                cols_map != cols_fac)
            {
                throw std::runtime_error("grad_param_map shape (" + std::to_string(rows_map) + "," +
                    std::to_string(cols_map) + ") and/or grad_param_factor shape (" +
                    std::to_string(rows_fac) + "," + std::to_string(cols_fac) + ") != (" +
                    std::to_string(NGAUSSIANS) + "," + std::to_string(NGAUSSPARAMS) + ")");
            }
        }
    }
    if(getlikelihood)
    {
        check_is_matrix(DATA);
        check_is_matrix(VARINVERSE);
        if(DATASIZE == 0 || VARINVERSE->size() == 0) throw std::runtime_error("gaussians_pixel_template "
            "can't compute loglikelihood with empty DATA or VARINVERSE");
    }
    if(!writeoutput && !getlikelihood)
    {
        if(output->size() == 0 && DATASIZE == 0 && VARINVERSE->size() == 0) throw std::runtime_error(
            "gaussians_pixel_template can't infer size of matrix without one of DATA or output.");
    }

    const ndarray & matcomparesize = DATASIZE ? *DATA : *output;
    const unsigned int XDIM = matcomparesize.shape(1);
    const unsigned int YDIM = matcomparesize.shape(0);
    const bool VARISMAT = getlikelihood ? VARINVERSE->size() > 1 : false;

    if(gradient_type == GradientType::jacobian &&
        !(output_jac_ref.shape(0) == YDIM && output_jac_ref.shape(1) == XDIM))
    {
        throw std::runtime_error("Data/output matrix dimensions [" + std::to_string(XDIM) + ',' +
            std::to_string(YDIM) + "] don't match Jacobian matrix dimensions [" +
            std::to_string(output_jac_ref.shape(1)) + ',' + std::to_string(output_jac_ref.shape(0)) + ']');
    }

    if(getlikelihood)
    {
        // The case of constant variance per pixel
        if(VARISMAT == 1 && (XDIM != VARINVERSE->shape(1) || YDIM != VARINVERSE->shape(0)))
        {
            throw std::runtime_error("Data matrix dimensions [" + std::to_string(XDIM) + ',' +
                std::to_string(YDIM) + "] don't match inverse variance dimensions [" +
                std::to_string(VARINVERSE->shape(1)) + ',' + std::to_string(VARINVERSE->shape(0)) + ']');
        }
    }
    if(writeoutput)
    {
        if(XDIM != output->shape(1) || YDIM != output->shape(0))
        {
            throw std::runtime_error("Data matrix dimensions [" + std::to_string(XDIM) + ',' +
                std::to_string(YDIM) + "] don't match output matrix dimensions [" +
                std::to_string(output->shape(1)) + ',' + std::to_string(output->shape(0)) + ']');
        }
    }
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;
    const double XBINHALF = XBIN/2.;

    GaussianTerms gaussterms(NGAUSSIANS);
    // These are to store pre-computed values for gradients and are unused otherwise
    // TODO: Move these into a simple class/struct
    const size_t ngaussgrad = NGAUSSIANS*(do_gradient);
    GradientTerms gradterms(ngaussgrad);
    std::vector<GaussWeights> gradweights(NGAUSSIANS*(gradient_type == GradientType::loglike),
        GaussWeights());
    const MatrixUnchecked GAUSSIANSREF = GAUSSIANS.unchecked<2>();
    for(size_t g = 0; g < NGAUSSIANS; ++g)
    {
        const double XCEN = GAUSSIANSREF(g, 0);
        const double YCEN = GAUSSIANSREF(g, 1);
        const double L = GAUSSIANSREF(g, 2);
        const Covar COVPSF = Covar{GAUSSIANSREF(g, 6), GAUSSIANSREF(g, 7), GAUSSIANSREF(g, 8)};
        const Covar COVSRC = Covar{GAUSSIANSREF(g, 3), GAUSSIANSREF(g, 4), GAUSSIANSREF(g, 5)};
        const Covar COV = convolution(COVSRC, COVPSF);
        // Deliberately omit luminosity for now
        const TermsGaussPDF TERMS = terms_from_covar(XBIN*YBIN, COV);

        auto yvals = gaussian_pixel_x_xx(YCEN, YMIN, YBIN, YDIM, TERMS.yy, TERMS.xy);

        gaussterms.set(g, TERMS.norm, XMIN - XCEN + XBINHALF, TERMS.xx,
            std::move(yvals.x_norm), std::move(yvals.xx));
        if(do_gradient)
        {
            gradterms.set(g, L, TERMS, COVSRC, COVPSF, COV, std::move(yvals.x));
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
    const MatrixUnchecked DATAREF = getlikelihood ? (*DATA).unchecked<2>() : GAUSSIANSREF;
    const MatrixUnchecked VARINVERSEREF = getlikelihood ? (*VARINVERSE).unchecked<2>() : GAUSSIANSREF;
    double loglike = 0;
    double model = 0;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < XDIM; i++)
    {
        for(size_t g = 0; g < NGAUSSIANS; ++g)
        {
            (*(gaussterms.xmc_norm))[g] = (*(gaussterms.xmc))[g]*(*(gaussterms.norms_xx))[g];
            (*(gaussterms.xsqs))[g] = (*(gaussterms.xmc_norm))[g]*(*(gaussterms.xmc))[g];
            if(do_gradient) (*(gradterms.xsnormxy))[g] = (*(gaussterms.xmc))[g]*(*(gradterms.xy_factor))[g];
        }
        for(unsigned int j = 0; j < YDIM; j++)
        {
            model = 0;
            for(size_t g = 0; g < NGAUSSIANS; ++g)
            {
                model += gaussian_pixel_add_all<gradient_type>(g, j, i, GAUSSIANSREF(g, 2), gaussterms,
                    output_jac_ref, grad_param_map_ref, grad_param_factor_ref, gradweights, gradterms);
            }
            gaussians_pixel_output<output_type>(outputref, model, j, i);
            gaussians_pixel_residual<output_type>(outputref, DATAREF, model, j, i);
            gaussians_pixel_add_like<getlikelihood, gradient_type>(loglike, model, DATAREF,
                VARINVERSEREF, j, i, VARISMAT);
            gaussians_pixel_add_like_grad<getlikelihood, gradient_type>(outputgradref, grad_param_map_ref,
                grad_param_factor_ref, NGAUSSIANS, gradweights, loglike, model, DATAREF, VARINVERSEREF, j, i,
                VARISMAT, gaussterms, gradterms);
        }
        for(size_t g = 0; g < NGAUSSIANS; ++g) (*(gaussterms.xmc))[g] += XBIN;
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
double loglike_gaussians_pixel_getlike(const ndarray & DATA, const ndarray & VARINVERSE,
    const paramsgauss & GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    ndarray & output, ndarray & grad, ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const GradientType gradient_type = get_gradient_type(grad);
    if(gradient_type == GradientType::loglike)
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::loglike>(
            GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA, &VARINVERSE, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
    else if (gradient_type == GradientType::jacobian)
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::jacobian>(
            GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA, &VARINVERSE, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
    else
    {
        return gaussians_pixel_template<output_type, get_likelihood, GradientType::none>(
            GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA, &VARINVERSE, &output, &grad,
            &grad_param_map, &grad_param_factor);
    }
}

// See loglike_gaussians_pixel for docs. This is just for explicit template instantiation.
template <OutputType output_type>
double loglike_gaussians_pixel_output(const ndarray & DATA, const ndarray & VARINVERSE,
    const paramsgauss & GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    ndarray & output, ndarray & grad, ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const bool get_likelihood = DATA.size() > 0;
    if(get_likelihood)
    {
        return loglike_gaussians_pixel_getlike<output_type, true>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
    else
    {
        return loglike_gaussians_pixel_getlike<output_type, false>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
}

/**
 * Compute the model and/or log-likehood and/or gradient (d(log-likehood)/dx) and/or Jacobian (dmodel/dx)
 * for a Gaussian mixture model.
 *
 * TODO: Consider override to compute LL and Jacobian, even if it's only useful for debug purposes.
 *
 * @param DATA 2D input image matrix.
 * @param VARINVERSE 2D inverse variance map of the same size as image.
 * @param GAUSSIANS N x 6 matrix of Gaussian parameters [cen_x, cen_y, flux, sigma_x, sigma_y, rho]
 * @param XMIN x coordinate of the left edge of the box.
 * @param XMAX x coordinate of the right edge of the box.
 * @param YMIN y coordinate of the bottom edge of the box.
 * @param YMAX y coordinate of the top edge of the box.
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
double loglike_gaussians_pixel(const ndarray & DATA, const ndarray & VARINVERSE,
    const paramsgauss & GAUSSIANS, const double XMIN, const double XMAX,
    const double YMIN, const double YMAX, bool to_add, ndarray & output, ndarray & grad,
    ndarray_s & grad_param_map, ndarray & grad_param_factor)
{
    const GradientType gradient_type = get_gradient_type(grad);
    const OutputType output_type = output.size() > 0 ? (
            gradient_type == GradientType::jacobian ? OutputType::residual : (
            to_add ? OutputType::add : OutputType::overwrite)) :
        OutputType::none;
    if(output_type == OutputType::overwrite)
    {
        return loglike_gaussians_pixel_output<OutputType::overwrite>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
    else if(output_type == OutputType::add)
    {
        return loglike_gaussians_pixel_output<OutputType::add>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
    else if(output_type == OutputType::residual)
    {
        return loglike_gaussians_pixel_output<OutputType::residual>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
    else
    {
        return loglike_gaussians_pixel_output<OutputType::none>(DATA, VARINVERSE, GAUSSIANS,
            XMIN, XMAX, YMIN, YMAX, output, grad, grad_param_map, grad_param_factor);
    }
}

ndarray make_gaussians_pixel(
    const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    ndarray mat({YDIM, XDIM});
    gaussians_pixel_template<OutputType::overwrite, false, GradientType::none>(
        GAUSSIANS, XMIN, XMAX, YMIN, YMAX, nullptr, nullptr, &mat);
    return mat;
}

void add_gaussians_pixel(
    const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    ndarray & output)
{
    gaussians_pixel_template<OutputType::add, true, GradientType::none>(
        GAUSSIANS, XMIN, XMAX, YMIN, YMAX, nullptr, nullptr, &output);
}
}
#endif
