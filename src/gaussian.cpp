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
typedef pybind11::detail::unchecked_mutable_reference<double, 2l> MatrixUncheckedMutable;

namespace multiprofit {

/*
  Note: This causes build errors on Solaris due to "ambiguous and overloaded" pow calls.
  These issues will need fixing before any integration into libprofit
  Author: Dan Taranu

  Semi-analytic integration for a Gaussian profile.

  I call this semi-analytic because there is an analytical solution for the integral
  of a 2D Gaussian over one dimension of a rectangular area, which is basically just
  the product of an exponential and error function (not too surprising). Unfortunately
  Wolfram Alpha wasn't able to integrate it over the second dimension. Perhaps a more
  clever person could find a useful normal integral as from here:

  http://www.tandfonline.com/doi/pdf/10.1080/03610918008812164

  xmod = xmid * INVREX + ymid * INVREY;
  ymod = (xmid * INVREY - ymid * INVREX)*INVAXRAT;
  return exp(-BN*(xmod*xmod + ymod * ymod -1));
  -> integral of
  exp(-BN*((x*INVREX + y*INVREY)^2 + (x*INVREY - y*INVREX)^2*INVAXRAT^2 -1));
  constants: BN = b, INVREX=c, INVREY=d, INVAXRAT^2 = a

  exp(-b*((x*c + y*d)^2 + a*(x*d - y*c)^2) - 1)

  Integrate[Exp[-(b ((x c + y d)^2 + a (x d - y c)^2)) - 1], x]
  = (Sqrt[Pi] Erf[(Sqrt[b] (c^2 x + a d^2 x - (-1 + a) c d y))/Sqrt[c^2 + a d^2]])/(2 Sqrt[b] Sqrt[c^2 + a d^2] E^((c^2 + a d^2 + a b c^4 y^2 + 2 a b c^2 d^2 y^2 + a b d^4 y^2)/(c^2 + a d^2)))
  = Erf((sqrt(b)*(c^2*x + a*d^2*x - (-1 + a)*c*d*y))/sqrt(c^2 + a*d^2))*exp((c^2 + a*d^2 + a*b*c^4*y^2 + 2*a*b*c^2*d^2*y^2 + a*b*d^4*y^2)/(c^2 + a*d^2)) dy * const
  = erf((sqb*(c2pad2tx - am1tcd*y))*isqc2pad2)*
    exp(-(c2pad2 + abc4*y2 + 2*abc2d2*y2 + abd4*y^2)/(c2pad2)) dy * const
  = erf((sqb*(c2pad2tx - am1tcd*y))*isqc2pad2)*
    exp(-1 - y^2*(abc4 + 2*abc2d2 + abd4)/c2pad2) dy * const

  where:

  const = Sqrt[Pi]/(2 Sqrt[b] Sqrt[c^2 + a d^2])
        = sqrt(pi)/(2*sqb)*isqc2pad2

  And:

  Integrate[Exp[-(b ((x c + y d)^2 + a (x d - y c)^2)) - 1], y]
  = (Sqrt[Pi] Erf[(Sqrt[b] (-((-1 + a) c d x) + a c^2 y + d^2 y))/Sqrt[a c^2 + d^2]])/(2 Sqrt[b] Sqrt[a c^2 + d^2] E^((d^2 + a (b c^4 x^2 + b d^4 x^2 + c^2 (1 + 2 b d^2 x^2)))/(a c^2 + d^2)))
  = Erf[(Sqrt[b] (-((-1 + a) c d x) + a c^2 y + d^2 y))/Sqrt[a c^2 + d^2]]*E^(-(d^2 + a (b c^4 x^2 + b d^4 x^2 + c^2 (1 + 2 b d^2 x^2)))/(a c^2 + d^2))*const
  = erf((sqb*(d2pac2ty - am1tcd*x))*isqd2pac2)*
    exp(-1 - x^2*(abc4 + 2*abc2d2 + abd4)/d2pac2) dx * const

  where:

  const = Sqrt[Pi]/(2 Sqrt[b] Sqrt[d^2 + a c^2])
        = sqrt(pi)/(2*sqb)*isqd2pac2

  (In other words, exactly the same but replacing c2pad2 with d2pac2 and isqc2pad2 with isqd2pac2)

 TODO: Integrate this into libprofit
*/

class Profit2DGaussianIntegrator{
  private:
    double a, b, c, d;
    double x2fac, y2fac, am1tcd, c2pad2, sqbisqc2pad2, d2pac2, sqbisqd2pac2;
    short i;
    double acc[2];
    double xintfac, yintfac;
    std::vector<double> rval;

    inline double valuexy(double x, double xyfac, double axfac1, double axfac2);
    template <bool isx> inline double value(double x, double axfac1, double axfac2);

    template <bool x>
    inline double integral(double y1, double y2,
      double axfac1, double axfac2, double leftval=0,
      double rightval = 0);

  public:
    const std::vector<double> & integral(double x1, double x2, double y1,
      double y2, double iacc=1e-5, double bottomval = 0, double topval = 0,
      double leftval = 0, double rightval = 0);

    // constants: BN = b, INVREX=c, INVREY=d, INVRAT = a
    Profit2DGaussianIntegrator(const double BN, const double INVREX, const double INVREY,
      const double INVAXRAT) : a(INVAXRAT*INVAXRAT),b(BN),c(INVREX),d(INVREY)
    {
      rval.resize(5);
      double sqb = sqrt(b);
      double ab = a*b;
      double c2 = c*c;
      double d2 = d*d;
      double cd = c*d;
      double c4 = c2*c2;
      double d4 = d2*d2;
      am1tcd = (-1.0 + a)*cd;
      c2pad2 = c2+a*d2;
      d2pac2 = d2+a*c2;
      double isqc2pad2 = 1.0/sqrt(c2pad2);
      double isqd2pac2 = 1.0/sqrt(d2pac2);
      y2fac = (ab*c4 + 2*ab*c2*d2 + ab*d4);
      x2fac = y2fac/d2pac2;
      y2fac /= c2pad2;
      xintfac = sqrt(M_PI)*isqc2pad2/(2.*sqb);
      yintfac = sqrt(M_PI)*isqd2pac2/(2.*sqb);
      sqbisqc2pad2 = sqb*isqc2pad2;
      sqbisqd2pac2 = sqb*isqd2pac2;
    }
};

template <typename T> inline constexpr
int signum(T x, std::false_type is_signed) {
    return T(0) < x;
}

template <typename T> inline constexpr
int signum(T x, std::true_type is_signed) {
    return (T(0) < x) - (x < T(0));
}

template <typename T> inline constexpr
int signum(T x) {
    return signum(x, std::is_signed<T>());
}

inline double Profit2DGaussianIntegrator::valuexy(double x, double xyfac, double axfac1, double axfac2)
{
    double zi = am1tcd*x;
    double x1 = (axfac2 - zi)*xyfac;
    double x2 = (axfac1 - zi)*xyfac;
    double sign = signum<double>(x2);
    x1 *= sign;
    x2 *= sign;
    if(x1 > 5) zi = erfc(x2) - erfc(x1);
    else zi = erf(x1) - erf(x2);
    zi *= sign*exp(-1.0 - y2fac*x*x);
    return(zi);
}

template <> inline double Profit2DGaussianIntegrator::value<true>(
  double x, double axfac1, double axfac2)
{
    return(valuexy(x, sqbisqc2pad2, axfac1, axfac2));
}

template <> inline double Profit2DGaussianIntegrator::value<false>(
  double x, double axfac1, double axfac2)
{
    return(valuexy(x, sqbisqd2pac2, axfac1, axfac2));
}

// integral from x1 to x2 (i.e. bottomval, topval) numerically integrated from y1 to y2
template <bool isx> inline double Profit2DGaussianIntegrator::integral(
  double y1, double y2, double axfac1, double axfac2,
  double bottomval, double topval)
{
    double dy = (y2 - y1)/2.;
    double ymid = (y2+y1)/2.;
    double zi = am1tcd*y1;
    /*
    std::cout << y1 << "-" << y2 << "," << ymid << " dy=" << dy << "; axfac(" << axfac1 << "," << axfac2 << "); ";
    std::cout <<  " erfargs(" << (axfac1 - zi)*sqbisqc2pad2 << "," << (axfac2 - zi)*sqbisqc2pad2 << "); ";
    std::cout <<  " erfs(" << erf((axfac1 - zi)*sqbisqc2pad2) << "," << erf((axfac2 - zi)*sqbisqc2pad2) << "); ";
    std::cout << " y2fac,exp(y2fac) " << y2fac << "," << exp(-1.0 - y2fac*y1*y1) << std::endl;
    */
    if(bottomval == 0)
    {
        bottomval = value<isx>(y1, axfac1, axfac2);
    }
    //std::cout << " " << bottomval;
    double cenval = value<isx>(ymid, axfac1, axfac2);
    //std::cout << " " << cenval;
    if(topval == 0)
    {
        topval = value<isx>(y2, axfac1, axfac2);
    }
    //std::cout << " " << topval;
    double z1 = (bottomval + cenval)*dy;
    if(z1 > acc[isx]) z1 = integral<isx>(y1, ymid, axfac1, axfac2, bottomval, cenval);
    double z2 = (topval + cenval)*dy;
    if(z2 > acc[isx]) z2 = integral<isx>(ymid, y2, axfac1, axfac2, cenval, topval);
    //std::cout << " : " << (z1+z2) << std::endl;
    return(z1+z2);
}

const std::vector<double> & Profit2DGaussianIntegrator::integral(
  double x1, double x2, double y1, double y2, double iacc,
  double bottomval, double topval, double leftval, double rightval)
{
  acc[true] = iacc*xintfac;
  acc[false] = iacc*yintfac;

  double c2pad2tx1 = x1*c2pad2;
  double c2pad2tx2 = x2*c2pad2;

  double d2pac2ty1 = y1*d2pac2;
  double d2pac2ty2 = y2*d2pac2;

  if(bottomval == 0)
  {
    bottomval = value<true>(y1, c2pad2tx1, c2pad2tx2);
    rval[1] = bottomval;
  }
  if(topval == 0)
  {
    topval = value<true>(y2, c2pad2tx1, c2pad2tx2);
    rval[2] = topval;
  }
  rval[0] = (integral<true>(y1,y2,c2pad2tx1,c2pad2tx2,bottomval,topval)*xintfac); // + rval[0])/2.;
  if(leftval == 0)
  {
    leftval = value<false>(x1, d2pac2ty1, d2pac2ty2);
    rval[3] = leftval;
  }
  if(rightval == 0)
  {
    rightval = value<false>(x2, d2pac2ty1, d2pac2ty2);
    rval[4] = rightval;
  }
  //rval[0] = integral<false>(x1,x2,(x2-x1)/2.,d2pac2ty1,d2pac2ty2,leftval,rightval)*yintfac;
  //std::cout << rval[0];
  //
  //std::cout << "," << rval[0] << std::endl;
  return rval;
}

Matrix make_gaussian(
    const double XCEN, const double YCEN, const double MAG, const double RE,
    const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM,
    const double ACC)
{
    //const double BN=R::qgamma(0.5, 2 * NSER,1,1,0);
    const double BN = 0.69314718055994528622676398299518;
    // This doesn't work for boxy or other generalized ellipses, for now
    const double BOX = 0;
    //const double RBOX=PI*(BOX+2.)/(4.*R::beta(1./(BOX+2.),1+1./(BOX+2.)));
    // for box = 0, R::beta(...) = pi/2, so rbox = pi*2/(4*pi/2) = 1
    // and this term: R::gammafn(1)/RBOX is also just one
    const double RBOX = 1;
    const double LUMTOT = pow(10.,(-0.4*MAG));
    // TODO: Figure out why this empirical factor of exp(1) is required to normalize properly
    // It's to cancel out the exp(-1) in the integrand, duh.
    const double Ie=exp(1.)*LUMTOT/(RE*RE*AXRAT*M_PI*((exp(BN))/BN)*RBOX);
    const double INVRE = 1.0/RE;

    Matrix mat({YDIM, XDIM});
    auto matref = mat.mutable_unchecked<2>();
    double x,y,xhi,yhi,angmod;
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;

    const double INVREY = sin(ANG*M_PI/180.)*INVRE;
    const double INVREX = cos(ANG*M_PI/180.)*INVRE*pow(-1.,angmod > 90);
    const double INVAXRAT = 1.0/AXRAT;
    Profit2DGaussianIntegrator gauss2(BN, INVREX, INVREY, INVAXRAT);
    // Want each pixels' integral to stop recursing only when less than 1e-3
    // But keep in mind we don't include the ie term, so the total over the
    // image won't be lumtot but lumtot/ie
    const double ACC2 = ACC*(LUMTOT/Ie);

    double bottomval=0;
    std::vector<double> leftvals(YDIM);

    x = XMIN-XCEN; xhi = x + XBIN;
    for(unsigned int i = 0; i < XDIM; i++)
    {
        y = YMIN - YCEN; yhi = y + YBIN;
        for(unsigned int j = 0; j < YDIM; j++)
        {
            /*
            if(i >= 13 && i <= 14 && j == 50)
            {
                std::cout << "i,j=" << i << "," << j << " x,y=" << x << "," << y <<
                    " xhi,yhi=" << xhi << "," << yhi << "bottomval,leftvals[j]=" <<
                    bottomval << "," << leftvals[j] << std::endl;
            }
            */
            const std::vector<double> & rval = gauss2.integral(x,xhi,y,yhi,ACC2,bottomval,0,leftvals[j]);
            /*
            if(i >= 13 && i <= 14 && j == 50)
            {
                return(mat);
                std::cout << rval[0] << "," << rval[1] << "," << rval[2] << "," << rval[3] << "," << rval[4] << std::endl;
            }
            */
            bottomval = rval[2];
            leftvals[j] = rval[4];
            matref(j, i) = rval[0]*Ie;
            y = yhi;
            yhi += YBIN;
        }
        bottomval = 0;
        x = xhi;
        xhi += XBIN;
    }

    return mat;
}

inline void check_is_matrix(const Matrix * mat, std::string name = "matrix")
{
    if(mat == nullptr) throw std::invalid_argument("Passed null " + name + " to check_is_matrix");
    if(mat->ndim() != 2)
    {
        throw std::invalid_argument("Passed " + name + " with ndim=" + std::to_string(mat->ndim()) + " !=2");
    }
}

inline void check_is_gaussians(const Matrix & mat)
{
    check_is_matrix(&mat, "Gaussian parameter matrix");
    if(mat.shape(1) != 6)
    {
        throw std::invalid_argument("Passed Gaussian parameter matrix with shape=[" +
            std::to_string(mat.shape(0)) + "!=6, " + std::to_string(mat.shape(1)) + "]");
    }
}


inline std::pair<std::vector<double>, std::vector<double>> gaussian_pixel_x_xx(
    const double XCEN, const double XMIN, const double XBIN, const unsigned int XDIM,
    const double XXNORMINV, const double XYNORMINV)
{
    const double XINIT = XMIN - XCEN + XBIN/2.;
    std::vector<double> x(XDIM);
    std::vector<double> xx(XDIM);
    for(unsigned int i = 0; i < XDIM; i++)
    {
        x[i] = XINIT + i*XBIN;
        xx[i] = x[i]*x[i]*XXNORMINV;
        x[i] *= XYNORMINV;
    }
    return {x, xx};
}

inline void gaussian_pixel(Matrix & mat, const double NORM, const double XCEN, const double YCEN,
    const double XMIN, const double YMIN, const double XBIN, const double YBIN,
    const double XXNORMINV, const double YYNORMINV, const double XYNORMINV)
{
    check_is_matrix(&mat);
    // don't ask me why these are reversed
    const unsigned int XDIM = mat.shape(1);
    const unsigned int YDIM = mat.shape(0);

    const auto YVALS = gaussian_pixel_x_xx(YCEN, YMIN, YBIN, YDIM, YYNORMINV, XYNORMINV);
    const std::vector<double> & Y = YVALS.first;
    const std::vector<double> & YY = YVALS.second;

    auto matref = mat.mutable_unchecked<2>();
    double x = XMIN-XCEN+XBIN/2.;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < XDIM; i++)
    {
        const double XSQ = x*x*XXNORMINV;
        for(unsigned int j = 0; j < YDIM; j++)
        {
           matref(j, i) = NORM*exp(-(XSQ + YY[j] - x*Y[j]));
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

// Evaluate a Gaussian on a grid given the three elements of the symmetric covariance matrix
// Actually, rho is scaled by sigx and sigy (i.e. the covariance is RHO*SIGX*SIGY)
Matrix make_gaussian_pixel_covar(const double XCEN, const double YCEN, const double L,
    const double SIGX, const double SIGY, const double RHO,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;
    const Covar COV {.sigx = SIGX, .sigy=SIGY, .rho=RHO};
    const TermsGaussPDF TERMS = terms_from_covar(L*XBIN*YBIN, COV);

    Matrix mat({YDIM, XDIM});
    gaussian_pixel(mat, TERMS.norm, XCEN, YCEN, XMIN, YMIN, XBIN, YBIN, TERMS.xx, TERMS.yy, TERMS.xy);

    return mat;
}

// Evaluate a Gaussian on a grid given R_e, the axis ratio and position angle
Matrix make_gaussian_pixel(
    const double XCEN, const double YCEN, const double L, const double R,
    const double ANG, const double AXRAT,
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

Matrix make_gaussian_pixel_sersic(
    const double XCEN, const double YCEN, const double L, const double R,
    const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    // I don't remember why this isn't just 1/(2*ln(2)) but anyway it isn't
    const double NORMRFAC = 0.69314718055994528622676398299518;
    const double XBIN=(XMAX-XMIN)/XDIM;
    const double YBIN=(YMAX-YMIN)/YDIM;
    const double NORM = L*NORMRFAC/(M_PI*AXRAT)/R/R*XBIN*YBIN;
    const double INVAXRATSQ = 1.0/AXRAT/AXRAT;

    Matrix mat({YDIM, XDIM});
    double x,y;
    const double XBINHALF=XBIN/2.;
    const double YBINHALF=YBIN/2.;

    const double INVREY = sin(ANG*M_PI/180.)*sqrt(NORMRFAC)/R;
    const double INVREX = cos(ANG*M_PI/180.)*sqrt(NORMRFAC)/R;

    unsigned int i=0,j=0,k=0;
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

template <bool writeoutput, bool add>
inline void gaussians_pixel_output(MatrixUncheckedMutable & output, double value, unsigned int dim1,
    unsigned int dim2);

template <>
inline void gaussians_pixel_output<false, false>(MatrixUncheckedMutable & output, double value,
    unsigned int dim1, unsigned int dim2) {}

template <>
inline void gaussians_pixel_output<false, true>(MatrixUncheckedMutable & output, double value,
    unsigned int dim1, unsigned int dim2) {}

template <>
inline void gaussians_pixel_output<true, false>(MatrixUncheckedMutable & output, double value,
    unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) = value;
}

template <>
inline void gaussians_pixel_output<true, true>(MatrixUncheckedMutable & output, double value,
    unsigned int dim1, unsigned int dim2)
{
    output(dim1, dim2) += value;
}

template <bool getlikelihood>
inline void gaussians_pixel_getlikelihood(double & loglike, const double model, const MatrixUnchecked & DATA,
    const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2, const bool VARISMAT);

template <>
inline void gaussians_pixel_getlikelihood<false>(double & loglike, const double model,
    const MatrixUnchecked & DATA, const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2,
    const bool VARISMAT) {}

template <>
inline void gaussians_pixel_getlikelihood<true>(double & loglike, const double model,
    const MatrixUnchecked & DATA, const MatrixUnchecked & VARINVERSE, unsigned int dim1, unsigned int dim2,
    const bool VARISMAT)
{
    double diff = DATA(dim1, dim2) - model;
    // TODO: Check what the performance penalty for using IDXVAR is and rewrite if it's worth it
    loglike += -diff*diff*VARINVERSE(dim1*VARISMAT, dim2*VARISMAT)/2.;
}

// Compute Gaussian mixtures with the option to write output and/or evaluate the log likehood
// TODO: Reconsider whether there's a better way to do this
// The template arguments ensure that there is no performance penalty to any of the versions of this function.
// However, some messy validation is required as a result.
template <bool writeoutput, bool getlikelihood, bool add>
double gaussians_pixel_template(const paramsgauss & GAUSSIANS,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const Matrix * const DATA, const Matrix * const VARINVERSE, Matrix * output = nullptr)
{
    check_is_gaussians(GAUSSIANS);
    const size_t DATASIZE = DATA == nullptr ? 0 : DATA->size();
    //std::cout << writeoutput << "," << getlikelihood << std::endl;
    std::unique_ptr<Matrix> matrixnull;
    if(writeoutput)
    {
        check_is_matrix(output);
        if(output->size() == 0) throw std::runtime_error("gaussians_pixel_template can't write to empty "
            "matrix");
    }
    else
    {
        auto nullshape = pybind11::array::ShapeContainer({0, 0});
        matrixnull = std::make_unique<Matrix>(nullshape);
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

    const Matrix & matcomparesize = DATASIZE ? *DATA : *output;
    const unsigned int XDIM = matcomparesize.shape(1);
    const unsigned int YDIM = matcomparesize.shape(0);
    const bool VARISMAT = getlikelihood ? VARINVERSE->size() > 1 : false;

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

    const size_t NGAUSSIANS = GAUSSIANS.shape(0);
    std::vector<std::vector<double>> ys;
    std::vector<std::vector<double>> yys;
    std::vector<double> xs;
    std::vector<double> normsxx;
    std::vector<double> norms;

    const MatrixUnchecked GAUSSIANSREF = GAUSSIANS.unchecked<2>();
    for(size_t g = 0; g < NGAUSSIANS; ++g)
    {
        const double XCEN = GAUSSIANSREF(g, 0); const double YCEN = GAUSSIANSREF(g, 1);
        const double L = GAUSSIANSREF(g, 2), R = GAUSSIANSREF(g, 3);
        const double ANG = GAUSSIANSREF(g, 4), AXRAT = GAUSSIANSREF(g, 5);
        const Covar COV = ellipse_to_covar(reff_to_sigma_gauss(R), AXRAT, degrees_to_radians(ANG));
        const TermsGaussPDF TERMS = terms_from_covar(L*XBIN*YBIN, COV);

        auto yvals = gaussian_pixel_x_xx(YCEN, YMIN, YBIN, YDIM, TERMS.yy, TERMS.xy);

        ys.push_back(yvals.first);
        yys.push_back(yvals.second);
        xs.push_back(XMIN-XCEN+XBIN/2.);
        normsxx.push_back(TERMS.xx);
        norms.push_back(TERMS.norm);
    }

    /*
        Somewhat ugly hack here to set refs to point at Gaussians if we know they won't be used
        I would love to replace these by unique_ptrs or something and get rid of matrixnull but I can't figure
         out how to construct an unchecked reference even after looking at the pybind11 source.
    */
    MatrixUncheckedMutable outputref = writeoutput ? (*output).mutable_unchecked<2>() :
        (*matrixnull).mutable_unchecked<2>();
    const MatrixUnchecked DATAREF = getlikelihood ? (*DATA).unchecked<2>() : GAUSSIANSREF;
    const MatrixUnchecked VARINVERSEREF = getlikelihood ? (*VARINVERSE).unchecked<2>() : GAUSSIANSREF;
    std::vector<double> xsqs(NGAUSSIANS);
    double loglike = 0;
    double model = 0;
    // TODO: Consider a version with cached xy, although it doubles memory usage
    for(unsigned int i = 0; i < XDIM; i++)
    {
        for(size_t g = 0; g < NGAUSSIANS; ++g) xsqs[g] = xs[g]*xs[g]*normsxx[g];
        for(unsigned int j = 0; j < YDIM; j++)
        {
            model = 0;
            for(size_t g = 0; g < NGAUSSIANS; ++g)
            {
                model += norms[g]*exp(-(xsqs[g] + yys[g][j] - xs[g]*ys[g][j]));
            }
            gaussians_pixel_output<writeoutput, add>(outputref, model, j, i);
            gaussians_pixel_getlikelihood<getlikelihood>(loglike, model, DATAREF, VARINVERSEREF, j, i,
                VARISMAT);
        }
        for(size_t g = 0; g < NGAUSSIANS; ++g) xs[g] += XBIN;
    }
    return loglike;
}

double loglike_gaussians_pixel(const Matrix & DATA, const Matrix & VARINVERSE,
    const paramsgauss & GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    Matrix & output)
{
    if(DATA.size() == 0)
    {
        if(output.size() == 0)
        {
            return gaussians_pixel_template<false, false, false>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA,
                &VARINVERSE, &output);
        }
        else
        {
            return gaussians_pixel_template<true, false, true>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA,
                &VARINVERSE, &output);
        }
    }
    else
    {
        if(output.size() == 0)
        {
            return gaussians_pixel_template<false, true, false>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA,
                &VARINVERSE, &output);
        }
        else
        {
            return gaussians_pixel_template<true, true, true>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, &DATA,
                &VARINVERSE, &output);
        }
    }
}

Matrix make_gaussians_pixel(
    const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM)
{
    Matrix mat({YDIM, XDIM});
    gaussians_pixel_template<true, false, false>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, nullptr, nullptr, &mat);
    return mat;
}

void add_gaussians_pixel(
    const paramsgauss& GAUSSIANS, const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    Matrix & output)
{
    gaussians_pixel_template<true, false, true>(GAUSSIANS, XMIN, XMAX, YMIN, YMAX, nullptr, nullptr, &output);
}
}
#endif
