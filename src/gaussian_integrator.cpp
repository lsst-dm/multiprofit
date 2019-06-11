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

#ifndef MULTIPROFIT_GAUSSIAN_INTEGRATOR_H
#include "gaussian_integrator.h"
#endif

#include <vector>

namespace multiprofit {

/*
  TODO: Use covariance ellipse parameters

  Semi-analytic integration for a Gaussian profile.

  I call this semi-analytic because there is an analytical solution for the integral
  of a 2D Gaussian over one dimension of a rectangular area, which is basically just
  the product of an exponential and error function (not too surprising). Unfortunately
  Wolfram Alpha wasn't able to integrate it over the second dimension. Perhaps a more
  clever person could find a useful normal integral as from here:

  http://www.tandfonline.com/doi/pdf/10.1080/03610918008812164

  xmod = xmid * r_eff_inv_x + ymid * r_eff_inv_y;
  ymod = (xmid * r_eff_inv_y - ymid * r_eff_inv_x)*axrat_inv;
  return exp(-b_n*(xmod*xmod + ymod * ymod -1));
  -> integral of
  exp(-b_n*((x*r_eff_inv_x + y*r_eff_inv_y)^2 + (x*r_eff_inv_y - y*r_eff_inv_x)^2*axrat_inv^2 -1));
  constants: b_n = b, r_eff_inv_x=c, r_eff_inv_y=d, axrat_inv^2 = a

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

    // constants: b_n = b, r_eff_inv_x=c, r_eff_inv_y=d, INVRAT = a
    Profit2DGaussianIntegrator(const double b_n, const double r_eff_inv_x, const double r_eff_inv_y,
                               const double axrat_inv) : a(axrat_inv*axrat_inv),b(b_n),c(r_eff_inv_x),d(r_eff_inv_y)
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
    /*
    double zi = am1tcd*y1;
    std::cout << y1 << "-" << y2 << "," << ymid << " dy=" << dy << ";axfac(" << axfac1 << "," <<
        axfac2 << "); ";
    std::cout <<  " erfargs(" << (axfac1 - zi)*sqbisqc2pad2 << "," << (axfac2 - zi)*sqbisqc2pad2 << "); ";
    std::cout <<  " erfs(" << erf((axfac1 - zi)*sqbisqc2pad2) << "," << erf((axfac2 - zi)*sqbisqc2pad2) <<
        "); ";
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
    //TODO: Use and/or implement a better integration method
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

ndarray make_gaussian(
        const double cen_x, const double cen_y, const double mag,
        const double r_eff, const double axrat, const double ang,
        const double x_min, const double x_max, const double y_min, const double y_max,
        const unsigned int dim_x, const unsigned int dim_y,
        const double acc)
{
    //const double b_n=R::qgamma(0.5, 2 * NSER,1,1,0);
    const double b_n = 0.69314718055994528622676398299518;
    // This doesn't work for boxy or other generalized ellipses, for now
    const double BOX = 0;
    //const double RBOX=PI*(BOX+2.)/(4.*R::beta(1./(BOX+2.),1+1./(BOX+2.)));
    // for box = 0, R::beta(...) = pi/2, so rbox = pi*2/(4*pi/2) = 1
    // and this term: R::gammafn(1)/RBOX is also just one
    const double RBOX = 1;
    const double l_tot = pow(10.,(-0.4*mag));
    // TODO: Figure out why this empirical factor of exp(1) is required to normalize properly
    // It's to cancel out the exp(-1) in the integrand, duh.
    const double Ie=exp(1.)*l_tot/(r_eff*r_eff*axrat*M_PI*((exp(b_n))/b_n)*RBOX);
    const double r_eff_inv = 1.0/r_eff;

    ndarray mat({dim_y, dim_x});
    auto matref = mat.mutable_unchecked<2>();
    double x,y,xhi,yhi,angmod;
    const double bin_x=(x_max-x_min)/dim_x;
    const double bin_y=(y_max-y_min)/dim_y;

    const double r_eff_inv_y = sin(ang*M_PI/180.)*r_eff_inv;
    const double r_eff_inv_x = cos(ang*M_PI/180.)*r_eff_inv*pow(-1.,angmod > 90);
    const double axrat_inv = 1.0/axrat;
    Profit2DGaussianIntegrator gauss2(b_n, r_eff_inv_x, r_eff_inv_y, axrat_inv);
    // Want each pixels' integral to stop recursing only when less than 1e-3
    // But keep in mind we don't include the ie term, so the total over the
    // image won't be lumtot but lumtot/ie
    const double ACC2 = acc*(l_tot/Ie);

    double bottomval=0;
    std::vector<double> leftvals(dim_y);

    x = x_min-cen_x; xhi = x + bin_x;
    for(unsigned int i = 0; i < dim_x; i++)
    {
        y = y_min - cen_y; yhi = y + bin_y;
        for(unsigned int j = 0; j < dim_y; j++)
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
                std::cout << rval[0] << "," << rval[1] << "," << rval[2] << "," <<
                    rval[3] << "," << rval[4] << std::endl;
            }
            */
            bottomval = rval[2];
            leftvals[j] = rval[4];
            matref(j, i) = rval[0]*Ie;
            y = yhi;
            yhi += bin_y;
        }
        bottomval = 0;
        x = xhi;
        xhi += bin_x;
    }

    return mat;
}
}