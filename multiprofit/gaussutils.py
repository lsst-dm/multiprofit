import numpy as np
import scipy.optimize as spopt


# https://www.wolframalpha.com/input/?i=Integrate+2*pi*x*exp(-x%5E2%2F(2*s%5E2))%2F(s*sqrt(2*pi))+dx+from+0+to+r
# The fraction of the total flux of a 2D Sersic profile contained within r
# For efficiency, we can replace r with r/sigma.
# Note the trivial renormalization allows dropping annoying sigma and pi constants
# It returns (0, 1) for inputs of (0, inf)
def gauss2dint(xdivsigma):
    return 1 - np.exp(-xdivsigma**2/2.)


# Compute the fraction of the integrated flux within x for a sum of Gaussians
# x is a length in arbitrary units
# Weightsizes is a list of tuples of the weight (flux) and size (r_eff in the units of x) of each gaussian
# Note that gauss2dint expects x/sigma, but size is re, so we pass x/re*re/sigma = x/sigma
# 0 > quant > 1 turns it into a function that returns zero
#   at the value of x containing a fraction quant of the total flux
# This is so you can use root finding algorithms to find x for a given quant (see below)
def multigauss2dint(x, weightsizes, quant=0):
    retosigma = np.sqrt(2.*np.log(2.))
    weightsumtox = 0
    weightsum = 0
    for weight, size in weightsizes:
        weightsumtox += weight*(gauss2dint(x/size*retosigma) if size > 0 else 1)
        weightsum += weight
    return weightsumtox/weightsum - quant


# Compute x_quant for a sum of Gaussians, where 0<quant<1
# There's probably an analytic solution to this if you care to work it out
# Weightsizes and quant are as above
# Choose xmin, xmax so that xmin < x_quant < xmax.
# Ideally we'd just set xmax=np.inf but brentq doesn't work then; a very large number like 1e5 suffices.
def multigauss2drquant(weightsizes, quant=0.5, xmin=0, xmax=1e5):
    if not 0 <= quant <= 1:
        raise ValueError('Quant {} not >=0 & <=1'.format(quant, quant))
    weightsumzerosize = 0
    weightsum = 0
    for weight, size in weightsizes:
        if not (size > 0):
            weightsumzerosize += weight
        weightsum += weight
    if weightsumzerosize/weightsum >= quant:
        return 0
    return spopt.brentq(multigauss2dint, a=xmin, b=xmax, args=(weightsizes, quant))
