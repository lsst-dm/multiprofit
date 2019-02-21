# This file is part of multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
import importlib
import matplotlib.pyplot as plt
import multiprofit as mpf
import multiprofit.utils as mpfutil
import numpy as np
from scipy import special, stats
import timeit

import multiprofit.objects as proobj


def logstretch(x, lower, factor=1):
    return np.log10(x-lower)*factor


def powstretch(x, lower, factor=1):
    return 10**(x*factor) + lower


def getlogstretch(lower, factor=1):
    return proobj.Transform(
        transform=functools.partial(logstretch, lower=lower, factor=factor),
        reverse=functools.partial(powstretch, lower=lower, factor=1./factor))


def logitlimited(x, lower, extent, factor=1):
    return special.logit((x-lower)/extent)*factor


def expitlimited(x, lower, extent, factor=1):
    return special.expit(x*factor)*extent + lower


def getlogitlimited(lower, upper, factor=1):
    return proobj.Transform(
        transform=functools.partial(logitlimited, lower=lower, extent=upper-lower, factor=factor),
        reverse=functools.partial(expitlimited, lower=lower, extent=upper-lower, factor=1./factor))


transformsref = {
    "none": proobj.Transform(),
    "log": proobj.Transform(transform=np.log, reverse=np.exp),
    "log10": proobj.Transform(transform=np.log10, reverse=functools.partial(np.power, 10.)),
    "inverse": proobj.Transform(transform=functools.partial(np.divide, 1.),
                                reverse=functools.partial(np.divide, 1.)),
    "logit": proobj.Transform(transform=special.logit, reverse=special.expit),
    "logitaxrat": getlogitlimited(1e-4, 1),
    "logitsersic":  getlogitlimited(0.3, 6.2),
    "logitmultigausssersic": getlogitlimited(0.3, 6.2),
    "logstretchmultigausssersic": getlogstretch(0.3, 6.2),
}


# TODO: Replace with a parameter factory and/or profile factory
limitsref = {
    "none": proobj.Limits(),
    "fraction": proobj.Limits(lower=0., upper=1., transformed=True),
    "fractionlog10": proobj.Limits(upper=0., transformed=True),
    "axratlog10": proobj.Limits(lower=-2., upper=0., transformed=True),
    "coninverse": proobj.Limits(lower=0.1, upper=0.9090909, transformed=True),
    "nserlog10": proobj.Limits(lower=np.log10(0.3), upper=np.log10(6.0), lowerinclusive=False,
                               upperinclusive=False, transformed=True),
}


class ImageEmpty:
    shape = (0, 0)

    def __init__(self, shape=(0, 0)):
        self.shape = shape


# For priors
def normlogpdfmean(x, mean=0., scale=1.):
    return stats.norm.logpdf(x - mean, scale=scale)


def truncnormlogpdfmean(x, mean=0., scale=1., a=-np.inf, b=np.inf):
    return stats.truncnorm.logpdf(x - mean, scale=scale, a=a, b=b)


def isfluxratio(param):
    return isinstance(param, proobj.FluxParameter) and param.isfluxratio


def ellipse_to_covar(ang, axrat, sigma):
    ang = np.radians(ang)
    sinang = np.sin(ang)
    cosang = np.cos(ang)
    majsq = sigma ** 2
    minsq = majsq * axrat ** 2
    sinangsq = sinang ** 2
    cosangsq = cosang ** 2
    sigxsq = majsq * cosangsq + minsq * sinangsq
    sigysq = majsq * sinangsq + minsq * cosangsq
    covxy = (majsq - minsq) * cosang * sinang
    covar = np.matrix([[sigxsq, covxy], [covxy, sigysq]])
    return covar


def getparamdefault(param, value=None, profile=None, fixed=False, isvaluetransformed=False,
                    sersiclogit=True, ismultigauss=False):
    transform = transformsref["none"]
    limits = limitsref["none"]
    name = param
    if param == "slope":
        if profile == "moffat":
            name = "con"
            transform = transformsref["inverse"]
            limits = limitsref["coninverse"]
            if value is None:
                value = 2.5
        elif profile == "sersic":
            name = "nser"
            if sersiclogit:
                if ismultigauss:
                    transform = transformsref["logitmultigausssersic"]
                else:
                    transform = transformsref["logitsersic"]
            else:
                transform = transformsref["log10"]
                limits = limitsref["nserlog10"]
            if value is None:
                value = 0.5
    elif param == "size":
        transform = transformsref["log10"]
        if profile == "moffat":
            name = "fwhm"
        elif profile == "sersic":
            name = "re"
    elif param == "axrat":
        transform = transformsref["logitaxrat"]

    if value is None:
        # TODO: Improve this (at least check limits)
        value = 0.
    elif not isvaluetransformed:
        value = transform.transform(value)

    param = proobj.Parameter(name, value, "", limits=limits,
                             transform=transform, transformed=True, fixed=fixed)
    return param


# TODO: Think about whether this should work with a dict of fluxfracs by band (and how)
def getcomponents(profile, fluxfracs, values={}, istransformedvalues=False):
    bands = list(fluxfracs.keys())
    bandref = bands[0]
    for band in bands:
        fluxfracsband = fluxfracs[band]
        fluxfracsband = np.array(fluxfracsband)
        sumfluxfracsband = np.sum(fluxfracsband)
        if any(np.logical_not(fluxfracsband > 0)) or not sumfluxfracsband < 1:
            raise RuntimeError('fluxfracsband={} has elements not > 0 or sum {} < 1'.format(
                fluxfracsband, sumfluxfracsband))
        if len(fluxfracsband) == 0:
            fluxfracsband = np.ones(1)
        else:
            fluxfracsband /= np.concatenate([np.array([1.0]), 1-np.cumsum(fluxfracsband[:-1])])
            fluxfracsband = np.append(fluxfracsband, 1)
        fluxfracs[band] = fluxfracsband
        ncompsband = len(fluxfracs[band])
        if band == bandref:
            ncomps = ncompsband
        elif ncompsband != ncomps:
            raise RuntimeError('getcomponents for profile {} has ncomps[{}]={} != ncomps[{}]={}'.format(
                profile, ncompsband, band, ncomps, bandref
            ))
    components = []
    isgaussian = profile == "gaussian"
    ismultigaussiansersic = profile.startswith('mgsersic')
    if ismultigaussiansersic:
        order = np.int(profile.split('mgsersic')[1])
    issoftened = profile == "lux" or profile == "luv"
    if isgaussian or ismultigaussiansersic:
        profile = "sersic"
        if 'nser' in values:
            values['nser'] = np.zeros_like(values['nser'])

    for compi in range(ncomps):
        islast = compi == (ncomps - 1)
        paramfluxescomp = [
            proobj.FluxParameter(
                band, "flux", special.logit(fluxfracs[band][compi]), None, limits=limitsref["none"],
                transform=transformsref["logit"], fixed=islast, isfluxratio=True)
            for band in bands
        ]
        params = [getparamdefault(param, valueslice[compi], profile,
                                  fixed=param == "slope" and isgaussian,
                                  isvaluetransformed=istransformedvalues,
                                  ismultigauss=ismultigaussiansersic)
                  for param, valueslice in values.items()]
        if ismultigaussiansersic or issoftened:
            components.append(proobj.MultiGaussianApproximationProfile(
                paramfluxescomp, profile=profile, parameters=params, order=order))
        else:
            components.append(proobj.EllipticalProfile(
                paramfluxescomp, profile=profile, parameters=params))

    return components


# Convenience function to get a model with 'standard' limits and transforms
def getmodel(
    fluxesbyband, modelstr, imagesize, sizes=None, axrats=None, angs=None, slopes=None, fluxfracs=None,
    offsetxy=None, name="", nexposures=1, engine="galsim", engineopts=None, istransformedvalues=False
):
    bands = list(fluxesbyband.keys())
    modelstrs = modelstr.split(",")

    profiles = {}
    ncomps = 0
    for modeldesc in modelstrs:
        profile, ncompsprof = modeldesc.split(":")
        ncompsprof = np.int(ncompsprof)
        profiles[profile] = ncompsprof
        ncomps += ncompsprof
    try:
        noneall = np.repeat(None, ncomps)
        sizes = np.array(sizes) if sizes is not None else noneall
        axrats = np.array(axrats) if axrats is not None else noneall
        angs = np.array(angs) if angs is not None else noneall
        slopes = np.array(slopes) if slopes is not None else noneall
        if fluxfracs is not None:
            fluxfracs = np.array(fluxfracs)
        # TODO: Verify lengths identical to bandscount
    except Exception as error:
        raise error

    # TODO: Figure out how this should work in multiband
    cenx, ceny = [x / 2.0 for x in imagesize]
    if offsetxy is not None:
        cenx += offsetxy[0]
        ceny += offsetxy[1]
    if nexposures > 0:
        exposures = []
        for band in bands:
            for _ in range(nexposures):
                exposures.append(proobj.Exposure(band, image=np.zeros(shape=imagesize),
                                                 maskinverse=None, sigmainverse=None))
        data = proobj.Data(exposures)
    else:
        data = None

    paramsastrometry = [
        proobj.Parameter("cenx", cenx, "pix", proobj.Limits(lower=0., upper=imagesize[0]),
                         transform=transformsref["none"]),
        proobj.Parameter("ceny", ceny, "pix", proobj.Limits(lower=0., upper=imagesize[1]),
                         transform=transformsref["none"]),
    ]
    modelastro = proobj.AstrometricModel(paramsastrometry)
    components = []

    if fluxfracs is None:
        fluxfracs = np.repeat(1.0/ncomps, ncomps)

    compnum = 0
    for profile, nprofiles in profiles.items():
        comprange = range(compnum, compnum + nprofiles)
        # TODO: Review whether this should change to support band-dependent fluxfracs
        fluxfracscomp = [fluxfracs[i] for i in comprange][:-1]
        fluxfracscomp = {band: fluxfracscomp for band in bands}
        if profile == "gaussian":
            for compi in comprange:
                if sizes[compi] is not None:
                    sizes[compi] /= 2.
        values = {
            "size": sizes[comprange],
            "axrat": axrats[comprange],
            "ang": angs[comprange],
        }
        if not profile == "lux" or profile == "luv":
            values["slope"] = slopes[comprange]

        components += getcomponents(profile, fluxfracscomp, values, istransformedvalues)
        compnum += nprofiles

    paramfluxes = [proobj.FluxParameter(
            band, "flux", np.log10(fluxesbyband[band]), None, limits=limitsref["none"],
            transform=transformsref["log10"], transformed=True, prior=None, fixed=False,
            isfluxratio=False)
        for bandi, band in enumerate(bands)
    ]
    modelphoto = proobj.PhotometricModel(components, paramfluxes)
    source = proobj.Source(modelastro, modelphoto, name)
    model = proobj.Model([source], data, engine=engine, engineopts=engineopts)
    return model


# Convenience function to evaluate a model and optionally plot with title, returning chi map only
def evaluatemodel(model, plot=False, title=None, **kwargs):
    _, _, chis, _ = model.evaluate(plot=plot, **kwargs)

    if plot:
        if title is not None:
            plt.suptitle(title)
        plt.show(block=False)
    return chis


# Convenience function to fit a model. kwargs are passed on to evaluatemodel
def fitmodel(model, modeller=None, modellib="scipy", modellibopts={'algo': "Nelder-Mead"}, printfinal=True,
             printsteps=100, plot=False, **kwargs):
    if modeller is None:
        modeller = proobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts)
    fit = modeller.fit(printfinal=printfinal, printsteps=printsteps)
    if printfinal:
        paramsall = model.getparameters(fixed=True)
        print("Param names:" + ",".join(["{:11s}".format(p.name) for p in paramsall]))
        print("All params: " + ",".join(["{:+.4e}".format(p.getvalue(transformed=False)) for p in paramsall]))
    # Conveniently sets the parameters to the right values too
    chis = evaluatemodel(model, plot=plot, params=fit["paramsbest"], **kwargs)
    fit["chisqred"] = mpfutil.getchisqred(chis)
    params = model.getparameters()
    for item in ['paramsbestall', 'paramsbestalltransformed', 'paramsallfixed']:
        fit[item] = []
    for param in params:
        fit["paramsbestall"].append(param.getvalue(transformed=False))
        fit["paramsbestalltransformed"].append(param.getvalue(transformed=True))
        fit["paramsallfixed"].append(param.fixed)

    return fit, modeller


# Convenience function to set an exposure object with optional defaults for the sigma (variance) map
# Can be used to nullify an exposure before saving to disk, for example
def setexposure(model, band, index=0, image=None, sigmainverse=None, psf=None, mask=None, meta=None,
                factorsigma=1):
    if band not in model.data.exposures:
        model.data.exposures[band] = [proobj.Exposure(band=band, image=None)]
    exposure = model.data.exposures[band][index]
    imageisempty = image is "empty"
    exposure.image = image if not imageisempty else ImageEmpty(exposure.image.shape)
    if imageisempty:
        exposure.sigmainverse = exposure.image
    else:
        if psf is None and image is not None and sigmainverse is None:
            sigmaimg = np.sqrt(np.var(image))
            exposure.sigmainverse = 1.0/(factorsigma*sigmaimg)
        else:
            exposure.sigmainverse = sigmainverse
    exposure.psf = psf
    exposure.mask = mask
    exposure.meta = {} if meta is None else meta
    return model


# TODO: Figure out multi-band operation here
# Get Gaussian component objects from profiles that are multi-Gaussian approximations (to e.g. Sersic)
# ncomponents is an array of ints specifying the number of Gaussian components in each physical component
def getmultigaussians(profiles, paramsinherit=None, ncomponents=None, fluxfracmin=1e-4):
    bands = set()
    if not 0 <= fluxfracmin < 1:
        raise ValueError('Invalid fluxfracmin not 0 <= {} < 1'.format(fluxfracmin))
    for profile in profiles:
        for band in profile.keys():
            bands.add(band)
    bands = list(bands)
    bandref = bands[0]
    params = ['mag', 're', 'axrat', 'ang', 'nser']
    fluxfracs = {}
    values = {}
    samefluxfracs = ncomponents is None or len(ncomponents) == 1
    for band in bands:
        values[band] = {
            'size' if name == 're' else 'slope' if name == 'nser' else name:
                [profile[band][name] for profile in profiles]
            for name in params
        }
        fluxfracs[band] = 10 ** (-0.4 * np.array(values[band]['mag']))
        if fluxfracmin > 0:
            fluxfracs[band] /= np.sum(fluxfracs[band])
            fluxfracs[band][fluxfracs[band] < fluxfracmin] = fluxfracmin
        fluxfracs[band] /= np.sum(fluxfracs[band])
        fluxfracs[band] = fluxfracs[band][:-1]
        del values[band]['mag']
        if samefluxfracs:
            isclosefluxfracs = np.isclose(fluxfracs[band], fluxfracs[bandref], rtol=0, atol=1e-12)
            if not np.all(isclosefluxfracs):
                raise RuntimeError(
                    'isclosefluxfracs={} so fluxfracs[{}]={} != fluxfracs[{}]={}; '
                    'band-dependent fluxfracs unsupported'.format(
                        isclosefluxfracs, band, fluxfracs[band], bandref, fluxfracs[bandref]))
        if not values[band] == values[bandref]:
            raise RuntimeError('values[{}]={} != values[{}]={}; band-dependent values unsupported'.format(
                band, values[band], bandref, values[bandref]))

    # Comparing dicts above is easier with list elements but we want np arrays in the end
    values = {key: np.array(value) for key, value in values[band].items()}

    # These are the Gaussian components
    componentsgauss = getcomponents('gaussian', fluxfracs, values=values)
    ncomponentsgauss = len(componentsgauss)
    if paramsinherit is not None and ncomponentsgauss > 1:
        if ncomponents is None:
            ncomponents = np.array([ncomponentsgauss])
        if np.sum(ncomponents) != ncomponentsgauss:
            raise ValueError('Number of Gaussian components={} != total number of Gaussian sub-components '
                             'in physical components={}; list={}'.format(
                ncomponentsgauss, np.sum(ncomponents), ncomponents))

        # Inheritee has every right to be a word
        componentinit = 0
        for ncompstoadd in ncomponents:
            paramsinheritees = {param.name: param for param in componentsgauss[componentinit].getparameters()
                                if param.name in paramsinherit}
            for param in paramsinheritees.values():
                param.inheritors = []
            for comp in componentsgauss[componentinit+1:componentinit+ncompstoadd]:
                for param in comp.getparameters():
                    if param.name in paramsinherit:
                        # This param will map onto the first component's param
                        param.fixed = True
                        paramsinheritees[param.name].inheritors.append(param)
            componentinit += ncompstoadd

    return componentsgauss


# Example usage:
# test = mpfutil.testgaussian(nbenchmark=1000)
# for x in test:
#   print('re={} q={:.2f} ang={:2.0f} {}'.format(x['reff'], x['axrat'], x['ang'], x['string']))
def testgaussian(xdim=25, ydim=35, reffs=[2.0, 5.0], angs=np.linspace(0, 90, 7),
                 axrats=[0.01, 0.1, 0.2, 0.5, 1], nbenchmark=0):
    results = []
    hasgs = importlib.util.find_spec('galsim') is not None
    if hasgs:
        import galsim as gs
    for reff in reffs:
        for ang in angs:
            for axrat in axrats:
                gaussmpfold = mpf.make_gaussian_pixel_sersic(xdim/2, ydim/2, 1, reff, ang, axrat,
                                                             0, xdim, 0, ydim, xdim, ydim)
                gaussmpf = mpf.make_gaussian_pixel(xdim/2, ydim/2, 1, reff, ang, axrat,
                                                   0, xdim, 0, ydim, xdim, ydim)
                oldtonew = np.sum(np.abs(gaussmpf-gaussmpfold))
                result = 'Old/new residual=({:.3e})'.format(oldtonew)
                if hasgs:
                    gaussgs = gs.Gaussian(flux=1, half_light_radius=reff*np.sqrt(axrat)).shear(
                        q=axrat, beta=ang*gs.degrees).drawImage(
                        nx=xdim, ny=ydim, scale=1, method='no_pixel').array
                    gstonew = np.sum(np.abs(gaussmpf-gaussgs))
                    result += '; GalSim/new residual=({:.3e})'.format(gstonew)
                if nbenchmark > 0:
                    argsmpf = ('(' + ','.join(np.repeat('{}', 12)) + ')').format(
                        xdim/2, ydim/2, 1, reff, ang, axrat, 0, xdim, 0, ydim, xdim, ydim)
                    timesmpf = {
                        key: np.min(timeit.repeat(
                            'x=mpf.make_gaussian_pixel' + value + argsmpf, setup='import multiprofit as mpf',
                            repeat=nbenchmark, number=1))
                        for key, value in {'old': '_sersic', 'new': ''}.items()
                    }
                    if hasgs:
                        timegs = np.min(timeit.repeat(
                            'x=gs.Gaussian(flux=1, half_light_radius={}).shear(q={}, beta={}*gs.degrees)'
                            '.drawImage(nx={}, ny={}, scale=1, method="no_pixel").array'.format(
                                reff*np.sqrt(axrat), axrat, ang, xdim, ydim
                            ),
                            setup='import galsim as gs', repeat=nbenchmark, number=1
                        ))
                    result += '; old/new' + ('/GalSim' if hasgs else '') + ' times=(' + ','.join(
                        ['{:.3e}'.format(x) for x in [timesmpf['old'], timesmpf['new']] + (
                            [timegs] if hasgs else [])]) + ')'
                results.append({
                    'string': result,
                    'xdim': xdim,
                    'ydim': ydim,
                    'reff': reff,
                    'axrat': axrat,
                    'ang': ang,
                })

    return results