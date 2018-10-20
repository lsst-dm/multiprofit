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

import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
from scipy import special, stats

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def absconservetotal(ndarray):
    shape = ndarray.shape
    ndarray.shape = np.prod(shape)
    if any(ndarray < 0):
        indices = np.argsort(ndarray)
        # Not sure if this is any faster than cumsum - probably if most pixels are positive
        indexarr = 0
        sumneg = 0
        while ndarray[indices[indexarr]] < 0:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        while sumneg < 0 and indexarr < ndarray.shape[0]:
            sumneg += ndarray[indices[indexarr]]
            ndarray[indices[indexarr]] = 0
            indexarr += 1
        ndarray[indices[indexarr-1]] = sumneg
        if indexarr == ndarray.shape[0]:
            raise RuntimeError("absconservetotal failed for array with sum {}".format(np.sum(ndarray)))
    ndarray.shape = shape
    return ndarray


# For priors
def normlogpdfmean(x, mean=0., scale=1.):
    return stats.norm.logpdf(x - mean, scale=scale)


def truncnormlogpdfmean(x, mean=0., scale=1., a=-np.inf, b=np.inf):
    return stats.truncnorm.logpdf(x - mean, scale=scale, a=a, b=b)


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


def getmodel(
    fluxesbyband, modelstr, imagesize, sizes=None, axrats=None, angs=None, slopes=None, fluxfracs=None,
    offsetxy=None, name="", nexposures=1, engine="galsim", engineopts=None, istransformedvalues=False
):
    bands = fluxesbyband.keys()
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

    cenx, ceny = [x / 2.0 for x in imagesize]
    if offsetxy is not None:
        cenx += offsetxy[0]
        ceny += offsetxy[1]
    if nexposures > 0:
        exposures = []
        for band in bands:
            for _ in range(nexposures):
                exposures.append(proobj.Exposure(band, image=np.zeros(shape=imagesize), maskinverse=None,
                                                 sigmainverse=None))
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
        fluxfracs = 1.0 / np.arange(ncomps, 0, -1)

    compnum = 0
    for profile, nprofiles in profiles.items():
        comprange = range(compnum, compnum + nprofiles)
        isgaussian = profile == "gaussian"
        ismultigaussiansersic = profile == "multigaussiansersic"
        issoftened = profile == "lux" or profile == "luv"
        if isgaussian:
            profile = "sersic"
            for compi in comprange:
                sizes[compi] /= 2.
        if ismultigaussiansersic:
            profile = "sersic"
        values = {
            "size": sizes,
            "axrat": axrats,
            "ang": angs,
        }
        if not issoftened:
            values["slope"] = slopes

        for compi in comprange:
            islast = compi == (ncomps - 1)
            paramfluxescomp = [
                proobj.FluxParameter(
                    band, "flux", special.logit(fluxfracs[compi]), None, limits=limitsref["none"],
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
                    paramfluxescomp, profile=profile, parameters=params))
            else:
                components.append(proobj.EllipticalProfile(
                    paramfluxescomp, profile=profile, parameters=params))

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


def getchisqred(chis):
    chisum = 0
    chicount = 0
    for chivals in chis:
        chisum += np.sum(chivals**2)
        chicount += len(chivals)**2
    return chisum/chicount


def fitmodel(model, modeller=None, modellib="scipy", modellibopts={'algo': "Nelder-Mead"}, printfinal=True,
             printsteps=100, plot=False, modelname=None, figure=None, title=None, axes=None,
             figurerow=None, modelnameappendparams=None, flipplot=False):
    if modeller is None:
        modeller = proobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts)
    fit = modeller.fit(printfinal=printfinal, printsteps=printsteps)
    if printfinal:
        paramsall = model.getparameters(fixed=True)
        print("Param names:" + ",".join(["{:11s}".format(p.name) for p in paramsall]))
        print("All params: " + ",".join(["{:+.4e}".format(p.getvalue(transformed=False)) for p in paramsall]))
    # Conveniently sets the parameters to the right values too
    if plot:
        modeldesc = None
        if modelnameappendparams is not None:
            modeldesc = ""
            modeldescs = {}
            for formatstring, param in modelnameappendparams:
                if '=' not in formatstring:
                    paramname = param.name
                    value = formatstring
                else:
                    paramname, value = formatstring.split('=')
                if paramname not in modeldescs:
                    modeldescs[paramname] = []
                modeldescs[paramname].append(value.format(param.getvalue(transformed=False)))

            for paramname, values in modeldescs.items():
                modeldesc += paramname + ':' + ','.join(values) + ';'
            # Remove the trailing colon
            modeldesc = modeldesc[:-1]
    else:
        modeldesc = None
    _, _, chis, _ = model.evaluate(params=fit["paramsbest"], plot=plot, modelname=modelname,
                                   modeldesc=modeldesc, figure=figure, axes=axes, figurerow=figurerow,
                                   flipplot=flipplot)
    if plot:
        if title is not None:
            plt.suptitle(title)
        plt.show(block=False)
    fit["chisqred"] = getchisqred(chis)
    params = model.getparameters()
    for item in ['paramsbestall', 'paramsbestalltransformed', 'paramsallfixed']:
        fit[item] = []
    for param in params:
        fit["paramsbestall"].append(param.getvalue(transformed=False))
        fit["paramsbestalltransformed"].append(param.getvalue(transformed=True))
        fit["paramsallfixed"].append(param.fixed)

    return fit, modeller


def setexposure(model, band, image=None, sigmainverse=None, psf=None, mask=None, factorsigma=1):
    if band not in model.data.exposures:
        model.data.exposures[band] = [proobj.Exposure(band=band, image=None)]
    exposure = model.data.exposures[band][0]
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
    return model