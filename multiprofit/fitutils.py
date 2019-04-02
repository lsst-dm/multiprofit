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

from collections import OrderedDict
import csv
import functools
import galsim as gs
import importlib
import io
import matplotlib.pyplot as plt
import multiprofit as mpf
from multiprofit.multigaussianapproxprofile import MultiGaussianApproximationProfile
import multiprofit.objects as mpfobj
import multiprofit.utils as mpfutil
import numpy as np
from scipy import special, stats
import sys
import timeit
import traceback


def logstretch(x, lower, factor=1.0):
    return np.log10(x-lower)*factor


def powstretch(x, lower, factor=1.0):
    return 10**(x*factor) + lower


def getlogstretch(lower, factor=1.0):
    return mpfobj.Transform(
        transform=functools.partial(logstretch, lower=lower, factor=factor),
        reverse=functools.partial(powstretch, lower=lower, factor=1./factor))


def logitlimited(x, lower, extent, factor=1.0):
    return special.logit((x-lower)/extent)*factor


def expitlimited(x, lower, extent, factor=1.0):
    return special.expit(x*factor)*extent + lower


def getlogitlimited(lower, upper, factor=1.0):
    return mpfobj.Transform(
        transform=functools.partial(logitlimited, lower=lower, extent=upper-lower, factor=factor),
        reverse=functools.partial(expitlimited, lower=lower, extent=upper-lower, factor=1./factor))


transformsref = {
    "none": mpfobj.Transform(),
    "log": mpfobj.Transform(transform=np.log, reverse=np.exp),
    "log10": mpfobj.Transform(transform=np.log10, reverse=functools.partial(np.power, 10.)),
    "inverse": mpfobj.Transform(transform=functools.partial(np.divide, 1.),
                                reverse=functools.partial(np.divide, 1.)),
    "logit": mpfobj.Transform(transform=special.logit, reverse=special.expit),
    "logitaxrat": getlogitlimited(1e-4, 1),
    "logitsersic":  getlogitlimited(0.3, 6.2),
    "logitmultigausssersic": getlogitlimited(0.3, 6.2),
}


# TODO: Replace with a parameter factory and/or profile factory
limitsref = {
    "none": mpfobj.Limits(),
    "fraction": mpfobj.Limits(lower=0., upper=1., transformed=True),
    "fractionlog10": mpfobj.Limits(upper=0., transformed=True),
    "axratlog10": mpfobj.Limits(lower=-2., upper=0., transformed=True),
    "coninverse": mpfobj.Limits(lower=0.1, upper=0.9090909, transformed=True),
    "nserlog10": mpfobj.Limits(lower=np.log10(0.3), upper=np.log10(6.0), lowerinclusive=False,
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
    return isinstance(param, mpfobj.FluxParameter) and param.isfluxratio


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
    elif param == "rscale":
        transform = transformsref['log10']

    if value is None:
        # TODO: Improve this (at least check limits)
        value = 0.
    elif not isvaluetransformed:
        value = transform.transform(value)

    param = mpfobj.Parameter(name, value, "", limits=limits,
                             transform=transform, transformed=True, fixed=fixed)
    return param


def getcomponents(profile, fluxes, values={}, istransformedvalues=False, isfluxesfracs=True):
    bands = list(fluxes.keys())
    bandref = bands[0]
    for band in bands:
        if isfluxesfracs:
            fluxfracsband = fluxes[band]
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
            fluxes[band] = fluxfracsband
        ncompsband = len(fluxes[band])
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

    transform = transformsref["logit"] if isfluxesfracs else transformsref["log10"]
    for compi in range(ncomps):
        islast = compi == (ncomps - 1)
        paramfluxescomp = [
            mpfobj.FluxParameter(
                band, "flux", transform.transform(fluxes[band][compi]), None, limits=limitsref["none"],
                transform=transform, fixed=islast, isfluxratio=isfluxesfracs)
            for band in bands
        ]
        params = [getparamdefault(param, valueslice[compi], profile,
                                  fixed=param == "slope" and isgaussian,
                                  isvaluetransformed=istransformedvalues,
                                  ismultigauss=ismultigaussiansersic)
                  for param, valueslice in values.items()]
        if ismultigaussiansersic or issoftened:
            components.append(MultiGaussianApproximationProfile(
                paramfluxescomp, profile=profile, parameters=params, order=order))
        else:
            components.append(mpfobj.EllipticalProfile(
                paramfluxescomp, profile=profile, parameters=params))

    return components


def getmodel(
    fluxesbyband, modelstr, imagesize, sizes=None, axrats=None, angs=None, slopes=None, fluxfracs=None,
    offsetxy=None, name="", nexposures=1, engine="galsim", engineopts=None, istransformedvalues=False,
    convertfluxfracs=False
):
    """
    Convenience function to get a multiprofit.objects.model with a single source with components with
    reasonable default parameters and transforms.

    :param fluxesbyband: Dict; key=band: value=np.array of fluxes per component if fluxfracs is None else
        source flux
    :param modelstr: String; comma-separated list of 'component_type:number'
    :param imagesize: Float[2]; the x- and y-size of the image
    :param sizes: Float[ncomponents]; Linear sizes of each component
    :param axrats: Float[ncomponents]; Axis ratios of each component
    :param angs: Float[ncomponents]; Position angle of each component
    :param slopes: Float[ncomponents]; Profile shape (e.g. Sersic n) of each components
    :param fluxfracs: Float[ncomponents]; The flux fraction for each component
    :param offsetxy: Float[2][ncomponents]; The x-y offsets relative to source center of each component
    :param name: String; a name for the source
    :param nexposures: Int > 0; the number of exposures in each band.
    :param engine: String; the rendering engine to pass to the multiprofit.objects.Model.
    :param engineopts: Dict; the rendering options to pass to the multiprofit.objects.Model.
    :param istransformedvalues: Boolean; are the provided initial values above already transformed?
    :param convertfluxfracs: Boolean; should the model have absolute fluxes per component instead of ratios?
    :return:
    """
    bands = list(fluxesbyband.keys())
    modelstrs = modelstr.split(",")

    profiles = {}
    ncomps = 0
    for modeldesc in modelstrs:
        # TODO: What should be done about modifiers?
        profilemodifiers = modeldesc.split("+")
        profile, ncompsprof = profilemodifiers[0].split(":")
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
                exposures.append(mpfobj.Exposure(band, image=np.zeros(shape=imagesize),
                                                 maskinverse=None, sigmainverse=None))
        data = mpfobj.Data(exposures)
    else:
        data = None

    paramsastrometry = [
        mpfobj.Parameter("cenx", cenx, "pix", mpfobj.Limits(lower=0., upper=imagesize[0]),
                         transform=transformsref["none"]),
        mpfobj.Parameter("ceny", ceny, "pix", mpfobj.Limits(lower=0., upper=imagesize[1]),
                         transform=transformsref["none"]),
    ]
    modelastro = mpfobj.AstrometricModel(paramsastrometry)
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

    paramfluxes = [mpfobj.FluxParameter(
            band, "flux", np.log10(fluxesbyband[band]), None, limits=limitsref["none"],
            transform=transformsref["log10"], transformed=True, prior=None, fixed=False,
            isfluxratio=False)
        for bandi, band in enumerate(bands)
    ]
    modelphoto = mpfobj.PhotometricModel(components, paramfluxes)
    if convertfluxfracs:
        modelphoto.convertfluxparameters(
            usefluxfracs=False, transform=transformsref['log10'], limits=limitsref["none"])
    source = mpfobj.Source(modelastro, modelphoto, name)
    model = mpfobj.Model([source], data, engine=engine, engineopts=engineopts)
    return model


# Convenience function to evaluate a model and optionally plot with title, returning chi map only
def evaluatemodel(model, plot=False, title=None, **kwargs):
    """
    Convenience function to evaluate a model and optionally at a title to the plot.
    :param model: multiprofit.Model
    :param plot: Boolean; generate plot?
    :param title: String; title to add on top of the plot.
    :param kwargs: Dict; additional arguments to pass to model.evaluate().
    :return: Chi maps for each exposure.
    """
    _, _, chis, _ = model.evaluate(plot=plot, **kwargs)

    if plot:
        if title is not None:
            plt.suptitle(title)
        plt.show(block=False)
    return chis


# Convenience function to fit a model. kwargs are passed on to evaluatemodel
def fitmodel(model, modeller=None, modellib="scipy", modellibopts={'algo': "Nelder-Mead"}, printfinal=True,
             printsteps=100, plot=False, dolinear=True, **kwargs):
    """
    Convenience function to fit a model with reasonable defaults.
    :param model: multiprofit.Model
    :param modeller: multiprofit.Modeller; default: new Modeller.
    :param modellib: String; the modelling library to use if modeller is None.
    :param modellibopts: Dict; options to pass to the modeller if modeller is None.
    :param printfinal: Boolean; print the final parameter value?
    :param printsteps: Integer; step interval between printing.
    :param plot: Boolean; plot final fit?
    :param kwargs: Dict; passed to evaluatemodel() after fitting is complete (e.g. plotting options).
    :return: Tuple of modeller.fit and modeller.
    """
    if modeller is None:
        modeller = mpfobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts)
    fit = modeller.fit(printfinal=printfinal, printsteps=printsteps, dolinear=dolinear)
    if printfinal:
        paramsall = model.getparameters(fixed=True)
        print("Param names:" + ",".join(["{:11s}".format(p.name) for p in paramsall]))
        print("All params: " + ",".join(["{:+.4e}".format(p.getvalue(transformed=False)) for p in paramsall]))
    # Conveniently sets the parameters to the right values too
    # TODO: Find a better way to ensure chis are returned than setting drawimage=True
    chis = evaluatemodel(model, plot=plot, params=fit["paramsbest"], drawimage=True, **kwargs)
    fit["chisqred"] = mpfutil.getchisqred(chis)
    params = model.getparameters()
    for item in ['paramsbestall', 'paramsbestalltransformed', 'paramsallfixed']:
        fit[item] = []
    for param in params:
        fit["paramsbestall"].append(param.getvalue(transformed=False))
        fit["paramsbestalltransformed"].append(param.getvalue(transformed=True))
        fit["paramsallfixed"].append(param.fixed)

    return fit, modeller


def fitpsf(modeltype, imgpsf, engines, band, psfmodelfits=None, sigmainverse=None, modellib="scipy",
           modellibopts={'algo': 'Nelder-Mead'}, plot=False, title='', modelname=None,
           label=None, printfinal=True, printsteps=100, figaxes=(None, None), figurerow=None, redo=True):
    if psfmodelfits is None:
        psfmodelfits = {}
    if modelname is None:
        modelname = modeltype
    # Fit the PSF
    numcomps = np.int(modeltype.split(":")[1])
    for engine, engineopts in engines.items():
        if engine not in psfmodelfits:
            psfmodelfits[engine] = {}
        if redo or modelname not in psfmodelfits[engine]:
            model = getpsfmodel(engine, engineopts, numcomps, band, modeltype, imgpsf,
                                sigmainverse=sigmainverse)
            psfmodelfits[engine][modelname] = {}
        else:
            model = psfmodelfits[engine][modelname]['modeller'].model
        model.name = '.'.join(['PSF', band, modelname])
        if redo or 'fit' not in psfmodelfits[engine][modelname]:
            psfmodelfits[engine][modelname]['fit'], psfmodelfits[engine][modelname]['modeller'] = \
                fitmodel(
                model, modellib=modellib, modellibopts=modellibopts, printfinal=printfinal,
                printsteps=printsteps, plot=plot, title=title, modelname=label,
                figure=figaxes[0], axes=figaxes[1], figurerow=figurerow)
        elif plot:
            exposure = model.data.exposures[band][0]
            isempty = isinstance(exposure.image, ImageEmpty)
            if isempty:
                setexposure(model, band, image=imgpsf, sigmainverse=sigmainverse)
            evaluatemodel(
                model, params=psfmodelfits[engine][modelname]['fit']['paramsbest'],
                plot=plot, title=title, modelname=label, figure=figaxes[0], axes=figaxes[1],
                figurerow=figurerow)
            if isempty:
                setexposure(model, band, image='empty')

    return psfmodelfits


# Engine is galsim; TODO: add options
def fitgalaxy(
        exposurespsfs, modelspecs, modellib=None, modellibopts=None, plot=False, name=None, models=None,
        fitsbyengine=None, redo=False, imgplotmaxs=None, imgplotmaxmulti=None, weightsband=None,
        fitfluxfracs=False, printsteps=100,
):
    """
    Convenience function to fit a galaxy given some exposures with PSFs.

    :param exposurespsfs: Iterable of tuple(mpfobj.Exposure, dict; key=psftype: value=mpfobj.PSF)
    :param modelspecs: Model specifications as returned by getmodelspecs
    :param modellib: string; Model fitting library
    :param modellibopts: dict; Model fitting library options
    :param plot: bool; Make plots?
    :param name: string; Name of the model for plot labelling
    :param models: dict; key=model name: value=mpfobj.Model
    :param fitsbyengine: dict; same format as return value.
    :param redo: bool; Redo any pre-existing fits in fitsbyengine?
    :param imgplotmaxs: dict; key=band: value=float (Maximum value when plotting images in this band)
    :param imgplotmaxmulti: float; Maximum value of summed images when plotting multi-band.
    :param weightsband: dict; key=band: value=float (Multiplicative weight when plotting multi-band RGB).
    :param fitfluxfracs: bool; fit component flux ratios instead of absolute fluxes?

    :return: fitsbyengine: dict; key=engine: value=dict; key=modelname: value=dict;
        key='fits': value=array of fit results, key='modeltype': value =
        fitsbyengine[engine][modelname] = {"fits": fits, "modeltype": modeltype}
        , models: tuple of complicated structures:

        modelinfos: dict; key=model name: value=dict; TBD
        models: dict; key=model name: value=mpfobj.Model
        psfmodels: dict: TBD
    """
    bands = OrderedDict()
    fluxes = {}
    npiximg = None
    paramnamesmomentsinit = ["axrat", "ang", "re"]
    initfrommoments = {paramname: 0 for paramname in paramnamesmomentsinit}
    for exposure, _ in exposurespsfs:
        band = exposure.band
        imgarr = exposure.image
        npiximgexp = imgarr.shape
        if npiximg is None:
            npiximg = npiximgexp
        elif npiximgexp != npiximg:
            'fitgalaxy exposure image shape={} not same as first={}'.format(npiximgexp, npiximg)
        if band not in bands:
            for paramname, value in zip(paramnamesmomentsinit, mpfutil.estimateellipse(imgarr)):
                # TODO: Find a way to average/median angles without annoying periodicity problems
                # i.e. average([1, 359]) = 180
                # Obvious solution is to convert to another parameterization like covariance, etc.
                if paramname == 'ang':
                    initfrommoments[paramname] = value
                else:
                    initfrommoments[paramname] += value**2
            bands[exposure.band] = None
        # TODO: Figure out what to do if given multiple exposures per band (TBD if we want to)
        fluxes[band] = np.sum(imgarr[exposure.maskinverse] if exposure.maskinverse is not None else imgarr)
    npiximg = np.flip(npiximg, axis=0)
    for paramname in initfrommoments:
        if paramname != 'ang':
            initfrommoments[paramname] = np.sqrt(initfrommoments[paramname]/len(bands))
    print('Bands:', bands, 'Moment init.:', initfrommoments)
    engine = 'galsim'
    engines = {
        engine: {
            "gsparams": gs.GSParams(
                kvalue_accuracy=1e-2, integration_relerr=1e-2, integration_abserr=1e-3,
                maximum_fft_size=32768)}
    }
    title = name if plot else None
    if modellib is None:
        modellib = "scipy"

    valuesmax = {
       "re": np.sqrt(np.sum((npiximg/2.)**2)),
    }
    valuesmin = {}
    for band in bands:
        valuesmin["flux_" + band] = 1e-4 * fluxes[band]
        valuesmax["flux_" + band] = 100 * fluxes[band]
    models = {} if (models is None) else models
    paramsfixeddefault = {}
    fitsbyengine = {} if ((models is None) or (fitsbyengine is None)) else fitsbyengine
    usemodellibdefault = modellibopts is None
    for engine, engineopts in engines.items():
        if engine not in fitsbyengine:
            fitsbyengine[engine] = {}
        fitsengine = fitsbyengine[engine]
        if plot:
            nrows = len(modelspecs)
            figures = {}
            axeses = {}
            for band in list(bands) + (['multi'] if len(bands) > 1 else []):
                ncols = 5
                # Change to landscape
                figure, axes = plt.subplots(nrows=min([ncols, nrows]), ncols=max([ncols, nrows]))
                if nrows > ncols:
                    axes = np.transpose(axes)
                # This keeps things consistent with the nrows>1 case
                if nrows == 1:
                    axes = np.array([axes])
                plt.suptitle(title + " {} model".format(engine))
                figures[band] = figure
                axeses[band] = axes
            if len(bands) == 1:
                figures = figures[band]
                axeses = axeses[band]
            plotascolumn = nrows > ncols
        else:
            figures = None
            axeses = None
            plotascolumn = None
        for modelidx, modelinfo in enumerate(modelspecs):
            modelname = modelinfo["name"]
            modeltype = modelinfo["model"]
            modeldefault = getmodel(
                fluxes, modeltype, npiximg, engine=engine, engineopts=engineopts,
                convertfluxfracs=not fitfluxfracs
            )
            paramsfixeddefault[modeltype] = [param.fixed for param in
                                             modeldefault.getparameters(fixed=True)]
            existsmodel = modeltype in models
            model = modeldefault if not existsmodel else models[modeltype]
            if not existsmodel:
                models[modeltype] = model
            psfname = modelinfo["psfmodel"] + ("_pixelated" if mpfutil.str2bool(
                modelinfo["psfpixel"]) else "")
            model.data.exposures = {band: [] for band in bands}
            for exposure, psfs in exposurespsfs:
                exposure.psf = psfs[engine][psfname]['object']
                model.data.exposures[exposure.band].append(exposure)
            plotmulti = plot and len(bands) > 1
            if not redo and modelname in fitsbyengine[engine] and \
                    'fits' in fitsbyengine[engine][modelname]:
                if plot:
                    valuesbest = fitsengine[modelname]['fits'][-1]['paramsbestalltransformed']
                    # TODO: consider how to avoid code repetition here and below
                    modelnameappendparams = []
                    for param, value in zip(model.getparameters(fixed=True), valuesbest):
                        param.setvalue(value, transformed=True)
                        if (param.name == "nser" and (
                                not param.fixed or param.getvalue(transformed=False) != 0.5)) or \
                            param.name == "re" or (isfluxratio(param) and param.getvalue(
                                transformed=False) < 1):
                            modelnameappendparams += [('{:.2f}', param)]
                    if title is not None:
                        plt.suptitle(title)
                    model.evaluate(
                        plot=plot, modelname=modelname, modelnameappendparams=modelnameappendparams,
                        figure=figures, axes=axeses, figurerow=modelidx, plotascolumn=plotascolumn,
                        plotmulti=plotmulti, imgplotmaxs=imgplotmaxs, imgplotmaxmulti=imgplotmaxmulti,
                        weightsband=weightsband)
                    plt.show(block=False)
            else:
                # Parse default overrides from model spec
                paramflagkeys = ['inherit', 'modify']
                paramflags = {key: [] for key in paramflagkeys}
                for flag in ['fixed', 'init']:
                    paramflags[flag] = {}
                    values = modelinfo[flag + 'params']
                    if values:
                        for flagvalue in values.split(";"):
                            if flag == "fixed":
                                paramflags[flag][flagvalue] = None
                            elif flag == "init":
                                value = flagvalue.split("=")
                                # TODO: improve input handling here or just revamp the whole system later
                                if value[1] in paramflagkeys:
                                    paramflags[value[1]].append(value[0])
                                else:
                                    valuesplit = [np.float(x) for x in value[1].split(',')]
                                    paramflags[flag][value[0]] = valuesplit
                # Initialize model from estimate of moments (size/ellipticity) or from another fit
                inittype = modelinfo['inittype']
                if inittype == 'moments':
                    print('Initializing from moments')
                    for param in model.getparameters(fixed=False):
                        if param.name in initfrommoments:
                            param.setvalue(initfrommoments[param.name], transformed=False)
                else:
                    model = initmodel(model, modeltype, inittype, models, modelspecs[0:modelidx], fitsengine,
                                      bands=bands, paramsinherit=paramflags['inherit'],
                                      paramsmodify=paramflags['modify'])

                # Reset parameter fixed status
                for param, fixed in zip(model.getparameters(fixed=True, modifiers=False),
                                        paramsfixeddefault[modeltype]):
                    if param.name not in paramflags['inherit']:
                        param.fixed = fixed
                # For printing parameter values when plotting
                modelnameappendparams = []
                # Now actually apply the overrides and the hardcoded maxima
                timesmatched = {}
                for param in model.getparameters(fixed=True):
                    isflux = isinstance(param, mpfobj.FluxParameter)
                    isfluxrat = isfluxratio(param)
                    paramname = param.name if not isflux else (
                        'flux' + ('ratio' if isfluxrat else '') + '_' + param.band)
                    if paramname in paramflags['fixed'] or (isflux and 'flux' in paramflags['fixed']):
                        param.fixed = True
                    # TODO: Figure out a better way to reset modifiers to be free
                    elif paramname == 'rscale':
                        param.fixed = False
                    if paramname in paramflags["init"]:
                        if paramname not in timesmatched:
                            timesmatched[paramname] = 0
                        # If there's only one input value, assume it applies to all instances of this param
                        idxparaminit = (0 if len(paramflags["init"][paramname]) == 1 else
                                        timesmatched[paramname])
                        param.setvalue(paramflags["init"][paramname][idxparaminit],
                                       transformed=False)
                        timesmatched[paramname] += 1
                    if plot and not param.fixed:
                        if paramname == "nser" or isfluxrat:
                            modelnameappendparams += [('{:.2f}', param)]
                    # Try to set a hard limit on params that need them with a logit transform
                    # This way even methods that don't respect bounds will have to until the transformed
                    # value reaches +/-inf, at least
                    if paramname in valuesmax:
                        valuemin = 0 if paramname not in valuesmin else valuesmin[paramname]
                        valuemax = valuesmax[paramname]
                        # Most scipy algos ignore limits, so we need to restrict the range manually
                        if modellib == 'scipy':
                            factor = 1/fluxes[param.band] if isflux else 1
                            value = np.max([np.min([param.getvalue(transformed=False), valuemax]), valuemin])
                            param.transform = getlogitlimited(valuemin, valuemax, factor=factor)
                            param.setvalue(value, transformed=False)
                        else:
                            transform = param.transform.transform
                            param.limits = mpfobj.Limits(
                                lower=transform(valuemin), upper=transform(valuemax),
                                transformed=True)
                    # Reset non-finite free param values
                    # This occurs e.g. at the limits of a logit transformed param
                    if not param.fixed:
                        paramval = param.getvalue(transformed=False)
                        paramvaltrans = param.getvalue(transformed=True)
                        if not np.isfinite(paramvaltrans):
                            # Get the next float in the direction of inf if -inf else -inf
                            # This works for nans too, otherwise we could use -paramval
                            # TODO: Deal with nans explicitly - they may not be recoverable
                            direction = -np.inf * np.sign(paramvaltrans)
                            # This is probably excessive but this ought to allow for a non-zero init. gradient
                            for _ in range(100):
                                paramval = np.nextafter(paramval, direction)
                            param.setvalue(paramval, transformed=False)

                paramvals = np.array([x.getvalue(transformed=False) for x in model.getparameters(fixed=True)])
                if not all(np.isfinite(paramvals)):
                    raise RuntimeError('Not all params finite for model {}:'.format(modelname), paramvals)

                print("Fitting model {:s} of type {:s} using engine {:s}".format(
                    modelname, modeltype, engine))
                model.name = modelname
                sys.stdout.flush()
                model.evaluate()
                try:
                    fits = []
                    dosecond = len(model.sources[0].modelphotometric.components) > 1 or not usemodellibdefault
                    if usemodellibdefault:
                        modellibopts = {
                            "algo": ("lbfgs" if modellib == "pygmo" else "L-BFGS-B") if dosecond else
                            ("neldermead" if modellib == "pygmo" else "Nelder-Mead")
                        }
                        if modellib == "scipy":
                            modellibopts['options'] = {'maxfun': 1e4}
                    fit1, modeller = fitmodel(
                        model, modellib=modellib, modellibopts=modellibopts, printfinal=True,
                        printsteps=printsteps, plot=plot and not dosecond, plotmulti=plotmulti,
                        figure=figures, axes=axeses, figurerow=modelidx, plotascolumn=plotascolumn,
                        modelname=modelname, modelnameappendparams=modelnameappendparams,
                        imgplotmaxs=imgplotmaxs, imgplotmaxmulti=imgplotmaxmulti, weightsband=weightsband,
                    )
                    fits.append(fit1)
                    if dosecond:
                        if usemodellibdefault:
                            modeller.modellibopts["algo"] = "neldermead" if modellib == "pygmo" else \
                                "Nelder-Mead"
                        fit2, _ = fitmodel(
                            model, modeller, printfinal=True, printsteps=printsteps,
                            plot=plot, plotmulti=plotmulti, figure=figures, axes=axeses,
                            figurerow=modelidx, plotascolumn=plotascolumn, modelname=modelname,
                            modelnameappendparams=modelnameappendparams, imgplotmaxs=imgplotmaxs,
                            imgplotmaxmulti=imgplotmaxmulti, weightsband=weightsband, dolinear=False,
                        )
                        fits.append(fit2)
                    fitsbyengine[engine][modelname] = {"fits": fits, "modeltype": modeltype}
                except Exception as e:
                    print("Error fitting galaxy {}:".format(name))
                    print(e)
                    trace = traceback.format_exc()
                    print(trace)
                    fitsbyengine[engine][modelname] = e, trace
    if plot:
        if len(bands) > 1:
            for figure in figures.values():
                plt.figure(figure.number)
                plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
        else:
            plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
        plt.show(block=False)

    return fitsbyengine, models


def fitgalaxyexposures(
        exposurespsfs, bands, modelspecs, results=None, plot=False, fitname=None, redo=False,
        redopsfs=False, resetimages=False, **kwargs
):
    """
    Fit a set of exposures and accompanying PSF images in the given bands with the requested model
    specifications.

    :param exposurespsfs: List of tuples (multiprofit.object.Exposure, nparray with PSF image)
    :param bands: List of bands
    :param modelspecs: List of dicts; as in getmodelspecs().
    :param results:
    :param plot: Boolean; generate plots?
    :param fitname: String; name of the galaxy/image to use as a title in plots
    :param redo: bool; Redo any pre-existing fits in fitsbyengine?
    :param redopsfs: Boolean; Redo any pre-existing PSF fits in results?
    :param resetimages: Boolean; reset all images in data structures to EmptyImages before returning results?
    :param kwargs: dict; keyword: value arguments to pass on to fitgalaxy()
    :return:
    """
    if results is None:
        results = {}
    metadata = {"bands": bands}
    # Having worked out what the image, psf and variance map are, fit PSFs and images
    psfs = results['psfs'] if 'psfs' in results else {}
    psfmodels = set([(x["psfmodel"], mpfutil.str2bool(x["psfpixel"])) for x in modelspecs])
    engine = 'galsim'
    engineopts = {
        "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
    }
    figure, axes = (None, None)
    psfrow = None
    if plot:
        npsfs = 0
        for psfmodeltype, _ in psfmodels:
            npsfs += psfmodeltype != "empirical"
        npsfs *= len(bands)
        if npsfs > 1:
            figure, axes = plt.subplots(nrows=min([5, npsfs]), ncols=max([5, npsfs]))
            psfrow = 0
    for idx, (exposure, psf) in enumerate(exposurespsfs):
        band = exposure.band
        if idx not in psfs:
            psfs[idx] = {engine: {}}
        for psfmodeltype, ispsfpixelated in psfmodels:
            psfname = psfmodeltype + ("_pixelated" if ispsfpixelated else "")
            label = psfmodeltype + (" pix." if ispsfpixelated else "") + " PSF"
            if psfmodeltype == "empirical":
                # TODO: Check if this works
                psfs[idx][engine][psfname] = {'object': mpfobj.PSF(
                    band=band, engine=engine, image=psf.image.array)}
            else:
                engineopts["drawmethod"] = "no_pixel" if ispsfpixelated else None
                refit = redopsfs or psfname not in psfs[idx][engine]
                if refit or plot:
                    if refit:
                        print('Fitting PSF band={} model={}'.format(band, psfname))
                    psfs[idx] = fitpsf(
                        psfmodeltype, psf.image.array, {engine: engineopts}, band=band,
                        psfmodelfits=psfs[idx], plot=plot, modelname=psfname, label=label, title=fitname,
                        figaxes=(figure, axes), figurerow=psfrow, redo=refit, printsteps=np.Inf)
                    if redo or 'object' not in psfs[idx][engine][psfname]:
                        psfs[idx][engine][psfname]['object'] = mpfobj.PSF(
                            band=band, engine=engine,
                            model=psfs[idx][engine][psfname]['modeller'].model.sources[0],
                            modelpixelated=ispsfpixelated)
                    if plot and psfrow is not None:
                        psfrow += 1
        exposurespsfs[idx] = (exposurespsfs[idx][0], psfs[idx])
    if plot:
        plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
    fitsbyengine = None if 'fits' not in results else results['fits']
    models = None if 'models' not in results else results['models']
    kwargs['redo'] = redo
    fits, models = fitgalaxy(
        exposurespsfs, modelspecs=modelspecs, name=fitname, plot=plot, models=models,
        fitsbyengine=fitsbyengine, **kwargs)
    if resetimages:
        for idx, psfsengine in psfs.items():
            if engine in psfsengine:
                for psfmodeltype, psf in psfsengine[engine].items():
                    setexposure(psf['modeller'].model, exposurespsfs[idx][0].band, image='empty')
        for modelname, model in models.items():
            for band in bands:
                setexposure(model, band, image='empty')
        for engine, modelfitinfo in fits.items():
            for modelname, modelfits in modelfitinfo.items():
                if 'fits' in modelfits:
                    for fit in modelfits["fits"]:
                        fit["fitinfo"]["log"] = None
                        # Don't try to pickle pygmo problems for some reason I forget
                        if hasattr(fit["result"], "problem"):
                            fit["result"]["problem"] = None
    results = {'fits': fits, 'models': models, 'psfs': psfs, 'metadata': metadata}
    return results


def getpsfmodel(engine, engineopts, numcomps, band, psfmodel, psfimage, sigmainverse=None, factorsigma=1,
                sizeinpixels=8.0, axrat=0.92):
    model = getmodel({band: 1}, psfmodel, np.flip(psfimage.shape, axis=0),
                     sizeinpixels*10**((np.arange(numcomps) - numcomps/2)/numcomps),
                     np.repeat(axrat, numcomps),
                     np.linspace(start=0, stop=180, num=numcomps + 2)[1:(numcomps + 1)],
                     engine=engine, engineopts=engineopts)
    for param in model.getparameters(fixed=False):
        param.fixed = isinstance(param, mpfobj.FluxParameter) and not param.isfluxratio
    setexposure(model, band, image=psfimage, sigmainverse=sigmainverse, factorsigma=factorsigma)
    return model


def getmodelspecs(filename=None):
    """
    Read a model specification file, the format of which is to be described.

    :param filename: String; path to the model specification file.
    :return: modelspecs; list of dicts of key specification name: value specification value.
    """
    if filename is None:
        modelspecs = io.StringIO("\n".join([
            ",".join(["name", "model", "fixedparams", "initparams", "inittype", "psfmodel", "psfpixel"]),
            ",".join(["gausspix", "sersic:1", "nser", "nser=0.5", "moments", "gaussian:2", "T"]),
            ",".join(["gauss", "sersic:1",  "nser", "nser=0.5", "moments", "gaussian:2", "F"]),
            ",".join(["gaussinitpix ", "sersic:1", "nser", "", "gausspix", "gaussian:2", "F"]),
        ]))
    with modelspecs if filename is None else open(filename, 'r') as filecsv:
        modelspecs = [row for row in csv.reader(filecsv)]
    header = modelspecs[0]
    del modelspecs[0]
    modelspecs = [{name: modelspec[i] for i, name in enumerate(header)} for modelspec in modelspecs]
    # TODO: Validate modelspecs
    return modelspecs


def initmodelfrommodelfits(model, modelfits, fluxfracs=None):
    # TODO: Come up with a better structure for parameter
    # TODO: Move to utils as a generic init model from other model(s) method
    chisqreds = [value['chisqred'] for value in modelfits]
    if fluxfracs is None:
        modelbest = chisqreds.index(min(chisqreds))
        fluxfracs = 1./np.array(chisqreds)
        fluxfracs = fluxfracs/np.sum(fluxfracs)
    if not len(model.sources) == 1:
        raise RuntimeError("Can't init model with multiple sources from fits")
    hasfluxfracs = len(model.sources[0].modelphotometric.fluxes) > 0
    if hasfluxfracs:
        total = 1
        for i, frac in enumerate(fluxfracs):
            fluxfracs[i] = frac/total
            total -= frac
        fluxfracs[-1] = 1.0
    print('Initializing from best model={} w/fluxfracs: {}'.format(modelfits[modelbest]['name'], fluxfracs))
    paramtreebest = modelfits[modelbest]['paramtree']
    fluxcensinit = paramtreebest[0][1][0] + paramtreebest[0][0]
    # Get fluxes and components for init
    fluxesinit = []
    compsinit = []
    for modelfit in modelfits:
        paramtree = modelfit['paramtree']
        for param, value in zip(modelfit['params'], modelfit['paramvals']):
            param.setvalue(value, transformed=False)
        sourcefluxes = paramtree[0][1][0]
        if (len(sourcefluxes) > 0) != hasfluxfracs:
            raise RuntimeError('Can\'t init model with hasfluxfracs={} and opposite for model fit')
        for iflux, flux in enumerate(sourcefluxes):
            fluxisflux = isinstance(flux, mpfobj.FluxParameter)
            if fluxisflux and not flux.isfluxratio:
                fluxesinit.append(flux)
            else:
                raise RuntimeError(
                    "paramtree[0][1][0][{}] (type={}) isFluxParameter={} and/or isfluxratio".format(
                        iflux, type(flux), fluxisflux))
        for comp in paramtree[0][1][1:-1]:
            compsinit.append([(param, param.getvalue(transformed=False)) for param in comp])
    params = model.getparameters(fixed=True, flatten=False)
    # Assume one source
    paramssrc = params[0]
    fluxcomps = paramssrc[1]
    fluxcens = fluxcomps[0] + paramssrc[0]
    # The first list is fluxes
    comps = [comp for comp in fluxcomps[1:-1]]
    # Check if fluxcens all length three with a total flux parameter and two centers named cenx and ceny
    # TODO: More informative errors; check fluxesinit
    bands = set([flux.band for flux in fluxesinit])
    nbands = len(bands)
    for name, fluxcen in {"init": fluxcensinit, "new": fluxcens}.items():
        lenfluxcensexpect = 2 + nbands
        errfluxcens = len(fluxcens) != lenfluxcensexpect
        errfluxcensinit = len(fluxcensinit) != lenfluxcensexpect
        errmsg = None if not (errfluxcens or errfluxcensinit) else \
            '{} len(fluxcen{})={} != {}=(2 x,y cens + nbands={})'.format(
                name, 'init' if errfluxcensinit else '', len(fluxcens) if errfluxcens else len(fluxcensinit),
                lenfluxcensexpect, nbands)
        if errmsg is not None:
            raise RuntimeError(errmsg)
        for idxband in range(nbands):
            fluxisflux = isinstance(fluxcen[0], mpfobj.FluxParameter)
            if not fluxisflux or fluxcen[0].isfluxratio:
                raise RuntimeError("{} fluxcen[0] (type={}) isFluxParameter={} or isfluxratio".format(
                    name, type(fluxcen[0]), fluxisflux))
        if not (fluxcen[nbands].name == "cenx" and fluxcen[nbands+1].name == "ceny"):
            raise RuntimeError("{}[{}:{}] names=({},{}) not ('cenx','ceny')".format(
                name, nbands, nbands+1, fluxcen[nbands].name, fluxcen[nbands+1].name))
    for paramset, paraminit in zip(fluxcens, fluxcensinit):
        paramset.setvalue(paraminit.getvalue(transformed=False), transformed=False)
    # Check if ncomps equal
    if len(comps) != len(compsinit):
        raise RuntimeError("Model {} has {} components but prereqs {} have a total of "
                           "{}".format(model.name, len(comps), [x['modeltype'] for x in modelfits],
                                       len(compsinit)))
    for idxcomp, (compset, compinit) in enumerate(zip(comps, compsinit)):
        if len(compset) != len(compinit):
            # TODO: More informative error plz
            raise RuntimeError(
                '[len(compset)={}, len(compinit)={}, len(fluxfracs)={}] not identical'.format(
                    len(compset), len(compinit), len(fluxfracs)
                ))
        for paramset, (paraminit, value) in zip(compset, compinit):
            if isinstance(paramset, mpfobj.FluxParameter):
                if hasfluxfracs:
                    if not paramset.isfluxratio:
                        raise RuntimeError('Component flux parameter is not ratio but should be')
                    paramset.setvalue(fluxfracs[idxcomp], transformed=False)
                else:
                    if paramset.isfluxratio:
                        raise RuntimeError('Component flux parameter is ratio but shouldn\'t be')
                    # Note this means that the new total flux will be some weighted sum of the best fits
                    # for each model that went into this, which may not be ideal. Oh well!
                    paramset.setvalue(paraminit.getvalue(transformed=False)*fluxfracs[idxcomp],
                                      transformed=False)
            else:
                if type(paramset) != type(paraminit):
                    # TODO: finish this
                    raise RuntimeError("Param types don't match")
                if paramset.name != paraminit.name:
                    # TODO: warn or throw?
                    pass
                paramset.setvalue(value, transformed=False)


def initmodel(model, modeltype, inittype, models, modelinfocomps, fitsengine, bands=None,
              paramsinherit=None, paramsmodify=None):
    """
    Initialize a multiprofit.objects.Model of a given modeltype with a method inittype.

    :param model: A multiprofit.objects.Model.
    :param modeltype: String; a valid model type, as defined in TODO: define it somewhere.
    :param inittype: String; a valid initialization type, as defined in TODO: define it somewhere.
    :param models: Dict; key modeltype: value existing multiprofit.objects.Model.
        TODO: review if/when this is necessary.
    :param modelinfocomps: Model specifications to map onto individual components of the model,
        e.g. to initialize a two-component model from two single-component fits.
    :param bands:
    :param fitsengine:
    :param paramsinherit:
    :param paramsmodify: 
    :return: A multiprofit.objects.Model initialized as requested; it may be the original model or a new one.
    """
    # TODO: Refactor into function
    if inittype.startswith("best"):
        if inittype == "best":
            modelnamecomps = []

            # Loop through all previous models and add ones of the same type
            for modelinfocomp in modelinfocomps:
                if modelinfocomp["model"] == modeltype:
                    modelnamecomps.append(modelinfocomp['name'])
        else:
            # TODO: Check this more thoroughly
            modelnamecomps = inittype.split(":")[1].split(";")
            print(modelnamecomps)
        chisqreds = [fitsengine[modelnamecomp]["fits"][-1]["chisqred"]
                     for modelnamecomp in modelnamecomps]
        inittype = modelnamecomps[np.argmin(chisqreds)]
    else:
        inittype = inittype.split(';')
        if len(inittype) > 1:
            # Example:
            # mg8devexppx,mgsersic8:2,nser,"nser=4,1",mg8dev2px;mg8exppx,gaussian:3,T
            # ... means init two mgsersic8 profiles from some combination of the m8dev and mg8exp fits
            modelfits = [{
                'paramvals': fitsengine[initname]['fits'][-1]['paramsbestall'],
                'paramtree': models[fitsengine[initname]['modeltype']].getparameters(
                    fixed=True, flatten=False),
                'params': models[fitsengine[initname]['modeltype']].getparameters(fixed=True),
                'chisqred': fitsengine[initname]['fits'][-1]['chisqred'],
                'modeltype': fitsengine[initname]['modeltype'],
                'name': initname, }
                for initname in inittype
            ]
            initmodelfrommodelfits(model, modelfits)
            inittype = None
        else:
            inittype = inittype[0]
            if inittype not in fitsengine:
                # TODO: Fail or fall back here?
                raise RuntimeError("Model {} can't find reference {} "
                                   "to initialize from".format(modeltype, inittype))
    if inittype and 'fits' in fitsengine[inittype]:
        paramvalsinit = fitsengine[inittype]["fits"][-1]["paramsbestall"]
        # TODO: Find a more elegant method to do this
        inittypemod = fitsengine[inittype]['modeltype'].split('+')
        inittypesplit = inittypemod[0].split(':')
        modeltypebase = modeltype.split('+')[0]
        ismgtogauss = (
            len(inittypesplit) == 2
            and modeltypebase in ['gaussian:' + str(order) for order in
                                  MultiGaussianApproximationProfile.weights['sersic']]
            and inittypesplit[0] in [
                'mgsersic' + str(order) for order in MultiGaussianApproximationProfile.weights['sersic']]
            and inittypesplit[1].isdecimal()
        )
        if ismgtogauss:
            ncomponents = np.repeat(np.int(inittypesplit[0].split('mgsersic')[1]), inittypesplit[1])
            nsources = len(model.sources)
            modelnew = model
            model = models[fitsengine[inittype]['modeltype']]
            componentsnew = []
        print('Initializing from best model=' + inittype +
              ' (MGA to {} GMM)'.format(ncomponents) if ismgtogauss else '')
        # For mgtogauss, first we turn the mgsersic model into a true GMM
        # Then we take the old model and get the parameters that don't depend on components (mostly source
        # centers) and set those as appropriate
        for i in range(1+ismgtogauss):
            paramsall = model.getparameters(fixed=True, modifiers=not ismgtogauss)
            if ismgtogauss:
                print('Paramvalsinit:', paramvalsinit)
                print('Paramnames: ', [x.name for x in paramsall])
            if len(paramvalsinit) != len(paramsall):
                raise RuntimeError('len(paramvalsinit)={} != len(params)={}, paramsall={}'.format(
                    len(paramvalsinit), len(paramsall), [x.name for x in paramsall]))
            for param, value in zip(paramsall, paramvalsinit):
                # The logic here is that we can't start off an MG Sersic at n=0.5 since it's just one Gauss.
                # It's possible that a Gaussian mixture is better than an n<0.5 fit, so start it close to 0.55
                # Note that getcomponents (called by getmultigaussians below) will ignore input values of nser
                # This prevents having a multigaussian model with components having n>0.5 (bad!)
                if ismgtogauss and param.name == 'nser' and value <= 0.55:
                    value = 0.55
                param.setvalue(value, transformed=False)
            if ismgtogauss:
                print('Param vals:', [param.getvalue(transformed=False) for param in model.getparameters()])
            # Set the ellipse parameters fixed the first time through
            # The second time through, uh, ...? TODO Remember what happens
            if ismgtogauss and i == 0:
                for idxsrc in range(nsources):
                    componentsnew.append(getmultigaussians(
                        model.sources[idxsrc].getprofiles(bands=bands, engine='libprofit'),
                        paramsinherit=paramsinherit, paramsmodify=paramsmodify, ncomponents=ncomponents,
                        source=modelnew.sources[idxsrc]))
                    componentsold = model.sources[idxsrc].modelphotometric.components
                    for modeli in [model, modelnew]:
                        modeli.sources[idxsrc].modelphotometric.components = []
                    paramvalsinit = [param.getvalue(transformed=False)
                                     for param in model.getparameters(fixed=True)]
                    model.sources[idxsrc].modelphotometric.components = componentsold
                model = modelnew
        if ismgtogauss:
            for idxsrc in range(len(componentsnew)):
                model.sources[idxsrc].modelphotometric.components = componentsnew[idxsrc]

    return model


# Convenience function to set an exposure object with optional defaults for the sigma (variance) map
# Can be used to nullify an exposure before saving to disk, for example
def setexposure(model, band, index=0, image=None, sigmainverse=None, psf=None, mask=None, meta=None,
                factorsigma=1):
    if band not in model.data.exposures:
        model.data.exposures[band] = [mpfobj.Exposure(band=band, image=None)]
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
def getmultigaussians(profiles, paramsinherit=None, paramsmodify=None, ncomponents=None, source=None):
    """
    Get Gaussian component objects from profiles that are multi-Gaussian approximations (to e.g. Sersic)

    :param profiles: Dict; key band: value: profiles as formatted by mpf.objects.model.getprofiles()
    :param paramsinherit: List of parameter names for Gaussians to inherit the values of (e.g. axrat, ang)
    :param ncomponents: Array of ints specifying the number of Gaussian components in each physical
        component. Defaults to the number of Gaussians used to represent profiles, i.e. each Gaussian has
        independent ellipse parameters.
    :return: List of new Gaussian components
    """
    bands = set()
    for profile in profiles:
        for band in profile.keys():
            bands.add(band)
    bands = list(bands)
    bandref = bands[0]
    params = ['mag', 're', 'axrat', 'ang', 'nser']
    values = {}
    fluxes = {}
    for band in bands:
        # Keep these as lists to make the check against values[bandref] below easier (no need to call np.all)
        values[band] = {
            'size' if name == 're' else 'slope' if name == 'nser' else name:
                [profile[band][name] for profile in profiles]
            for name in params
        }
        # magtoflux needs a numpy array
        fluxes[band] = mpfutil.magtoflux(np.array(values[band]['mag']))
        # Ensure that only band-independent parameters are compared
        del values[band]['mag']
        if not values[band] == values[bandref]:
            raise RuntimeError('values[{}]={} != values[{}]={}; band-dependent values unsupported'.format(
                band, values[band], bandref, values[bandref]))

    # Comparing dicts above is easier with list elements but we want np arrays in the end
    values = {key: np.array(value) for key, value in values[band].items()}

    # These are the Gaussian components
    componentsgauss = getcomponents('gaussian', fluxes, values=values, isfluxesfracs=False)
    ncomponentsgauss = len(componentsgauss)
    if (paramsinherit is not None or paramsmodify is not None) and ncomponentsgauss > 1:
        if paramsinherit is None:
            paramsinherit = []
        if paramsmodify is None:
            paramsmodify = []
        if ncomponents is None:
            ncomponents = [ncomponentsgauss]
        ncomponents = np.array(ncomponents)
        if np.sum(ncomponents) != ncomponentsgauss:
            raise ValueError(
                'Number of Gaussian components={} != total number of Gaussian sub-components in physical '
                'components={}; list={}'.format(ncomponentsgauss, np.sum(ncomponents), ncomponents))

        # Inheritee has every right to be a word
        componentinit = 0
        paramsmodifycomps = {}
        if paramsmodify is not None:
            modifiers = source.modelphotometric.modifiers
            ncomponentsold = len(ncomponents)
            for nameofparam in paramsmodify:
                modifiersoftype = [modifier for modifier in modifiers if modifier.name == nameofparam]
                nmodifiers = len(modifiersoftype)
                if nmodifiers == 0:
                    if nameofparam == 'rscale':
                        paramsmodifycomps[nameofparam] = [
                            getparamdefault(nameofparam, value=1, isvaluetransformed=False, fixed=False)
                            for _ in range(ncomponentsold)
                        ]
                        for param in paramsmodifycomps[nameofparam]:
                            source.modelphotometric.modifiers.append(param)
                elif nmodifiers == ncomponentsold:
                    paramsmodifycomps[nameofparam] = modifiersoftype
                    if nameofparam == 'rscale':
                        for modifier in modifiersoftype:
                            modifier.setvalue(1, transformed=False)
                else:
                    raise RuntimeError('Expected to find {} modifiers of type {} but found {}'.format(
                        ncomponentsold, nameofparam, nmodifiers
                    ))
        for idxcomp, ncompstoadd in enumerate(ncomponents):
            paramsinheritees = {
                param.name: param for param in componentsgauss[componentinit].getparameters()
                if param.name in paramsinherit}
            for param in paramsinheritees.values():
                param.inheritors = []
            paramsmodifycomp = [paramoftype[idxcomp] for paramoftype in paramsmodifycomps.values()]
            for compoffset in range(ncompstoadd):
                comp = componentsgauss[componentinit+compoffset]
                for param in comp.getparameters():
                    if compoffset > 0 and param.name in paramsinherit:
                        # This param will map onto the first component's param
                        param.fixed = True
                        paramsinheritees[param.name].inheritors.append(param)
                    if param.name == 're':
                        param.modifiers += paramsmodifycomp
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
                    functions = {
                        'old': 'mpf.make_gaussian_pixel_sersic' + argsmpf,
                        'new': 'mpf.make_gaussian_pixel' + argsmpf,
                        'like': 'mpf.loglike_gaussians_pixel(data, data, '
                                'np.array([[' +
                                ','.join(np.repeat('{}', 6)).format(xdim/2, ydim/2, 1, reff, ang, axrat) +
                                ']]), 0, {}, 0, {}, np.zeros(({}, {})))'.format(
                                    xdim, ydim, ydim, xdim),
                    }
                    timesmpf = {
                        key: np.min(timeit.repeat(
                            callstr,
                            setup='import multiprofit as mpf' + (
                                '; import numpy as np; data=np.zeros([0, 0])' if key == 'like' else ''),
                            repeat=nbenchmark, number=1))
                        for key, callstr in functions.items()
                    }
                    if hasgs:
                        timegs = np.min(timeit.repeat(
                            'x=gs.Gaussian(flux=1, half_light_radius={}).shear(q={}, beta={}*gs.degrees)'
                            '.drawImage(nx={}, ny={}, scale=1, method="no_pixel").array'.format(
                                reff*np.sqrt(axrat), axrat, ang, xdim, ydim
                            ),
                            setup='import galsim as gs', repeat=nbenchmark, number=1
                        ))
                    mpffuncs = list(timesmpf.keys())
                    times = [timesmpf[x] for x in mpffuncs] + [timegs] if hasgs else []
                    result += ';' + '/'.join(mpffuncs) + ('/GalSim' if hasgs else '') + ' times=(' + \
                              ','.join(['{:.3e}'.format(x) for x in times]) + ')'
                results.append({
                    'string': result,
                    'xdim': xdim,
                    'ydim': ydim,
                    'reff': reff,
                    'axrat': axrat,
                    'ang': ang,
                })

    return results
