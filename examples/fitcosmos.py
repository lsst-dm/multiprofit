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
import astropy as ap
import copy
import csv
import galsim as gs
import inspect
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize as spopt
import sys
import traceback

from collections import OrderedDict

import multiprofit.fitutils as mpffit
import multiprofit.objects as mpfobj
import multiprofit.utils as mpfutil

options = {
    "algos":       {"default": {"scipy": "BFGS", "pygmo": "lbfgs"}},
    "backgrounds": {"default": [1.e3]},
    "engines":     {"avail": ["galsim", "libprofit"], "default": "galsim"},
    "bands":       {"default": [""]},
    "galaxyfluxes":     {"default": [1.e5]},
    "galaxyfluxmults":  {"default": [1.]},
    "galaxyradii":      {"default": [5.]},
    "galaxycenoffsets": {"default": [[0., 0.], [-0.5, -0.5]]},
    "imagesize":   {"default": [60]},
    "optlibs":     {"avail": ["pygmo", "scipy"], "default": ["scipy"]},
    "psfaxrats":   {"default": [0.95]},
    "psffluxes":   {"default": [1.e4]},
    "psfradii":    {"default": [4.]},
    "psfsizes":    {"default": [21]},
    "psffit":      {"default": False},
    "psfmodeluse": {"default": False},
}


# Fairly standard moment of inertia estimate of ellipse orientation and size
# TODO: compare with galsim's convenient calculateHLR/FWHM
# TODO: replace with the stack's method (in meas_?)
def getellipseestimate(img, denoise=True):
    imgmeas = mpfutil.absconservetotal(np.copy(img)) if denoise else img
    y, x = np.nonzero(imgmeas)
    flux = imgmeas[y, x]
    y = y - imgmeas.shape[0]/2.
    x = x - imgmeas.shape[1]/2.
    inertia = np.zeros((2,2))
    inertia[0, 0] = np.sum(flux*x**2)
    inertia[0, 1] = np.sum(flux*x*y)
    inertia[1, 0] = inertia[0, 1]
    inertia[1, 1] = np.sum(flux*y**2)
    evals, evecs = np.linalg.eig(inertia)
    idxevalmax = np.argmax(evals)
    axrat = evals[1-idxevalmax]/evals[idxevalmax]
    ang = np.degrees(np.arctan2(evecs[1, idxevalmax], evecs[0, idxevalmax])) - 90
    if ang < 0:
        ang += 360
    return axrat, ang, np.sqrt(evals[idxevalmax]/np.sum(flux))


def getpsfmodel(engine, engineopts, numcomps, band, psfmodel, psfimage, sigmainverse=None, factorsigma=1):
    model = mpffit.getmodel({band: 1}, psfmodel, np.flip(psfimage.shape, axis=0),
                             8.0 * 10 ** ((np.arange(numcomps) - numcomps / 2) / numcomps),
                             np.repeat(0.95, numcomps),
                             np.linspace(start=0, stop=180, num=numcomps + 2)[1:(numcomps + 1)],
                             engine=engine, engineopts=engineopts)
    for param in model.getparameters(fixed=False):
        param.fixed = isinstance(param, mpfobj.FluxParameter) and not param.isfluxratio
    mpffit.setexposure(model, band, image=psfimage, sigmainverse=sigmainverse, factorsigma=factorsigma)
    return model


def getmodelspecs(filename=None):
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
        if redo or 'fit' not in psfmodelfits[engine][modelname]:
            psfmodelfits[engine][modelname]['fit'], psfmodelfits[engine][modelname]['modeller'] = \
                mpffit.fitmodel(
                model, modellib=modellib, modellibopts=modellibopts, printfinal=printfinal,
                printsteps=printsteps, plot=plot, title=title, modelname=label,
                figure=figaxes[0], axes=figaxes[1], figurerow=figurerow)
        elif plot:
            exposure = model.data.exposures[band][0]
            isempty = isinstance(exposure.image, mpffit.ImageEmpty)
            if isempty:
                mpffit.setexposure(model, band, image=imgpsf, sigmainverse=sigmainverse)
            mpffit.evaluatemodel(
                model, params=psfmodelfits[engine][modelname]['fit']['paramsbest'],
                plot=plot, title=title, modelname=label, figure=figaxes[0], axes=figaxes[1],
                figurerow=figurerow)
            if isempty:
                mpffit.setexposure(model, band, image='empty')

    return psfmodelfits


def initmodelfrommodelfits(model, modelfits):
    # TODO: Come up with a better structure for parameter
    # TODO: Move to utils as a generic init model from other model(s) method
    chisqreds = [value['chisqred'] for value in modelfits]
    modelbest = chisqreds.index(min(chisqreds))
    fluxfracs = 1./np.array(chisqreds)
    fluxfracs = fluxfracs/np.sum(fluxfracs)
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
        for iflux, flux in enumerate(paramtree[0][1][0]):
            fluxisflux = isinstance(flux, mpfobj.FluxParameter)
            if fluxisflux and not flux.isfluxratio:
                fluxesinit.append(flux)
            else:
                raise RuntimeError(
                    "paramtree[0][1][0][{}] (type={}) isFluxParameter={} and/or isfluxratio".format(
                        iflux, type(flux), fluxisflux))
        for comp in paramtree[0][1][1:len(paramtree[0][1])]:
            compsinit.append([(param, param.getvalue(transformed=False)) for param in comp])
    params = model.getparameters(fixed=True, flatten=False)
    # Assume one source
    paramssrc = params[0]
    fluxcomps = paramssrc[1]
    fluxcens = fluxcomps[0] + paramssrc[0]
    comps = [comp for comp in fluxcomps[1:len(fluxcomps)]]
    # Check if fluxcens all length three with a total flux parameter and two centers named cenx and ceny
    # TODO: More informative errors; check fluxesinit
    bands = set([flux.band for flux in fluxesinit])
    nbands = len(bands)
    for name, fluxcen in {"init": fluxcensinit, "new": fluxcens}.items():
        lenfluxcensexpect = 2 + nbands
        errfluxcens = len(fluxcens) != lenfluxcensexpect
        errfluxcensinit = len(fluxcensinit) != lenfluxcensexpect
        errmsg = None if not (errfluxcens or errfluxcensinit) else\
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
                if not paramset.isfluxratio:
                    raise RuntimeError('Component flux parameter is not ratio')
                paramset.setvalue(fluxfracs[idxcomp], transformed=False)
            else:
                if type(paramset) != type(paraminit):
                    # TODO: finish this
                    raise RuntimeError("Param types don't match")
                if paramset.name != paraminit.name:
                    # TODO: warn or throw?
                    pass
                paramset.setvalue(value, transformed=False)


def initmodel(model, modeltype, inittype, models, modelinfocomps, bands, fitsengine, paramsinherit=None):
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
        print(chisqreds)
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
        inittypesplit = fitsengine[inittype]['modeltype'].split(':')
        ismgtogauss = (modeltype in ['gaussian:' + str(order) for order in
                       mpfobj.MultiGaussianApproximationProfile.weights['sersic']] and
                       len(inittypesplit) == 2 and inittypesplit[0] in
                       ['mgsersic' + str(order) for order in
                        mpfobj.MultiGaussianApproximationProfile.weights['sersic']] and
                       inittypesplit[1].isdecimal()
                       )
        if ismgtogauss:
            ncomponents = np.repeat(np.int(inittypesplit[0].split('mgsersic')[1]), inittypesplit[1])
            modelnew = model
            model = models[fitsengine[inittype]['modeltype']]
        print('Initializing from best model=' + inittype +
              ' (MGA to {} GMM)'.format(ncomponents) if ismgtogauss else '')
        for i in range(1+ismgtogauss):
            if ismgtogauss:
                print('Paramvalsinit:', paramvalsinit)
            paramsall = model.getparameters(fixed=True)
            if len(paramvalsinit) != len(paramsall):
                raise RuntimeError('len(paramvalsinit)={} != len(params)={}'.format(
                    len(paramvalsinit), len(paramsall)))
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
                componentsnew = mpffit.getmultigaussians(
                    model.getprofiles(bands, engine='libprofit'), paramsinherit=paramsinherit,
                    ncomponents=ncomponents)
                componentsold = model.sources[0].modelphotometric.components
                for modeli in [model, modelnew]:
                    modeli.sources[0].modelphotometric.components = []
                paramvalsinit = [param.getvalue(transformed=False)
                                 for param in model.getparameters(fixed=True)]
                model.sources[0].modelphotometric.components = componentsold
                model = modelnew
        if ismgtogauss:
            model.sources[0].modelphotometric.components = componentsnew

    return model


# Engine is galsim; TODO: add options
def fitgalaxy(
        exposurespsfs, modelspecs, modellib=None, modellibopts=None, plot=False, name=None, models=None,
        fitsbyengine=None, redo=True, imgplotmaxs=None, imgplotmaxmulti=None, weightsband=None
):
    """

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
            for paramname, value in zip(paramnamesmomentsinit, getellipseestimate(imgarr)):
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

    valuesmax = {
       "re": np.sqrt(np.sum((npiximg/2.)**2)),
    }
    for band in bands:
        valuesmax["flux_" + band] = 10 * fluxes[band]
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
            modeldefault = mpffit.getmodel(
                fluxes, modeltype, npiximg, engine=engine, engineopts=engineopts
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
                            param.name == "re" or (mpffit.isfluxratio(param) and param.getvalue(
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
                paramflags = {'inherit': []}
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
                                if value[1] == 'inherit':
                                    paramflags['inherit'].append(value[0])
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
                    model = initmodel(model, modeltype, inittype, models, modelspecs[0:modelidx], bands,
                                      fitsengine, paramflags['inherit'])

                # Reset parameter fixed status
                for param, fixed in zip(model.getparameters(fixed=True), paramsfixeddefault[modeltype]):
                    if param.name not in paramflags['inherit']:
                        param.fixed = fixed
                # For printing parameter values when plotting
                modelnameappendparams = []
                # Now actually apply the overrides and the hardcoded maxima
                timesmatched = {}
                for param in model.getparameters(fixed=True):
                    isflux = isinstance(param, mpfobj.FluxParameter)
                    isfluxrat = mpffit.isfluxratio(param)
                    paramname = param.name if not isflux else (
                            'flux' + ('ratio' if isfluxrat else '') + '_' + param.band)
                    if paramname in paramflags["fixed"]:
                        param.fixed = True
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
                        valuemax = valuesmax[paramname]
                        # Most scipy algos ignore limits, so we need to restrict the range manually
                        if modellib == 'scipy':
                            factor = 1/fluxes[param.band] if isflux else 1
                            value = np.min([param.getvalue(transformed=False), valuemax])
                            param.transform = mpffit.getlogitlimited(0, valuemax, factor=factor)
                            param.setvalue(value, transformed=False)
                        else:
                            transform = param.transform.transform
                            param.limits = mpfobj.Limits(
                                lower=transform(0), upper=transform(valuesmaxband[paramname]),
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

                print("Fitting model {:s} of type {:s} using engine {:s}".format(modelname, modeltype, engine))
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
                    fit1, modeller = mpffit.fitmodel(
                        model, modellib=modellib, modellibopts=modellibopts, printfinal=True, printsteps=100,
                        plot=plot and not dosecond, plotmulti=plotmulti, figure=figures, axes=axeses,
                        figurerow=modelidx, plotascolumn=plotascolumn, modelname=modelname,
                        modelnameappendparams=modelnameappendparams, imgplotmaxs=imgplotmaxs,
                        imgplotmaxmulti=imgplotmaxmulti, weightsband=weightsband,
                    )
                    fits.append(fit1)
                    if dosecond:
                        if usemodellibdefault:
                            modeller.modellibopts["algo"] = "neldermead" if modellib == "pygmo" else \
                                "Nelder-Mead"
                        fit2, _ = mpffit.fitmodel(
                            model, modeller, printfinal=True, printsteps=100, plot=plot, plotmulti=plotmulti,
                            figure=figures, axes=axeses, figurerow=modelidx, plotascolumn=plotascolumn,
                            modelname=modelname, modelnameappendparams=modelnameappendparams,
                            imgplotmaxs=imgplotmaxs, imgplotmaxmulti=imgplotmaxmulti, weightsband=weightsband,
                        )
                        fits.append(fit2)
                    fitsbyengine[engine][modelname] = {"fits": fits, "modeltype": modeltype}
                except Exception as e:
                    print("Error fitting id={}:".format(idnum))
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


# Take a source (HST) image, convolve with a target (HSC) PSF, shift, rotate, and scale in amplitude until
# they sort of match.
# TODO: Would any attempt to incorporate the HST PSF improve things?
# This obviously won't work well if imgsrc's PSF isn't much smaller than imgtarget's
def imgoffsetchisq(x, args, returnimg=False):
    img = gs.Convolve(
        gs.InterpolatedImage(args['imgsrc']*10**x[0]).rotate(args['anglehst']),
        args['psf']).shift(x[1], x[2]).drawImage(nx=args['nx'], ny=args['ny'], scale=args['scale'])
    chisq = np.sum((img.array - args['imgtarget']) ** 2 / args['vartarget'])
    if returnimg:
        return img
    return chisq


# Fit the transform for a single COSMOS F814W image to match the HSC-I band image
def fitcosmosgalaxytransform(ra, dec, imghst, imgpsfgs, sizeCutout, cutouthsc, varhsc, scalehsc, plot=False):
    # Use Sophie's code to make our own cutout for comparison to the catalog
    # TODO: Double check the origin of these images; I assume they're the rotated and rescaled v2.0 mosaics
    cutouthst = make_cutout.cutout_HST(ra, dec, width=np.ceil(sizeCutout * scalehsc), return_data=True)
    hdr = cutouthst[0][0][1].header
    # 0.03" scale: check hdr["CD1_1"] and hdr["CD2_2"]
    if 'ORIENTAT' in hdr:
        print('hdr[\'ORIENTAT\']={}'.format(hdr['ORIENTAT']))
    imghstrot = gs.Image(cutouthst[0][0][1].data, scale=0.03)

    def getoffsetdiff(x, returnimg=False):
        img = gs.InterpolatedImage(imghstrot).rotate(-x[0] * gs.radians).shift(
            x[1], x[2]).drawImage(
            nx=imghst.array.shape[1], ny=imghst.array.shape[0],
            scale=imghst.scale)
        if returnimg:
            return img
        return np.sum(np.abs(img.array - imghst.array))

    result = spopt.minimize(getoffsetdiff, [-np.pi / 2, 0, 0], method="Nelder-Mead")
    result2 = spopt.minimize(getoffsetdiff, result.x + [np.pi, 0, 0], method="Nelder-Mead")
    [anglehst, xoffset, yoffset] = (result if result.fun < result2.fun else result2).x

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(np.log10(imghstrot.array))
        ax[0, 0].set_title("HST rotated")
        ax[1, 0].imshow(np.log10(imghst.array))
        ax[1, 0].set_title("COSMOS GalSim")
        ax[0, 1].imshow(np.log10(getoffsetdiff([anglehst, xoffset, yoffset], returnimg=True).array))
        ax[0, 1].set_title("HST re-rotated+shifted")

    scalefluxhst2hsc = np.sum(cutouthsc)/np.sum(imghst.array)
    args = {
        "imgsrc": imghst,
        "imgtarget": cutouthsc,
        "vartarget": varhsc,
        "psf": imgpsfgs,
        "nx": sizeCutout,
        "ny": sizeCutout,
        "scale": scalehsc,
        "anglehst": anglehst*gs.radians,
    }
    result = spopt.minimize(imgoffsetchisq, [np.log10(scalefluxhst2hsc), 0, 0], method="Nelder-Mead",
                            args=args)
    print("Offsetchisq fit params:", result.x)

    if plot:
        ax[1, 1].imshow(np.log10(
            gs.InterpolatedImage(imghst * 10 ** result.x[0]).rotate(anglehst * gs.radians).shift(
                -xoffset, -yoffset).drawImage(nx=imghst.array.shape[1], ny=imghst.array.shape[0],
                                              scale=imghst.scale).array))
        ax[1, 1].set_title("COSMOS GalSim rotated+shifted")

    return result, anglehst, xoffset, yoffset


# PSFmodels: array of tuples (modelname, ispixelated)
def fitcosmosgalaxy(
        idcosmosgs, srcs, modelspecs, rgcfits, rgcat, ccat, results={}, plot=False, redo=True, redopsfs=False,
        modellib="scipy", modellibopts=None, hst2hscmodel=None, hscbands=['HSC-I'], resetimages=False,
        imgplotmaxs=None, imgplotmaxmulti=None, weightsband=None):
    if results is None:
        results = {}
    np.random.seed(idcosmosgs)
    radec = rgcfits[idcosmosgs][1:3]
    imghst = rgcat.getGalImage(idcosmosgs)
    scalehst = rgcfits[idcosmosgs]['PIXEL_SCALE']
    bandhst = rgcat.band[idcosmosgs]
    psfhst = rgcat.getPSF(idcosmosgs)
    coaddshsc = None
    if "hsc" in srcs or "hst2hsc" in srcs:
        # Get the HSC dataRef
        spherePoint = geom.SpherePoint(radec[0], radec[1], geom.degrees)
        patch = skymap[tract].findPatch(spherePoint).getIndex()
        patch = ",".join([str(x) for x in patch])
        # HSC-I will be the reference for matching as it's closest to F814W
        bandref = 'HSC-I'
        dataRefs = {band: butler.dataRef(
            "deepCoadd", dataId={"tract": 9813, "patch": patch, "filter": band})
            for band in set(hscbands + [bandref])}
        # Get the coadd
        coaddshsc = {key: dataRef.get("deepCoadd_calexp") for key, dataRef in dataRefs.items()}
        scalehsc = coaddshsc[bandref].getWcs().getPixelScale().asArcseconds()
        # Get the measurements
        measCat = dataRefs[bandref].get("deepCoadd_meas")
        # Get and verify match
        distsq = ((radec[0] - np.degrees(measCat["coord_ra"])) ** 2 +
                  (radec[1] - np.degrees(measCat["coord_dec"])) ** 2)
        row = np.int(np.argmin(distsq))
        idHsc = measCat["id"][row]
        dist = np.sqrt(distsq[row]) * 3600
        print('Source distance={:.2e}"'.format(dist))
        # TODO: Threshold distance?
        if dist > 1:
            raise RuntimeError("Nearest HSC source at distance {:.3e}>1; aborting".format(dist))
        # Determine the HSC cutout size (larger than HST due to bigger PSF)
        sizeCutout = np.int(4 + np.ceil(np.max(imghst.array.shape) * scalehst / scalehsc))
        sizeCutout += np.int(sizeCutout % 2)

    # This does all of the necessary setup for each src. It should persist somehow
    for src in srcs:
        srcname = src
        bands = [rgcat.band[idcosmosgs]] if src == "hst" else hscbands
        metadata = {"bands": bands}
        exposures = {}
        if src == "hst":
            exposures[bands[0]] = mpfobj.Exposure(
                bands[0], imghst.array,
                sigmainverse=np.array([np.power(rgcat.getNoiseProperties(idcosmosgs)[2], -0.5)]))
            psfimgs = {bands[0]: psfhst}
        elif src.startswith("hsc") or src == 'hst2hsc':
            psfimgs = {band: coaddshsc[band].getPsf() for band in hscbands}
            for band in bands:
                coadd = coaddshsc[band]
                psf = psfimgs[band]
                scalehscpsf = psf.getWcs(0).getPixelScale().asArcseconds()
                imgpsf = psf.computeImage().array
                imgpsfgs = gs.InterpolatedImage(gs.Image(imgpsf, scale=scalehscpsf))

                useNoiseReplacer = True
                if useNoiseReplacer:
                    measCat = dataRefs[band].get("deepCoadd_meas")
                    noiseReplacer = rebuildNoiseReplacer(coadd, measCat)
                    print('noiseReplacer band={} seedMult={}, mean+/-std={:4f}+/-{:4f}'.format(
                        band, noiseReplacer.noiseSeedMultiplier, noiseReplacer.noiseGenMean,
                        noiseReplacer.noiseGenStd))
                    noiseReplacer.insertSource(idHsc)
                cutouthsc = make_cutout.make_cutout_lsst(
                    spherePoint, coadd, size=np.floor_divide(sizeCutout, 2))
                idshsc = cutouthsc[4]
                var = coadd.getMaskedImage().getVariance().array[
                      idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]]

                if src == "hst2hsc":
                    # The COSMOS GalSim catalog is in the original HST frame, which is rotated by
                    # 10-12 degrees from RA/Dec axes; fit for this
                    result, anglehst, offsetxhst, offsetyhst = fitcosmosgalaxytransform(
                        radec[0], radec[1], imghst, imgpsfgs, sizeCutout, cutouthsc[0], var, scalehsc,
                        plot=plot
                    )
                    fluxscale = (10 ** result.x[0])
                    metadata["lenhst2hsc"] = scalehst/scalehsc
                    metadata["fluxscalehst2hsc"] = fluxscale
                    metadata["anglehst2hsc"] = anglehst
                    metadata["offsetxhst2hsc"] = offsetxhst
                    metadata["offsetyhst2hsc"] = offsetyhst

                    realgalaxy = ccat.makeGalaxy(index=idcosmosgs, gal_type="real")

                    # Assuming that these images match, add HSC noise back in
                    if hst2hscmodel is None:
                        # TODO: Fix this as it's not working by default
                        img = imgoffsetchisq(result.x, returnimg=True, imgref=cutouthsc[0],
                                             psf=imgpsfgs, nx=sizeCutout, ny=sizeCutout, scale=scalehsc)
                        img = gs.Convolve(img, imgpsfgs).drawImage(nx=sizeCutout, ny=sizeCutout,
                                                                   scale=scalehsc) * fluxscale
                        # The PSF is now HSTPSF*HSCPSF, and "truth" is the deconvolved HST image/model
                        psf = gs.Convolve(imgpsfgs, psfhst.rotate(anglehst * gs.degrees)).drawImage(
                            nx=imgpsf.shape[1], ny=imgpsf.shape[0], scale=scalehscpsf
                        )
                        psf /= np.sum(psf.array)
                        psf = gs.InterpolatedImage(psf)
                    else:
                        fits = results['hst']['fits']['galsim']
                        if hst2hscmodel == 'best':
                            chisqredsmodel = {model: fit['fits'][-1]['chisqred'] for model, fit in fits.items()}
                            modeltouse = min(chisqredsmodel, key=chisqredsmodel.get)
                        else:
                            modeltouse = hst2hscmodel
                        modeltype = fits[modeltouse]['modeltype']
                        srcname = "_".join([src, hst2hscmodel])
                        # In fact there wasn't really any need to store a model since we
                        # can reconstruct it, but let's go ahead and use the unpickled one
                        paramsbest = fits[modeltouse]['fits'][-1]['paramsbestalltransformed']
                        # Apply all of the same rotations and shifts directly to the model
                        modeltouse = results['hst']['models'][modeltype]
                        metadata["hst2hscmodel"] = modeltouse
                        scalefactor = scalehst / scalehsc
                        imghstshape = imghst.array.shape
                        # I'm pretty sure that these should all be converted to arcsec units
                        # TODO: Verify above
                        for param, value in zip(modeltouse.getparameters(fixed=True), paramsbest):
                            param.setvalue(value, transformed=True)
                            valueset = param.getvalue(transformed=False)
                            if param.name == "cenx":
                                valueset = (scalehst * (valueset - imghstshape[1] / 2) + result.x[1]
                                            + sizeCutout / 2)
                            elif param.name == "ceny":
                                valueset = (scalehst * (valueset - imghstshape[0] / 2) + result.x[2]
                                            + sizeCutout / 2)
                            elif param.name == "ang":
                                valueset += np.degrees(anglehst)
                            elif param.name == "re":
                                valueset *= scalehst
                            param.setvalue(valueset, transformed=False)
                        exposuremodel = modeltouse.data.exposures[bandhst][0]
                        exposuremodel.image = mpffit.ImageEmpty((sizeCutout, sizeCutout))
                        # Save the GalSim model object
                        modeltouse.evaluate(keepmodels=True, getlikelihood=False, drawimage=False)
                        img = gs.Convolve(exposuremodel.meta['model'], imgpsfgs).drawImage(
                            nx=sizeCutout, ny=sizeCutout, scale=scalehsc).array * fluxscale
                        psf = imgpsfgs

                    noisetoadd = np.random.normal(scale=np.sqrt(var))
                    img += noisetoadd

                    if plot:
                        fig2, ax2 = plt.subplots(nrows=2, ncols=3)
                        ax2[0, 0].imshow(np.log10(cutouthsc[0]))
                        ax2[0, 0].set_title("HSC {}".format(band))
                        imghst2hsc = gs.Convolve(
                            realgalaxy.rotate(anglehst * gs.radians).shift(
                                result.x[1], result.x[2]
                            ), imgpsfgs).drawImage(
                            nx=sizeCutout, ny=sizeCutout, scale=scalehsc)
                        imghst2hsc += noisetoadd
                        imgsplot = (img.array, "my naive"), (imghst2hsc.array, "GS RealGal")
                        descpre = "HST {} - {}"
                        for imgidx, (imgit, desc) in enumerate(imgsplot):
                            ax2[1, 1 + imgidx].imshow(np.log10(imgit))
                            ax2[1, 1 + imgidx].set_title(descpre.format(bandhst, desc))
                            ax2[0, 1 + imgidx].imshow(np.log10(imgit))
                            ax2[0, 1 + imgidx].set_title((descpre + " + noise").format(
                                bandhst, desc))
                else:
                    # TODO: Use the mask properly
                    # mask = exposure.getMaskedImage().getMask()
                    # mask = mask.array[idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]]
                    if useNoiseReplacer:
                        img = copy.deepcopy(coadd.getMaskedImage().getImage().getArray()[
                                                idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]])
                    else:
                        footprint = measCat[row].getFootprint()
                        # TODO: There is presumably a much better way of doing this
                        img = copy.deepcopy(coadd.maskedImage.image) * 0
                        footprint.getSpans().copyImage(coadd.maskedImage.image, img)
                    psf = imgpsfgs
                    mask = img != 0
                exposures[band] = mpfobj.Exposure(band, img, sigmainverse=1.0/np.sqrt(var))
                psfimgs[band] = psf
        else:
            raise RuntimeError('Unknown src ' + src)

        # Having worked out what the image, psf and variance map are, fit PSFs and images
        if srcname not in results:
            results[srcname] = {}
        psfs = results[srcname]['psfs'] if 'psfs' in results[srcname] else {}
        psfmodels = set([(x["psfmodel"], mpfutil.str2bool(x["psfpixel"])) for x in modelspecs])
        engine = 'galsim'
        engineopts = {
            "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
        }
        fitname = "COSMOS #{}".format(idcosmosgs)
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
        for band, psf in psfimgs.items():
            if band not in psfs:
                psfs[band] = {engine: {}}
            for psfmodeltype, ispsfpixelated in psfmodels:
                psfname = psfmodeltype + ("_pixelated" if ispsfpixelated else "")
                label = psfmodeltype + (" pix." if ispsfpixelated else "") + " PSF"
                if psfmodeltype == "empirical":
                    # TODO: Check if this works
                    psfs[band][engine][psfname] = {'object': mpfobj.PSF(
                        band=band, engine=engine, image=psf.image.array)}
                else:
                    engineopts["drawmethod"] = "no_pixel" if ispsfpixelated else None
                    refit = redopsfs or psfname not in psfs[band][engine]
                    if refit or plot:
                        if refit:
                            print('Fitting PSF model {}'.format(psfname))
                        psfs[band] = fitpsf(
                            psfmodeltype, psf.image.array, {engine: engineopts}, band=band,
                            psfmodelfits=psfs[band], plot=plot, modelname=psfname, label=label, title=fitname,
                            figaxes=(figure, axes), figurerow=psfrow, redo=refit)
                        if redo or 'object' not in psfs[band][engine][psfname]:
                            psfs[band][engine][psfname]['object'] = mpfobj.PSF(
                                band=band, engine=engine,
                                model=psfs[band][engine][psfname]['modeller'].model.sources[0],
                                modelpixelated=ispsfpixelated)
                        if plot and psfrow is not None:
                            psfrow += 1
        if plot:
            plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.02, hspace=0.15)
        fitsbyengine = None if 'fits' not in results[srcname] else results[srcname]['fits']
        models = None if 'models' not in results[srcname] else results[srcname]['models']
        exposurespsfs = [(exposures[band], psfs[band]) for band in bands]
        fits, models = fitgalaxy(
            exposurespsfs, modelspecs=modelspecs, name=fitname, modellib=modellib, plot=plot, models=models,
            fitsbyengine=fitsbyengine, redo=redo, imgplotmaxs=imgplotmaxs, imgplotmaxmulti=imgplotmaxmulti,
            weightsband=weightsband)
        if resetimages:
            for band, psfsband in psfs.items():
                if engine in psfsband:
                    for psfmodeltype, psf in psfsband[engine].items():
                        mpffit.setexposure(psf['modeller'].model, band, image='empty')
            for modelname, model in models.items():
                for band in bands:
                    mpffit.setexposure(model, band, image='empty')
            for engine, modelfitinfo in fits.items():
                for modelname, modelfits in modelfitinfo.items():
                    if 'fits' in modelfits:
                        for fit in modelfits["fits"]:
                            fit["fitinfo"]["log"] = None
                            # Don't try to pickle pygmo problems for some reason I forget
                            if hasattr(fit["result"], "problem"):
                                fit["result"]["problem"] = None
        results[srcname] = {'fits': fits, 'models': models, 'psfs': psfs, 'metadata': metadata}

    return results


def main(args):
    modelspecs = getmodelspecs(None if args.modelspecfile is None else os.path.expanduser(args.modelspecfile))
    print('Loading COSMOS catalog at ' + os.path.join(args.catalogpath, args.catalogfile))
    try:
        rgcat = gs.RealGalaxyCatalog(args.catalogfile, dir=args.catalogpath)
    except Exception as e:
        print("Failed to load RealGalaxyCatalog {} in directory {}".format(
            args.catalogfile, args.catalogpath))
        print("Exception:", e)
        raise e
    try:
        ccat = gs.COSMOSCatalog(args.catalogfile, dir=args.catalogpath)
    except Exception as e:
        print("Failed to load COSMOSCatalog {} in directory {}".format(
            args.catalogfile, args.catalogpath))
        print("Not using COSMOSCatalog")
        ccat = None

    if args.file is not None:
        args.file = os.path.expanduser(args.file)
    if args.file is not None and os.path.isfile(args.file):
            with open(args.file, 'rb') as f:
                data = pickle.load(f)
    else:
        data = {}

    if args.plot:
        mpl.rcParams['image.origin'] = 'lower'

    rgcfits = ap.io.fits.open(os.path.join(args.catalogpath, args.catalogfile))[1].data
    srcs = ['hst'] if args.fithst else []
    if args.fithsc or args.fithst2hsc:
        from modelling_research import make_cutout
        import lsst.afw.geom as geom
        import lsst.daf.persistence as dafPersist
        from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
        from lsst.afw.table import SourceTable
        from lsst.meas.base.measurementInvestigationLib import makeRerunCatalog
        from lsst.meas.base import (SingleFrameMeasurementConfig,
                                    SingleFrameMeasurementTask)
        tract = 9813
        butler = dafPersist.Butler('/datasets/hsc/repo/rerun/RC/w_2019_02/DM-16110/')
        dataId = {"tract": tract}
        skymap = butler.get("deepCoadd_skyMap", dataId=dataId)
    if args.fithsc:
        srcs += ["hsc"]
    if args.fithst2hsc:
        srcs += ["hst2hsc"]
    bands = (['F814W'] if args.fithst else []) + (args.hscbands if (args.fithsc or args.fithst2hsc) else [])
    for argname, values in {'imgplotmaxs': args.imgplotmaxs, 'weightsband': args.weightsband}.items():
        if values is not None:
            if len(bands) != len(values):
                raise ValueError('len({}={})={} != len(bands={})={}'.format(
                    argname, values, len(values), bands, len(bands)))
            #values = {key: value for key, value in zip(bands, values)}

    nfit = 0
    for index in args.indices:
        idrange = [np.int(x) for x in index.split(",")]
        for idnum in range(idrange[0], idrange[0 + (len(idrange) > 1)] + (len(idrange) == 1)):
            print("Fitting COSMOS galaxy with ID: {}".format(idnum))
            try:
                fits = fitcosmosgalaxy(idnum, srcs=srcs, modelspecs=modelspecs, rgcfits=rgcfits, rgcat=rgcat,
                                       ccat=ccat, plot=args.plot, redo=args.redo, redopsfs=args.redopsfs,
                                       resetimages=True, hst2hscmodel=args.hst2hscmodel,
                                       hscbands=args.hscbands, modellib=args.modellib,
                                       results=data[idnum] if idnum in data else None,
                                       imgplotmaxs=args.imgplotmaxs, imgplotmaxmulti=args.imgplotmaxmulti,
                                       weightsband=args.weightsband)
                data[idnum] = fits
            except Exception as e:
                print("Error fitting id={}:".format(idnum))
                print(e)
                trace = traceback.format_exc()
                print(trace)
                if idnum not in data:
                    data[idnum] = {'error': e, 'trace': trace}
            nfit += 1
            if args.write and args.file is not None and (nfit % args.nwrite) == 0:
                with open(args.file, 'wb') as f:
                    pickle.dump(data, f)

    if args.write and args.file is not None:
        with open(args.file, 'wb') as f:
            pickle.dump(data, f)
    if args.plot:
        input("Press Enter to finish")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyProFit HST COSMOS galaxy modelling test')

    signature = inspect.signature(fitgalaxy)
    defaults = {
        k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty
    }

    flags = {
        'catalogpath': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog path'},
        'catalogfile': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog filename'},
        'file':        {'type': str, 'nargs': '?', 'default': None, 'help': 'Filename for input/output'},
        'fithsc':      {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HSC I band image'},
        'fithst':      {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image'},
        'fithst2hsc':  {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image convolved '
                                                                            'to HSC seeing'},
        'hscbands':    {'type': str, 'nargs': '*', 'default': ['HSC-I'], 'help': 'HSC Bands to fit'},
        'hst2hscmodel': {'type': str, 'default': None, 'help': 'HST model fit to use for mock HSC image'},
        'imgplotmaxs':  {'type': float, 'nargs': '*', 'default': None,
                         'help': 'Max. flux for scaling single-band images. F814W first if fitting HST, '
                                 'then HSC bands.'},
        'imgplotmaxmulti': {'type': float, 'default': None, 'help': 'Max. flux for scaling color images'},
        'indices':     {'type': str, 'nargs': '*', 'default': None, 'help': 'Galaxy catalog index'},
        'modelspecfile': {'type': str, 'default': None, 'help': 'Model specification file'},
        'modellib':    {'type': str,   'nargs': '?', 'default': 'scipy', 'help': 'Optimization libraries'},
        'modellibopts':{'type': str,   'nargs': '?', 'default': None, 'help': 'Model fitting options'},
        'nwrite':      {'type': int, 'default': 5, 'help': 'Number of galaxies to fit before writing file'},
#        'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
        'plot':        {'type': mpfutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
#        'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
        'redo':        {'type': mpfutil.str2bool, 'default': True, 'help': 'Redo existing fits'},
        'redopsfs':    {'type': mpfutil.str2bool, 'default': False, 'help': 'Redo existing PSF fits'},
        'weightsband': {'type': float, 'nargs': '*', 'default': None,
                        'help': 'Multiplicative weights for scaling images in multi-band RGB'},
        'write':       {'type': mpfutil.str2bool, 'default': True, 'help': 'Write file?'},
    }

    for key, value in flags.items():
        if key in options:
            default = options[key]["default"]
        else:
            default = value['default']
        if 'help' in value:
            value['help'] += ' (default: ' + str(default) + ')'
        value["default"] = default
        parser.add_argument('-' + key, **value)

    argsparsed = parser.parse_args()
    argsparsed.catalogpath = os.path.expanduser(argsparsed.catalogpath)
    main(argsparsed)
