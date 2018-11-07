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

import multiprofit.objects as mpfobj
import multiprofit.util as mpfutil

options = {
    "algos":       {"default": {"scipy": "BFGS", "pygmo": "lbfgs"}},
    "backgrounds": {"default": [1.e3]},
    "engines":     {"avail": ["galsim", "libprofit"], "default": "galsim"},
    "bands":       {"default": [""]},
    "galaxyfluxes":     {"default": [1.e5]},
    "galaxyfluxmults":  {"default": [1.]},
    "galaxyradii":      {"default": [5.]},
    "galaxycenoffsets": {"default": [[0., 0.], [-0.5, -0.5]]},
    "imagesizes":  {"default": [60]},
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


def isfluxratio(param):
    return isinstance(param, mpfobj.FluxParameter) and param.isfluxratio


def getpsfmodel(engine, engineopts, numcomps, band, psfmodel, psfimage, sigmainverse=None, factorsigma=1):
    model = mpfutil.getmodel({band: 1}, psfmodel, np.flip(psfimage.shape, axis=0),
                             8.0 * 10 ** ((np.arange(numcomps) - numcomps / 2) / numcomps),
                             np.repeat(0.8, numcomps),
                             np.linspace(start=0, stop=180, num=numcomps + 2)[1:(numcomps + 1)],
                             engine=engine, engineopts=engineopts)
    for param in model.getparameters(fixed=False):
        param.fixed = isinstance(param, mpfobj.FluxParameter) and not param.isfluxratio
    mpfutil.setexposure(model, band, image=psfimage, sigmainverse=sigmainverse, factorsigma=factorsigma)
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


def fitpsf(imgpsf, psfmodel, engines, band, sigmainverse=None, modellib="scipy",
           modellibopts={'algo': 'Nelder-Mead'}, plot=False, title="", modelname="", printfinal=True,
           printsteps=100, figaxes=(None, None), figurerow=None):
    psfmodels = {}
    # Fit the PSF
    numcomps = np.int(psfmodel.split(":")[1])
    for engine, engineopts in engines.items():
        model = getpsfmodel(engine, engineopts, numcomps, band, psfmodel, imgpsf,
                            sigmainverse=sigmainverse)
        fit = mpfutil.fitmodel(model, modellib=modellib, modellibopts=modellibopts, printfinal=printfinal,
                               printsteps=printsteps, plot=plot, title=title, modelname=modelname + " PSF",
                               figure=figaxes[0], axes=figaxes[1], figurerow=figurerow)
        psfmodels[engine] = {"model": model, "fit": fit}
    return psfmodels


def initmodelfrommodelfits(model, modelfits):
    # TODO: Come up with a better structure for parameter
    # TODO: Move to utils as a generic init model from other model(s) method
    chisqreds = [value['chisqred'] for value in modelfits]
    modelbest = chisqreds.index(min(chisqreds))
    print('Initializing from best model=' + modelfits[modelbest]['name'])
    paramtreebest = modelfits[modelbest]['paramtree']
    fluxcensinit = paramtreebest[0][1][0] + paramtreebest[0][0]
    # Get fluxes and components for init
    fluxesinit = []
    compsinit = []
    for modelfit in modelfits:
        paramtree = modelfit['paramtree']
        for param, value in zip(modelfit['params'], modelfit['paramvals']):
            param.setvalue(value, transformed=False)
        for flux in paramtree[0][1][0]:
            fluxesinit.append(flux)
        for comp in paramtree[0][1][1:len(paramtree[0][1])]:
            compsinit.append([(param, param.getvalue(transformed=False)) for param in comp])
    params = model.getparameters(fixed=True, flatten=False)
    # Assume one source
    paramssrc = params[0]
    fluxcomps = paramssrc[1]
    fluxcens = fluxcomps[0] + paramssrc[0]
    comps = [comp for comp in fluxcomps[1:len(fluxcomps)]]
    # Check if fluxcens all length three with a total flux parameter and two centers named
    #  cenx and ceny
    # TODO: More informative errors
    for name, fluxcen in {"init": fluxcensinit, "new": fluxcens}.items():
        if len(fluxcen) != 3:
            raise RuntimeError("{} len(fluxcen[)={} != 3".format(name, len(fluxcen)))
        fluxisflux = isinstance(fluxcen[0], mpfobj.FluxParameter)
        if not fluxisflux or fluxcen[0].isfluxratio:
            raise RuntimeError("{} fluxcen[0] (type={}) isFluxParameter={} or isfluxratio".format(
                name, type(fluxcen[0]), fluxisflux))
        if not (fluxcen[1].name == "cenx" and fluxcen[2].name == "ceny"):
            raise RuntimeError("{}[1:2] names=({},{}) not ('cenx','ceny')".format(
                name, fluxcen[1].name, fluxcen[2].name))
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
            raise RuntimeError("Comps not same length")
        for paramset, (paraminit, value) in zip(compset, compinit):
            if isinstance(paramset, mpfobj.FluxParameter):
                # TODO: Should this be checked? Eventually we should override it smartly
                ratio = paramset.getvalue(transformed=False)
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
        print('Initializing from best model=' + inittype)
        paramvalsinit = fitsengine[inittype]["fits"][-1]["paramsbestall"]
        # TODO: Find a more elegant method to do this
        ismgtogauss = (modeltype == 'gaussian:8' and
                       fitsengine[inittype]['modeltype'] == 'multigaussiansersic:1')
        if ismgtogauss:
            modelnew = model
            model = models['multigaussiansersic:1']
        for i in range(1+ismgtogauss):
            paramsall = model.getparameters(fixed=True)
            if len(paramvalsinit) != len(paramsall):
                raise RuntimeError('len(paramvalsinit)={} != len(params)={}'.format(
                    len(paramvalsinit), len(paramsall)))
            for param, value in zip(paramsall, paramvalsinit):
                param.setvalue(value, transformed=False)
            if ismgtogauss and i == 0:
                componentsnew = mpfutil.getmultigaussians(
                    model.getprofiles(bands, engine='libprofit'), paramsinherit=paramsinherit)
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


def fitgalaxy(imgs, psfs, sigmainverses, bands, modelspecs, masks={}, modellib=None, modellibopts=None,
              plot=False, name=None, models=None, fitsbyengine=None, redoall=True,
              ):
    """

    :param img: ndarray; 2D Image
    :param psfs: Collection of mpfutil.PSF object
    :param sigmainverse: ndarray; 2D Inverse sigma image ndarr
    :param band: string; Filter/passband name
    :param mask: ndarray; 2D Inverse mask image (1=include, 0=omit)
    :param modelspecs: Model specifications as returned by getmodelspecs
    :param modellib: string; Model fitting library
    :param modellibopts: dict; Model fitting library options
    :param plot: bool; Make plots?
    :param name: string; Name of the model for plot labelling

    :return: fitsbyengine, models: tuple of complicated structures:
        modelinfos: dict; key=model name: value=dict; TBD
        models: dict; key=engine name: value=dict(key=model type: value=mpfobj.Model of that type)
        psfmodels: dict: TBD
    """
    print("Bands:", bands)
    print(imgs.keys())
    initfrommoments = {
        band: {
            name: value for name, value in zip(["axrat", "ang", "re"],
                                               getellipseestimate(imgs[band].array))
        } for band in bands
    }
    engines = {
        "galsim": {"gsparams": gs.GSParams(kvalue_accuracy=1e-2, integration_relerr=1e-2,
                                           integration_abserr=1e-3, maximum_fft_size=32768)}
    }
    title = name if plot else None
    img = imgs[bands[0]]
    npiximg = np.flip(img.array.shape, axis=0)
    fluxes = {band: np.sum(imgs[band].array[mask] if band in masks and masks[band] is not None else
                           imgs[band].array) for band in bands}

    valuesmax = {
        band: {
            "re": np.sqrt(np.sum((npiximg/2.)**2)),
            "flux": 10*fluxes[band],
        } for band in bands
    }
    # TODO: Remove when finished implementing multiband, maybe also img = ... above
    band = bands[0]
    psfs = psfs[band]
    mask = masks[band] if band in masks else None
    sigmainverse = sigmainverses[band]
    models = {} if (models is None) or redoall else models
    paramsfixeddefault = {}
    fitsbyengine = {} if ((models is None) or (fitsbyengine is None) or redoall) else fitsbyengine
    usemodellibdefault = modellibopts is None
    for engine, engineopts in engines.items():
        if (engine not in fitsbyengine) or redoall:
            fitsbyengine[engine] = {}
        fitsengine = fitsbyengine[engine]
        if plot:
            nrows = len(modelspecs)
            # Change to landscape
            figure, axes = plt.subplots(nrows=min([5, nrows]), ncols=max([5, nrows]))
            if nrows > 5:
                axes = np.transpose(axes)
            # This keeps things consistent with the nrows>1 case
            if nrows == 1:
                axes = np.array([axes])
            plt.suptitle(title + " {} model".format(engine))
            flipplot = nrows > 5
        else:
            figure = None
            axes = None
            flipplot = None
        for modelidx, modelinfo in enumerate(modelspecs):
            modelname = modelinfo["name"]
            modeltype = modelinfo["model"]
            modeldefault = mpfutil.getmodel(
                {band: fluxes[band]}, modeltype, npiximg, engine=engine, engineopts=engineopts
            )
            paramsfixeddefault[modeltype] = [param.fixed for param in
                                             modeldefault.getparameters(fixed=True)]
            existsmodel = modeltype in models
            model = modeldefault if not existsmodel else models[modeltype]
            if not existsmodel:
                models[modeltype] = model
            psfname = modelinfo["psfmodel"] + ("_pixelated" if mpfutil.str2bool(
                modelinfo["psfpixel"]) else "")
            mpfutil.setexposure(model, band, image=img.array, sigmainverse=sigmainverse,
                                psf=psfs[psfname]["object"], mask=mask)
            if not redoall and modelname in fitsbyengine[engine] and \
                    'fits' in fitsbyengine[engine][modelname]:
                if plot:
                    valuesbest = fitsengine[modelname]['fits'][-1]['paramsbestalltransformed']
                    # TODO: consider how to avoid code repetition here and below
                    modeldescs = {x: [] for x in ['f', 'n', 'r']}
                    formats = {x: '{:.1f}' if x == 'r' else '{:.2f}' for x in ['f', 'n', 'r']}
                    for param, value in zip(model.getparameters(fixed=True), valuesbest):
                        param.setvalue(value, transformed=True)
                        if param.name == "nser":
                            modeldescs['n'].append(param)
                        elif param.name == "re":
                            modeldescs['r'].append(param)
                        elif isfluxratio(param) and param.getvalue(transformed=False) < 1:
                            modeldescs['f'].append(param)
                    for key in ['n', 'r', 'f']:
                        if len(modeldescs[key]) > 3:
                            del modeldescs[key]
                    modeldescs = [paramname + '=' + ','.join(
                        [formats[paramname] .format(param.getvalue(transformed=False)) for param in params])
                        for paramname, params in modeldescs.items() if params]
                    modeldescs = ';'.join(modeldescs)
                    if title is not None:
                        plt.suptitle(title)
                    model.evaluate(plot=plot, modelname=modelname,
                                   modeldesc=modeldescs if modeldescs else None, figure=figure, axes=axes,
                                   figurerow=modelidx, flipplot=flipplot)
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
                inittype = modelinfo["inittype"]
                if inittype == "moments":
                    print('Initializing from moments')
                    for param in model.getparameters(fixed=False):
                        if param.name in initfrommoments[band]:
                            param.setvalue(initfrommoments[band][param.name], transformed=False)
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
                    paramname = param.name if not isinstance(param, mpfobj.FluxParameter) else \
                        'flux' + 'ratio' if param.isfluxratio else ''
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
                    isfluxrat = isfluxratio(param)
                    if plot and not param.fixed:
                        if paramname == "nser":
                            modelnameappendparams += [("n={:.2f}", param)]
                        elif isfluxrat:
                            modelnameappendparams += [("f={:.2f}", param)]
                    # Try to set a hard limit on params that need them with a logit transform
                    # This way even methods that don't respect bounds will have to until the transformed
                    # value reaches +/-inf, at least
                    valuesmaxband = valuesmax[band]
                    if paramname in valuesmaxband and not isfluxrat:
                        # Most scipy algos ignore limits, so we need to restrict the range manually
                        if modellib == 'scipy':
                            factor = 1/fluxes[band] if param.name == 'flux' else 1
                            value = np.min([param.getvalue(transformed=False), valuesmaxband[paramname]])
                            param.transform = mpfutil.getlogitlimited(0, valuesmaxband[paramname],
                                                                      factor=factor)
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
                    dosecond = (len(model.sources[0].modelphotometric.components) > 1) or not usemodellibdefault
                    if usemodellibdefault:
                        modellibopts = {
                            "algo": ("cobyla" if modellib == "pygmo" else "COBYLA") if dosecond else
                            ("neldermead" if modellib == "pygmo" else "Nelder-Mead")
                        }
                        if modellib == "scipy":
                            modellibopts['options'] = {'maxfun': 1e4}
                    fit1, modeller = mpfutil.fitmodel(model, modellib=modellib, modellibopts=modellibopts,
                                                      printfinal=True, printsteps=100,
                                                      plot=plot and not dosecond,
                                                      figure=figure, axes=axes, figurerow=modelidx,
                                                      flipplot=flipplot, modelname=modelname,
                                                      modelnameappendparams=modelnameappendparams
                                                      )
                    fits.append(fit1)
                    if dosecond:
                        if usemodellibdefault:
                            modeller.modellibopts["algo"] = "neldermead" if modellib == "pygmo" else \
                                "Nelder-Mead"
                        fit2, _ = mpfutil.fitmodel(model, modeller, printfinal=True, printsteps=100,
                                                   plot=plot, figure=figure, axes=axes, figurerow=modelidx,
                                                   flipplot=flipplot, modelname=modelname,
                                                   modelnameappendparams=modelnameappendparams)
                        fits.append(fit2)
                    fitsbyengine[engine][modelname] = {"fits": fits, "modeltype": modeltype}
                except Exception as e:
                    print("Error fitting id={}:".format(idnum))
                    print(e)
                    trace = traceback.format_exc()
                    print(trace)
                    fitsbyengine[engine][modelname] = e, trace
    if plot:
        plt.show(block=False)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
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
    # 0.03" scale: check cutouthst[0][0][1].header["CD1_1"] and ["CD2_2"]
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
def fitcosmosgalaxy(idcosmosgs, srcs, modelspecs, results={}, plot=False, redo=True,
                    modellib="scipy", modellibopts=None, hst2hscmodel=None, hscbands=['HSC-I'],
                    resetimages=False, resetfitlogs=False):
    if results is None:
        results = {}
    np.random.seed(idcosmosgs)
    radec = rgcfits[idcosmosgs][1:3]
    imghst = rgcat.getGalImage(idcosmosgs)
    scalehst = rgcfits[idcosmosgs]['PIXEL_SCALE']
    bandhst = rgcat.band[idcosmosgs]
    psfhst = rgcat.getPSF(idcosmosgs)
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
        exposures = {key: dataRef.get("deepCoadd_calexp") for key, dataRef in dataRefs.items()}
        scalehsc = exposures[bandref].getWcs().getPixelScale().asArcseconds()
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
        # TODO: None of this actually works in multiband, which it should for HSC one day
        bands = [rgcat.band[idcosmosgs]] if src == "hst" else hscbands
        metadata = {"bands": bands}
        masks = {}
        if src == "hst":
            imgs = {bands[0]: imghst}
            psfsband = {bands[0]: psfhst}
            sigmainverses = {bands[0]: [np.power(rgcat.getNoiseProperties(idcosmosgs)[2], -0.5)]}
        elif src.startswith("hsc") or src == 'hst2hsc':
            imgs = {}
            psfsband = {band: exposures[band].getPsf() for band in hscbands}
            sigmainverses = {}
            for band in hscbands:
                exposure = exposures[band]
                psf = psfsband[band]
                scalehscpsf = psf.getWcs(0).getPixelScale().asArcseconds()
                imgpsf = psf.computeImage().array
                imgpsfgs = gs.InterpolatedImage(gs.Image(imgpsf, scale=scalehscpsf))

                useNoiseReplacer = True
                if useNoiseReplacer:
                    noiseReplacer = rebuildNoiseReplacer(exposure, measCat)
                    noiseReplacer.insertSource(idHsc)
                cutouthsc = make_cutout.make_cutout_lsst(
                    spherePoint, exposure, size=np.floor_divide(sizeCutout, 2))
                idshsc = cutouthsc[4]
                var = exposure.getMaskedImage().getVariance().array[
                      idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]]

                if src == "hst2hsc":
                    # The COSMOS GalSim catalog is in the original HST frame, which is rotated by
                    # 10-12 degrees from RA/Dec axes; fit for this
                    result, anglehst, offsetxhst, offsetyhst = fitcosmosgalaxytransform(
                        radec[0], radec[1], imghst, imgpsfgs, sizeCutout, cutouthsc[0], var, scalehsc, plot=plot
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
                        # TODO: Store the name of the PyProFit model somewhere
                        # In fact there wasn't really any need to store a model since we
                        # can reconstruct it, but let's go ahead and use the unpickled one
                        paramsbest = fits[modeltouse]['fits'][-1]['paramsbestalltransformed']
                        # Apply all of the same rotations and shifts directly to the model
                        modeltouse = results['hst']['models'][modeltype]
                        metadata["hst2hscmodel"] = modeltouse
                        scalefactor = scalehst / scalehsc
                        imghstshape = imghst.array.shape
                        # I'm pretty sure that these should all be converted to arcsec units
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
                        exposuremodel.image = mpfutil.ImageEmpty((sizeCutout, sizeCutout))
                        # Save the GalSim model object
                        modeltouse.evaluate(keepmodels=True, getlikelihood=False, drawimage=False)
                        img = gs.Convolve(exposuremodel.meta['model'], imgpsfgs).drawImage(
                            nx=sizeCutout, ny=sizeCutout, scale=scalehsc) * fluxscale
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
                        img = copy.deepcopy(exposure.maskedImage.image.array[idshsc[3]: idshsc[2],
                                            idshsc[1]: idshsc[0]])
                    else:
                        footprint = measCat[row].getFootprint()
                        # TODO: There is presumably a much better way of doing this
                        img = copy.deepcopy(exposure.maskedImage.image)*0
                        footprint.getSpans().copyImage(exposure.maskedImage.image, img)
                    psf = imgpsfgs
                    mask = img != 0
                    img = gs.Image(img, scale=scalehsc)
                imgs[band] = img
                psfsband[band] = psf
                sigmainverses[band] = 1.0 / np.sqrt(var)
        else:
            raise RuntimeError('Unknown src ' + src)

        # Having worked out what the image, psf and variance map are, fit PSFs and images
        if srcname not in results:
            results[srcname] = {}
        psfs = {}
        psfmodels = set([(x["psfmodel"], mpfutil.str2bool(x["psfpixel"])) for x in modelspecs])
        engineopts = {
            "gsparams": gs.GSParams(kvalue_accuracy=1e-3, integration_relerr=1e-3, integration_abserr=1e-5)
        }
        fitname = "COSMOS #{}".format(idcosmosgs)
        figure, axes = (None, None)
        psfrow = None
        if plot:
            npsfs = 0
            for psfmodelname, _ in psfmodels:
                npsfs += psfmodelname != "empirical"
            if npsfs > 1:
                figure, axes = plt.subplots(nrows=min([5, npsfs]), ncols=max([5, npsfs]))
                psfrow = 0
        for band, psf in psfsband.items():
            psfs[band] = {}
            for psfmodelname, ispsfpixelated in psfmodels:
                psfname = psfmodelname + ("_pixelated" if ispsfpixelated else "")
                if psfmodelname == "empirical":
                    psfmodel = psf
                    psfexposure = mpfobj.PSF(band=band, engine="galsim", image=psf.image.array)
                else:
                    engineopts["drawmethod"] = "no_pixel" if ispsfpixelated else None
                    # TODO: Allow plotting of PSF fit after the fact
                    if redo or 'psfs' not in results[srcname] or band not in results[srcname]['psfs'] \
                            or psfname not in results[srcname]['psfs'][band]:
                        print('Fitting PSF model {}'.format(psfmodelname))
                        psfmodel = fitpsf(psf.image.array, psfmodelname, {"galsim": engineopts}, band=band,
                                          plot=plot, modelname=psfmodelname, title=fitname,
                                          figaxes=(figure, axes), figurerow=psfrow)["galsim"]
                        psfexposure = mpfobj.PSF(band=band, engine="galsim", model=psfmodel["model"].sources[0],
                                                 modelpixelated=ispsfpixelated)
                        if plot and psfrow is not None:
                            psfrow += 1
                    else:
                        psfmodel = results[srcname]['psfs'][band][psfname]['model']
                        psfexposure = results[srcname]['psfs'][band][psfname]['object']
                psfs[band][psfname] = {"model": psfmodel, "object": psfexposure}
        fitsbyengine = None if redo or 'fits' not in results[srcname] else results[srcname]['fits']
        models = None if redo or 'models' not in results[srcname] else results[srcname]['models']
        fits, models = fitgalaxy(
            imgs=imgs, psfs=psfs, sigmainverses=sigmainverses, masks=masks, bands=bands,
            modelspecs=modelspecs, name=fitname, modellib=modellib, plot=plot, models=models,
            fitsbyengine=fitsbyengine, redoall=redo)
        if resetimages:
            for psfmodelname, psf in psfs.items():
                if "model" in psf:
                    mpfutil.setexposure(psf["model"]["model"], band, "empty")
            for modelname, model in models.items():
                mpfutil.setexposure(model, band, "empty")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyProFit HST COSMOS galaxy modelling test')

    signature = inspect.signature(fitgalaxy)
    defaults = {
        k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty
    }

    flags = {
        'catalogpath': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog path'},
        'catalogfile': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog filename'},
        'fileout':     {'type': str, 'nargs': '?', 'default': None, 'help': 'File prefix to output results'},
        'fithsc':      {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HSC I band image'},
        'fithst':      {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image'},
        'fithst2hsc':  {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image convolved '
                                                                            'to HSC seeing'},
        'hscbands':    {'type': str, 'nargs': '*', 'default': ['HSC-I'], 'help': 'HSC Bands to fit'},
        'hst2hscmodel': {'type': str, 'default': None, 'help': 'HST model fit to use for mock HSC image'},
        'indices':     {'type': str, 'nargs': '*', 'default': None, 'help': 'Galaxy catalog index'},
        'modelspecfile': {'type': str, 'default': None, 'help': 'Model specification file'},
        'modellib':    {'type': str,   'nargs': '?', 'default': 'scipy', 'help': 'Optimization libraries'},
        'modellibopts':{'type': str,   'nargs': '?', 'default': None, 'help': 'Model fitting options'},
        'nwrite':      {'type': int, 'default': 5, 'help': 'Number of galaxies to fit before writing file'},
#        'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
        'plot':        {'type': mpfutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
#        'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
        'redo':        {'type': mpfutil.str2bool, 'default': True, 'help': 'Redo existing fits'},
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

    args = parser.parse_args()
    args.catalogpath = os.path.expanduser(args.catalogpath)
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

    if args.fileout is not None:
        args.fileout = os.path.expanduser(args.fileout)
    if args.fileout is not None and os.path.isfile(args.fileout):
            with open(args.fileout, 'rb') as f:
                data = pickle.load(f)
    else:
        data = {}

    if args.plot:
        mpl.rcParams['image.origin'] = 'lower'

    rgcfits = ap.io.fits.open(os.path.join(args.catalogpath, args.catalogfile))[1].data
    srcs = ["hst"] if args.fithst else []
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
        butler = dafPersist.Butler("/datasets/hsc/repo/rerun/RC/w_2018_30/DM-15120/")
        dataId = {"tract": tract}
        skymap = butler.get("deepCoadd_skyMap", dataId=dataId)
    if args.fithsc:
        srcs += ["hsc"]
    if args.fithst2hsc:
        srcs += ["hst2hsc"]

    nfit = 0
    for index in args.indices:
        idrange = [np.int(x) for x in index.split(",")]
        for idnum in range(idrange[0], idrange[0 + (len(idrange) > 1)] + (len(idrange) == 1)):
            print("Fitting COSMOS galaxy with ID: {}".format(idnum))
            try:
                fits = fitcosmosgalaxy(idnum, srcs=srcs, modelspecs=modelspecs, plot=args.plot,
                                       redo=args.redo, resetimages=True, resetfitlogs=True,
                                       hst2hscmodel=args.hst2hscmodel, hscbands=args.hscbands,
                                       modellib=args.modellib,
                                       results=data[idnum] if idnum in data else None)
                data[idnum] = fits
            except Exception as e:
                print("Error fitting id={}:".format(idnum))
                print(e)
                trace = traceback.format_exc()
                print(trace)
                if idnum not in data:
                    data[idnum] = {'error': e, 'trace': trace}
            nfit += 1
            if args.fileout is not None and (nfit % args.nwrite) == 0:
                with open(args.fileout, 'wb') as f:
                    pickle.dump(data, f)

    if args.fileout is not None:
        with open(args.fileout, 'wb') as f:
            pickle.dump(data, f)
    if args.plot:
        input("Press Enter to finish")
