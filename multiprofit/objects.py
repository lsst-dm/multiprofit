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

from abc import ABCMeta, abstractmethod
import astropy.visualization as apvis
import copy
import galsim as gs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprofit as mpf
import multiprofit.gaussutils as mpfgauss
import multiprofit.utils as mpfutil
import multiprofit.asinhstretchsigned as mpfasinh
import numpy as np
import pyprofit as pyp
import scipy.stats as spstats
import scipy.optimize as spopt
import seaborn as sns
import sys
import time


# TODO: Implement WCS
# The smart way for this to work would be to specify sky coordinates and angular sizes for objects. This
# way you could give a model some exposures with WCS and it would automagically convert to pixel coordinates.

# TODO: Make this a class?
def getgsparams(engineopts):
    if engineopts is None:
        gsparams = gs.GSParams()
    else:
        gsparams = engineopts["gsparams"]
    return gsparams


class Exposure:
    """
        A class to hold an image, sigma map, bad pixel mask and reference to a PSF model/image.
        TODO: Decide whether this should be mutable and implement getters/setters if so; or use afw class
    """
    def __init__(self, band, image, maskinverse=None, sigmainverse=None, psf=None, calcinvmask=None,
                 meta=None):
        if psf is not None and not isinstance(psf, PSF):
            raise TypeError("Exposure (band={}) PSF type={:s} not instanceof({:s})".format(
                band, type(psf), type(PSF)))
        self.band = band
        self.image = image
        extraargs = {
            'sigmainverse': sigmainverse,
            'maskinverse': maskinverse,
        }
        for key, value in extraargs.items():
            if value is not None and not (value.shape == image.shape or
                                          (key == 'sigmainverse' and np.prod(np.shape(value)) == 1)):
                raise ValueError('Exposure input {:s} shape={} not same as image.shape={}'.format(
                    key, value.shape, image.shape))
        self.maskinverse = maskinverse
        self.sigmainverse = sigmainverse
        self.psf = psf
        self.calcinvmask = calcinvmask
        self.meta = dict() if meta is None else meta


class Data:
    """
        A class that holds exposures, mostly for convenience to pass to a model.
        Currently a dict keyed by band with items lists of exposures.
        Also enforces that all exposures are the same size right now, assuming that they're pixel matched.
        This will be relaxed once WCS is implemented.
        TODO: Should this use multiband exposures? Dunno
    """
    def __init__(self, exposures):
        self.exposures = {}
        self.nx = None
        for i, exposure in enumerate(exposures):
            if not isinstance(exposure, Exposure):
                raise TypeError(
                    "exposure[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(exposure), type(Exposure))
                )
            else:
                if self.nx is None:
                    self.nx = exposure.image.shape[0]
                    self.ny = exposure.image.shape[1]
                else:
                    if exposure.image.shape[0] != self.nx or exposure.image.shape[1] != self.ny:
                        raise ValueError(
                            "Mismatch in exposure[{:d}] (band={:s}] shape={} vs expected {}".format(
                                i, band, exposure.image.shape, (self.nx, self.ny)))
                band = exposure.band
                if band not in self.exposures:
                    self.exposures[band] = []
                self.exposures[band].append(exposure)


class PSF:
    """
        Either a model or an image. The model can be non-pixellated - i.e. a function describing the PDF of
        photons scattering to a given position from their origin - or pixellated - i.e. a function
        describing the convolution of the PSF with a pixel at a given position.

        Has convenience functions to convert to/from galsim format.
    """

    def get(self, usemodel=None):
        if usemodel is None:
            usemodel = self.usemodel
        if usemodel:
            return self.model
        return self.getimage()

    def getimageshape(self):
        if self.image is None:
            return None
        if isinstance(self.image, gs.InterpolatedImage):
            return self.image.image.array.shape
        return self.image.shape

    # TODO: support rescaling of the PSF if it's a galsim interpolatedimage?
    def getimage(self, engine=None, size=None, engineopts=None):
        if engine is None:
            engine = self.engine
        if size is None and self.model is None:
            size = self.getimageshape()
        if self.image is None or self.getimageshape() != size:
            if self.model is None:
                raise RuntimeError("Can't get new PSF image without a model")
            exposure = Exposure(self.band, np.zeros(shape=size), None, None)
            data = Data(exposures=[exposure])
            model = Model(sources=[self.model], data=data)
            # TODO: Think about the consequences of making a new astrometry vs resetting the old one
            # It's necessary because the model needs to be centered and it might not be
            astro = model.sources[0].modelastrometric
            model.sources[0].modelastrometric = AstrometricModel([
                Parameter("cenx", value=size[0]/2.),
                Parameter("ceny", value=size[1]/2.),
            ])
            model.evaluate(keepimages=True, getlikelihood=False)
            self.image = data.exposures[self.band][0].meta["modelimage"]
            model.sources[0].modelastrometric = astro

        # TODO: There's more torturous logic needed here if we're to support changing engines on the fly
        if engine != self.engine:
            if engine == "galsim":
                gsparams = getgsparams(engineopts)
                self.image = gs.InterpolatedImage(gs.ImageD(self.image, scale=1), gsparams=gsparams)
            else:
                if self.engine == "galsim":
                    self.image = self.image.image.array
            self.engine = engine
        return self.image

    def __init__(self, band, image=None, model=None, engine=None, usemodel=False, modelpixelated=False):
        self.band = band
        self.model = model
        self.image = image
        self.modelpixelated = modelpixelated
        if model is None:
            if image is None:
                raise ValueError("PSF must be initialized with either a model or engine but both are none")
            if usemodel:
                raise ValueError("PSF usemodel==True but no model specified")
            if (engine == "galsim") and (isinstance(image, gs.InterpolatedImage) or
                                         isinstance(image, gs.Image)):
                self.engine = engine
            else:
                if not isinstance(image, np.ndarray):
                    raise ValueError("PSF image must be an ndarray or galsim.Image/galsim.InterpolatedImage"
                                     " if using galsim")
                self.engine = "libprofit"
                self.image = self.getimage(engine=engine)
        else:
            if image is not None:
                raise ValueError("PSF initialized with a model cannot be initialized with an image as well")
            if not isinstance(model, Source):
                raise ValueError("PSF model (type={:s}) not instanceof({:s})".format(
                    type(model), type(Source)))
            self.engine = engine
        self.usemodel = usemodel


def gaussianprofilestomatrix(profiles):
    return np.array(
        [
            np.array([p['cenx'], p['ceny'], mpfutil.magtoflux(p['mag']), p['re'], p['ang'], p['axrat']])
            for p in profiles
        ]
    )


def _sidecolorbar(axis, figure, img, vertical=True, showlabels=True):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right' if vertical else 'bottom', size='5%', pad=0.05)
    cbar = figure.colorbar(img, cax=cax, ax=axis, orientation='vertical' if vertical else 'horizontal')
    if not showlabels:
        (cbar.ax.set_yticklabels if vertical else cbar.ax.set_xticklabels)([])
    return cbar


class Model:
    likefuncs = {
        "normal": "normal",
        "gaussian": "normal",
        "student-t": "t",
        "t": "t",
    }

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in cls.ENGINES:
            raise ValueError("Unknown Model rendering engine {:s}".format(engine))

    @classmethod
    def _labelfigureaxes(cls, axes, chisqred, modelname='Model', modeldesc=None, labelimg='Image',
                         isfirstmodel=False, islastmodel=False, plotascolumn=False, labeldiffpostfix=None,
                         labelchipostfix=None):
        if labeldiffpostfix is None:
            labeldiffpostfix = ''
        if labelchipostfix is None:
            labelchipostfix = ''
        (axes[0].set_title if plotascolumn else axes[0].set_ylabel)(labelimg)
        # Check if the modelname is informative as it's redundant otherwise
        if modelname != "Model":
            (axes[1].set_title if plotascolumn else axes[1].set_ylabel)(modelname)
        if modeldesc is not None:
            (axes[2].set_title if plotascolumn else axes[2].set_ylabel)(modeldesc)
        labelchisq = r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred)
        (axes[3].set_title if plotascolumn else axes[3].set_ylabel)(labelchisq)
        axes[4].set_xlabel(r'$\chi=$(Data-Model)/$\sigma$')
        if plotascolumn:
            # TODO: What to do here?
            if not (isfirstmodel or islastmodel):
                for i in range(1, 5):
                    axes[i].tick_params(labelleft=False)
        else:
            axes[4].yaxis.tick_right()
            for i in range(1, 5):
                if i != 4:
                    axes[i].set_yticklabels([])
                axes[i].yaxis.set_label_position('right')
                if not islastmodel:
                    axes[i].set_xticklabels([])
        if isfirstmodel:
            labels = ["Data", "Model", "Data-Model" + labeldiffpostfix,
                      r'$\chi=$(Data-Model)/$\sigma$' + labelchipostfix, 'PDF']
            for axis, label in enumerate(labels):
                (axes[axis].set_ylabel if plotascolumn or axis == 4 else axes[axis].set_title)(label)

    @classmethod
    def _plotexposurescolor(cls, images, modelimages, chis, figaxes, bands=None, bandstring=None,
                            maximg=None, modelname='Model', modeldesc=None, modelnameappendparams=None,
                            isfirstmodel=True, islastmodel=True, plotascolumn=False, originimg='bottom',
                            weightsimgs=None, asinhscale=16, imgdiffscale=0.05):
        if bands is None:
            bands = []
        if bandstring is None:
            bandstring = ','.join(bands) if bands is not None else ''
        if modelnameappendparams is not None:
            if modeldesc is None:
                modeldesc = ''
            modeldesc += Model._formatmodelparams(modelnameappendparams, bands)
        # TODO: verify lengths
        axes = figaxes[1]
        imagesall = [np.copy(images), np.copy(modelimages)]
        for imagesoftype in imagesall:
            for idx, img in enumerate(imagesoftype):
                imagesoftype[idx] = np.clip(img, 0, np.Inf)
            if weightsimgs is not None:
                for img, weight in zip(imagesoftype, weightsimgs):
                    img *= weight
        if maximg is None:
            maximg = np.max([np.max(np.sum(imgs)) for imgs in imagesall])
        for i, imagesoftype in enumerate(imagesall):
            rgb = apvis.make_lupton_rgb(imagesoftype[0], imagesoftype[1], imagesoftype[2],
                                        stretch=maximg/asinhscale, Q=asinhscale)
            axes[i].imshow(rgb, origin=originimg)
        (axes[0].set_title if plotascolumn else axes[0].set_ylabel)(bandstring)
        # TODO: Verify if the image limits are working as intended
        imgsdiff = [data-model for data, model in zip(imagesall[0], imagesall[1])]
        rgb = apvis.make_lupton_rgb(imgsdiff[0], imgsdiff[1], imgsdiff[2], minimum=-maximg*imgdiffscale,
                                    stretch=imgdiffscale*maximg/asinhscale, Q=asinhscale)
        axes[2].imshow(rgb, origin=originimg)
        # Check if the modelname is informative as it's redundant otherwise
        if modelname != "Model":
            (axes[1].set_title if plotascolumn else axes[1].set_ylabel)(modelname)
        if modeldesc is not None:
            (axes[2].set_title if plotascolumn else axes[2].set_ylabel)(modeldesc)
        # The chi (data-model)/error map clipped at +/- 10 sigma
        rgb = np.clip(np.stack(chis, axis=2)/20 + 0.5, 0, 1)
        axes[3].imshow(rgb, origin=originimg)
        # Residual histogram compared to a normal distribution
        chi = np.array([])
        for chiband in chis:
            chi = np.append(chi, chiband)
        nbins = np.max([100, np.int(np.round(np.sum(~np.isnan(chi))/300))])
        sns.distplot(chi[~np.isnan(chi)], bins=nbins, ax=axes[4],
                     hist_kws={"log": True, "histtype": "step"},
                     kde_kws={"kernel": "tri", "gridsize": nbins/2}).set(
            xlim=(-5, 5), ylim=(1e-4, 1)
        )
        # axes[4].hist(chi[~np.isnan(chi)], bins=100, log=True, density=True, histtype="step", fill=False)
        x = np.linspace(-5., 5., int(1e4) + 1, endpoint=True)
        axes[4].plot(x, spstats.norm.pdf(x))
        chisqred = np.sum(chi*chi)/len(chi)
        Model._labelfigureaxes(axes, chisqred, modelname=modelname, modeldesc=modeldesc,
                               labelimg=bandstring, isfirstmodel=isfirstmodel, islastmodel=islastmodel,
                               plotascolumn=plotascolumn,
                               labeldiffpostfix=' (lim. +/- {}*max(image)'.format(imgdiffscale),
                               labelchipostfix=r' ($\pm 10\sigma$)')

    # Takes an iterable of tuples of formatstring, param and returns a sensibly formatted summary string
    # TODO: Should probably be tuples of param, formatstring
    # TODO: Add options for the definition of 'sensibly formatted'
    @classmethod
    def _formatmodelparams(cls, modelparamsformat, bands=None):
        if bands is None:
            bands = {}
        modeldescs = {}
        paramdescs = {'nser': 'n', 'reff': 'r'}
        for formatstring, param in modelparamsformat:
            isflux = isinstance(param, FluxParameter)
            isfluxratio = isflux and param.isfluxratio
            if '=' not in formatstring:
                paramname = 'f' if isfluxratio else (
                    paramdescs[param.name] if param.name in paramdescs else param.name)
                value = formatstring
            else:
                paramname, value = formatstring.split('=')
            if paramname not in modeldescs:
                modeldescs[paramname] = []
            if not isfluxratio or not param.fixed and param.band in bands:
                modeldescs[paramname].append(value.format(param.getvalue(transformed=False)))
        # Show the flux ratio if there is only one (otherwise it's painful to work out)
        # Show other parameters if there <= 3 of them otherwise the string is too long
        # TODO: Make a more elegant solution for describing models
        modeldesc = ';'.join(
            [paramname + ':' + ','.join(values) for paramname, values in modeldescs.items()
             if len(values) <= (1 + 2 * (paramname != 'f'))])
        return modeldesc

    def evaluate(self, params=None, data=None, bands=None, engine=None, engineopts=None,
                 paramstransformed=True, getlikelihood=True, likelihoodlog=True, keeplikelihood=False,
                 keepimages=False, keepmodels=False, plot=False,
                 plotmulti=False, figure=None, axes=None, figurerow=None, modelname="Model", modeldesc=None,
                 modelnameappendparams=None, drawimage=False, scale=1, clock=False, plotascolumn=False,
                 imgplotmaxs=None, imgplotmaxmulti=None, weightsband=None, dolinearfitprep=False,
                 comparelikelihoods=False):
        """
        Evaluate a model, plot and/or benchmark, and optionally return the likelihood and derived images.

        :param params: ndarray; optional new values for all model free parameters in the order returned by
            getparameters. Defaults to use existing values.
        :param data: multiprofit.data; optional override to evaluate model on a different set of data. May
            not work as expected unless the data has exposures of the same size and in the same order as
            the default (self.data).
        :param bands: iterable; bandpass filters to use for evaluating the model. Defaults to use all bands
            in data.
        :param engine: A valid rendering engine
        :param engineopts: dict; engine options.
        :param paramstransformed: bool; are the values in params already transformed?
        :param getlikelihood: bool; return the model likelihood?
        :param likelihoodlog: bool; return the natural logarithm of the likelihood?
        :param keeplikelihood: bool; store each exposure's likelihood in its metadata?
        :param keepimages: bool; store each exposure's model image in its metadata?
        :param keepmodels: bool; store each exposure's model specification in its metadata?
        :param plot: bool; plot the model and residuals for each exposure?
        :param plotmulti: bool; plot colour images if fitting multiband? Ignored otherwise.
        :param figure: matplotlib figure handle. If data has multiple bands, must be a dict keyed by band.
        :param axes: iterable of matplotlib axis handles. If data has multiple bands, must be a dict keyed by
            band.
        :param figurerow: non-negative integer; the index of the axis handle to plot this model on.
        :param modelname: string; a name for the model to appear in plots.
        :param modeldesc: string; a description of the model to appear in plots.
        :param modelnameappendparams: iterable of multiprofit.Parameter; parameters whose values should
            appear in the plot (if possible).
        :param drawimage: bool; draw (evaluate) the model image for each exposure? May be overruled if
            getlikelihood or plot is True.
        :param scale: float; Spatial scale factor for drawing images. User beware; this may not work
            properly and should be replaced by exposure WCS soon.
        :param clock: bool; benchmark model evaluation?
        :param plotascolumn: bool; are the plots arranged vertically?
        :param imgplotmaxs: dict; key band: value maximum for image/model plots.
        :param imgplotmaxmulti: float; maximum for multiband image/model plots.
        :param weightsband: dict; key band: weight for scaling each band in multiband plots.
        :param dolinearfitprep: bool; do prep work to fit a linear model?
        :param comparelikelihoods: bool; compare likelihoods from C++ vs Python (if both exist)
        :return: likelihood: float; the (log) likelihood
            params: ndarray of floats; parameter values
            chis: list of residual images for each fit exposure, where chi = (data-model)/sigma
            times: Dict of floats, with time in
        """
        times = {}
        if clock:
            timenow = time.time()
        if engine is None:
            engine = self.engine
        Model._checkengine(engine)
        if engineopts is None:
            engineopts = self.engineopts
        if data is None:
            data = self.data

        paramobjects = self.getparameters(free=True, fixed=False)
        if params is None:
            params = [param.value for param in paramobjects]
        else:
            paramslen = len(params)
            if not len(paramobjects) == paramslen:
                raise RuntimeError("Length of parameters[{:d}] != # of free params=[{:d}]".format(
                    len(paramobjects), paramslen
                ))
            for paramobj, paramval in zip(paramobjects, params):
                paramobj.setvalue(paramval, transformed=paramstransformed)

        if getlikelihood:
            likelihood = 1.
            if likelihoodlog:
                likelihood = 0.
        else:
            likelihood = None

        if bands is None:
            bands = data.exposures.keys()
        if plot:
            plotslen = 0
            singlemodel = figure is None or axes is None or figurerow is None
            if singlemodel:
                for band in bands:
                    plotslen += len(data.exposures[band])
                nrows = plotslen + plotmulti
                figure, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(10, 2*nrows), dpi=100)
                if plotslen == 1:
                    axes.shape = (1, 5)
                figurerow = 0
            figaxes = (figure, axes)
        else:
            figaxes = None
        if plot and (figaxes is None or any(x is None for x in figaxes)):
            raise RuntimeError("Plot is true but there are None figaxes: {}".format(figaxes))
        chis = []
        imgclips = []
        modelclips = []
        if plotmulti:
            weightsimgs = []
        if clock:
            times["setup"] = time.time() - timenow
            timenow = time.time()

        for band in bands:
            # TODO: Check band
            for idxexposure, exposure in enumerate(data.exposures[band]):
                profiles, metamodel, times = self._getexposuremodelsetup(
                    exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
                image, model, timesmodel, likelihoodexposure = self.getexposuremodel(
                    exposure, profiles=profiles, metamodel=metamodel, engine=engine,
                    engineopts=engineopts, drawimage=drawimage or plot or comparelikelihoods, scale=scale,
                    clock=clock, times=times, dolinearfitprep=dolinearfitprep,
                    getlikelihood=getlikelihood, likelihoodlog=likelihoodlog)
                if clock:
                    nameexposure = '_'.join([band, str(idxexposure)])
                    times['_'.join([nameexposure, 'modeltotal'])] = time.time() - timenow
                    for timename, timevalue in timesmodel.items():
                        times['_'.join([nameexposure, timename])] = timevalue
                    timenow = time.time()
                if keepimages:
                    exposure.meta["modelimage"] = np.array(image)
                if keepmodels:
                    exposure.meta["model"] = model
                if plot:
                    nrows = (axes[band] if plotmulti else axes).shape[0]
                    if plotmulti:
                        figaxes = (figure[band], axes[band][figurerow])
                    else:
                        figaxes = (figure, axes[figurerow])
                    isfirstmodel = figurerow is None or figurerow == 0
                    islastmodel = figurerow is None or axes is None or ((figurerow + 1) == nrows)
                else:
                    isfirstmodel = None
                    islastmodel = None

                if plot or (getlikelihood and likelihoodexposure is None) or (
                        comparelikelihoods and likelihoodexposure is not None):
                    likelihoodnew, chi, imgclip, modelclip = \
                        self.getexposurelikelihood(
                            exposure, image, log=likelihoodlog, figaxes=figaxes,
                            modelname=modelname, modeldesc=modeldesc,
                            modelnameappendparams=modelnameappendparams, plotascolumn=plotascolumn,
                            isfirstmodel=isfirstmodel, islastmodel=islastmodel,
                            maximg=imgplotmaxs[band] if (imgplotmaxs is not None and band in
                                                         imgplotmaxs) else None
                        )
                    if likelihoodexposure is None:
                        likelihoodexposure = likelihoodnew
                    elif not np.isclose(likelihoodnew, likelihoodexposure):
                        # TODO: Think harder about what to do here
                        raise RuntimeError(
                            'getexposuremodel vs getexposurelikelihood likelihoods differ significantly '
                            '({:5e} vs {:5e})'.format(likelihoodnew, likelihoodexposure))
                elif drawimage:
                    chi = (image - exposure.image)*exposure.sigmainverse
                else:
                    chi = None
                if getlikelihood or plot:
                    if clock:
                        times['_'.join([nameexposure, 'like'])] = time.time() - timenow
                        timenow = time.time()
                    if keeplikelihood:
                        exposure.meta["likelihood"] = likelihoodexposure
                        exposure.meta["likelihoodlog"] = likelihoodlog
                    if likelihoodlog:
                        likelihood += likelihoodexposure
                    else:
                        likelihood *= likelihoodexposure
                    chis.append(chi)
                    if plot:
                        if not plotmulti:
                            figurerow += 1
                        if plotmulti:
                            imgclips.append(imgclip)
                            modelclips.append(modelclip)
                            weightsimgs.append(
                                weightsband[band] if weightsband is not None and band in weightsband else 1)
        # Color images! whooo
        if plot:
            if plotmulti:
                if singlemodel:
                    Model._plotexposurescolor(
                        imgclips, modelclips, chis, (figure, axes[figurerow]),
                        bands=bands, modelname=modelname, modeldesc=modeldesc,
                        modelnameappendparams=modelnameappendparams, isfirstmodel=isfirstmodel,
                        islastmodel=islastmodel, plotascolumn=plotascolumn, maximg=imgplotmaxmulti,
                        weightsimgs=weightsimgs)
                else:
                    Model._plotexposurescolor(
                        imgclips, modelclips, chis, (figure['multi'], axes['multi'][figurerow]),
                        bands=bands, modelname=modelname, modeldesc=modeldesc,
                        modelnameappendparams=modelnameappendparams, isfirstmodel=isfirstmodel,
                        islastmodel=islastmodel, plotascolumn=plotascolumn, maximg=imgplotmaxmulti,
                        weightsimgs=weightsimgs)
                figurerow += 1
        if clock:
            print(','.join(['{}={:.2e}'.format(name, value) for name, value in times.items()]))
        return likelihood, params, chis, times

    def getexposurelikelihood(
            self, exposure, modelimage, log=True, likefunc=None, figaxes=None, maximg=None, minimg=None,
            modelname="Model", modeldesc=None, modelnameappendparams=None, isfirstmodel=True,
            islastmodel=True, plotascolumn=False, originimg='bottom', normdiff=None, normchi=None
    ):
        if likefunc is None:
            likefunc = self.likefunc
        hasmask = exposure.maskinverse is not None
        if figaxes is not None:
            if modelnameappendparams is not None:
                if modeldesc is None:
                    modeldesc = ""
                modeldesc += Model._formatmodelparams(modelnameappendparams, {exposure.band})

            axes = figaxes[1]
            if hasmask:
                xlist = np.arange(0, modelimage.shape[1])
                ylist = np.arange(0, modelimage.shape[0])
                x, y = np.meshgrid(xlist, ylist)
            if maximg is None:
                if hasmask:
                    maximg = np.max([np.max(exposure.image[exposure.maskinverse]),
                                     np.max(modelimage[exposure.maskinverse])])
                else:
                    maximg = np.max([np.max(exposure.image), np.max(modelimage)])
            if minimg is None:
                minimg = np.min([0, np.min(exposure.image), np.min(modelimage)])
            # The original image and model image
            norm = apvis.ImageNormalize(vmin=minimg, vmax=maximg, stretch=apvis.AsinhStretch(1e-2))
            showlabels = islastmodel
            for i, img in enumerate([exposure.image, modelimage]):
                imgobj = axes[i].imshow(img, cmap='gray', origin=originimg, norm=norm)
                _sidecolorbar(axes[i], figaxes[0], imgobj, vertical=plotascolumn, showlabels=showlabels)
                if hasmask:
                    z = exposure.maskinverse
                    axes[i].contour(x, y, z)
            # The difference map
            chi = exposure.image - modelimage
            if normdiff is None:
                diffabsmax = np.max(exposure.image)/10
                normdiff = apvis.ImageNormalize(vmin=-diffabsmax, vmax=diffabsmax,
                                                stretch=mpfasinh.AsinhStretchSigned(0.1))
            imgdiff = axes[2].imshow(chi, cmap='gray', origin=originimg, norm=normdiff)
            _sidecolorbar(axes[2], figaxes[0], imgdiff, vertical=plotascolumn, showlabels=showlabels)
            if hasmask:
                axes[2].contour(x, y, z)
            # The chi (data-model)/error map
            chi *= exposure.sigmainverse
            chisqred = mpfutil.getchisqred([chi[exposure.maskinverse] if hasmask else chi])
            if normchi is None:
                chiabsmax = np.max(np.abs(chi))
                if chiabsmax < 1:
                    chiabsmax = np.ceil(chiabsmax*100)/100
                else:
                    chiabsmax = 10
                normchi = apvis.ImageNormalize(vmin=-chiabsmax, vmax=chiabsmax,
                                               stretch=mpfasinh.AsinhStretchSigned(0.1))
            imgchi = axes[3].imshow(chi, cmap='RdYlBu_r', origin=originimg, norm=normchi)
            _sidecolorbar(axes[3], figaxes[0], imgchi, vertical=plotascolumn, showlabels=showlabels)
            if hasmask:
                axes[3].contour(x, y, z, colors="green")
            # Residual histogram compared to a normal distribution
            nbins = np.max([100, np.int(np.round(np.sum(~np.isnan(chi))/300))])
            sns.distplot(chi[~np.isnan(chi)], bins=nbins, ax=axes[4],
                         hist_kws={"log": True, "histtype": "step"},
                         kde_kws={"kernel": "tri", "gridsize": nbins/2}).set(
                xlim=(-5, 5), ylim=(1e-4, 1)
            )
            # axes[4].hist(chi[~np.isnan(chi)], bins=100, log=True, density=True, histtype="step", fill=False)
            x = np.linspace(-5., 5., int(1e4) + 1, endpoint=True)
            axes[4].plot(x, spstats.norm.pdf(x))
            Model._labelfigureaxes(axes, chisqred, modelname=modelname, modeldesc=modeldesc,
                                   labelimg='Band={}'.format(exposure.band), isfirstmodel=isfirstmodel,
                                   islastmodel=islastmodel, plotascolumn=plotascolumn)
        else:
            if hasmask:
                chi = (exposure.image[exposure.maskinverse] - modelimage[exposure.maskinverse]) * \
                    exposure.sigmainverse[exposure.maskinverse]
            else:
                chi = (exposure.image - modelimage)*exposure.sigmainverse
        if likefunc == "t":
            variance = chi.var()
            dof = 2. * variance / (variance - 1.)
            dof = max(min(dof, float('inf')), 0)
            likelihood = np.sum(spstats.t.logpdf(chi, dof))
        elif likefunc == "normal":
            likelihood = np.sum(spstats.norm.logpdf(chi) + np.log(exposure.sigmainverse))
        else:
            raise ValueError("Unknown likelihood function {:s}".format(self.likefunc))

        if not log:
            likelihood = np.exp(likelihood)

        return likelihood, chi, exposure.image, modelimage

    def _getexposuremodelsetup(self, exposure, engine=None, engineopts=None, clock=False, times=None):
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        Model._checkengine(engine)
        if engine == "galsim":
            gsparams = getgsparams(engineopts)
        band = exposure.band
        if clock:
            timenow = time.time()
            if times is None:
                times = {}
        else:
            times = None
        profiles = self.getprofiles(bands=[band], engine=engine)
        if clock:
            times['getprofiles'] = time.time() - timenow
            timenow = time.time()
        haspsf = exposure.psf is not None
        psfispixelated = haspsf and exposure.psf.model is not None and exposure.psf.modelpixelated
        profilesgaussian = [comp.isgaussian()
                            for comps in [src.modelphotometric.components for src in self.sources]
                            for comp in comps]
        allgaussian = not haspsf or (exposure.psf.model is not None and
            all([comp.isgaussian() for comp in exposure.psf.model.modelphotometric.components]))
        anygaussian = allgaussian and any(profilesgaussian)
        allgaussian = allgaussian and all(profilesgaussian)
        # Use fast, efficient Gaussian evaluators only if everything's a Gaussian mixture model
        usefastgauss = allgaussian and (
                psfispixelated or
                (engine == 'galsim' and self.engineopts is not None and
                 "drawmethod" in self.engineopts and self.engineopts["drawmethod"] == 'no_pixel') or
                (engine == 'libprofit' and self.engineopts is not None and
                 "drawmethod" in self.engineopts and self.engineopts["drawmethod"] == 'rough'))
        if usefastgauss and engine != 'libprofit':
            engine = 'libprofit'
            # Down below we'll get the libprofit format profiles anyway, unless not haspsf
            if not haspsf:
                profiles = self.getprofiles(bands=[band], engine="libprofit")

        if profiles:
            # Do analytic convolution of any Gaussian model with the Gaussian PSF
            # This turns each resolved profile into N_PSF_components Gaussians
            if anygaussian and haspsf:
                # Note that libprofit uses the irritating GALFIT convention, so reverse that
                varsgauss = ["re", "axrat", "ang"]
                # Ellipse transformation functions expect a sigma argument, not re
                varmap = {var: var if var != 're' else 'sigma' for var in varsgauss}
                covarprofilessrc = [
                    (
                        mpfgauss.ellipsetocovar(**{varmap[var]: profile[band][var] + 90*(var == "ang")
                                                   for var in varsgauss})
                        if profile[band]["nser"] == 0.5 else None,
                        profile[band]
                    )
                    for profile in self.getprofiles(bands=[band], engine="libprofit")
                ]

                profilesnew = []
                if clock:
                    times['modelallgausseig'] = 0
                for idxpsf, profilepsf in enumerate(exposure.psf.model.getprofiles(
                        bands=[band], engine="libprofit")):
                    fluxfrac = 10**(-0.4*profilepsf[band]["mag"])
                    covarpsf = mpfgauss.ellipsetocovar(
                        **{varmap[var]: profilepsf[band][var] + 90*(var == "ang") for var in varsgauss}
                    )
                    for idxsrc, (covarsrc, profile) in enumerate(covarprofilessrc):
                        if covarsrc is not None:
                            if clock:
                                timeeig = time.time()
                            reff, axrat, ang = mpfgauss.covartoellipse(
                                covarpsf[0, 0] + covarsrc[0, 0],
                                covarpsf[1, 1] + covarsrc[1, 1],
                                covarpsf[0, 1] + covarsrc[0, 1],
                            )
                            if clock:
                                times['modelallgausseig'] += time.time() - timeeig
                            if engine == "libprofit":
                                # Needed because each PSF component will loop over the same profile object
                                profile = copy.copy(profile)
                                profile["re"] = reff
                                profile["axrat"] = axrat
                                profile["mag"] += mpfutil.fluxtomag(fluxfrac)
                                profile["ang"] = ang
                            else:
                                profile = {
                                    "profile": gs.Gaussian(flux=10**(-0.4*profile["mag"])*fluxfrac,
                                                           fwhm=2*reff*np.sqrt(axrat), gsparams=gsparams),
                                    "shear": gs.Shear(q=axrat, beta=ang*gs.degrees),
                                    "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                                }
                            profile["pointsource"] = True
                            profile["resolved"] = True
                            profile = {band: profile}
                        else:
                            profile = profiles[idxsrc]
                        # TODO: Remember what the point of this check is
                        if covarsrc is not None or idxpsf == 0:
                            profilesnew.append(profile)
                profiles = profilesnew
                if clock:
                    times['modelallgauss'] = time.time() - timenow
                    timenow = time.time()

        if clock:
            times['modelsetup'] = time.time() - timenow

        metamodel = {
            'allgaussian': allgaussian,
            'haspsf': haspsf,
            'profiles': profiles,
            'psfispixelated': psfispixelated,
            'usefastgauss': usefastgauss,
        }
        return profiles, metamodel, times

    def getexposuremodel(
            self, exposure, profiles=None, metamodel=None, engine=None, engineopts=None, drawimage=True,
            scale=1, clock=False, times=None, dolinearfitprep=False, getlikelihood=False, likelihoodlog=True):
        """
            Draw model image for one exposure with one PSF

            Returns the image and the engine-dependent model used to draw it.
            If getlikelihood == True, it will try to get the likelihood
        """
        # TODO: Rewrite this completely at some point as validating inputs is a pain
        if profiles is None or metamodel is None:
            profiles, metamodel, times = self._getexposuremodelsetup(
                exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
        likelihood = None
        ny, nx = exposure.image.shape
        band = exposure.band
        image = None
        haspsf = metamodel['haspsf']
        if dolinearfitprep:
            exposure.meta['modelimagefixed'] = None
            exposure.meta['modelimagesparamsfree'] = []

        if clock:
            times = metamodel['times'] if 'times' in metamodel else []
            timenow = time.time()
            if times is None:
                times = {}

        if engine == "galsim":
            gsparams = getgsparams(engineopts)

        if drawimage:
            image = np.zeros_like(exposure.image)

        model = {}

        # TODO: Do this in a smarter way
        if metamodel['usefastgauss']:
            profilesleft = []
            profilestodraw = []
            profileslinearfit = {}
            for profile in profiles:
                params = profile[band]
                # Verify that it's a Gaussian
                if 'profile' in params and params['profile'] == 'sersic' and 'nser' in params and \
                        params['nser'] == 0.5:
                    if dolinearfitprep and not params['fluxparameter'].fixed:
                        idfluxparam = id(params['fluxparameter'])
                        if idfluxparam not in profileslinearfit:
                            profileslinearfit[idfluxparam] = ([], params['fluxparameter'])
                        profileslinearfit[idfluxparam][0].append(profile)
                    else:
                        profilestodraw += [profile]
                else:
                    profilesleft += [profile]
            profiles = profilesleft
            model['multiprofit'] = {key: value for key, value in zip(
                ['profiles' + x for x in ['todraw', 'left', 'linearfit']],
                [profilestodraw, profilesleft, profileslinearfit]
            )}
            # If these are the only profiles, just get the likelihood
            if profilestodraw and not profilesleft and not profileslinearfit:
                varinverse = exposure.sigmainverse**2
                varisscalar = np.isscalar(varinverse)
                # TODO: Do this in a prettier way while avoiding recalculating loglike_gaussian_pixel
                if 'likeconst' not in exposure.meta:
                    exposure.meta['likeconst'] = np.sum(np.log(exposure.sigmainverse/np.sqrt(2.*np.pi)))
                    if varisscalar:
                        exposure.meta['likeconst'] *= np.prod(exposure.image.shape)
                    print('Setting exposure.meta[\'likeconst\']=', exposure.meta['likeconst'])
                if varisscalar:
                    # I think this is the easiest way to reshape it into a 1x1 matrix
                    varinverse = np.zeros((1, 1)) + varinverse
                profilemat = gaussianprofilestomatrix([profile[band] for profile in profilestodraw])
                likelihood = exposure.meta['likeconst'] + mpf.loglike_gaussians_pixel(
                    data=exposure.image, varinverse=varinverse, gaussians=profilemat,
                    xmin=0, xmax=exposure.image.shape[1], ymin=0, ymax=exposure.image.shape[0],
                    output=image if drawimage else np.zeros([0, 0]), grad=np.zeros([0, 0]))
                if not likelihoodlog:
                    likelihood = 10**likelihood
            else:
                if profilestodraw:
                    image = mpf.make_gaussians_pixel(
                        gaussianprofilestomatrix([profile[band] for profile in profilestodraw]),
                        0, exposure.image.shape[1], 0, exposure.image.shape[0],
                        exposure.image.shape[1], exposure.image.shape[0])
                    if dolinearfitprep:
                        exposure.meta['modelimagefixed'] = np.copy(image)
                # Ensure identical order until all dicts are ordered
                for idflux in np.sort(list(profileslinearfit.keys())):
                    profilesflux, paramflux = profileslinearfit[idflux]
                    # Draw all of the profiles together if there are multiple; otherwise
                    # make_gaussian_pixel is faster
                    if len(profilesflux) > 1:
                        imgprofiles = mpf.make_gaussians_pixel(
                            gaussianprofilestomatrix([profile[band] for profile in profilesflux]),
                            0, exposure.image.shape[1], 0, exposure.image.shape[0],
                            exposure.image.shape[1], exposure.image.shape[0])
                    else:
                        profile = profilesflux[0]
                        params = profile[band]
                        imgprofiles = np.array(mpf.make_gaussian_pixel(
                            params['cenx'], params['ceny'], mpfutil.magtoflux(params['mag']), params['re'],
                            params['ang'], params['axrat'], 0, nx, 0, ny, nx, ny))
                    if dolinearfitprep:
                        exposure.meta['modelimagesparamsfree'] += [(imgprofiles, paramflux)]
                    if image is None:
                        image = np.copy(imgprofiles)
                    else:
                        image += imgprofiles

        if profiles:
            if engine == 'libprofit':
                if dolinearfitprep:
                    profilesfree = []
                    fluxparameters = []
                profilespro = {}
                for profile in profiles:
                    profile = profile[band]
                    profiletype = profile["profile"]
                    if profiletype not in profilespro:
                        profilespro[profiletype] = []
                    del profile["profile"]
                    if not profile["pointsource"]:
                        profile["convolve"] = haspsf
                        if not profile["resolved"]:
                            raise RuntimeError("libprofit can't handle non-point sources that aren't resolved"
                                               "(i.e. profiles with size==0)")
                    del profile["pointsource"]
                    del profile["resolved"]
                    # TODO: Find a better way to do this
                    for coord in ["x", "y"]:
                        nameold = "cen" + coord
                        profile[coord + "cen"] = profile[nameold]
                        del profile[nameold]
                    dolinearprofile = dolinearfitprep and profile["fluxparameter"].fixed
                    if dolinearprofile:
                        profilesfree += [{}]
                        fluxparameters += [profile["fluxparameter"]]
                    del profile["fluxparameter"]
                    profiledict = profilesfree[:-1] if dolinearprofile else profilespro
                    profiledict[profiletype] += [profile]

                profit_model = {
                    'width': nx,
                    'height': ny,
                    'magzero': 0.0,
                    'profiles': profilespro,
                }
                if metamodel['psfispixelated']:
                    profit_model['rough'] = True
                if haspsf:
                    shape = exposure.psf.getimageshape()
                    if shape is None:
                        shape = [1 + np.int(x/2) for x in np.floor([nx, ny])]
                    profit_model["psf"] = exposure.psf.getimage(engine, size=shape, engineopts=engineopts)

                if exposure.calcinvmask is not None:
                    profit_model['calcmask'] = exposure.calcinvmask
                if clock:
                    times['modelsetup'] = time.time() - timenow
                    timenow = time.time()
                if drawimage or getlikelihood:
                    imagepro = np.array(pyp.make_model(profit_model)[0])
                    if image is None:
                        image = imagepro
                    else:
                        image += imagepro
                    if dolinearfitprep:
                        if exposure.meta['modelimagefixed'] is None:
                            exposure.meta['modelimagefixed'] = np.copy(image)
                        else:
                            exposure.meta['modelimagefixed'] += image
                        for fluxparameter, profilesfree in zip(fluxparameters, profilesfree):
                            profit_model['profiles'] = profilesfree
                            imageprofile = np.array(pyp.make_model(profit_model)[0])
                            image += imageprofile
                            exposure.meta['modelimagesparamsfree'].append((imageprofile, fluxparameter))
                model[engine] = profit_model
            elif engine == "galsim":
                if clock:
                    timesetupgs = time.time()
                    times['modelsetupprofilegs'] = 0
                model[engine] = None
                # The logic here is to attempt to avoid GalSim's potentially excessive memory usage when
                # performing FFT convolution by splitting the profiles up into big and small ones
                # Perfunctory experiments have shown that this can reduce the required FFT size
                profilesgs = {key: {False: [], True: []} for key in ["small", "big", "linearfitprep"]}
                cenimg = gs.PositionD(nx/2., ny/2.)
                for profile in profiles:
                    profile = profile[band]
                    if not profile["pointsource"] and not profile["resolved"]:
                        profile["pointsource"] = True
                        raise RuntimeError("Converting small profiles to point sources not implemented yet")
                        # TODO: Finish this
                    else:
                        profilegs = profile["profile"].shear(
                            profile["shear"]).shift(
                            profile["offset"] - cenimg)
                    # TODO: Revise this when image scales are taken into account
                    dolinearprofile = dolinearfitprep and not profile["fluxparameter"].fixed
                    sizebinname = "linearfitprep" if dolinearprofile else (
                        "big" if profilegs.original.half_light_radius > 1 else "small")
                    convolve = haspsf and not profile["pointsource"]
                    profilesgs[sizebinname][convolve] += [
                        (profilegs, profile["fluxparameter"]) if dolinearprofile else profilegs
                    ]

                if haspsf:
                    psfgs = exposure.psf.model
                    if psfgs is None:
                        profilespsf = exposure.psf.getimage(engine=engine)
                    else:
                        psfgs = psfgs.getprofiles(bands=[band], engine=engine)
                        profilespsf = None
                        for profile in psfgs:
                            profile = profile[band]
                            profilegs = profile["profile"].shear(profile["shear"])
                            # TODO: Think about whether this would ever be necessary
                            #.shift(profile["offset"] - cenimg)
                            if profilespsf is None:
                                profilespsf = profilegs
                            else:
                                profilespsf += profilegs
                else:
                    if any([value[True] for key, value in profilesgs.items()]):
                        raise RuntimeError("Model (band={}) has profiles to convolve but no PSF, "
                                           "profiles are: {}".format(exposure.band, profilesgs))
                # TODO: test this, and make sure that it works in all cases, not just gauss. mix
                # Has a PSF and it's a pixelated analytic model, so all sources must use no_pixel
                if metamodel['psfispixelated']:
                    method = "no_pixel"
                else:
                    if (self.engineopts is not None and "drawmethod" in self.engineopts and
                            self.engineopts["drawmethod"] is not None):
                        method = self.engineopts["drawmethod"]
                    else:
                        method = "fft"
                if clock:
                    times['modelsetupprofilegs'] = time.time() - timesetupgs
                haspsfimage = haspsf and psfgs is None
                if clock:
                    times['modelsetup'] = time.time() - timenow
                    timenow = time.time()
                for profiletype, profilesoftype in profilesgs.items():
                    for convolve, profilesgsbin in profilesoftype.items():
                        if profilesgsbin:
                            # Pixel convolution is included with PSF images so no_pixel should be used
                            # TODO: Implement oversampled convolution here (or use libprofit?)
                            # Otherwise, flux conservation may be very poor
                            methodbin = 'no_pixel' if convolve and haspsfimage else method
                            linearprofile = profiletype == 'linearfitprep'
                            profilesdraw = profilesgsbin if linearprofile else [profilesgsbin[0]]
                            if not linearprofile:
                                for profiletoadd in profilesgsbin[1:]:
                                    profilesdraw[0] += profiletoadd
                            for profiledraw in profilesdraw:
                                if linearprofile:
                                    fluxparameter = profiledraw[1]
                                    profiledraw = profiledraw[0]
                                if convolve:
                                    profiledraw = gs.Convolve(profiledraw, profilespsf, gsparams=gsparams)
                                if model[engine] is None:
                                    model[engine] = profiledraw
                                else:
                                    model[engine] += profiledraw
                                if drawimage or getlikelihood or dolinearfitprep:
                                    try:
                                        imagegs = profiledraw.drawImage(
                                            method=methodbin, nx=nx, ny=ny, scale=scale).array
                                    # Somewhat ugly hack - catch RunTimeErrors which are usually excessively
                                    # large FFTs and then try to evaluate the model in real space or give
                                    # up if it's any other error
                                    except RuntimeError as e:
                                        try:
                                            if method == "fft":
                                                imagegs = profiledraw.drawImage(
                                                    method='real_space', nx=nx, ny=ny, scale=scale).array
                                            else:
                                                raise e
                                        except Exception as e:
                                            raise e
                                    except Exception as e:
                                        print("Exception attempting to draw image from profiles:", profiledraw)
                                        raise e
                                    if dolinearfitprep:
                                        if linearprofile:
                                            exposure.meta['modelimagesparamsfree'].append(
                                                (np.copy(imagegs), fluxparameter))
                                        else:
                                            if exposure.meta['modelimagefixed'] is None:
                                                exposure.meta['modelimagefixed'] = np.copy(imagegs)
                                            else:
                                                exposure.meta['modelimagefixed'] += imagegs
                                    if image is None:
                                        image = imagegs
                                    else:
                                        image += imagegs
            else:
                errmsg = "engine is None" if engine is None else "engine type " + engine + " not implemented"
                raise RuntimeError(errmsg)

        if clock:
            times['modeldraw'] = time.time() - timenow

        if image is not None:
            sumnotfinite = np.sum(~np.isfinite(image))
            if sumnotfinite > 0:
                raise RuntimeError("{}.getexposuremodel() got {:d} non-finite pixels out of {:d}".format(
                    type(self), sumnotfinite, np.prod(image.shape)
                ))

        return image, model, times, likelihood

    def getlimits(self, free=True, fixed=True, transformed=True):
        """
        Get parameter limits.

        :param free: Bool; return limits for free parameters?
        :param fixed: Bool; return limits for fixed parameters?
        :param transformed: Bool; return transformed limit values?
        :return:
        """
        params = self.getparameters(free=free, fixed=fixed)
        return [param.getlimits(transformed=transformed) for param in params]

    def getprofiles(self, **kwargs):
        """
        Get engine-dependent representations of profiles to be rendered.

        :param kwargs: keyword arguments passed to Source.getprofiles()
        :return: List of profiles
        """
        if 'engine' not in kwargs or kwargs['engine'] is None:
            kwargs['engine'] = self.engine
        if 'engineopts' not in kwargs or kwargs['engine'] is None:
            kwargs['engineopts'] = self.engineopts
        self._checkengine(kwargs['engine'])
        profiles = []
        for src in self.sources:
            profiles += src.getprofiles(**kwargs)
        return profiles

    def getparameters(self, free=True, fixed=True, flatten=True, modifiers=True):
        params = []
        for src in self.sources:
            paramssrc = src.getparameters(free=free, fixed=fixed, flatten=flatten, modifiers=modifiers)
            if flatten:
                params += paramssrc
            else:
                params.append(paramssrc)
        return params

    def getparamnames(self, free=True, fixed=True):
        names = []
        for i, src in enumerate(self.sources):
            srcname = src.name
            if srcname == "":
                srcname = str(i)
            names += [".".join([srcname, paramname]) for paramname in \
                      src.getparamnames(free=free, fixed=fixed)]
        return names

    def getpriorvalue(self, free=True, fixed=True, log=True):
        return np.sum(np.array(
            [param.getprior(log=log) for param in self.getparameters(free=free, fixed=fixed)]
        ))

    # TODO: implement
    def getpriormodes(self, free=True, fixed=True):
        params = self.getparameters(free=free, fixed=fixed)
        return [param.getprior.mode for param in params]

    def getlikelihood(self, params=None, data=None, log=True, **kwargs):
        likelihood = self.evaluate(params, data, likelihoodlog=log, **kwargs)[0]
        return likelihood

    def __init__(self, sources, data=None, likefunc="normal", engine=None, engineopts=None, name=""):
        if engine is None:
            engine = "libprofit"
        for i, source in enumerate(sources):
            if not isinstance(source, Source):
                raise TypeError(
                    "Model source[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(source), type(Source))
                )
        if data is not None and not isinstance(data, Data):
            raise TypeError(
                "Model data (type={:s}) is not an instance of {:s}".format(
                    str(type(data)), str(Data))
            )
        self.sources = sources
        self.data = data
        Model._checkengine(engine)
        self.engine = engine
        self.engineopts = engineopts
        self.likefunc = likefunc
        self.name = name


class ModellerPygmoUDP:
    def fitness(self, x):
        return [-self.modeller.evaluate(x, returnlponly=True, timing=self.timing)]

    def get_bounds(self):
        return self.boundslower, self.boundsupper

    def gradient(self, x):
        # TODO: Fix this; it doesn't actually work now that the import is conditional
        # Putting an import statement here is probably a terrible idea
        return pg.estimate_gradient(self.fitness, x)

    def __init__(self, modeller, boundslower, boundsupper, timing=False):
        self.modeller = modeller
        self.boundslower = boundslower
        self.boundsupper = boundsupper
        self.timing = timing

    def __deepcopy__(self, memo):
        """
            This is some rather ugly code to make a deep copy of a modeller without copying the model's
            data, which should be shared between all copies of a model (and its modeller).
        """
        modelself = self.modeller.model
        model = Model(sources=copy.deepcopy(modelself.sources, memo=memo), data=modelself.data,
                      likefunc=copy.copy(modelself.likefunc), engine=copy.copy(modelself.engine),
                      engineopts=copy.copy(modelself.engineopts))
        modeller = Modeller(model=model, modellib=copy.copy(self.modeller.modellib),
                            modellibopts=copy.copy(self.modeller.modellibopts))
        modeller.fitinfo = copy.copy(self.modeller.fitinfo)
        copied = self.__class__(modeller=modeller, boundslower=copy.copy(self.boundslower),
                                boundsupper=copy.copy(self.boundsupper), timing=copy.copy(self.timing))
        memo[id(self)] = copied
        return copied


class Modeller:
    """
        A class that does the fitting for a Model.
        Some useful things it does is define optimization functions called by libraries and optionally
        store info that they don't track (mainly per-iteration info, including parameter values,
        running time, separate priors and likelihoods rather than just the objective, etc.).
    """
    def evaluate(self, paramsfree=None, timing=False, returnlponly=False, returnlog=True, plot=False,
                 dolinearfitprep=False, comparelikelihoods=False):

        if timing:
            tinit = time.time()
        # TODO: Attempt to prevent/detect defeating this by modifying fixed/free params?
        prior = self.fitinfo["priorLogfixed"] + self.model.getpriorvalue(free=True, fixed=False)
        # TODO: Clarify that setting drawimage = True forces evaluation of likelihoods with both C++/Python
        #  (if possible)
        likelihood = self.model.getlikelihood(
            paramsfree, plot=plot, dolinearfitprep=dolinearfitprep, comparelikelihoods=comparelikelihoods)
        # return LL, LP, etc.
        if returnlponly:
            rv = likelihood + prior
            if not returnlog:
                rv = np.exp(rv)
        else:
            if not returnlog:
                likelihood = np.exp(likelihood)
                prior = np.exp(prior)
            rv = likelihood, prior
        if timing:
            tstep = time.time() - tinit
            rv += tstep
        loginfo = {
            "params": paramsfree,
            "likelihood": likelihood,
            "prior": prior,
        }
        if timing:
            loginfo["time"] = tstep
            loginfo["tinit"] = tinit
        if "log" in self.fitinfo:
            self.fitinfo["log"].append(loginfo)
        if ("printsteps" in self.fitinfo and
                self.fitinfo["printsteps"] is not None):
            stepnum = len(self.fitinfo["log"])
            if stepnum % self.fitinfo["printsteps"] == 0:
                print(stepnum, rv, loginfo, flush=True)
        return rv

    def fit(self, paramsinit=None, printfinal=True, timing=None, maxwalltime=np.Inf, printsteps=None,
            dolinear=True, dolinearonly=False):
        self.fitinfo["priorLogfixed"] = self.model.getpriorvalue(free=False, fixed=True, log=True)
        self.fitinfo["log"] = []
        self.fitinfo["printsteps"] = printsteps

        paramsfree = self.model.getparameters(fixed=False)
        if paramsinit is None:
            paramsinit = np.array([param.getvalue(transformed=True) for param in paramsfree])

        paramnames = [param.name for param in paramsfree]
        isfreeflux = [isinstance(param, FluxParameter) for param in paramsfree]
        isanyfluxfree = any(isfreeflux)
        # Is this a linear problem? True iff all free params are fluxes
        islinear = all(isfreeflux)
        # Do a linear fit at all? No if all fluxes are fixed
        dolinear = dolinear and isanyfluxfree
        # TODO: The mechanism here is TBD
        dolinearonly = dolinear and dolinearonly

        timestart = time.time()
        likelihood = self.evaluate(paramsinit, dolinearfitprep=dolinear, comparelikelihoods=True)
        print("Param names   :".format(paramnames))
        print("Initial params: {}".format(paramsinit))
        print("Initial likelihood in t={:.3e}: {}".format(time.time() - timestart, likelihood))
        sys.stdout.flush()

        if dolinear:
            # If this isn't a linear problem, fix all free non-flux parameters and do a linear fit first
            if not islinear:
                for param in paramsfree:
                    if not isinstance(param, FluxParameter):
                        param.fixed = True
            print("Beginning linear fit")
            tinit = time.time()
            datasizes = []
            # TODO: This should be an input argument
            bands = np.sort(list(self.model.data.exposures.keys()))
            params = []
            for band in bands:
                paramsband = None
                for exposure in self.model.data.exposures[band]:
                    datasizes.append(np.sum(exposure.maskinverse)
                                     if exposure.maskinverse is not None else
                                     exposure.image.size)
                    paramsexposure = [x[1] for x in exposure.meta['modelimagesparamsfree']]
                    if paramsband is None:
                        paramsband = paramsexposure
                    elif paramsband != paramsexposure:
                        raise RuntimeError('Got different linear fit params '
                                           'in two exposures: {} vs {}'.format(paramsband, paramsexposure))
                if paramsband is not None:
                    params += paramsband
            nparams = len(params)
            datasize = np.sum(datasizes)
            # Matrix of vectors for each variable component
            x = np.zeros([datasize, nparams])
            y = np.zeros(datasize)
            idxbegin = 0
            idxexp = 0
            idxparam = 0
            for band in bands:
                exposures = self.model.data.exposures[band]
                for exposure in exposures:
                    idxend = idxbegin+datasizes[idxexp]
                    maskinv = exposure.maskinverse
                    for idxfree, (imgfree, _) in enumerate(exposure.meta['modelimagesparamsfree']):
                        x[idxbegin:idxend, idxparam + idxfree] = (
                            imgfree if maskinv is None else imgfree[maskinv]).flat
                    img = exposure.image
                    imgfixed = exposure.meta['modelimagefixed']
                    y[idxbegin:idxend] = (
                        (img if maskinv is None else img[maskinv]) if imgfixed is None else
                        (img - imgfixed if maskinv is None else img[maskinv] - imgfixed[maskinv])
                    ).flat
                    idxbegin = idxend
                    idxexp += 1
                if exposures:
                    idxparam += idxfree + 1
            likelihood = likelihood[0]
            fluxratiosprint = None
            fitmethods = {
                'scipy.optimize.nnls': [None],
                # TODO: Confirm that nnls really performs best
                #'scipy.optimize.lsq_linear': ['bvls'],
                #'numpy.linalg.lstsq': [None, 1e-2, 0.1, 1],
            }
            for method, fitparams in fitmethods.items():
                for fitparam in fitparams:
                    valuesinit = [param.getvalue(transformed=False) for param in params]
                    if method == 'scipy.optimize.nnls':
                        fluxratios = spopt.nnls(x, y)[0]
                    elif method == 'scipy.optimize.lsq_linear':
                        fluxratios = spopt.lsq_linear(x, y, bounds=(0, np.Inf), method=fitparam).x
                    elif method == 'numpy.linalg.lstsq':
                        fluxratios = np.linalg.lstsq(x, y, rcond=fitparam)[0]
                    else:
                        raise ValueError('Unknown linear fit method ' + method)
                    for fluxratio, param, valueinit in zip(fluxratios, params, valuesinit):
                        ratio = np.max([1e-3, fluxratio])
                        # TODO: See if there is a better alternative to setting values outside transform range
                        # Perhaps leave an option to change the transform to log10?
                        for frac in np.linspace(1, 0, 10+1):
                            param.setvalue(valueinit*(frac*ratio + 1-frac), transformed=False)
                            if np.isfinite(param.getvalue(transformed=False)):
                                break
                    likelihoodnew = self.evaluate(returnlponly=True)
                    if likelihoodnew > likelihood:
                        fluxratiosprint = fluxratios
                        methodbest = method
                        likelihood = likelihoodnew
                    else:
                        for valueinit, param in zip(valuesinit, params):
                            param.setvalue(valueinit, transformed=False)
            print("Model '{}' linear fit elapsed time: {:.3e}".format(self.model.name, time.time() - tinit))
            if fluxratiosprint is None:
                print("Linear fit failed to improve on initial parameters")
            else:
                paramsinit = np.array([param.getvalue(transformed=True) for param in paramsfree])
                print("Final likelihood: {}".format(self.evaluate(returnlponly=True)))
                print("New initial parameters from method {}:\n".format(methodbest), paramsinit)
                print("Linear flux ratios: {}".format(fluxratiosprint))
            if not islinear:
                for param in paramsfree:
                    param.fixed = False

        timerun = 0.0
        if not dolinearonly:
            limits = self.model.getlimits(fixed=False, transformed=True)
            algo = self.modellibopts["algo"]
            if self.modellib == "scipy":
                def neg_like_model(paramsi, modeller):
                    return -modeller.evaluate(paramsi, timing=timing, returnlponly=True)

                tinit = time.time()
                result = spopt.minimize(neg_like_model, paramsinit, method=algo, bounds=np.array(limits),
                                        options={} if 'options' not in self.modellibopts else
                                        self.modellibopts['options'], args=(self, ))
                timerun += time.time() - tinit
                paramsbest = result.x

            elif self.modellib == "pygmo":
                import pygmo as pg
                algocmaes = algo == "cmaes"
                algonlopt = not algocmaes
                if algocmaes:
                    uda = pg.cmaes()
                elif algonlopt:
                    uda = pg.nlopt(algo)
                    uda.ftol_abs = 1e-3
                    if np.isfinite(maxwalltime) and maxwalltime > 0:
                        uda.maxtime = maxwalltime
                    nloptopts = ["stopval", "ftol_rel", "ftol_abs", "xtol_rel", "xtol_abs", "maxeval"]
                    for nloptopt in nloptopts:
                        if nloptopt in self.modellibopts and self.modellibopts[nloptopt] is not None:
                            setattr(uda, nloptopt, self.modellibopts[nloptopt])

                algo = pg.algorithm(uda)
                #        algo.extract(pg.nlopt).ftol_rel = 1e-6
                if algonlopt:
                    algo.extract(pg.nlopt).ftol_abs = 1e-3

                if "verbosity" in self.modellibopts and self.modellibopts["verbosity"] is not None:
                    algo.set_verbosity(self.modellibopts["verbosity"])
                limitslower = np.zeros(len(limits))
                limitsupper = np.zeros(len(limits))
                for i, limit in enumerate(limits):
                    limitslower[i] = limit[0]
                    limitsupper[i] = limit[1]

                udp = ModellerPygmoUDP(modeller=self, boundslower=limitslower, boundsupper=limitsupper,
                                       timing=timing)
                problem = pg.problem(udp)
                pop = pg.population(prob=problem, size=0)
                if algocmaes:
                    npop = 5
                    npushed = 0
                    while npushed < npop:
                        try:
                            #pop.push_back(init + np.random.normal(np.zeros(np.sum(data.tofit)),
                            #                                      data.sigmas[data.tofit]))
                            npushed += 1
                        except:
                            pass
                else:
                    pop.push_back(paramsinit)
                tinit = time.time()
                result = algo.evolve(pop)
                timerun += time.time() - tinit
                paramsbest = result.champion_x
            else:
                raise RuntimeError("Unknown optimization library " + self.modellib)

            if printfinal:
                print(
                    "Model '{}' nonlinear fit elapsed time: {:.3e}".format(self.model.name, timerun))
                print("Final likelihood: {}".format(self.evaluate(paramsbest)))
                print("Parameter names:        " + ",".join(["{:11s}".format(i) for i in paramnames]))
                print("Transformed parameters: " + ",".join(["{:+1.4e}".format(i) for i in paramsbest]))
                # TODO: Finish this
                #print("Parameters (linear): " + ",".join(["{:.4e}".format(i) for i in paramstransformed]))

            result = {
                "fitinfo": copy.copy(self.fitinfo),
                "params": self.model.getparameters(),
                "paramnames": paramnames,
                "paramsbest": paramsbest,
                "result": result,
                "time": timerun,
            }
            return result

    # TODO: Should constraints be implemented?
    def __init__(self, model, modellib, modellibopts, constraints=None):
        self.model = model
        self.modellib = modellib
        self.modellibopts = modellibopts
        self.constraints = constraints

        # Scratch space, I guess...
        self.fitinfo = {
            "priorLogfixed": np.log(1.0)
        }


class Source:
    """
        A model of a source, like a galaxy or star, or even the PSF (TBD).
    """
    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    def getparameters(self, free=None, fixed=None, flatten=True, modifiers=True, time=None):
        astrometry = self.modelastrometric.getposition(time)
        paramobjects = [
            self.modelastrometric.getparameters(free, fixed, time),
            self.modelphotometric.getparameters(free, fixed, flatten=flatten, modifiers=modifiers,
                astrometry=astrometry)
        ]
        params = []
        for paramobject in paramobjects:
            if flatten:
                params += paramobject
            else:
                params.append(paramobject)
        return params

    # TODO: Finish this
    def getparamnames(self, free=None, fixed=None):
        return self.modelastrometric.getparamnames(free, fixed) + \
            self.modelphotometric.getparameters(free, fixed)

    def getprofiles(self, engine, bands, time=None, engineopts=None):
        """

        :param bands: List of bands
        :param engine: Valid rendering engine
        :param engineopts: Dict of engine options
        :return:
        """
        self._checkengine(engine)
        cenx, ceny = self.modelastrometric.getposition(time=time)
        return self.modelphotometric.getprofiles(engine=engine, bands=bands, cenx=cenx, ceny=ceny, time=time,
                                                 engineopts=engineopts)

    def __init__(self, modelastrometry, modelphotometry, name=""):
        self.name = name
        self.modelphotometric = modelphotometry
        self.modelastrometric = modelastrometry


class PhotometricModel:
    def convertfluxparameters(self, usefluxfracs, **kwargs):
        if usefluxfracs and self.fluxes:
            raise RuntimeError("Tried to convert model to use flux fracs but self.fluxes not empty, "
                               "so it must already be using flux fracs")
        profiles = self.getprofiles(engine='libprofit', bands=self.fluxes.keys(), cenx=0, ceny=0)
        for profilescomp, comp in zip(profiles, self.components):
            for i, flux in enumerate(comp.fluxes):
                bandprofile = profilescomp[flux.band]
                if flux.isfluxratio is usefluxfracs:
                    raise RuntimeError(
                        'Tried to convert component with isfluxratio={} already == usefluxfracs={}'.format(
                            flux.isfluxratio, usefluxfracs
                        ))
                if usefluxfracs:
                    if flux.band in self.fluxes:
                        self.fluxes[flux.band].append(flux)
                    else:
                        self.fluxes[flux.band] = np.array([flux])
                else:
                    flux = FluxParameter(
                        band=flux.band, value=0, name=flux.name, isfluxratio=False,
                        unit=self.fluxes[flux.band].unit, **kwargs)
                    flux.setvalue(mpfutil.magtoflux(bandprofile['mag']), transformed=False)
                    comp.fluxes[i] = flux
        if usefluxfracs:
            # Convert fluxes to fractions
            for band, fluxes in self.fluxes.items():
                values = np.zeros(len(fluxes))
                fixed = []
                units = []
                for idx, flux in enumerate(fluxes):
                    values[idx] = flux.getvalue(transformed=False)
                    fixed[idx] = flux.fixed
                    units[idx] = flux.units
                if all(fixed):
                    fixed = True
                elif not any(fixed):
                    fixed = False
                else:
                    raise RuntimeError('Fixed={} invalid; must be all true or all false'.format(fixed))
                total = np.sum(values)
                idxlast = len(fluxes) - 1
                for idx, flux in enumerate(fluxes):
                    islast = idx == idxlast
                    value = 1 if islast else values[i]/total
                    fluxfrac = FluxParameter(
                        band=flux.band, value=value, isfluxratio=True, unit=None,
                        fixed=True if islast else fixed, **kwargs)
                    comp.fluxes[i] = fluxfrac
                    total -= values[i]
                self.fluxes[band] = FluxParameter(
                    band=band, value=np.sum(values), fixed=fixed, name=flux.name, isfluxratio=False, **kwargs)
        else:
            self.fluxes = {}

    def getparameters(self, free=True, fixed=True, flatten=True, modifiers=True, astrometry=None):
        paramsflux = [flux for flux in self.fluxes.values() if
                      (flux.fixed and fixed) or (not flux.fixed and free)]
        params = paramsflux if flatten else [paramsflux]
        for comp in self.components:
            paramscomp = comp.getparameters(free=free, fixed=fixed)
            if flatten:
                params += paramscomp
            else:
                params.append(paramscomp)
        if modifiers:
            modifiers = ([param for param in self.modifiers if
                          (param.fixed and fixed) or (not param.fixed and free)])
            if flatten:
                params += modifiers
            else:
                params.append(modifiers)
        return params

    # TODO: Determine how the astrometric model is supposed to interact here
    def getprofiles(self, engine, bands, cenx, ceny, time=None, engineopts=None):
        """
        :param engine: Valid rendering engine
        :param bands: List of bands
        :param cenx: X coordinate
        :param ceny: Y coordinate
        :param time: A time for variable sources. Not implemented yet.
        :param engineopts: Dict of engine options
        :return: List of dicts by band
        """
        # TODO: Check if this should skip entirely instead of adding a None for non-included bands
        if bands is None:
            bands = self.fluxes.keys()
        bandfluxes = {band: self.fluxes[band].getvalue(transformed=False) if
                      band in self.fluxes else None for band in bands}
        profiles = []
        for comp in self.components:
            profiles += comp.getprofiles(bandfluxes, engine, cenx, ceny, engineopts=engineopts)
        for flux in comp.fluxes:
            if flux.isfluxratio and flux.getvalue(transformed=False) != 1.:
                raise RuntimeError('Non-unity flux ratio for final component')
        return profiles

    def __init__(self, components, fluxes=None, modifiers=None):
        for i, comp in enumerate(components):
            if not isinstance(comp, Component):
                raise TypeError("PhotometricModel component[{:d}](type={:s}) "
                                "is not an instance of {:s}".format(
                    i, type(comp), type(Component)))
        if fluxes is None:
            fluxes = []
        for i, flux in enumerate(fluxes):
            if not isinstance(flux, FluxParameter):
                raise TypeError("PhotometricModel flux[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(flux), type(FluxParameter)))
        bandscomps = [[flux.band for flux in comp.fluxes] for comp in components]
        # TODO: Check if component has a redundant mag or no specified flux ratio
        if not mpfutil.allequal(bandscomps):
            raise ValueError(
                "Bands of component fluxes in PhotometricModel components not all equal: {}".format(
                    bandscomps))
        bandsfluxes = [flux.band for flux in fluxes]
        if any([band not in bandscomps[0] for band in bandsfluxes]):
            raise ValueError("Bands of fluxes in PhotometricModel fluxes not all in fluxes of the "
                             "components: {} not all in {}".format(bandsfluxes, bandscomps[0]))
        self.components = components
        self.fluxes = {flux.band: flux for flux in fluxes}
        self.modifiers = [] if modifiers is None else modifiers


# TODO: Implement and use, with optional WCS attached?
class Position:
    def __init__(self, x, y):
        for key, value in {"x": x, "y": y}:
            if not isinstance(value, Parameter):
                raise TypeError("Position[{:s}](type={:s}) is not an instance of {:s}".format(
                    key, type(value), type(Parameter)))
        self.x = x
        self.y = y


class AstrometricModel:
    """
        The astrometric model for this source.
        TODO: Implement moving models, or at least think about how to do it
    """

    def getparameters(self, free=True, fixed=True, time=None):
        return [value for value in self.params.values() if
                (value.fixed and fixed) or (not value.fixed and free)]

    def getposition(self, time=None):
        return self.params["cenx"].getvalue(transformed=False), self.params["ceny"].getvalue(
            transformed=False)

    def __init__(self, params):
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise TypeError("Mag[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(param), type(Parameter)))
            # TODO: Check if component has a redundant mag or no specified flux ratio
        self.params = {param.name: param for param in params}


# TODO: Store position and/or astrometry
class Component(object, metaclass=ABCMeta):
    """
        A component of a source, which can be extended or not. This abstract class only stores fluxes.
        TODO: Implement shape model or at least alternative angle/axis ratio implementations (w/boxiness)
        It could be isophotal twisting, a 3D shape, etc.
    """
    optional = ["cenx", "ceny"]

    @abstractmethod
    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        """
            bandfluxes is a dict of bands with item flux in a linear scale or None if components independent
            Return is dict keyed by band with lists of engine-dependent profiles.
            galsim are GSObjects
            libprofit are dicts with a "profile" key
        """
        pass

    @abstractmethod
    def getparameters(self, free=True, fixed=True):
        pass

    @abstractmethod
    def isgaussian(self):
        pass

    def __init__(self, fluxes, name=""):
        for i, param in enumerate(fluxes):
            if not isinstance(param, Parameter):
                raise TypeError(
                    "Component param[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, str(type(param)), str(type(Parameter)))
                )
        self.fluxes = fluxes
        self.name = name


class EllipticalProfile(Component):
    """
        Class for any profile with a (generalized) ellipse shape.
        TODO: implement boxiness for libprofit; not sure if galsim does generalized ellipses?
    """
    profilesavailable = ["moffat", "sersic"]
    mandatoryshape = ["ang", "axrat"]
    # TODO: Consider adopting gs's flexible methods of specifying re, fwhm, etc.
    mandatory = {
        "moffat": mandatoryshape + ["con", "fwhm"],
        "sersic": mandatoryshape + ["nser", "re"],
    }

    ENGINES = ["galsim", "libprofit"]
    splinelogeps = 1e-12

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    # TODO: Should the parameters be stored as a dict? This method is the only reason why it's useful now
    def isgaussian(self):
        return (self.profile == "sersic" and self.parameters["nser"].getvalue() == 0.5) \
            or (self.profile == "moffat" and np.isinf(self.parameters["con"].getvalue()))

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if
                (value.fixed and fixed) or (not value.fixed and free)] + \
            [value for value in self.parameters.values() if
                (value.fixed and fixed) or (not value.fixed and free)]

    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        """
        :param bandfluxes: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param engineopts: Dict of engine options
        :return: Dict by band with list of profiles
        """
        self._checkengine(engine)
        isgaussian = self.isgaussian()

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Asked for EllipticalProfile (profile={:s}, name={:s}) model for band={:s} not in "
                    "bands with fluxes {}".format(self.profile, self.name, band, fluxesbands))

        profiles = {}
        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            if fluxesbands[band].isfluxratio:
                fluxratio = copy.copy(flux)
                if not 0 <= fluxratio <= 1:
                    raise ValueError("flux ratio not 0 <= {} <= 1".format(fluxratio))
                flux *= bandfluxes[band]
                bandfluxes[band] *= (1.0-fluxratio)
            profile = {}
            for param in self.parameters.values():
                value = param.getvalue(transformed=False)
                if param.name == "re":
                    for modifier in param.modifiers:
                        if modifier.name == "rscale":
                            value *= modifier.getvalue(transformed=False)
                profile[param.name] = value

            if not 0 < profile["axrat"] <= 1:
                if profile["axrat"] > 1:
                    profile["axrat"] = 1
                elif profile["axrat"] <= 0:
                    profile["axrat"] = 1e-3
                else:
                    raise ValueError("axrat {} ! >0 and <=1".format(profile["axrat"]))

            cens = {"cenx": cenx, "ceny": ceny}

            # Does this profile have a non-zero size?
            # TODO: Review cutoff - it can't be zero for galsim or it will request huge FFTs
            resolved = not (
                (self.profile == "sersic" and profile["re"] < 0*(engine == "galsim")) or
                (self.profile == "moffat" and profile["fwhm"] < 0*(engine == "galsim"))
            )
            for key, value in cens.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = copy.copy(value)
            if resolved:
                if engine == "galsim":
                    axrat = profile["axrat"]
                    axratsqrt = np.sqrt(axrat)
                    gsparams = getgsparams(engineopts)
                    if isgaussian:
                        if self.profile == "sersic":
                            fwhm = 2.*profile["re"]
                        else:
                            fwhm = profile["fwhm"]
                        profilegs = gs.Gaussian(
                            flux=flux, fwhm=fwhm*axratsqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "sersic":
                        if profile["nser"] < 0.3 or profile["nser"] > 6.2:
                            print("Warning: Sersic n {:.3f} not >= 0.3 and <= 6.2; "
                                  "GalSim could fail.".format(profile["nser"]))
                        profilegs = gs.Sersic(
                            flux=flux, n=profile["nser"],
                            half_light_radius=profile["re"]*axratsqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "moffat":
                        profilegs = gs.Moffat(
                            flux=flux, beta=profile["con"],
                            fwhm=profile["fwhm"]*axratsqrt,
                            gsparams=gsparams
                        )
                    profile = {
                        "profile": profilegs,
                        "shear": gs.Shear(q=axrat, beta=(profile["ang"] + 90.)*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    }
                elif engine == "libprofit":
                    profile["mag"] = mpfutil.fluxtomag(flux)
                    # TODO: Review this. It might not be a great idea because Sersic != Moffat integration
                    # libprofit should be able to handle Moffats with infinite con
                    if self.profile != "sersic" and self.isgaussian():
                        profile["profile"] = "sersic"
                        profile["nser"] = 0.5
                        if self.profile == "moffat":
                            profile["re"] = profile["fwhm"]/2.0
                            del profile["fwhm"]
                            del profile["con"]
                        else:
                            raise RuntimeError("No implentation for turning profile {:s} into gaussian".format(
                                profile))
                    else:
                        profile["profile"] = self.profile
                else:
                    raise ValueError("Unimplemented rendering engine {:s}".format(engine))
            else:
                profile["flux"] = flux
            # This profile is part of a point source *model*
            profile["pointsource"] = False
            profile["resolved"] = resolved
            profile["fluxparameter"] = fluxesbands[band]
            profiles[band] = profile
        return [profiles]

    @classmethod
    def _checkparameters(cls, parameters, profile):
        mandatory = {param: False for param in EllipticalProfile.mandatory[profile]}
        paramnamesneeded = mandatory.keys()
        paramnames = [param.name for param in parameters]
        errors = []
        # Not as efficient as early exit if true but oh well
        if len(paramnames) > len(set(paramnames)):
            errors.append("Parameters array not unique")
        # Check if these parameters are known (in mandatory)
        for param in parameters:
            if isinstance(param, FluxParameter):
                errors.append("Param {:s} is {:s}, not {:s}".format(param.name, type(FluxParameter),
                                                                    type(Parameter)))
            if param.name in paramnamesneeded:
                mandatory[param.name] = True
            elif param.name not in Component.optional:
                errors.append("Unknown param {:s}".format(param.name))

        for paramname, found in mandatory.items():
            if not found:
                errors.append("Missing mandatory param {:s}".format(paramname))
        if errors:
            errorstr = "Errors validating params of component (profile={:s}):\n" + \
                       "\n".join(errors) + "\nPassed params:" + str(parameters)
            raise ValueError(errorstr)

    def __init__(self, fluxes, name="", profile="sersic", parameters=None):
        if profile not in EllipticalProfile.profilesavailable:
            raise ValueError("Profile type={:s} not in available: ".format(profile) + str(
                EllipticalProfile.profilesavailable))
        self._checkparameters(parameters, profile)
        self.profile = profile
        Component.__init__(self, fluxes, name)
        self.parameters = {param.name: param for param in parameters}


class PointSourceProfile(Component):

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    @classmethod
    def isgaussian(cls):
        return False

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)]

    # TODO: default PSF could be unit image?
    def getprofiles(self, bandfluxes, engine, cenx, ceny, psf=None):
        """
        Get engine-dependent representations of profiles to be rendered.

        :param bandfluxes: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param psf: A PSF (required, despite the default)
        :return: List of engine-dependent profiles
        """
        self._checkengine(engine)
        if not isinstance(psf, PSF):
            raise TypeError("psf type {} must be a {}".format(type(psf), PSF))

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Called PointSourceProfile (name={:s}) getprofiles() for band={:s} not in "
                    "bands with fluxes {}", self.name, band, fluxesbands)

        # TODO: Think of the best way to do this
        # TODO: Ensure that this is getting copies - it isn't right now
        profiles = psf.model.getprofiles(engine=engine, bands=bandfluxes.keys())
        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            for profile in profiles[band]:
                if engine == "galsim":
                    profile["profile"].flux *= flux
                elif engine == "libprofit":
                    profile["mag"] -= 2.5 * np.log10(flux)
                profile["pointsource"] = True
            else:
                raise ValueError("Unimplemented PointSourceProfile rendering engine {:s}".format(engine))
        return profiles

    def __init__(self, fluxes, name=""):
        Component.__init__(fluxes=fluxes, name=name)


class Transform:
    @classmethod
    def null(cls, value):
        return value

    def __init__(self, transform=None, reverse=None):
        if transform is None or reverse is None:
            if transform is not reverse:
                raise ValueError(
                    "One of transform (type={:s}) and reverse (type={:s}) is {:s} but "
                    "both or neither must be".format(type(transform), type(reverse), type(None))
                 )
            else:
                transform = self.null
                reverse = self.null
        self.transform = transform
        self.reverse = reverse
        # TODO: Verify if forward(reverse(x)) == reverse(forward(x)) +/- error for x in ???


class Limits:
    """
        Limits for a Parameter.
    """
    def within(self, value):
        return self.lower <= value <= self.upper

    def __init__(self, lower=-np.inf, upper=np.inf, lowerinclusive=True, upperinclusive=True,
                 transformed=True):
        isnanlower = np.isnan(lower)
        isnanupper = np.isnan(upper)
        if isnanlower or isnanupper:
            raise ValueError("Limits lower,upper={},{} finite check={},{}".format(
                lower, upper, isnanlower, isnanupper))
        if not upper >= lower:
            raise ValueError("Limits upper={} !>= lower{}".format(lower, upper))
        # TODO: Should pass in the transform and check if lower
        if not lowerinclusive:
            lower = np.nextafter(lower, lower+1.)
        if not upperinclusive:
            upper = np.nextafter(upper, upper-1.)
        self.lower = lower
        self.upper = upper
        self.transformed = transformed


# TODO: This class needs loads of sanity checks and testing
class Parameter:
    """
        A parameter with all the info about itself that you would need when fitting.
    """
    def getvalue(self, transformed=False):
        value = copy.copy(self.value)
        if transformed and not self.transformed:
            value = self.transform.transform(value)
        elif not transformed and self.transformed:
            value = self.transform.reverse(value)
        return value

    def getprior(self, log=True):
        if self.prior is None:
            prior = 1.0
            if log:
                prior = 0.
        else:
            prior = self.prior.getvalue(param=self, log=log)
        return prior

    def getlimits(self, transformed=False):
        lower = self.limits.lower
        upper = self.limits.upper
        if transformed and not self.limits.transformed:
            lower = self.transform.transform(lower)
            upper = self.transform.transform(upper)
        elif not transformed and self.limits.transformed:
            lower = self.transform.reverse(lower)
            upper = self.transform.reverse(upper)
        return lower, upper

    def setvalue(self, value, transformed=False):
        if not transformed:
            if value < self.limits.lower:
                value = self.limits.lower
        if transformed and not self.transformed:
            value = self.transform.reverse(value)
        elif not transformed and self.transformed:
            value = self.transform.transform(value)
        self.value = value
        # TODO: Error checking, etc. There are probably better ways to do this
        for param in self.inheritors:
            param.value = self.value

    def __init__(self, name, value, unit="", limits=None, transform=Transform(), transformed=True, prior=None,
                 fixed=False, inheritors=None, modifiers=None):
        if prior is not None and not isinstance(prior, Prior):
            raise TypeError("prior (type={:s}) is not an instance of {:s}".format(type(prior), type(Prior)))
        if limits is None:
            limits = Limits(transformed=transformed)
        if limits.transformed != transformed:
            raise ValueError("limits.transformed={} != Param[{:s}].transformed={}".format(
                limits.transformed, name, transformed
            ))
        self.fixed = fixed
        self.name = name
        self.value = value
        self.unit = unit
        self.limits = limits
        self.transform = transform
        self.transformed = transformed
        self.prior = prior
        # List of parameters that should inherit values from this one
        self.inheritors = [] if inheritors is None else inheritors
        # List of parameters that can modify this parameter's value - user decides what to do with them
        self.modifiers = [] if modifiers is None else modifiers


class FluxParameter(Parameter):
    """
        A flux, magnitude or flux ratio, all of which one could conceivably fit.
        TODO: name seems a bit redundant, but I don't want to commit to storing the band as a string now
    """
    def __init__(self, band, name, value, unit, limits, transform=None, transformed=True, prior=None,
                 fixed=None, isfluxratio=None):
        if isfluxratio is None:
            isfluxratio = False
        Parameter.__init__(self, name=name, value=value, unit=unit, limits=limits, transform=transform,
                           transformed=transformed, prior=prior, fixed=fixed)
        self.band = band
        self.isfluxratio = isfluxratio


class Prior:
    """
        A prior probability distribution function.
        Not an ecclesiastical superior usually of lower rank than an abbot.

        TODO: I'm not sure how to enforce proper normalization without implementing specific subclasses
        e.g. Even in a flat prior, if the limits change the normalization does too.
    """
    def getvalue(self, param, log):
        if not isinstance(param, Parameter):
            raise TypeError(
                "param(type={:s}) is not an instance of {:s}".format(type(param), type(Parameter)))

        if self.transformed != self.limits.transformed:
            raise ValueError("Prior must have same transformed flag as its Limits")

        paramval = param.getvalue(transformed=self.transformed)
        if self.limits.within(paramval):
            prior = self.func(paramval)
        else:
            prior = 0.0
        if log and not self.log:
            return np.log(prior)
        elif not log and self.log:
            return np.exp(prior)
        return prior

    def __init__(self, func, log, transformed, mode, limits):
        if not isinstance(limits, Limits):
            "Prior limits(type={:s}) is not an instance of {:s}".format(type(limits), type(Limits))
        # TODO: how to type check this?
        self.func = func
        self.log = log
        self.transformed = transformed
        self.mode = mode
        self.limits = limits
