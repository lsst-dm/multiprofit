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
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprofit as mpf
import multiprofit.utils as mpfutil
import multiprofit.asinhstretchsigned as mpfasinh
import numpy as np
import pyprofit as pyp
import scipy.stats as spstats
import scipy.optimize as spopt
import scipy.interpolate as spinterp
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


def getprofilegausscovar(ang, axrat, re):
    ang = np.radians(ang)
    sinang = np.sin(ang)
    cosang = np.cos(ang)
    majsq = (2.*re)**2
    minsq = majsq*axrat**2
    sinangsq = sinang**2
    cosangsq = cosang**2
    sigxsq = majsq*cosangsq + minsq*sinangsq
    sigysq = majsq*sinangsq + minsq*cosangsq
    covxy = (majsq-minsq)*cosang*sinang
    covar = np.matrix([[sigxsq, covxy], [covxy, sigysq]])
    return covar


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
                            weightsband=None, asinhscale=40, imgdiffscale=0.05, asinhscalediff=10):
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
            if weightsband is not None:
                for img, weight in zip(imagesoftype, weightsband):
                    img *= weight
        if maximg is None:
            maximg = np.max([np.max(np.sum(imgs)) for imgs in imagesall])
        for i, imagesoftype in enumerate(imagesall):
            # One could use astropy.visualization.make_lupton_rgb instead but I can't quite figure out
            # how to get it to scale consistently
            hsv = mplcol.rgb_to_hsv(np.stack(imagesoftype, axis=2))
            # Rescale the values in the HSV image to the desired max.
            hsv[:, :, 2] *= np.max(np.sum(imagesoftype, axis=2))/maximg/np.max(hsv[:, :, 2])
            # Apply the asinh scaling
            hsv[:, :, 2] = np.arcsinh(np.clip(hsv[:, :, 2], 0, 1)*asinhscale)/np.arcsinh(asinhscale)
            axes[i].imshow(mplcol.hsv_to_rgb(hsv), origin=originimg)
        (axes[0].set_title if plotascolumn else axes[0].set_ylabel)(bandstring)
        # TODO: Verify if this is strictly correct - it should produce a diff. image with zero at 50% gray
        imgsdiff = [data-model for data, model in zip(imagesall[0], imagesall[1])]
        rgb = 0.5*(1 + np.clip(np.stack(imgsdiff, axis=2)/(maximg*imgdiffscale), -1, 1))
        hsv = mplcol.rgb_to_hsv(rgb)
        hsv[:, :, 2] = 0.5 + np.arcsinh(2*(np.clip(hsv[:, :, 2], 0, 1) - 0.5)*asinhscalediff)/np.arcsinh(
            asinhscalediff)/2
        axes[2].imshow(mplcol.hsv_to_rgb(hsv), origin=originimg)
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
        chisqred = np.sum(chi**2)/len(chi)
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
                 likelihoodonly=False, keepimages=False, keepmodels=False, plot=False,
                 plotmulti=False, figure=None, axes=None, figurerow=None, modelname="Model", modeldesc=None,
                 modelnameappendparams=None, drawimage=True, scale=1, clock=False, plotascolumn=False,
                 imgplotmaxs=None, imgplotmaxmulti=None, weightsband=None):
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
        :param likelihoodonly: bool; compute only the likelihood using faster C++ code? Overrides plot.
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
        :param drawimage: bool; draw (evaluate) the model image for each exposure? Ignored if getlikelihood
            is True.
        :param scale: float; Spatial scale factor for drawing images. User beware; this may not work
            properly and should be replaced by exposure WCS soon.
        :param clock: bool; benchmark model evaluation?
        :param plotascolumn: bool; are the plots arranged vertically?
        :param imgplotmaxs: dict; key band: value maximum for image/model plots.
        :param imgplotmaxmulti: float; maximum for multiband image/model plots.
        :param weightsband: dict; key band: weight for scaling each band in multiband plots.
        :return:
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
        if clock:
            times["setup"] = time.time() - timenow
            timenow = time.time()

        for band in bands:
            # TODO: Check band
            for idxexposure, exposure in enumerate(data.exposures[band]):
                profiles, metamodel, times = self._getexposuremodelsetup(
                    exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
                if likelihoodonly and metamodel['allgaussian'] and metamodel['usefastgauss']:
                    profilemat = np.array([np.array([
                            p['cenx'], p['ceny'], 10**(-0.4*p['mag']), p['re'], p['ang'], p['axrat']
                        ])
                        for p in [profile[band] for profile in profiles]])
                    varinverse = exposure.sigmainverse**2
                    varisscalar = not hasattr(varinverse, 'shape') or varinverse.shape == ()
                    # TODO: Do this in a prettier way while avoiding recalculating loglike_gaussian_pixel
                    if 'likeconst' not in exposure.meta:
                        exposure.meta['likeconst'] = np.sum(np.log(exposure.sigmainverse/np.sqrt(2.*np.pi)))
                        if varisscalar or np.prod(exposure.sigmainverse.shape) == 1:
                            exposure.meta['likeconst'] *= np.prod(exposure.image.shape)
                        print('Setting exposure.meta[\'likeconst\']=', exposure.meta['likeconst'])
                    if varisscalar:
                        # I think this is the easiest way to reshape it into a 1x1 matrix
                        varinverse = np.zeros((1, 1)) + varinverse
                    likelihoodexposure = exposure.meta['likeconst'] + mpf.loglike_gaussian_pixel(
                        exposure.image, varinverse, profilemat, 0, exposure.image.shape[1],
                        0, exposure.image.shape[0])
                    if not likelihoodlog:
                        likelihoodexposure = 10**likelihoodexposure
                    chi = None
                else:
                    image, model, timesmodel = self.getexposuremodel(
                        exposure, profiles=profiles, metamodel=metamodel, engine=engine,
                        engineopts=engineopts, drawimage=drawimage, scale=scale, clock=clock, times=times)
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
                    if getlikelihood or plot:
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

                        likelihoodexposure, chi, imgclip, modelclip = \
                            self.getexposurelikelihood(
                                exposure, image, log=likelihoodlog, figaxes=figaxes,
                                modelname=modelname, modeldesc=modeldesc,
                                modelnameappendparams=modelnameappendparams, plotascolumn=plotascolumn,
                                isfirstmodel=isfirstmodel, islastmodel=islastmodel,
                                maximg=imgplotmaxs[band] if (imgplotmaxs is not None and band in
                                                             imgplotmaxs) else None
                            )
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
        # Color images! whooo
        if plot:
            if plotmulti:
                if singlemodel:
                    Model._plotexposurescolor(
                        imgclips, modelclips, chis, (figure, axes[figurerow]),
                        bands=bands, modelname=modelname, modeldesc=modeldesc,
                        modelnameappendparams=modelnameappendparams, isfirstmodel=isfirstmodel,
                        islastmodel=islastmodel, plotascolumn=plotascolumn, maximg=imgplotmaxmulti,
                        weightsband=weightsband)
                else:
                    Model._plotexposurescolor(
                        imgclips, modelclips, chis, (figure['multi'], axes['multi'][figurerow]),
                        bands=bands, modelname=modelname, modeldesc=modeldesc,
                        modelnameappendparams=modelnameappendparams, isfirstmodel=isfirstmodel,
                        islastmodel=islastmodel, plotascolumn=plotascolumn, maximg=imgplotmaxmulti,
                        weightsband=weightsband)
                figurerow += 1
        if clock:
            print(','.join(['{}={:.2e}'.format(name, value) for name, value in times.items()]))
        return likelihood, params, chis, times

    def getexposurelikelihood(self, exposure, modelimage, log=True, likefunc=None,
                              figaxes=None, maximg=None, minimg=None, modelname="Model", modeldesc=None,
                              modelnameappendparams=None, isfirstmodel=True, islastmodel=True,
                              plotascolumn=False, originimg='bottom', normdiff=None, normchi=None):
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
            if hasmask:
                chi = chi[exposure.maskinverse]
                chisqred = (np.sum(chi**2)/np.sum(exposure.maskinverse))
            else:
                chisqred = np.sum(chi**2)/np.prod(chi.shape)
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
        profiles = self.getprofiles([band], engine=engine)
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
                profiles = self.getprofiles([band], engine="libprofit")

        if profiles:
            # Do analytic convolution of any Gaussian model with the Gaussian PSF
            # This turns each resolved profile into N_PSF_components Gaussians
            if anygaussian and haspsf:
                # Note that libprofit uses the irritating GALFIT convention, so reverse that
                varsgauss = ["ang", "axrat", "re"]
                covarprofilessrc = [(getprofilegausscovar
                                     (**{var: profile[band][var] + 90*(var == "ang") for var in varsgauss}) if
                                     profile[band]["nser"] == 0.5 else None,
                                     profile[band]) for
                                    profile in self.getprofiles([band], engine="libprofit")]

                profilesnew = []
                if clock:
                    times['modelallgausseig'] = 0
                for idxpsf, profilepsf in enumerate(exposure.psf.model.getprofiles(
                        bands=[band], engine="libprofit")):
                    fluxfrac = 10**(-0.4*profilepsf[band]["mag"])
                    covarpsf = getprofilegausscovar(**{var: profilepsf[band][var] + 90*(var == "ang") for
                                                       var in varsgauss})
                    for idxsrc, (covarsrc, profile) in enumerate(covarprofilessrc):
                        if covarsrc is not None:
                            if clock:
                                timeeig = time.time()
                            eigenvals, eigenvecs = np.linalg.eigh(covarpsf + covarsrc)
                            if clock:
                                times['modelallgausseig'] += time.time() - timeeig
                            indexmaj = np.argmax(eigenvals)
                            fwhm = np.sqrt(eigenvals[indexmaj])
                            axrat = np.sqrt(eigenvals[1-indexmaj])/fwhm
                            ang = np.degrees(np.arctan2(eigenvecs[1, indexmaj], eigenvecs[0, indexmaj]))
                            if engine == "libprofit":
                                # Needed because each PSF component will loop over the same profile object
                                profile = copy.copy(profile)
                                profile["re"] = fwhm/2.0
                                profile["axrat"] = axrat
                                profile["mag"] += mpfutil.fluxtomag(fluxfrac)
                                profile["ang"] = ang
                            else:
                                profile = {
                                    "profile": gs.Gaussian(flux=10**(-0.4*profile["mag"])*fluxfrac,
                                                           fwhm=fwhm*np.sqrt(axrat), gsparams=gsparams),
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

    def getexposuremodel(self, exposure, profiles=None, metamodel=None, engine=None, engineopts=None,
                         drawimage=True, scale=1, clock=False, times=None):
        """
            Draw model image for one exposure with one PSF

            Returns the image and the engine-dependent model used to draw it
        """
        # TODO: Rewrite this completely at some point as validating inputs is a pain
        if profiles is None or metamodel is None:
            profiles, metamodel, times = self._getexposuremodelsetup(
                exposure, engine=engine, engineopts=engineopts, clock=clock, times=times)
        ny, nx = exposure.image.shape
        band = exposure.band
        haspsf = metamodel['haspsf']

        if clock:
            times = metamodel['times'] if 'times' in metamodel else []
            timenow = time.time()
            if times is None:
                times = {}

        if engine == "galsim":
            gsparams = getgsparams(engineopts)

        # TODO: Do this in a smarter way
        if metamodel['usefastgauss']:
            profilesrunning = []
            image = None
            model = None
            for profile in profiles:
                params = profile[band]
                profilesrunning.append(profile)
                if len(profilesrunning) == 8:
                    paramsall = [x[band] for x in profilesrunning]
                    imgprofile = np.array(mpf.make_gaussian_mix_8_pixel(
                        params['cenx'], params['ceny'],
                        mpfutil.magtoflux(paramsall[0]['mag']), mpfutil.magtoflux(paramsall[1]['mag']),
                        mpfutil.magtoflux(paramsall[2]['mag']), mpfutil.magtoflux(paramsall[3]['mag']),
                        mpfutil.magtoflux(paramsall[4]['mag']), mpfutil.magtoflux(paramsall[5]['mag']),
                        mpfutil.magtoflux(paramsall[6]['mag']), mpfutil.magtoflux(paramsall[7]['mag']),
                        paramsall[0]['re'], paramsall[1]['re'], paramsall[2]['re'], paramsall[3]['re'],
                        paramsall[4]['re'], paramsall[5]['re'], paramsall[6]['re'], paramsall[7]['re'],
                        paramsall[0]['ang'], paramsall[1]['ang'], paramsall[2]['ang'],
                        paramsall[3]['ang'], paramsall[4]['ang'], paramsall[5]['ang'],
                        paramsall[6]['ang'], paramsall[7]['ang'],
                        paramsall[0]['axrat'], paramsall[1]['axrat'], paramsall[2]['axrat'],
                        paramsall[3]['axrat'], paramsall[4]['axrat'], paramsall[5]['axrat'],
                        paramsall[6]['axrat'], paramsall[7]['axrat'],
                        0, nx, 0, ny, nx, ny))
                    if image is None:
                        image = imgprofile
                    else:
                        image += imgprofile
                    profilesrunning = []
            if profilesrunning:
                for profile in profilesrunning:
                    params = profile[band]
                    imgprofile = np.array(mpf.make_gaussian_pixel(
                        params['cenx'], params['ceny'], mpfutil.magtoflux(params['mag']), params['re'],
                        params['ang'], params['axrat'], 0, nx, 0, ny, nx, ny))
                    if image is None:
                        image = imgprofile
                    else:
                        image += imgprofile
            profiles = []

        if profiles:
            if engine == 'libprofit':
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
                    profilespro[profiletype] += [profile]

                profit_model = {
                    'width': nx,
                    'height': ny,
                    'magzero': 0.0,
                    'profiles': profilespro,
                }
                if psfispixelated:
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
                if drawimage:
                    image = np.array(pyp.make_model(profit_model)[0])
                model = profit_model
            elif engine == "galsim":
                if clock:
                    timesetupgs = time.time()
                    times['modelsetupprofilegs'] = 0
                profilesgs = {
                    True: {"small": None, "big": None},
                    False: {"all": None},
                }
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
                        convolve = haspsf and not profile["pointsource"]
                    # TODO: Revise this when image scales are taken into account
                    profiletype = ("all" if not convolve else (
                        "big" if profilegs.original.half_light_radius > 1 else "small"))
                    if profilesgs[convolve][profiletype] is None:
                        profilesgs[convolve][profiletype] = profilegs
                    else:
                        profilesgs[convolve][profiletype] += profilegs
                profilesgs[False] = profilesgs[False]["all"]

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
                    profilesgsconv = None
                    for sizebin, profilesbin in profilesgs[True].items():
                        if profilesbin is not None:
                            profilesbin = gs.Convolve(profilesbin, profilespsf, gsparams=gsparams)
                            if profilesgsconv is None:
                                profilesgsconv = profilesbin
                            else:
                                profilesgsconv += profilesbin
                    profilesgs[True] = profilesgsconv
                else:
                    if any([x is not None for x in profilesgs[True].values()]):
                        raise RuntimeError("Model (band={}) has profiles to convolve but no PSF".format(
                            exposure.band))
                    profilesgs[True] = None
                    model = profilesgs[False]
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
                # Has a PSF, but it's not an analytic model, so it must be an image and therefore pixel
                # convolution is already included for extended objects, which must use no_pixel
                try:
                    if clock:
                        times['modelsetup'] = time.time() - timenow
                        timenow = time.time()
                    haspsfimage = haspsf and psfgs is None and profilesgs[True] is not None
                    if haspsfimage:
                        model = profilesgs[True]
                        if drawimage:
                            imagegs = profilesgs[True].drawImage(method="no_pixel", nx=nx, ny=ny, scale=scale)
                        if profilesgs[False] is not None:
                            model += profilesgs[False]
                            if drawimage:
                                imagegs += profilesgs[False].drawImage(method=method, nx=nx, ny=ny,
                                                                       scale=scale)
                    else:
                        if profilesgs[True] is not None:
                            if profilesgs[False] is not None:
                                profilesgs = profilesgs[True] + profilesgs[False]
                            else:
                                profilesgs = profilesgs[True]
                        else:
                            profilesgs = profilesgs[False]
                        if drawimage:
                            imagegs = profilesgs.drawImage(method=method, nx=nx, ny=ny, scale=scale)
                        model = profilesgs
                # This is not a great hack - catch RunTimeErrors which are usually excessively large FFTs
                # and then try to evaluate the model in real space or give up if it's anything else
                except RuntimeError as e:
                    try:
                        if method == "fft":
                            if haspsfimage:
                                imagegs = profilesgs[True].drawImage(
                                              method="real_space", nx=nx, ny=ny, scale=scale) + \
                                          profilesgs[False].drawImage(
                                              method=method, nx=nx, ny=ny, scale=scale)
                                model = profilesgs[True] + profilesgs[False]
                            else:
                                imagegs = profilesgs.drawImage(method="real_space", nx=nx, ny=ny, scale=scale)
                                model = profilesgs
                        else:
                            raise e
                    except Exception as e:
                        raise e
                except Exception as e:
                    print("Exception attempting to draw image from profiles:", profilesgs)
                    raise e
                # TODO: Check that this isn't destroyed
                if drawimage:
                    image = imagegs.array
            else:
                errmsg = "engine is None" if engine is None else "engine type " + engine + " not implemented"
                raise RuntimeError(errmsg)

        if clock:
            times['modeldraw'] = time.time() - timenow
        if not drawimage:
            return None, model, times

        sumnotfinite = np.sum(~np.isfinite(image))
        if sumnotfinite > 0:
            raise RuntimeError("{}.getexposuremodel() got {:d} non-finite pixels out of {:d}".format(
                type(self), sumnotfinite, np.prod(image.shape)
            ))

        return image, model, times

    def getlimits(self, free=True, fixed=True, transformed=True):
        params = self.getparameters(free=free, fixed=fixed)
        return [param.getlimits(transformed=transformed) for param in params]

    def getprofiles(self, bands, engine=None, engineopts=None):
        """
        :param bands: List of bands
        :param engine: Valid rendering engine
        :param engineopts: Dict of engine options
        :return: List of profiles
        """
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        self._checkengine(engine)
        profiles = []
        for src in self.sources:
            profiles += src.getprofiles(engine=engine, bands=bands, engineopts=engineopts)
        return profiles

    def getparameters(self, free=True, fixed=True, flatten=True):
        params = []
        for src in self.sources:
            paramssrc = src.getparameters(free=free, fixed=fixed, flatten=flatten)
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
        return [-self.modeller.evaluate(x, returnlponly=True, timing=self.timing, likelihoodonly=True)]

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
    def evaluate(self, paramsfree=None, timing=False, returnlponly=False, likelihoodonly=False,
                 returnlog=True, plot=False):

        if timing:
            tinit = time.time()
        # TODO: Attempt to prevent/detect defeating this by modifying fixed/free params?
        prior = self.fitinfo["priorLogfixed"] + self.model.getpriorvalue(free=True, fixed=False)
        likelihood = self.model.getlikelihood(paramsfree, plot=plot, likelihoodonly=likelihoodonly)
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
                print(stepnum, rv, loginfo)
                sys.stdout.flush()
        return rv

    def fit(self, paramsinit=None, printfinal=True, timing=None, maxwalltime=np.Inf, printsteps=None):
        self.fitinfo["priorLogfixed"] = self.model.getpriorvalue(free=False, fixed=True, log=True)
        self.fitinfo["log"] = []
        self.fitinfo["printsteps"] = printsteps

        if paramsinit is None:
            paramsinit = [param.getvalue(transformed=True) for param in self.model.getparameters(fixed=False)]
            paramsinit = np.array(paramsinit)
            # TODO: Why did I think I would want to do this?
            #paramsinit = self.model.getpriormodes(free=True, fixed=False)

        paramnames = [param.name for param in self.model.getparameters(fixed=False)]
        print("Param names:\n", paramnames)
        print("Initial parameters:\n", paramsinit)
        print("Evaluating initial parameters:\n", self.evaluate(paramsinit))

        sys.stdout.flush()

        timerun = 0.0

        limits = self.model.getlimits(fixed=False, transformed=True)
        algo = self.modellibopts["algo"]

        if self.modellib == "scipy":
            def neg_like_model(params, modeller):
                return -modeller.evaluate(params, timing=timing, returnlponly=True, likelihoodonly=True)

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
            print("Elapsed time: {:.1f}".format(timerun))
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

    def getparameters(self, free=None, fixed=None, flatten=True, time=None):
        astrometry = self.modelastrometric.getposition(time)
        paramobjects = [
            self.modelastrometric.getparameters(free, fixed, time),
            self.modelphotometric.getparameters(free, fixed, flatten=flatten, astrometry=astrometry)
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
        if usefluxfracs:
            raise RuntimeError('Converting model to use fluxfracs not supported yet')
        else:
            profiles = self.getprofiles('libprofit', self.fluxes.keys(), 0, 0)
            for profilescomp, comp in zip(profiles, self.components):
                for i, flux in enumerate(comp.fluxes):
                    bandprofile = profilescomp[flux.band]
                    if not flux.isfluxratio:
                        raise RuntimeError('Tried to convert component without fluxfracs from fluxfracs')
                    flux = FluxParameter(
                        band=flux.band, value=0, name=flux.name, isfluxratio=False,
                        unit=self.fluxes[flux.band].unit, **kwargs)
                    flux.setvalue(mpfutil.magtoflux(bandprofile['mag']), transformed=False)
                    comp.fluxes[i] = flux
        self.fluxes = {}

    def getparameters(self, free=True, fixed=True, flatten=True, astrometry=None):
        paramsflux = [flux for flux in self.fluxes.values() if
                      (flux.fixed and fixed) or (not flux.fixed and free)]
        params = paramsflux if flatten else [paramsflux]
        for comp in self.components:
            paramscomp = comp.getparameters(free=free, fixed=fixed)
            if flatten:
                params += paramscomp
            else:
                params.append(paramscomp)
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

    def __init__(self, components, fluxes=None):
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
                # bandfluxes[band] -= flux
                # TODO: Is subtracting as above best? Should be more accurate, but mightn't guarantee flux>=0
                bandfluxes[band] *= (1.0-fluxratio)
            profile = {param.name: param.getvalue(transformed=False) for param in
                       self.parameters.values()}
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


class MultiGaussianApproximationProfile(Component):
    # TODO: Figure out a better way to split functionality between this and EllipticalProfile
    # Maybe have a generic EllipticalProfile and SingleEllipticalProfile?
    """
        Class for multi-Gaussian profiles with (generalized) ellipse shape(s).
    """
    profilesavailable = ["sersic"]
    mandatoryshape = ["ang", "axrat"]
    ENGINES = ["galsim", "libprofit"]
    mandatoryshape = ["ang", "axrat"]
    mandatory = {
        "sersic": mandatoryshape + ["nser", "re"],
    }

    # These weights are derived here:
    # https://github.com/lsst-dm/modelling_research/blob/master/jupyternotebooks/multigaussian_sersic1d.ipynb
    weights = {
        "sersic": {
            4: {
                0.5: (
                    mpfutil.normalize(
                        np.array([1., 0., 0., 0.])),
                    np.array([1.0000000000e+00, 8.6800000000e-01, 5.5600000000e-01, 2.7200000000e-01]),
                ),
                0.501: (
                    mpfutil.normalize(
                        np.array([9.8945700264e-01, 9.9340700388e-03, 5.9654256544e-04, 1.2384756796e-05])),
                    np.array([1.0018152162e+00, 8.6829891694e-01, 5.5601124505e-01, 2.7194641925e-01]),
                ),
                0.502: (
                    mpfutil.normalize(
                        np.array([9.7915845320e-01, 1.9623440160e-02, 1.1927485937e-03, 2.5358041869e-05])),
                    np.array([1.0036304324e+00, 8.6859783387e-01, 5.5602249010e-01, 2.7189283849e-01]),
                ),
                0.503: (
                    mpfutil.normalize(
                        np.array([9.6909614755e-01, 2.9076336691e-02, 1.7886076942e-03, 3.8908060978e-05])),
                    np.array([1.0054456487e+00, 8.6889675081e-01, 5.5603373515e-01, 2.7183925774e-01]),
                ),
                0.504: (
                    mpfutil.normalize(
                        np.array([9.5926219793e-01, 3.8300684239e-02, 2.3840804072e-03, 5.3037419541e-05])),
                    np.array([1.0072608649e+00, 8.6919566775e-01, 5.5604498020e-01, 2.7178567698e-01]),
                ),
                0.505: (
                    mpfutil.normalize(
                        np.array([9.4964907955e-01, 4.7304034601e-02, 2.9791457766e-03, 6.7740074277e-05])),
                    np.array([1.0090760811e+00, 8.6949458468e-01, 5.5605622525e-01, 2.7173209623e-01]),
                ),
                0.506: (
                    mpfutil.normalize(
                        np.array([9.4024957357e-01, 5.6093640927e-02, 3.5737695079e-03, 8.3015990936e-05])),
                    np.array([1.0108912973e+00, 8.6979350162e-01, 5.5606747030e-01, 2.7167851547e-01]),
                ),
                0.507: (
                    mpfutil.normalize(
                        np.array([9.3105678877e-01, 6.4676421037e-02, 4.1679292284e-03, 9.8860967198e-05])),
                    np.array([1.0127065136e+00, 8.7009241855e-01, 5.5607871535e-01, 2.7162493472e-01]),
                ),
                0.508: (
                    mpfutil.normalize(
                        np.array([9.2206414921e-01, 7.3058966723e-02, 4.7616185406e-03, 1.1526552325e-04])),
                    np.array([1.0145217298e+00, 8.7039133549e-01, 5.5608996040e-01, 2.7157135396e-01]),
                ),
                0.509: (
                    mpfutil.normalize(
                        np.array([9.1326529121e-01, 8.1247682677e-02, 5.3547918794e-03, 1.3223423568e-04])),
                    np.array([1.0163369460e+00, 8.7069025243e-01, 5.5610120545e-01, 2.7151777321e-01]),
                ),
                0.51: (
                    mpfutil.normalize(
                        np.array([9.0465415940e-01, 8.9248645803e-02, 5.9474342283e-03, 1.4976056908e-04])),
                    np.array([1.0181521622e+00, 8.7098916936e-01, 5.5611245050e-01, 2.7146419246e-01]),
                ),
                0.512: (
                    mpfutil.normalize(
                        np.array([8.8797205161e-01, 1.0471043423e-01, 7.1310424559e-03, 1.8647171091e-04])),
                    np.array([1.0217825947e+00, 8.7158700324e-01, 5.5613494059e-01, 2.7135703095e-01]),
                ),
                0.514: (
                    mpfutil.normalize(
                        np.array([8.7197402933e-01, 1.1948833341e-01, 8.3122605259e-03, 2.2537673275e-04])),
                    np.array([1.0254130271e+00, 8.7218483711e-01, 5.5615743069e-01, 2.7124986944e-01]),
                ),
                0.516: (
                    mpfutil.normalize(
                        np.array([8.5661973094e-01, 1.3362289316e-01, 9.4909271779e-03, 2.6644872161e-04])),
                    np.array([1.0290434596e+00, 8.7278267098e-01, 5.5617992079e-01, 2.7114270793e-01]),
                ),
                0.518: (
                    mpfutil.normalize(
                        np.array([8.4187191031e-01, 1.4715152809e-01, 1.0666905708e-02, 3.0965589274e-04])),
                    np.array([1.0326738920e+00, 8.7338050486e-01, 5.5620241089e-01, 2.7103554642e-01]),
                ),
                0.52: (
                    mpfutil.normalize(
                        np.array([8.2769605865e-01, 1.6010893375e-01, 1.1840026965e-02, 3.5498064219e-04])),
                    np.array([1.0363043244e+00, 8.7397833873e-01, 5.5622490099e-01, 2.7092838491e-01]),
                ),
                0.525: (
                    mpfutil.normalize(
                        np.array([7.9455497749e-01, 1.9020815824e-01, 1.4759495284e-02, 4.7736898718e-04])),
                    np.array([1.0453804056e+00, 8.7547292341e-01, 5.5628112624e-01, 2.7066048114e-01]),
                ),
                0.53: (
                    mpfutil.normalize(
                        np.array([7.6435585744e-01, 2.1737336399e-01, 1.7658366124e-02, 6.1241244531e-04])),
                    np.array([1.0544564867e+00, 8.7696750809e-01, 5.5633735149e-01, 2.7039257736e-01]),
                ),
                0.535: (
                    mpfutil.normalize(
                        np.array([7.3673150348e-01, 2.4197382835e-01, 2.0534948388e-02, 7.5971977755e-04])),
                    np.array([1.0635325678e+00, 8.7846209277e-01, 5.5639357673e-01, 2.7012467359e-01]),
                ),
                0.54: (
                    mpfutil.normalize(
                        np.array([7.1137288409e-01, 2.6432043281e-01, 2.3387773784e-02, 9.1890931808e-04])),
                    np.array([1.0726086489e+00, 8.7995667746e-01, 5.5644980198e-01, 2.6985676982e-01]),
                ),
                0.545: (
                    mpfutil.normalize(
                        np.array([6.8801812381e-01, 2.8467668137e-01, 2.6215589953e-02, 1.0896048655e-03])),
                    np.array([1.0816847300e+00, 8.8145126214e-01, 5.5650602723e-01, 2.6958886605e-01]),
                ),
                0.55: (
                    mpfutil.normalize(
                        np.array([6.6644385745e-01, 3.0326737752e-01, 2.9017327056e-02, 1.2714379704e-03])),
                    np.array([1.0907608111e+00, 8.8294584682e-01, 5.5656225248e-01, 2.6932096228e-01]),
                ),
                0.56: (
                    mpfutil.normalize(
                        np.array([6.2789648133e-01, 3.3589734166e-01, 3.4539094688e-02, 1.6670823192e-03])),
                    np.array([1.1089129733e+00, 8.8593501618e-01, 5.5667470297e-01, 2.6878515473e-01]),
                ),
                0.57: (
                    mpfutil.normalize(
                        np.array([5.9448786957e-01, 3.6346156025e-01, 3.9947517585e-02, 2.1030525873e-03])),
                    np.array([1.1270651355e+00, 8.8892418555e-01, 5.5678715347e-01, 2.6824934719e-01]),
                ),
                0.58: (
                    mpfutil.normalize(
                        np.array([5.6527551984e-01, 3.8690886198e-01, 4.5238935286e-02, 2.5766828937e-03])),
                    np.array([1.1452172978e+00, 8.9191335491e-01, 5.5689960396e-01, 2.6771353964e-01]),
                ),
                0.59: (
                    mpfutil.normalize(
                        np.array([5.3953208499e-01, 4.0697128630e-01, 5.0411196453e-02, 3.0854322571e-03])),
                    np.array([1.1633694600e+00, 8.9490252428e-01, 5.5701205446e-01, 2.6717773209e-01]),
                ),
                0.6: (
                    mpfutil.normalize(
                        np.array([5.1668620161e-01, 4.2422360341e-01, 5.5463315784e-02, 3.6268791954e-03])),
                    np.array([1.1815216222e+00, 8.9789169364e-01, 5.5712450495e-01, 2.6664192455e-01]),
                ),
                0.625: (
                    mpfutil.normalize(
                        np.array([4.7758685902e-01, 4.4871014896e-01, 6.8837799255e-02, 4.8651927687e-03])),
                    np.array([1.2213388674e+00, 9.0522947018e-01, 5.5593716167e-01, 2.6448724702e-01]),
                ),
                0.65: (
                    mpfutil.normalize(
                        np.array([4.4790973196e-01, 4.6472790091e-01, 8.1183680488e-02, 6.1786866405e-03])),
                    np.array([1.2597206861e+00, 9.1227538845e-01, 5.5420949557e-01, 2.6202170433e-01]),
                ),
                0.675: (
                    mpfutil.normalize(
                        np.array([4.2468720925e-01, 4.7528820509e-01, 9.2482622270e-02, 7.5419633935e-03])),
                    np.array([1.2968687054e+00, 9.1901538645e-01, 5.5211228246e-01, 2.5929320267e-01]),
                ),
                0.7: (
                    mpfutil.normalize(
                        np.array([4.0619261279e-01, 4.8211835269e-01, 1.0275368140e-01, 8.9353531141e-03])),
                    np.array([1.3328732728e+00, 9.2538839592e-01, 5.4973718519e-01, 2.5638262685e-01]),
                ),
                0.725: (
                    mpfutil.normalize(
                        np.array([3.9117459271e-01, 4.8639603270e-01, 1.1208423379e-01, 1.0345140795e-02])),
                    np.array([1.3678589145e+00, 9.3142767823e-01, 5.4718845799e-01, 2.5336660763e-01]),
                ),
                0.75: (
                    mpfutil.normalize(
                        np.array([3.7883214393e-01, 4.8886181980e-01, 1.2054759480e-01, 1.1758441469e-02])),
                    np.array([1.4018952939e+00, 9.3712739357e-01, 5.4450683796e-01, 2.5028075100e-01]),
                ),
                0.775: (
                    mpfutil.normalize(
                        np.array([3.6858115238e-01, 4.9002935491e-01, 1.2822402555e-01, 1.3165467156e-02])),
                    np.array([1.4350485124e+00, 9.4249753902e-01, 5.4172689777e-01, 2.4715619082e-01]),
                ),
                0.8: (
                    mpfutil.normalize(
                        np.array([3.5999400190e-01, 4.9025638794e-01, 1.3519106673e-01, 1.4558543438e-02])),
                    np.array([1.4673735837e+00, 9.4755130911e-01, 5.3887419185e-01, 2.4401511404e-01]),
                ),
                0.825: (
                    mpfutil.normalize(
                        np.array([3.5275638884e-01, 4.8979354452e-01, 1.4151859016e-01, 1.5931476486e-02])),
                    np.array([1.4989145047e+00, 9.5229840735e-01, 5.3596493005e-01, 2.4087532891e-01]),
                ),
                0.85: (
                    mpfutil.normalize(
                        np.array([3.4661930227e-01, 4.8882885283e-01, 1.4727213167e-01, 1.7279713237e-02])),
                    np.array([1.5297163559e+00, 9.5675535334e-01, 5.3301353941e-01, 2.3774833319e-01]),
                ),
                0.875: (
                    mpfutil.normalize(
                        np.array([3.4139475178e-01, 4.8749519173e-01, 1.5251013392e-01, 1.8599922570e-02])),
                    np.array([1.5598149092e+00, 9.6093516491e-01, 5.3003111821e-01, 2.3464487823e-01]),
                ),
                0.9: (
                    mpfutil.normalize(
                        np.array([3.3693483823e-01, 4.8589217444e-01, 1.5728351471e-01, 1.9889472624e-02])),
                    np.array([1.5892416384e+00, 9.6484911931e-01, 5.2702460930e-01, 2.3157178352e-01]),
                ),
                0.925: (
                    mpfutil.normalize(
                        np.array([3.3311881357e-01, 4.8409461947e-01, 1.6163971512e-01, 2.1146851843e-02])),
                    np.array([1.6180273453e+00, 9.6851160006e-01, 5.2400218926e-01, 2.2853547559e-01]),
                ),
                0.95: (
                    mpfutil.normalize(
                        np.array([3.2985357508e-01, 4.8215861242e-01, 1.6561775424e-01, 2.2370058269e-02])),
                    np.array([1.6461952712e+00, 9.7193082380e-01, 5.2096478580e-01, 2.2553624276e-01]),
                ),
                0.975: (
                    mpfutil.normalize(
                        np.array([3.2705820815e-01, 4.8012679182e-01, 1.6925584733e-01, 2.3559152698e-02])),
                    np.array([1.6737719728e+00, 9.7512079513e-01, 5.1792043407e-01, 2.2258062250e-01]),
                ),
                1.0: (
                    mpfutil.normalize(
                        np.array([3.2466847784e-01, 4.7803210251e-01, 1.7258586760e-01, 2.4713552046e-02])),
                    np.array([1.7007792632e+00, 9.7809127671e-01, 5.1487137336e-01, 2.1966932363e-01]),
                ),
                1.1: (
                    mpfutil.normalize(
                        np.array([3.1821477023e-01, 4.6944264972e-01, 1.8335597803e-01, 2.8986602026e-02])),
                    np.array([1.8034938165e+00, 9.8796648656e-01, 5.0267797561e-01, 2.0848694531e-01]),
                ),
                1.2: (
                    mpfutil.normalize(
                        np.array([3.1519204595e-01, 4.6105964089e-01, 1.9101500145e-01, 3.2733311709e-02])),
                    np.array([1.8984714894e+00, 9.9502366952e-01, 4.9057127084e-01, 1.9805003387e-01]),
                ),
                1.3: (
                    mpfutil.normalize(
                        np.array([3.1432860593e-01, 4.5319073488e-01, 1.9647826914e-01, 3.6002390043e-02])),
                    np.array([1.9865281111e+00, 9.9969455805e-01, 4.7861874939e-01, 1.8832607467e-01]),
                ),
                1.4: (
                    mpfutil.normalize(
                        np.array([3.1489512079e-01, 4.4590727991e-01, 2.0035001581e-01, 3.8847583497e-02])),
                    np.array([2.0682658389e+00, 1.0023112528e+00, 4.6685531166e-01, 1.7926310694e-01]),
                ),
                1.5: (
                    mpfutil.normalize(
                        np.array([3.1644196302e-01, 4.3919486381e-01, 2.0304297828e-01, 4.1320194886e-02])),
                    np.array([2.1441585351e+00, 1.0031405656e+00, 4.5529969615e-01, 1.7080392320e-01]),
                ),
                1.6: (
                    mpfutil.normalize(
                        np.array([3.1868206214e-01, 4.3300475565e-01, 2.0484577176e-01, 4.3467410445e-02])),
                    np.array([2.2145442254e+00, 1.0023921643e+00, 4.4396221165e-01, 1.6289631422e-01]),
                ),
                1.7: (
                    mpfutil.normalize(
                        np.array([3.2141661149e-01, 4.2728356657e-01, 2.0596836872e-01, 4.5331453223e-02])),
                    np.array([2.2797211146e+00, 1.0002492514e+00, 4.3285136654e-01, 1.5549262413e-01]),
                ),
                1.8: (
                    mpfutil.normalize(
                        np.array([3.2450372508e-01, 4.2197913345e-01, 2.0656682602e-01, 4.6950315445e-02])),
                    np.array([2.3399462224e+00, 9.9687259873e-01, 4.2197602006e-01, 1.4855234090e-01]),
                ),
                1.9: (
                    mpfutil.normalize(
                        np.array([3.2782474774e-01, 4.1703843886e-01, 2.0677020639e-01, 4.8366607011e-02])),
                    np.array([2.3954526149e+00, 9.9243861765e-01, 4.1137384028e-01, 1.4206176868e-01]),
                ),
                2.0: (
                    mpfutil.normalize(
                        np.array([3.3128672873e-01, 4.1242201447e-01, 2.0667672958e-01, 4.9614527222e-02])),
                    np.array([2.4465543239e+00, 9.8711540712e-01, 4.0107011168e-01, 1.3599717559e-01]),
                ),
                2.1: (
                    mpfutil.normalize(
                        np.array([3.3480773477e-01, 4.0809304875e-01, 2.0636991637e-01, 5.0729300115e-02])),
                    np.array([2.4935840432e+00, 9.8108195933e-01, 3.9110438291e-01, 1.3034765947e-01]),
                ),
                2.2: (
                    mpfutil.normalize(
                        np.array([3.3824474964e-01, 4.0400630511e-01, 2.0596694378e-01, 5.1782001471e-02])),
                    np.array([2.5371137448e+00, 9.7470582393e-01, 3.8164575988e-01, 1.2518661787e-01]),
                ),
                2.3: (
                    mpfutil.normalize(
                        np.array([3.4154001680e-01, 4.0014083300e-01, 2.0551993522e-01, 5.2799214983e-02])),
                    np.array([2.5776523647e+00, 9.6817460552e-01, 3.7272302287e-01, 1.2049124896e-01]),
                ),
                2.4: (
                    mpfutil.normalize(
                        np.array([3.4462155040e-01, 3.9647171906e-01, 2.0508756596e-01, 5.3819164576e-02])),
                    np.array([2.6157607079e+00, 9.6172950359e-01, 3.6440804188e-01, 1.1627023797e-01]),
                ),
                2.5: (
                    mpfutil.normalize(
                        np.array([3.4743061538e-01, 3.9297757432e-01, 2.0471664490e-01, 5.4875165409e-02])),
                    np.array([2.6519979386e+00, 9.5559201385e-01, 3.5676044559e-01, 1.1252623389e-01]),
                ),
                2.6: (
                    mpfutil.normalize(
                        np.array([3.4967753819e-01, 3.8961227234e-01, 2.0459321850e-01, 5.6116970967e-02])),
                    np.array([2.6880626669e+00, 9.5066322139e-01, 3.5023361900e-01, 1.0948440882e-01]),
                ),
                2.7: (
                    mpfutil.normalize(
                        np.array([3.5177955585e-01, 3.8641279396e-01, 2.0446411884e-01, 5.7343531359e-02])),
                    np.array([2.7224001800e+00, 9.4582839068e-01, 3.4410642827e-01, 1.0670704905e-01]),
                ),
                2.8: (
                    mpfutil.normalize(
                        np.array([3.5319024425e-01, 3.8330557458e-01, 2.0466754604e-01, 5.8836635130e-02])),
                    np.array([2.7580928560e+00, 9.4277497993e-01, 3.3927833449e-01, 1.0466777695e-01]),
                ),
                2.9: (
                    mpfutil.normalize(
                        np.array([3.5378909446e-01, 3.8026742773e-01, 2.0527342180e-01, 6.0670056004e-02])),
                    np.array([2.7964529303e+00, 9.4202609384e-01, 3.3595005870e-01, 1.0343690690e-01]),
                ),
                3.0: (
                    mpfutil.normalize(
                        np.array([3.5433518510e-01, 3.7736713030e-01, 2.0582911847e-01, 6.2468566130e-02])),
                    np.array([2.8336134738e+00, 9.4133657412e-01, 3.3284256153e-01, 1.0231209515e-01]),
                ),
                3.1: (
                    mpfutil.normalize(
                        np.array([3.5408315961e-01, 3.7451250882e-01, 2.0677913224e-01, 6.4625199334e-02])),
                    np.array([2.8743074645e+00, 9.4315430327e-01, 3.3122820579e-01, 1.0194775480e-01]),
                ),
                3.2: (
                    mpfutil.normalize(
                        np.array([3.5394142199e-01, 3.7179082231e-01, 2.0759163388e-01, 6.6676121820e-02])),
                    np.array([2.9132258233e+00, 9.4461371375e-01, 3.2955559590e-01, 1.0152897182e-01]),
                ),
                3.3: (
                    mpfutil.normalize(
                        np.array([3.5305658132e-01, 3.6909677926e-01, 2.0876704539e-01, 6.9079594029e-02])),
                    np.array([2.9561537781e+00, 9.4863034763e-01, 3.2931721623e-01, 1.0181154872e-01]),
                ),
                3.4: (
                    mpfutil.normalize(
                        np.array([3.5192719487e-01, 3.6647515631e-01, 2.1001638061e-01, 7.1581268212e-02])),
                    np.array([3.0001702025e+00, 9.5363280939e-01, 3.2966495376e-01, 1.0235436427e-01]),
                ),
                3.5: (
                    mpfutil.normalize(
                        np.array([3.5060254214e-01, 3.6392322866e-01, 2.1131414032e-01, 7.4160088878e-02])),
                    np.array([3.0451387321e+00, 9.5950945004e-01, 3.3052314087e-01, 1.0311659760e-01]),
                ),
                3.6: (
                    mpfutil.normalize(
                        np.array([3.4912428096e-01, 3.6143851508e-01, 2.1263958405e-01, 7.6797619914e-02])),
                    np.array([3.0909310508e+00, 9.6616182047e-01, 3.3182712939e-01, 1.0406449740e-01]),
                ),
                3.7: (
                    mpfutil.normalize(
                        np.array([3.4752854894e-01, 3.5901873588e-01, 2.1397544966e-01, 7.9477265519e-02])),
                    np.array([3.1374206755e+00, 9.7350002889e-01, 3.3352041677e-01, 1.0516982075e-01]),
                ),
                3.8: (
                    mpfutil.normalize(
                        np.array([3.4584552795e-01, 3.5666183181e-01, 2.1530800221e-01, 8.2184638032e-02])),
                    np.array([3.1844907916e+00, 9.8144530745e-01, 3.3555479492e-01, 1.0640926144e-01]),
                ),
                3.9: (
                    mpfutil.normalize(
                        np.array([3.4362667421e-01, 3.5429634831e-01, 2.1688691349e-01, 8.5190063993e-02])),
                    np.array([3.2361730736e+00, 9.9188531513e-01, 3.3884215229e-01, 1.0822186544e-01]),
                ),
                4.0: (
                    mpfutil.normalize(
                        np.array([3.4184964448e-01, 3.5205834937e-01, 2.1817572509e-01, 8.7916281061e-02])),
                    np.array([3.2841593573e+00, 1.0008659833e+00, 3.4144548069e-01, 1.0967514839e-01]),
                ),
                4.1: (
                    mpfutil.normalize(
                        np.array([3.4005124040e-01, 3.4987797080e-01, 2.1943432458e-01, 9.0636464215e-02])),
                    np.array([3.3324038190e+00, 1.0102607769e+00, 3.4427950089e-01, 1.1121285477e-01]),
                ),
                4.2: (
                    mpfutil.normalize(
                        np.array([3.3780191983e-01, 3.4768053656e-01, 2.2089659664e-01, 9.3620946971e-02])),
                    np.array([3.3851409074e+00, 1.0220325225e+00, 3.4828215083e-01, 1.1328173586e-01]),
                ),
                4.3: (
                    mpfutil.normalize(
                        np.array([3.3601330133e-01, 3.4560948604e-01, 2.2207355647e-01, 9.6303656152e-02])),
                    np.array([3.4336805116e+00, 1.0321249127e+00, 3.5150240078e-01, 1.1495552057e-01]),
                ),
                4.4: (
                    mpfutil.normalize(
                        np.array([3.3424114354e-01, 3.4359102799e-01, 2.2320863101e-01, 9.8959197467e-02])),
                    np.array([3.4822167800e+00, 1.0424956754e+00, 3.5488215174e-01, 1.1668476113e-01]),
                ),
                4.5: (
                    mpfutil.normalize(
                        np.array([3.3249408218e-01, 3.4162359263e-01, 2.2429988001e-01, 1.0158244518e-01])),
                    np.array([3.5306750929e+00, 1.0531083728e+00, 3.5840316213e-01, 1.1846250846e-01]),
                ),
                4.6: (
                    mpfutil.normalize(
                        np.array([3.3077912016e-01, 3.3970523159e-01, 2.2534617890e-01, 1.0416946935e-01])),
                    np.array([3.5789852971e+00, 1.0639311710e+00, 3.6205013302e-01, 1.2028303071e-01]),
                ),
                4.7: (
                    mpfutil.normalize(
                        np.array([3.2910208683e-01, 3.3783437795e-01, 2.2634683555e-01, 1.0671669966e-01])),
                    np.array([3.6270819206e+00, 1.0749342719e+00, 3.6580889367e-01, 1.2214120589e-01]),
                ),
                4.8: (
                    mpfutil.normalize(
                        np.array([3.2746771584e-01, 3.3600943950e-01, 2.2730162001e-01, 1.0922122465e-01])),
                    np.array([3.6749037219e+00, 1.0860906741e+00, 3.6966693242e-01, 1.2403261813e-01]),
                ),
                4.9: (
                    mpfutil.normalize(
                        np.array([3.2624487697e-01, 3.3430423712e-01, 2.2802322150e-01, 1.1142766441e-01])),
                    np.array([3.7179708914e+00, 1.0953376653e+00, 3.7265528866e-01, 1.2550566759e-01]),
                ),
                5.0: (
                    mpfutil.normalize(
                        np.array([3.2469494436e-01, 3.3256655511e-01, 2.2889385343e-01, 1.1384464709e-01])),
                    np.array([3.7650917059e+00, 1.1067330284e+00, 3.7668264430e-01, 1.2745436857e-01]),
                ),
                5.1: (
                    mpfutil.normalize(
                        np.array([3.2354062374e-01, 3.3094533523e-01, 2.2954545188e-01, 1.1596858915e-01])),
                    np.array([3.8073867328e+00, 1.1161838350e+00, 3.7982690822e-01, 1.2898167857e-01]),
                ),
                5.2: (
                    mpfutil.normalize(
                        np.array([3.2208450763e-01, 3.2928880600e-01, 2.3033432453e-01, 1.1829236184e-01])),
                    np.array([3.8536403099e+00, 1.1277433902e+00, 3.8398921289e-01, 1.3097589146e-01]),
                ),
                5.3: (
                    mpfutil.normalize(
                        np.array([3.2100646759e-01, 3.2774571866e-01, 2.3091792559e-01, 1.2032988816e-01])),
                    np.array([3.8950303683e+00, 1.1373335145e+00, 3.8725931976e-01, 1.3254720884e-01]),
                ),
                5.4: (
                    mpfutil.normalize(
                        np.array([3.2027896321e-01, 3.2631406031e-01, 2.3131414720e-01, 1.2209282928e-01])),
                    np.array([3.9315853626e+00, 1.1449534426e+00, 3.8963891892e-01, 1.3369808741e-01]),
                ),
                5.5: (
                    mpfutil.normalize(
                        np.array([3.1926336064e-01, 3.2484343677e-01, 2.3184319316e-01, 1.2405000943e-01])),
                    np.array([3.9720090223e+00, 1.1546382291e+00, 3.9301530391e-01, 1.3530812924e-01]),
                ),
                5.6: (
                    mpfutil.normalize(
                        np.array([3.1858250670e-01, 3.2348118508e-01, 2.3219636882e-01, 1.2573993941e-01])),
                    np.array([4.0076071518e+00, 1.1623413089e+00, 3.9549599166e-01, 1.3649697272e-01]),
                ),
                5.7: (
                    mpfutil.normalize(
                        np.array([3.1792217796e-01, 3.2215146340e-01, 2.3253123511e-01, 1.2739512352e-01])),
                    np.array([4.0427226616e+00, 1.1700759446e+00, 3.9802186821e-01, 1.3770308584e-01]),
                ),
                5.8: (
                    mpfutil.normalize(
                        np.array([3.1728261246e-01, 3.2085314979e-01, 2.3284835496e-01, 1.2901588279e-01])),
                    np.array([4.0773540484e+00, 1.1778366476e+00, 4.0058956475e-01, 1.3892518114e-01]),
                ),
                5.9: (
                    mpfutil.normalize(
                        np.array([3.1666408556e-01, 3.1958517525e-01, 2.3314829004e-01, 1.3060244916e-01])),
                    np.array([4.1115006498e+00, 1.1856179310e+00, 4.0319577590e-01, 1.4016208250e-01]),
                ),
                6.0: (
                    mpfutil.normalize(
                        np.array([3.1633788451e-01, 3.1841919978e-01, 2.3329982319e-01, 1.3194309251e-01])),
                    np.array([4.1409057150e+00, 1.1914075491e+00, 4.0490227294e-01, 1.4097923007e-01]),
                ),
                6.15: (
                    mpfutil.normalize(
                        np.array([3.1599635660e-01, 3.1675771729e-01, 2.3344694983e-01, 1.3379897628e-01])),
                    np.array([4.1820500944e+00, 1.1991175754e+00, 4.0706101860e-01, 1.4201564715e-01]),
                ),
                6.3: (
                    mpfutil.normalize(
                        np.array([3.1566659474e-01, 3.1515443900e-01, 2.3357978807e-01, 1.3559917819e-01])),
                    np.array([4.2222276786e+00, 1.2068576369e+00, 4.0929437183e-01, 1.4308291759e-01]),
                ),
                6.45: (
                    mpfutil.normalize(
                        np.array([3.1559853443e-01, 3.1367747961e-01, 2.3357997276e-01, 1.3714401320e-01])),
                    np.array([4.2572411244e+00, 1.2126102163e+00, 4.1066204676e-01, 1.4374733334e-01]),
                ),
                6.6: (
                    mpfutil.normalize(
                        np.array([3.1577453573e-01, 3.1232227982e-01, 2.3345859196e-01, 1.3844459249e-01])),
                    np.array([4.2871343136e+00, 1.2163609353e+00, 4.1115794342e-01, 1.4400826897e-01]),
                ),
                6.8: (
                    mpfutil.normalize(
                        np.array([3.1615162465e-01, 3.1063558605e-01, 2.3322383402e-01, 1.3998895528e-01])),
                    np.array([4.3228602263e+00, 1.2200215999e+00, 4.1128396422e-01, 1.4410941934e-01]),
                ),
                7.0: (
                    mpfutil.normalize(
                        np.array([3.1673813265e-01, 3.0909602483e-01, 2.3288611110e-01, 1.4127973142e-01])),
                    np.array([4.3529043052e+00, 1.2216422195e+00, 4.1056166866e-01, 1.4382225816e-01]),
                ),
                7.2: (
                    mpfutil.normalize(
                        np.array([3.1775635772e-01, 3.0776545301e-01, 2.3234364820e-01, 1.4213454107e-01])),
                    np.array([4.3729586302e+00, 1.2191186098e+00, 4.0802622143e-01, 1.4271029990e-01]),
                ),
                7.4: (
                    mpfutil.normalize(
                        np.array([3.1896494236e-01, 3.0656756874e-01, 2.3171244184e-01, 1.4275504706e-01])),
                    np.array([4.3872282722e+00, 1.2144236572e+00, 4.0459360870e-01, 1.4119856069e-01]),
                ),
                7.6: (
                    mpfutil.normalize(
                        np.array([3.2036561864e-01, 3.0549662996e-01, 2.3099307346e-01, 1.4314467793e-01])),
                    np.array([4.3956132281e+00, 1.2074877237e+00, 4.0023879131e-01, 1.3928059243e-01]),
                ),
                7.8: (
                    mpfutil.normalize(
                        np.array([3.2196444937e-01, 3.0454758594e-01, 2.3018380480e-01, 1.4330415989e-01])),
                    np.array([4.3979592256e+00, 1.1982273959e+00, 3.9493334561e-01, 1.3694918085e-01]),
                ),
                8.0: (
                    mpfutil.normalize(
                        np.array([3.2401528949e-01, 3.0378290082e-01, 2.2916820523e-01, 1.4303360447e-01])),
                    np.array([4.3893063163e+00, 1.1843287450e+00, 3.8765576491e-01, 1.3375549788e-01]),
                ),
            },
            8: {
                0.5: (
                    mpfutil.normalize(np.array([1., 0, 0, 0, 0, 0, 0, 0])),
                    np.array([1., 9.0907169503e-01, 7.8061774471e-01, 5.6632815891e-01,
                              4.0699848261e-01, 2.7428165889e-01, 1.6667433497e-01, 8.1936830104e-02]),
                ),
                0.501: (
                    mpfutil.normalize(np.array(
                        [9.8309081062e-01, 1.4087009237e-02, 2.4916406163e-03, 2.4668023661e-04,
                         7.3745899515e-05, 7.3233075902e-06, 2.6828887191e-06, 1.0719227828e-07])),
                    np.array([1.0022362632e+00, 9.1034401393e-01, 7.8116382234e-01, 5.6655450917e-01,
                              4.0699848261e-01, 2.7417207769e-01, 1.6654118188e-01, 8.1838662954e-02]),
                ),
                0.502: (
                    mpfutil.normalize(np.array(
                        [9.6663028145e-01, 2.7700592068e-02, 4.9990098784e-03, 5.0273560468e-04,
                         1.4675979308e-04, 1.5055502427e-05, 5.3505768485e-06, 2.1512466830e-07])),
                    np.array([1.0044730508e+00, 9.1161557118e-01, 7.8170919194e-01, 5.6678049827e-01,
                              4.0699848261e-01, 2.7406275865e-01, 1.6640840035e-01, 8.1740808813e-02]),
                ),
                0.503: (
                    mpfutil.normalize(np.array(
                        [9.5060305253e-01, 4.0857769233e-02, 7.5207581369e-03, 7.6773816414e-04,
                         2.1921385256e-04, 2.3120652785e-05, 8.0247997539e-06, 3.2263403005e-07])),
                    np.array([1.0067103642e+00, 9.1288636876e-01, 7.8225385585e-01, 5.6700612752e-01,
                              4.0699848261e-01, 2.7395370061e-01, 1.6627598862e-01, 8.1643266061e-02]),
                ),
                0.504: (
                    mpfutil.normalize(np.array(
                        [9.3499330605e-01, 5.3577093716e-02, 1.0054040662e-02, 1.0419495572e-03,
                         2.9082514962e-04, 3.1712507187e-05, 1.0631785117e-05, 4.4057661952e-07])),
                    np.array([1.0089482026e+00, 9.1415640864e-01, 7.8279781637e-01, 5.6723139821e-01,
                              4.0699848261e-01, 2.7384490244e-01, 1.6614394491e-01, 8.1546033092e-02]),
                ),
                0.505: (
                    mpfutil.normalize(np.array(
                        [9.1978663802e-01, 6.5875009211e-02, 1.2596889736e-02, 1.3252955933e-03,
                         3.6162779264e-04, 4.0777171354e-05, 1.3184022145e-05, 5.7844986723e-07])),
                    np.array([1.0111865650e+00, 9.1542569277e-01, 7.8334107580e-01, 5.6745631160e-01,
                              4.0699848261e-01, 2.7373636301e-01, 1.6601226749e-01, 8.1449108310e-02]),
                ),
                0.506: (
                    mpfutil.normalize(np.array(
                        [9.0497024754e-01, 7.7764895442e-02, 1.5149366619e-02, 1.6168280315e-03,
                         4.3206062801e-04, 5.0140682061e-05, 1.5755349674e-05, 7.0570695410e-07])),
                    np.array([1.0134254504e+00, 9.1669422309e-01, 7.8388363644e-01, 5.6768086897e-01,
                              4.0699848261e-01, 2.7362808119e-01, 1.6588095460e-01, 8.1352490130e-02]),
                ),
                0.507: (
                    mpfutil.normalize(np.array(
                        [8.9052988922e-01, 8.9263922450e-02, 1.7708231070e-02, 1.9171022956e-03,
                         5.0176176043e-04, 5.9970283261e-05, 1.8275338331e-05, 8.4758433427e-07])),
                    np.array([1.0156648580e+00, 9.1796200154e-01, 7.8442550056e-01, 5.6790507160e-01,
                              4.0699848261e-01, 2.7352005585e-01, 1.6575000453e-01, 8.1256176980e-02]),
                ),
                0.508: (
                    mpfutil.normalize(np.array(
                        [8.7645376939e-01, 1.0038451549e-01, 2.0273367304e-02, 2.2253623113e-03,
                         5.7107404161e-04, 7.0125599132e-05, 2.0793626493e-05, 9.9223547090e-07])),
                    np.array([1.0179047869e+00, 9.1922903005e-01, 7.8496667043e-01, 5.6812892073e-01,
                              4.0699848261e-01, 2.7341228590e-01, 1.6561941557e-01, 8.1160167299e-02]),
                ),
                0.509: (
                    mpfutil.normalize(np.array(
                        [8.6272908630e-01, 1.1114182942e-01, 2.2842249206e-02, 2.5419745249e-03,
                         6.3968876885e-04, 8.0774130388e-05, 2.3246514064e-05, 1.1511378976e-06])),
                    np.array([1.0201452361e+00, 9.2049531053e-01, 7.8550714830e-01, 5.6835241762e-01,
                              4.0699848261e-01, 2.7330477022e-01, 1.6548918599e-01, 8.1064459535e-02]),
                ),
                0.51: (
                    mpfutil.normalize(np.array(
                        [8.4934469629e-01, 1.2154790138e-01, 2.5414362440e-02, 2.8663645915e-03,
                         7.0790665304e-04, 9.1766084840e-05, 2.5689755528e-05, 1.3128085204e-06])),
                    np.array([1.0223862047e+00, 9.2176084490e-01, 7.8604693641e-01, 5.6857556352e-01,
                              4.0699848261e-01, 2.7319750772e-01, 1.6535931412e-01, 8.0969052148e-02]),
                ),
                0.512: (
                    mpfutil.normalize(np.array(
                        [8.2355249852e-01, 1.4135612329e-01, 3.0563410006e-02, 3.5377502180e-03,
                         8.4322152425e-04, 1.1481963468e-04, 3.0522667684e-05, 1.6541427379e-06])),
                    np.array([1.0268696968e+00, 9.2428968288e-01, 7.8712445227e-01, 5.6902080726e-01,
                              4.0699848261e-01, 2.7298373789e-01, 1.6510063678e-01, 8.0779132403e-02]),
                ),
                0.514: (
                    mpfutil.normalize(np.array(
                        [7.9899316921e-01, 1.5990380647e-01, 3.5710887171e-02, 4.2384189024e-03,
                         9.7711604708e-04, 1.3931148321e-04, 3.5271270163e-05, 2.0194418417e-06])),
                    np.array([1.0313552560e+00, 9.2681555911e-01, 7.8819923570e-01, 5.6946466178e-01,
                              4.0699848261e-01, 2.7277096777e-01, 1.6484337022e-01, 8.0590395959e-02]),
                ),
                0.516: (
                    mpfutil.normalize(np.array(
                        [7.7558943163e-01, 1.7727738776e-01, 4.0848854081e-02, 4.9668992165e-03,
                         1.1099035731e-03, 1.6516342092e-04, 3.9951716791e-05, 2.4085992607e-06])),
                    np.array([1.0358428751e+00, 9.2933848856e-01, 7.8927130424e-01, 5.6990713680e-01,
                              4.0699848261e-01, 2.7255918881e-01, 1.6458750131e-01, 8.0402830884e-02]),
                ),
                0.518: (
                    mpfutil.normalize(np.array(
                        [7.5327013658e-01, 1.9355619523e-01, 4.5970324903e-02, 5.7217595931e-03,
                         1.2418748103e-03, 1.9231841549e-04, 4.4569809178e-05, 2.8206549857e-06])),
                    np.array([1.0403325471e+00, 9.3185848607e-01, 7.9034067523e-01, 5.7034824193e-01,
                              4.0699848261e-01, 2.7234839259e-01, 1.6433301709e-01, 8.0216425408e-02]),
                ),
                0.52: (
                    mpfutil.normalize(np.array(
                        [7.3196919232e-01, 2.0881407192e-01, 5.1068574515e-02, 6.5018106068e-03,
                         1.3732305680e-03, 2.2072873063e-04, 4.9138994140e-05, 3.2523462096e-06])),
                    np.array([1.0448242651e+00, 9.3437556634e-01, 7.9140736581e-01, 5.7078798666e-01,
                              4.0699848261e-01, 2.7213857077e-01, 1.6407990475e-01, 8.0031167928e-02]),
                ),
                0.525: (
                    mpfutil.normalize(np.array(
                        [6.8278301465e-01, 2.4292409615e-01, 6.3676783770e-02, 8.5539752337e-03,
                         1.7003591408e-03, 2.9698308173e-04, 6.0356445549e-05, 4.4315291824e-06])),
                    np.array([1.0560624656e+00, 9.4065559947e-01, 7.9406247663e-01, 5.7188145825e-01,
                              4.0699848261e-01, 2.7161822553e-01, 1.6345304341e-01, 7.9572971987e-02]),
                ),
                0.53: (
                    mpfutil.normalize(np.array(
                        [6.3875507340e-01, 2.7198914397e-01, 7.6033064829e-02, 1.0737137406e-02,
                         2.0283922703e-03, 3.8004136368e-04, 7.1417886623e-05, 5.7288694628e-06])),
                    np.array([1.0673133028e+00, 9.4691771495e-01, 7.9670120149e-01, 5.7296663012e-01,
                              4.0699848261e-01, 2.7110379338e-01, 1.6283448496e-01, 7.9121705340e-02]),
                ),
                0.535: (
                    mpfutil.normalize(np.array(
                        [5.9918808196e-01, 2.9677542159e-01, 8.8082893011e-02, 1.3034751655e-02,
                         2.3599928831e-03, 4.6930724393e-04, 8.2410922273e-05, 7.1407388649e-06])),
                    np.array([1.0785766716e+00, 9.5316213221e-01, 7.9932379468e-01, 5.7404364267e-01,
                              4.0699848261e-01, 2.7059515228e-01, 1.6222404265e-01, 7.8677199293e-02]),
                ),
                0.54: (
                    mpfutil.normalize(np.array(
                        [5.6349849628e-01, 3.1791867916e-01, 9.9787377137e-02, 1.5431669887e-02,
                         2.6974630300e-03, 5.6423257658e-04, 9.3423425058e-05, 8.6585015208e-06])),
                    np.array([1.0898524684e+00, 9.5938906599e-01, 8.0193050422e-01, 5.7511263265e-01,
                              4.0699848261e-01, 2.7009218384e-01, 1.6162153560e-01, 7.8239290788e-02]),
                ),
                0.545: (
                    mpfutil.normalize(np.array(
                        [5.3119503339e-01, 3.3594909797e-01, 1.1111995514e-01, 1.7913961609e-02,
                         3.0428527023e-03, 6.6427281862e-04, 1.0455643613e-04, 1.0269938465e-05])),
                    np.array([1.1011405919e+00, 9.6559872645e-01, 8.0452157205e-01, 5.7617373328e-01,
                              4.0699848261e-01, 2.6959477312e-01, 1.6102678861e-01, 7.7807822162e-02]),
                ),
                0.55: (
                    mpfutil.normalize(np.array(
                        [5.0186142640e-01, 3.5131169403e-01, 1.2206307201e-01, 2.0469217092e-02,
                         3.3977246723e-03, 7.6902605266e-04, 1.1586668614e-04, 1.1973059914e-05])),
                    np.array([1.1124409422e+00, 9.7179131932e-01, 8.0709723426e-01, 5.7722707438e-01,
                              4.0699848261e-01, 2.6910280858e-01, 1.6043963190e-01, 7.7382640919e-02]),
                ),
                0.56: (
                    mpfutil.normalize(np.array(
                        [4.5073790392e-01, 3.7547361660e-01, 1.4274718868e-01, 2.5754127142e-02,
                         4.1411315564e-03, 9.9111737243e-04, 1.3929326547e-04, 1.5621467014e-05])),
                    np.array([1.1350779325e+00, 9.8412610389e-01, 8.1220325793e-01, 5.7931098086e-01,
                              4.0699848261e-01, 2.6813478777e-01, 1.5928743610e-01, 7.6550555182e-02]),
                ),
                0.57: (
                    mpfutil.normalize(np.array(
                        [4.0785785248e-01, 3.9276483697e-01, 1.6182242700e-01, 3.1207637577e-02,
                         4.9356424877e-03, 1.2279060332e-03, 1.6413305380e-04, 1.9564395427e-05])),
                    np.array([1.1577626743e+00, 9.9639498193e-01, 8.1725035376e-01, 5.8136532669e-01,
                              4.0699848261e-01, 2.6718729131e-01, 1.5816369065e-01, 7.5741909117e-02]),
                ),
                0.58: (
                    mpfutil.normalize(np.array(
                        [3.7153148809e-01, 4.0489676054e-01, 1.7932862668e-01, 3.6765184855e-02,
                         5.7860931467e-03, 1.4773544501e-03, 1.9072454716e-04, 2.3767684661e-05])),
                    np.array([1.1804944278e+00, 1.0085994528e+00, 8.2224021804e-01, 5.8339103895e-01,
                              4.0699848261e-01, 2.6625953525e-01, 1.5706721212e-01, 7.4955648308e-02]),
                ),
                0.59: (
                    mpfutil.normalize(np.array(
                        [3.4048043140e-01, 4.1312661743e-01, 1.9533862662e-01, 4.2373984655e-02,
                         6.6949390946e-03, 1.7378698767e-03, 2.1932542340e-04, 2.8205501710e-05])),
                    np.array([1.2032724773e+00, 1.0207409564e+00, 8.2717446997e-01, 5.8538900026e-01,
                              4.0699848261e-01, 2.6535077843e-01, 1.5599688562e-01, 7.4190782895e-02]),
                ),
                0.6: (
                    mpfutil.normalize(np.array(
                        [3.1372329256e-01, 4.1838892696e-01, 2.0994265280e-01, 4.7991121392e-02,
                         7.6628074417e-03, 2.0082132442e-03, 2.5012708760e-04, 3.2858515965e-05])),
                    np.array([1.2260961300e+00, 1.0328208769e+00, 8.3205465644e-01, 5.8736005161e-01,
                              4.0699848261e-01, 2.6446031949e-01, 1.5495165976e-01, 7.3446382611e-02]),
                ),
                0.625: (
                    mpfutil.normalize(np.array(
                        [2.6099719492e-01, 4.2275873840e-01, 2.4094764459e-01, 6.1858674645e-02,
                         1.0333654133e-02, 2.7211565707e-03, 3.3759375082e-04, 4.5342994264e-05])),
                    np.array([1.2833504226e+00, 1.0627598647e+00, 8.4402813520e-01, 5.9217512270e-01,
                              4.0699848261e-01, 2.6230995013e-01, 1.5244203081e-01, 7.1669297996e-02]),
                ),
                0.65: (
                    mpfutil.normalize(np.array(
                        [2.2255710120e-01, 4.1964078694e-01, 2.6528960995e-01, 7.5200146771e-02,
                         1.3333013103e-02, 3.4795798994e-03, 4.4081020756e-04, 5.8951924994e-05])),
                    np.array([1.3408756591e+00, 1.0923416226e+00, 8.5569420844e-01, 5.9683849500e-01,
                              4.0699848261e-01, 2.6026040245e-01, 1.5006913872e-01, 7.0002434997e-02]),
                ),
                0.675: (
                    mpfutil.normalize(np.array(
                        [1.9362331316e-01, 4.1274100861e-01, 2.8430601328e-01, 8.7808324085e-02,
                         1.6610115988e-02, 4.2775717437e-03, 5.6000775566e-04, 7.3645380633e-05])),
                    np.array([1.3986626227e+00, 1.1215839185e+00, 8.6707215232e-01, 6.0136051589e-01,
                              4.0699848261e-01, 2.5830333518e-01, 1.4782068946e-01, 6.8435098592e-02]),
                ),
                0.7: (
                    mpfutil.normalize(np.array(
                        [1.7126594507e-01, 4.0404691053e-01, 2.9909801838e-01, 9.9583864066e-02,
                         2.0109494619e-02, 5.1114689786e-03, 6.9485978320e-04, 8.9438569740e-05])),
                    np.array([1.4567027409e+00, 1.1505030139e+00, 8.7817937248e-01, 6.0575047484e-01,
                              4.0699848261e-01, 2.5643137456e-01, 1.4568589847e-01, 6.6957979020e-02]),
                ),
                0.725: (
                    mpfutil.normalize(np.array(
                        [1.5360782153e-01, 3.9464579833e-01, 3.1054479446e-01, 1.1049425719e-01,
                         2.3777561692e-02, 5.9786496787e-03, 8.4474182712e-04, 1.0637528435e-04])),
                    np.array([1.5149880192e+00, 1.1791138408e+00, 8.8903164529e-01, 6.1001674496e-01,
                              4.0699848261e-01, 2.5463797213e-01, 1.4365525919e-01, 6.5562930470e-02]),
                ),
                0.75: (
                    mpfutil.normalize(np.array(
                        [1.3940075454e-01, 3.8513629592e-01, 3.1933927013e-01, 1.2054767394e-01,
                         2.7565695764e-02, 6.8769128425e-03, 1.0088829868e-03, 1.2451388725e-04])),
                    np.array([1.5735109829e+00, 1.2074301525e+00, 8.9964332072e-01, 6.1416690157e-01,
                              4.0699848261e-01, 2.5291728764e-01, 1.4172035363e-01, 6.4242791372e-02]),
                ),
                0.775: (
                    mpfutil.normalize(np.array(
                        [1.2778826451e-01, 3.7584414361e-01, 3.2602546393e-01, 1.2977616332e-01,
                         3.1431419324e-02, 7.8041692968e-03, 1.1864600849e-03, 1.4391592234e-04])),
                    np.array([1.6322646278e+00, 1.2354646525e+00, 9.1002749419e-01, 6.1820782198e-01,
                              4.0699848261e-01, 2.5126409175e-01, 1.3987369630e-01, 6.2991237363e-02]),
                ),
                0.8: (
                    mpfutil.normalize(np.array(
                        [1.1816644753e-01, 3.6693989578e-01, 3.3103051455e-01, 1.3822498822e-01,
                         3.5338551735e-02, 8.7583121202e-03, 1.3766491797e-03, 1.6464087983e-04])),
                    np.array([1.6912423757e+00, 1.2632291058e+00, 9.2019615278e-01, 6.2214576994e-01,
                              4.0699848261e-01, 2.4967368486e-01, 1.3810860478e-01, 6.1802660163e-02]),
                ),
                0.825: (
                    mpfutil.normalize(np.array(
                        [1.1009934273e-01, 3.5850471394e-01, 3.3469070109e-01, 1.4594581545e-01,
                         3.9256854522e-02, 9.7371742451e-03, 1.5786541629e-03, 1.8674385878e-04])),
                    np.array([1.7504380364e+00, 1.2907344353e+00, 9.3016030047e-01, 6.2598646774e-01,
                              4.0699848261e-01, 2.4814182879e-01, 1.3641909160e-01, 6.0672067129e-02]),
                ),
                0.85: (
                    mpfutil.normalize(np.array(
                        [1.0326557970e-01, 3.5056829259e-01, 3.3727166915e-01, 1.5299249841e-01,
                         4.3161426759e-02, 1.0738540143e-02, 1.7917199410e-03, 2.1027331046e-04])),
                    np.array([1.8098457733e+00, 1.3179908059e+00, 9.3993006588e-01, 6.2973515774e-01,
                              4.0699848261e-01, 2.4666468910e-01, 1.3479977348e-01, 5.9594997451e-02]),
                ),
                0.875: (
                    mpfutil.normalize(np.array(
                        [9.7423912383e-02, 3.4313078327e-01, 3.3898426511e-01, 1.5941840593e-01,
                         4.7032060966e-02, 1.1760160418e-02, 2.0151406821e-03, 2.3527124063e-04])),
                    np.array([1.8694600738e+00, 1.3450076975e+00, 9.4951479537e-01, 6.3339665549e-01,
                              4.0699848261e-01, 2.4523878609e-01, 1.3324579473e-01, 5.8567451847e-02]),
                ),
                0.9: (
                    mpfutil.normalize(np.array(
                        [9.2390237228e-02, 3.3617591771e-01, 3.3999656404e-01, 1.6527489454e-01,
                         5.0852565130e-02, 1.2799789712e-02, 2.2482589806e-03, 2.6177265423e-04])),
                    np.array([1.9292757222e+00, 1.3717939694e+00, 9.5892313392e-01, 6.3697539550e-01,
                              4.0699848261e-01, 2.4386095287e-01, 1.3175276218e-01, 5.7585833246e-02]),
                ),
                0.925: (
                    mpfutil.normalize(np.array(
                        [8.8021978134e-02, 3.2967878455e-01, 3.4044315871e-01, 1.7061043240e-01,
                         5.4610162054e-02, 1.3855212679e-02, 2.4904650251e-03, 2.8980643964e-04])),
                    np.array([1.9892877762e+00, 1.3983579171e+00, 9.6816309563e-01, 6.4047547107e-01,
                              4.0699848261e-01, 2.4252829955e-01, 1.3031668973e-01, 5.6646896507e-02]),
                ),
                0.95: (
                    mpfutil.normalize(np.array(
                        [8.4207252704e-02, 3.2361045944e-01, 3.4043231855e-01, 1.7547016950e-01,
                         5.8294939748e-02, 1.4924270998e-02, 2.7411934590e-03, 3.1939560181e-04])),
                    np.array([2.0494915452e+00, 1.4247073224e+00, 9.7724212546e-01, 6.4390066898e-01,
                              4.0699848261e-01, 2.4123818220e-01, 1.2893395081e-01, 5.5747705586e-02]),
                ),
                0.975: (
                    mpfutil.normalize(np.array(
                        [8.0857203429e-02, 3.1794076520e-01, 3.4005150474e-01, 1.7989578654e-01,
                         6.1899379840e-02, 1.6004882352e-02, 2.9999198800e-03, 3.5055801062e-04])),
                    np.array([2.1098825713e+00, 1.4508494976e+00, 9.8616715342e-01, 6.4725449979e-01,
                              4.0699848261e-01, 2.3998817614e-01, 1.2760123754e-01, 5.4885596884e-02]),
                ),
                1.0: (
                    mpfutil.normalize(np.array(
                        [7.7900489670e-02, 3.1263985956e-01, 3.3937166369e-01, 1.8392552231e-01,
                         6.5417944499e-02, 1.7095056297e-02, 3.2661568475e-03, 3.8330712707e-04])),
                    np.array([2.1704566125e+00, 1.4767913255e+00, 9.9494464239e-01, 6.5054022453e-01,
                              4.0699848261e-01, 2.3877605265e-01, 1.2631552536e-01, 5.4058147753e-02]),
                ),
                1.025: (
                    mpfutil.normalize(np.array(
                        [7.6952700357e-02, 3.0727072639e-01, 3.3806252649e-01, 1.8739161187e-01,
                         6.8202569142e-02, 1.8172957079e-02, 3.5271339749e-03, 4.1977469774e-04])),
                    np.array([2.2246007451e+00, 1.4987431217e+00, 1.0027090445e+00, 6.5219634543e-01,
                              4.0628168801e-01, 2.3742412651e-01, 1.2510112648e-01, 5.3290608702e-02]),
                ),
                1.05: (
                    mpfutil.normalize(np.array(
                        [7.5951490248e-02, 3.0215921072e-01, 3.3682448688e-01, 1.9065840105e-01,
                         7.0904528998e-02, 1.9249523626e-02, 3.7943353746e-03, 4.5802310098e-04])),
                    np.array([2.2794637900e+00, 1.5209999838e+00, 1.0106072058e+00, 6.5389107618e-01,
                              4.0558893861e-01, 2.3610532396e-01, 1.2392825164e-01, 5.2558180376e-02]),
                ),
                1.075: (
                    mpfutil.normalize(np.array(
                        [7.5135748841e-02, 2.9749043462e-01, 3.3546536698e-01, 1.9358256131e-01,
                         7.3462853842e-02, 2.0303293013e-02, 4.0625317696e-03, 4.9720962336e-04])),
                    np.array([2.3339962749e+00, 1.5427814621e+00, 1.0181712176e+00, 6.5537529000e-01,
                              4.0477367330e-01, 2.3472627683e-01, 1.2273485458e-01, 5.1825853206e-02]),
                ),
                1.1: (
                    mpfutil.normalize(np.array(
                        [7.4498936511e-02, 2.9320210890e-01, 3.3398176183e-01, 1.9621335234e-01,
                         7.5896652559e-02, 2.1336738803e-02, 4.3328997907e-03, 5.3754926888e-04])),
                    np.array([2.3880979256e+00, 1.5640577695e+00, 1.0254444683e+00, 6.5670779125e-01,
                              4.0388057030e-01, 2.3332077650e-01, 1.2154952986e-01, 5.1101902033e-02]),
                ),
                1.135: (
                    mpfutil.normalize(np.array(
                        [7.3781546972e-02, 2.8779141761e-01, 3.3188702034e-01, 1.9942751998e-01,
                         7.9070960103e-02, 2.2738828766e-02, 4.7078885593e-03, 5.9481767571e-04])),
                    np.array([2.4635595453e+00, 1.5932581651e+00, 1.0351200522e+00, 6.5822680688e-01,
                              4.0243572178e-01, 2.3124840302e-01, 1.1982999923e-01, 5.0086639820e-02]),
                ),
                1.165: (
                    mpfutil.normalize(np.array(
                        [7.3255209231e-02, 2.8349066695e-01, 3.3001114663e-01, 2.0193081227e-01,
                         8.1698254601e-02, 2.3932622436e-02, 5.0353253230e-03, 6.4596256847e-04])),
                    np.array([2.5283061848e+00, 1.6180366599e+00, 1.0433454396e+00, 6.5959011063e-01,
                              4.0128064060e-01, 2.2954064160e-01, 1.1841431935e-01, 4.9258660768e-02]),
                ),
                1.2: (
                    mpfutil.normalize(np.array(
                        [7.3049471911e-02, 2.7922670694e-01, 3.2777409366e-01, 2.0424611111e-01,
                         8.4376160956e-02, 2.5219119825e-02, 5.4031544571e-03, 7.0518114381e-04])),
                    np.array([2.6018171474e+00, 1.6452422737e+00, 1.0517342977e+00, 6.6033371673e-01,
                              3.9941275973e-01, 2.2729102882e-01, 1.1665885708e-01, 4.8244979488e-02]),
                ),
                1.225: (
                    mpfutil.normalize(np.array(
                        [7.2924434642e-02, 2.7636899184e-01, 3.2617310111e-01, 2.0576519823e-01,
                         8.6225063920e-02, 2.6126662214e-02, 5.6678426147e-03, 7.4870541818e-04])),
                    np.array([2.6544035115e+00, 1.6645338608e+00, 1.0576864274e+00, 6.6090989163e-01,
                              3.9813579492e-01, 2.2572975028e-01, 1.1544154676e-01, 4.7552398948e-02]),
                ),
                1.25: (
                    mpfutil.normalize(np.array(
                        [7.2878372178e-02, 2.7372507995e-01, 3.2457764447e-01, 2.0711095747e-01,
                         8.7973563948e-02, 2.7011369908e-02, 5.9304555740e-03, 7.9255650193e-04])),
                    np.array([2.7066254051e+00, 1.6834170100e+00, 1.0633867607e+00, 6.6135506613e-01,
                              3.9681235914e-01, 2.2415713241e-01, 1.1422435723e-01, 4.6865499137e-02]),
                ),
                1.275: (
                    mpfutil.normalize(np.array(
                        [7.2900064724e-02, 2.7126977308e-01, 3.2298627118e-01, 2.0831410943e-01,
                         8.9630084705e-02, 2.7871360953e-02, 6.1914750802e-03, 8.3686084817e-04])),
                    np.array([2.7584803155e+00, 1.7019220657e+00, 1.0688677183e+00, 6.6169363066e-01,
                              3.9544448122e-01, 2.2257783114e-01, 1.1301698786e-01, 4.6188041940e-02]),
                ),
                1.3: (
                    mpfutil.normalize(np.array(
                        [7.2979341729e-02, 2.6899437999e-01, 3.2142049900e-01, 2.0938091404e-01,
                         9.1191523992e-02, 2.8703037565e-02, 6.4489271228e-03, 8.8137655603e-04])),
                    np.array([2.8099956910e+00, 1.7200600517e+00, 1.0741141966e+00, 6.6190223258e-01,
                              3.9401454055e-01, 2.2097674237e-01, 1.1180871339e-01, 4.5518187402e-02]),
                ),
                1.333: (
                    mpfutil.normalize(np.array(
                        [7.3183531359e-02, 2.6622654631e-01, 3.1935400320e-01, 2.1060770733e-01,
                         9.3134084452e-02, 2.9767976028e-02, 6.7853773974e-03, 9.4077392683e-04])),
                    np.array([2.8772823086e+00, 1.7433867156e+00, 1.0807222068e+00, 6.6204790231e-01,
                              3.9209039282e-01, 2.1886715969e-01, 1.1023231316e-01, 4.4652846949e-02]),
                ),
                1.367: (
                    mpfutil.normalize(np.array(
                        [7.3487650222e-02, 2.6368130331e-01, 3.1728975108e-01, 2.1164342964e-01,
                         9.4962024290e-02, 3.0809475699e-02, 7.1244411775e-03, 1.0019245866e-03])),
                    np.array([2.9458863132e+00, 1.7666767337e+00, 1.0870376420e+00, 6.6191991667e-01,
                              3.8997933697e-01, 2.1665397927e-01, 1.0860925861e-01, 4.3771069938e-02]),
                ),
                1.4: (
                    mpfutil.normalize(np.array(
                        [7.3869856966e-02, 2.6142876206e-01, 3.1531641556e-01, 2.1248342756e-01,
                         9.6611627440e-02, 3.1780193876e-02, 7.4479301542e-03, 1.0617863755e-03])),
                    np.array([3.0116383837e+00, 1.7886160797e+00, 1.0928185226e+00, 6.6164541195e-01,
                              3.8788452017e-01, 2.1450781557e-01, 1.0705495422e-01, 4.2939156538e-02]),
                ),
                1.433: (
                    mpfutil.normalize(np.array(
                        [7.4328242690e-02, 2.5938643401e-01, 3.1338642759e-01, 2.1316962669e-01,
                         9.8136224123e-02, 3.2707434813e-02, 7.7642093822e-03, 1.1214007049e-03])),
                    np.array([3.0765848120e+00, 1.8098861415e+00, 1.0982259900e+00, 6.6119448681e-01,
                              3.8572311817e-01, 2.1234935339e-01, 1.0550848966e-01, 4.2116388403e-02]),
                ),
                1.467: (
                    mpfutil.normalize(np.array(
                        [7.4872130853e-02, 2.5746831740e-01, 3.1143663917e-01, 2.1373580324e-01,
                         9.9593558843e-02, 3.3625122131e-02, 8.0851260910e-03, 1.1833022694e-03])),
                    np.array([3.1425944917e+00, 1.8311293882e+00, 1.1034494396e+00, 6.6058831914e-01,
                              3.8346504735e-01, 2.1014371612e-01, 1.0394711183e-01, 4.1296657743e-02]),
                ),
                1.5: (
                    mpfutil.normalize(np.array(
                        [7.5475995304e-02, 2.5580085499e-01, 3.0959618736e-01, 2.1414861124e-01,
                         1.0088311645e-01, 3.4466212022e-02, 8.3863846098e-03, 1.2426380301e-03])),
                    np.array([3.2056749438e+00, 1.8509983599e+00, 1.1080953316e+00, 6.5978514332e-01,
                              3.8117892784e-01, 2.0797174159e-01, 1.0242946309e-01, 4.0505571318e-02]),
                ),
                1.55: (
                    mpfutil.normalize(np.array(
                        [7.6502550682e-02, 2.5354880371e-01, 3.0687977586e-01, 2.1456974992e-01,
                         1.0265739141e-01, 3.5675680025e-02, 8.8329264259e-03, 1.3331219657e-03])),
                    np.array([3.2994227931e+00, 1.8798737903e+00, 1.1145220254e+00, 6.5832978192e-01,
                              3.7767101636e-01, 2.0472023935e-01, 1.0018979020e-01, 3.9356761547e-02]),
                ),
                1.6: (
                    mpfutil.normalize(np.array(
                        [7.7659640625e-02, 2.5160547066e-01, 3.0425539667e-01, 2.1476783169e-01,
                         1.0422121322e-01, 3.6802497423e-02, 9.2643975227e-03, 1.4235521771e-03])),
                    np.array([3.3907987775e+00, 1.9072096944e+00, 1.1201757114e+00, 6.5655601831e-01,
                              3.7408162114e-01, 2.0149226610e-01, 9.8004352643e-02, 3.8254670807e-02]),
                ),
                1.65: (
                    mpfutil.normalize(np.array(
                        [7.8931342445e-02, 2.4992357670e-01, 3.0171842461e-01, 2.1477754279e-01,
                         1.0559902455e-01, 3.7853939452e-02, 9.6820144816e-03, 1.5141349720e-03])),
                    np.array([3.4797254302e+00, 1.9330494055e+00, 1.1251070506e+00, 6.5450298569e-01,
                              3.7043681992e-01, 1.9830296249e-01, 9.5880915235e-02, 3.7202542831e-02]),
                ),
                1.7: (
                    mpfutil.normalize(np.array(
                        [8.0316450199e-02, 2.4847646967e-01, 2.9925866462e-01, 2.1461892325e-01,
                         1.0680473696e-01, 3.8833724373e-02, 1.0086021036e-02, 1.6050098864e-03])),
                    np.array([3.5659821303e+00, 1.9573463532e+00, 1.1293146082e+00, 6.5217764373e-01,
                              3.6674282182e-01, 1.9515682520e-01, 9.3821522044e-02, 3.6201613818e-02]),
                ),
                1.75: (
                    mpfutil.normalize(np.array(
                        [8.1792739839e-02, 2.4721611839e-01, 2.9687267009e-01, 2.1432474982e-01,
                         1.0786563839e-01, 3.9752220066e-02, 1.0479168661e-02, 1.6966947385e-03])),
                    np.array([3.6496319553e+00, 1.9802223376e+00, 1.1328922108e+00, 6.4964332531e-01,
                              3.6303886453e-01, 1.9207521841e-01, 9.1836795706e-02, 3.5255752554e-02]),
                ),
                1.8: (
                    mpfutil.normalize(np.array(
                        [8.3351702690e-02, 2.4611430142e-01, 2.9455220920e-01, 2.1391460573e-01,
                         1.0879847905e-01, 4.0615967685e-02, 1.0863087155e-02, 1.7896470696e-03])),
                    np.array([3.7306079604e+00, 2.0017116295e+00, 1.1358814976e+00, 6.4693054226e-01,
                              3.5934370681e-01, 1.8906845047e-01, 8.9931445695e-02, 3.4366897568e-02]),
                ),
                1.8452: (
                    mpfutil.normalize(np.array(
                        [8.4690418058e-02, 2.4499663850e-01, 2.9243076853e-01, 2.1357193298e-01,
                         1.0970349732e-01, 4.1464012154e-02, 1.1252890196e-02, 1.8898422607e-03])),
                    np.array([3.8030355049e+00, 2.0212284065e+00, 1.1390572974e+00, 6.4504670150e-01,
                              3.5652992231e-01, 1.8677753901e-01, 8.8522131675e-02, 3.3772539140e-02]),
                ),
                1.893: (
                    mpfutil.normalize(np.array(
                        [8.6268380180e-02, 2.4410876298e-01, 2.9029725056e-01, 2.1304087852e-01,
                         1.1044271963e-01, 4.2234851844e-02, 1.1620818904e-02, 1.9863373736e-03])),
                    np.array([3.8760980063e+00, 2.0397751312e+00, 1.1412332930e+00, 6.4239149877e-01,
                              3.5318176699e-01, 1.8414199214e-01, 8.6904763730e-02, 3.3059879077e-02]),
                ),
                1.942: (
                    mpfutil.normalize(np.array(
                        [8.7934133461e-02, 2.4328016593e-01, 2.8815677809e-01, 2.1243234807e-01,
                         1.1112132440e-01, 4.2991916450e-02, 1.1995116441e-02, 2.0882171592e-03])),
                    np.array([3.9485387128e+00, 2.0576847387e+00, 1.1430695257e+00, 6.3960142547e-01,
                              3.4981275648e-01, 1.8153666466e-01, 8.5330623472e-02, 3.2383811008e-02]),
                ),
                2.0: (
                    mpfutil.normalize(np.array(
                        [8.9786551909e-02, 2.4210106985e-01, 2.8558295875e-01, 2.1177213071e-01,
                         1.1202974355e-01, 4.3990700969e-02, 1.2502640792e-02, 2.2342034665e-03])),
                    np.array([4.0333353005e+00, 2.0792055401e+00, 1.1460213576e+00, 6.3713731777e-01,
                              3.4655738268e-01, 1.7902175966e-01, 8.3868948324e-02, 3.1840955642e-02]),
                ),
                2.044: (
                    mpfutil.normalize(np.array(
                        [9.1301957044e-02, 2.4139389885e-01, 2.8371463599e-01, 2.1116596260e-01,
                         1.1256844305e-01, 4.4658864885e-02, 1.2856909009e-02, 2.3393285737e-03])),
                    np.array([4.0944704801e+00, 2.0938548002e+00, 1.1473987381e+00, 6.3482553563e-01,
                              3.4386194054e-01, 1.7699616012e-01, 8.2693177817e-02, 3.1382289258e-02]),
                ),
                2.1: (
                    mpfutil.normalize(np.array(
                        [9.3065382654e-02, 2.4023047887e-01, 2.8126728442e-01, 2.1048767010e-01,
                         1.1340973474e-01, 4.5646379434e-02, 1.3388576606e-02, 2.5044931731e-03])),
                    np.array([4.1722649038e+00, 2.1134211544e+00, 1.1502793656e+00, 6.3291535695e-01,
                              3.4126243863e-01, 1.7503188302e-01, 8.1614864264e-02, 3.1060199486e-02]),
                ),
                2.15: (
                    mpfutil.normalize(np.array(
                        [9.4560665893e-02, 2.3907453536e-01, 2.7906268506e-01, 2.0991004396e-01,
                         1.1421607553e-01, 4.6592364088e-02, 1.3910229972e-02, 2.6734001343e-03])),
                    np.array([4.2407794410e+00, 2.1310612983e+00, 1.1533780371e+00, 6.3177072474e-01,
                              3.3941875419e-01, 1.7363718122e-01, 8.0895869548e-02, 3.0919426066e-02]),
                ),
                2.2: (
                    mpfutil.normalize(np.array(
                        [9.6028518114e-02, 2.3788946552e-01, 2.7686687533e-01, 2.0932325507e-01,
                         1.1501950473e-01, 4.7558407790e-02, 1.4457011914e-02, 2.8569615308e-03])),
                    np.array([4.3078088076e+00, 2.1484065686e+00, 1.1566250205e+00, 6.3089362725e-01,
                              3.3783301914e-01, 1.7244418222e-01, 8.0313038134e-02, 3.0855843128e-02]),
                ),
                2.25: (
                    mpfutil.normalize(np.array(
                        [9.7470398493e-02, 2.3668081800e-01, 2.7468209559e-01, 2.0872621991e-01,
                         1.1581640109e-01, 4.8541160065e-02, 1.5027612045e-02, 3.0552948069e-03])),
                    np.array([4.3733554241e+00, 2.1654572079e+00, 1.1600060126e+00, 6.3026269866e-01,
                              3.3648302866e-01, 1.7143359093e-01, 7.9851408625e-02, 3.0858934533e-02]),
                ),
                2.3: (
                    mpfutil.normalize(np.array(
                        [9.8889091627e-02, 2.3545475050e-01, 2.7251038588e-01, 2.0811732384e-01,
                         1.1660264907e-01, 4.9536933260e-02, 1.5620444560e-02, 3.2684212673e-03])),
                    np.array([4.4374176927e+00, 2.1821996431e+00, 1.1635002124e+00, 6.2985372602e-01,
                              3.3534616120e-01, 1.7058665943e-01, 7.9497067166e-02, 3.0919392113e-02]),
                ),
                2.35: (
                    mpfutil.normalize(np.array(
                        [1.0028533292e-01, 2.3421673847e-01, 2.7035450425e-01, 2.0749554501e-01,
                         1.1737529596e-01, 5.0542325903e-02, 1.6233944343e-02, 3.4963131519e-03])),
                    np.array([4.5000020129e+00, 2.1986350310e+00, 1.1670911245e+00, 6.2964634637e-01,
                              3.3440248537e-01, 1.6988674041e-01, 7.9237722794e-02, 3.1029137360e-02]),
                ),
                2.4: (
                    mpfutil.normalize(np.array(
                        [1.0166002192e-01, 2.3296987374e-01, 2.6821637086e-01, 2.0686191605e-01,
                         1.1813193981e-01, 5.1554435824e-02, 1.6866573154e-02, 3.7388686379e-03])),
                    np.array([4.5611299918e+00, 2.2147646425e+00, 1.1707691398e+00, 6.2962335682e-01,
                              3.3363435510e-01, 1.6931896869e-01, 7.9062329480e-02, 3.1180991276e-02]),
                ),
                2.45: (
                    mpfutil.normalize(np.array(
                        [1.0301380953e-01, 2.3171819539e-01, 2.6609819914e-01, 2.0621611457e-01,
                         1.1887039516e-01, 5.2570453009e-02, 1.7516850242e-02, 3.9959829505e-03])),
                    np.array([4.6208228082e+00, 2.2305898027e+00, 1.1745208042e+00, 6.2976766696e-01,
                              3.3302562238e-01, 1.6887020365e-01, 7.8961503011e-02, 3.1369070707e-02]),
                ),
                2.5: (
                    mpfutil.normalize(np.array(
                        [1.0434667717e-01, 2.3046410911e-01, 2.6400142175e-01, 2.0555926083e-01,
                         1.1958933776e-01, 5.3588202632e-02, 1.8183451128e-02, 4.2675396129e-03])),
                    np.array([4.6791134755e+00, 2.2461191985e+00, 1.1783397851e+00, 6.3006658874e-01,
                              3.3256311146e-01, 1.6852944693e-01, 7.8927226712e-02, 3.1588428026e-02]),
                ),
                2.55: (
                    mpfutil.normalize(np.array(
                        [1.0566067184e-01, 2.2921303327e-01, 2.6192774405e-01, 2.0488960950e-01,
                         1.2028636846e-01, 5.4604654370e-02, 1.8864649102e-02, 4.5532694101e-03])),
                    np.array([4.7360096495e+00, 2.2613349155e+00, 1.1822032180e+00, 6.3050051791e-01,
                              3.3223081812e-01, 1.6828481685e-01, 7.8951447628e-02, 3.1834441327e-02]),
                ),
                2.6: (
                    mpfutil.normalize(np.array(
                        [1.0663061559e-01, 2.2745849245e-01, 2.5964517834e-01, 2.0436860610e-01,
                         1.2130437859e-01, 5.5923644186e-02, 1.9740214167e-02, 4.9288705730e-03])),
                    np.array([4.7970061271e+00, 2.2803015813e+00, 1.1890070616e+00, 6.3307311259e-01,
                              3.3338463451e-01, 1.6903101454e-01, 7.9605696314e-02, 3.2456364347e-02]),
                ),
                2.65: (
                    mpfutil.normalize(np.array(
                        [1.0790557056e-01, 2.2621886621e-01, 2.5761869750e-01, 2.0367233833e-01,
                         1.2195285249e-01, 5.6934476185e-02, 2.0451267627e-02, 5.2459310922e-03])),
                    np.array([4.8513257908e+00, 2.2949999186e+00, 1.1929968082e+00, 6.3377421628e-01,
                              3.3329580065e-01, 1.6895976256e-01, 7.9732103854e-02, 3.2746172362e-02]),
                ),
                2.7: (
                    mpfutil.normalize(np.array(
                        [1.0916065205e-01, 2.2498629638e-01, 2.5561907907e-01, 2.0296754679e-01,
                         1.2257791626e-01, 5.7938884730e-02, 2.1173112678e-02, 5.5765120366e-03])),
                    np.array([4.9043712073e+00, 2.3094181753e+00, 1.1970194368e+00, 6.3458200588e-01,
                              3.3330793029e-01, 1.6896082545e-01, 7.9900586569e-02, 3.3053266578e-02]),
                ),
                2.75: (
                    mpfutil.normalize(np.array(
                        [1.1008444908e-01, 2.2327735208e-01, 2.5340785630e-01, 2.0238676034e-01,
                         1.2350278648e-01, 5.9239679667e-02, 2.2095060009e-02, 6.0060560484e-03])),
                    np.array([4.9617054917e+00, 2.3276741994e+00, 1.2040126981e+00, 6.3752656632e-01,
                              3.3478996844e-01, 1.6993382919e-01, 8.0683460395e-02, 3.3725853600e-02]),
                ),
                2.8: (
                    mpfutil.normalize(np.array(
                        [1.1130256368e-01, 2.2206871038e-01, 2.5146278144e-01, 2.0166019969e-01,
                         1.2407490850e-01, 6.0226911651e-02, 2.2837963326e-02, 6.3659613192e-03])),
                    np.array([5.0123574283e+00, 2.3415859718e+00, 1.2081076172e+00, 6.3853444592e-01,
                              3.3498546135e-01, 1.7006268720e-01, 8.0923919786e-02, 3.4060491447e-02]),
                ),
                2.85: (
                    mpfutil.normalize(np.array(
                        [1.1220074338e-01, 2.2040401673e-01, 2.4930663492e-01, 2.0104169873e-01,
                         1.2493053945e-01, 6.1503513038e-02, 2.3782704733e-02, 6.8301490227e-03])),
                    np.array([5.0673631713e+00, 2.3593519738e+00, 1.2151688819e+00, 6.4166361061e-01,
                              3.3663257234e-01, 1.7114745588e-01, 8.1767184064e-02, 3.4753608041e-02]),
                ),
                2.9: (
                    mpfutil.normalize(np.array(
                        [1.1308692527e-01, 2.1876701579e-01, 2.4717939847e-01, 2.0040280720e-01,
                         1.2574853032e-01, 6.2765221618e-02, 2.4737699842e-02, 7.3124014987e-03])),
                    np.array([5.1212891084e+00, 2.3768980296e+00, 1.2222642154e+00, 6.4487729962e-01,
                              3.3835302626e-01, 1.7227977187e-01, 8.2634402516e-02, 3.5453447574e-02]),
                ),
                2.95: (
                    mpfutil.normalize(np.array(
                        [1.1425537366e-01, 2.1760738089e-01, 2.4532216450e-01, 1.9964738926e-01,
                         1.2623603774e-01, 6.3714296381e-02, 2.5502860637e-02, 7.7144969257e-03])),
                    np.array([5.1685573726e+00, 2.3900620555e+00, 1.2264140867e+00, 6.4611837614e-01,
                              3.3876464178e-01, 1.7255731719e-01, 8.2956029128e-02, 3.5816189477e-02]),
                ),
                3.0: (
                    mpfutil.normalize(np.array(
                        [1.1512004551e-01, 2.1602259133e-01, 2.4325646168e-01, 1.9897828593e-01,
                         1.2698319194e-01, 6.4940948770e-02, 2.6470606715e-02, 8.2278681347e-03])),
                    np.array([5.2202744700e+00, 2.4070910785e+00, 1.2335136248e+00, 6.4945365553e-01,
                              3.4060010334e-01, 1.7376728150e-01, 8.3863632609e-02, 3.6527471879e-02]),
                ),
                3.05: (
                    mpfutil.normalize(np.array(
                        [1.1597603466e-01, 2.1446903244e-01, 2.4122283549e-01, 1.9829262197e-01,
                         1.2769190000e-01, 6.6147103833e-02, 2.7443267828e-02, 8.7572037825e-03])),
                    np.array([5.2709410709e+00, 2.4238712126e+00, 1.2406093168e+00, 6.5283996166e-01,
                              3.4248386203e-01, 1.7500877486e-01, 8.4786459362e-02, 3.7241868618e-02]),
                ),
                3.1: (
                    mpfutil.normalize(np.array(
                        [1.1682327549e-01, 2.1294678841e-01, 2.3922179756e-01, 1.9759210514e-01,
                         1.2836298994e-01, 6.7331851687e-02, 2.8419406682e-02, 9.3017850934e-03])),
                    np.array([5.3205873987e+00, 2.4404050386e+00, 1.2476961525e+00, 6.5627146080e-01,
                              3.4441115790e-01, 1.7627853183e-01, 8.5722658846e-02, 3.7958672257e-02]),
                ),
                3.15: (
                    mpfutil.normalize(np.array(
                        [1.1791931900e-01, 2.1185627919e-01, 2.3748530895e-01, 1.9681559060e-01,
                         1.2874711731e-01, 6.8221432892e-02, 2.9200460543e-02, 9.7544915254e-03])),
                    np.array([5.3639723435e+00, 2.4527440049e+00, 1.2519177111e+00, 6.5777262574e-01,
                              3.4505590302e-01, 1.7670998382e-01, 8.6123277006e-02, 3.8345208220e-02]),
                ),
                3.2: (
                    mpfutil.normalize(np.array(
                        [1.1874159838e-01, 2.1038521507e-01, 2.3554812627e-01, 1.9609760389e-01,
                         1.2935521717e-01, 6.9366597752e-02, 3.0180228075e-02, 1.0325413402e-02])),
                    np.array([5.4117238630e+00, 2.4688349151e+00, 1.2589983675e+00, 6.6129517461e-01,
                              3.4706894200e-01, 1.7803544069e-01, 8.7086292086e-02, 3.9067758155e-02]),
                ),
                3.25: (
                    mpfutil.normalize(np.array(
                        [1.1955490715e-01, 2.0894489038e-01, 2.3364407234e-01, 1.9536925413e-01,
                         1.2992860305e-01, 7.0488734974e-02, 3.1160007761e-02, 1.0909530209e-02])),
                    np.array([5.4585475724e+00, 2.4846949676e+00, 1.2660619546e+00, 6.6485043614e-01,
                              3.4911482362e-01, 1.7938190215e-01, 8.8058614860e-02, 3.9791255852e-02]),
                ),
                3.3: (
                    mpfutil.normalize(np.array(
                        [1.2035894604e-01, 2.0753504215e-01, 2.3177347256e-01, 1.9463180441e-01,
                         1.3046827304e-01, 7.1587533143e-02, 3.2138792937e-02, 1.1506135726e-02])),
                    np.array([5.5044785584e+00, 2.5003304861e+00, 1.2731057625e+00, 6.6843459236e-01,
                              3.5119081978e-01, 1.8074764735e-01, 8.9039347408e-02, 4.0515387900e-02]),
                ),
                3.35: (
                    mpfutil.normalize(np.array(
                        [1.2115419218e-01, 2.0615509329e-01, 2.2993537788e-01, 1.9388697112e-01,
                         1.3097564604e-01, 7.2662851165e-02, 3.3115439886e-02, 1.2114428439e-02])),
                    np.array([5.5495324864e+00, 2.5157417771e+00, 1.2801283448e+00, 6.7204544968e-01,
                              3.5329429773e-01, 1.8213062261e-01, 9.0027338718e-02, 4.1239870850e-02]),
                ),
                3.4: (
                    mpfutil.normalize(np.array(
                        [1.2194010012e-01, 2.0480463414e-01, 2.2813066047e-01, 1.9313573237e-01,
                         1.3145163960e-01, 7.3714337390e-02, 3.4089146264e-02, 1.2733749648e-02])),
                    np.array([5.5937486897e+00, 2.5309389611e+00, 1.2871275756e+00, 6.7567916387e-01,
                              3.5542268340e-01, 1.8352951659e-01, 9.1022085410e-02, 4.1964551740e-02]),
                ),
                3.45: (
                    mpfutil.normalize(np.array(
                        [1.2271679126e-01, 2.0348313844e-01, 2.2635874198e-01, 1.9237936436e-01,
                         1.3189749153e-01, 7.4742082356e-02, 3.5059028879e-02, 1.3363361193e-02])),
                    np.array([5.6371509832e+00, 2.5459257457e+00, 1.2941024292e+00, 6.7933376534e-01,
                              3.5757415065e-01, 1.8494301007e-01, 9.2022843779e-02, 4.2689251287e-02]),
                ),
                3.5: (
                    mpfutil.normalize(np.array(
                        [1.2348421641e-01, 2.0219007573e-01, 2.2461937267e-01, 1.9161899885e-01,
                         1.3231440362e-01, 7.5746083321e-02, 3.6024285201e-02, 1.4002564198e-02])),
                    np.array([5.6797665163e+00, 2.5607077696e+00, 1.3010519273e+00, 6.8300709141e-01,
                              3.5974680430e-01, 1.8636987221e-01, 9.3029040552e-02, 4.3413847119e-02]),
                ),
                3.55: (
                    mpfutil.normalize(np.array(
                        [1.2424269973e-01, 2.0092510736e-01, 2.2291222875e-01, 1.9085554443e-01,
                         1.3270338934e-01, 7.6726270318e-02, 3.6984124800e-02, 1.4650635282e-02])),
                    np.array([5.7216133126e+00, 2.5752862911e+00, 1.3079731089e+00, 6.8669613262e-01,
                              3.6193835992e-01, 1.8780871178e-01, 9.4040057156e-02, 4.4138218449e-02]),
                ),
                3.6: (
                    mpfutil.normalize(np.array(
                        [1.2499190637e-01, 1.9968730238e-01, 2.2123681947e-01, 1.9009007645e-01,
                         1.3306576562e-01, 7.7683187886e-02, 3.7938031883e-02, 1.5306909928e-02])),
                    np.array([5.7627256419e+00, 2.5896713573e+00, 1.3148684086e+00, 6.9040093047e-01,
                              3.6414857894e-01, 1.8925924496e-01, 9.5055564287e-02, 4.4862260965e-02]),
                ),
                3.65: (
                    mpfutil.normalize(np.array(
                        [1.2573176524e-01, 1.9847597296e-01, 2.1959281696e-01, 1.8932379628e-01,
                         1.3340263296e-01, 7.8616914330e-02, 3.8885387505e-02, 1.5970713767e-02])),
                    np.array([5.8031293506e+00, 2.6038701081e+00, 1.3217383735e+00, 6.9411996181e-01,
                              3.6637578974e-01, 1.9072050284e-01, 9.6075158409e-02, 4.5585915607e-02]),
                ),
                3.7: (
                    mpfutil.normalize(np.array(
                        [1.2646305704e-01, 1.9729132514e-01, 2.1797991077e-01, 1.8855679385e-01,
                         1.3371477378e-01, 7.9527322669e-02, 3.9825450713e-02, 1.6641366042e-02])),
                    np.array([5.8428298493e+00, 2.6178766687e+00, 1.3285757967e+00, 6.9784880404e-01,
                              3.6861737786e-01, 1.9219092415e-01, 9.7098285314e-02, 4.6309119118e-02]),
                ),
                3.75: (
                    mpfutil.normalize(np.array(
                        [1.2718511680e-01, 1.9613214067e-01, 2.1639764460e-01, 1.8779049881e-01,
                         1.3400350178e-01, 8.0415031735e-02, 4.0757848240e-02, 1.7318217370e-02])),
                    np.array([5.8818675998e+00, 2.6317064173e+00, 1.3353863421e+00, 7.0158859854e-01,
                              3.7087332294e-01, 1.9367040029e-01, 9.8124694603e-02, 4.7031789460e-02]),
                ),
                3.8: (
                    mpfutil.normalize(np.array(
                        [1.2789854715e-01, 1.9499807335e-01, 2.1484533224e-01, 1.8702524657e-01,
                         1.3426977985e-01, 8.1280206817e-02, 4.1682119953e-02, 1.8000694078e-02])),
                    np.array([5.9202509209e+00, 2.6453574729e+00, 1.3421669494e+00, 7.0533735334e-01,
                              3.7314230377e-01, 1.9515818943e-01, 9.9154193319e-02, 4.7753948042e-02]),
                ),
                3.85: (
                    mpfutil.normalize(np.array(
                        [1.2842780769e-01, 1.9361534521e-01, 2.1313202385e-01, 1.8625537170e-01,
                         1.3465832295e-01, 8.2323607041e-02, 4.2776120770e-02, 1.8811400791e-02])),
                    np.array([5.9625696784e+00, 2.6622666168e+00, 1.3514490204e+00, 7.1085520547e-01,
                              3.7660561559e-01, 1.9742654944e-01, 1.0067859995e-01, 4.8777439146e-02]),
                ),
                3.9: (
                    mpfutil.normalize(np.array(
                        [1.2912868611e-01, 1.9253705987e-01, 2.1164152228e-01, 1.8549125228e-01,
                         1.3487685924e-01, 8.3140071516e-02, 4.3680864584e-02, 1.9503684108e-02])),
                    np.array([5.9996657443e+00, 2.6755408084e+00, 1.3581472204e+00, 7.1460423716e-01,
                              3.7888871564e-01, 1.9892341102e-01, 1.0171000379e-01, 4.9496722902e-02]),
                ),
                3.95: (
                    mpfutil.normalize(np.array(
                        [1.2982086138e-01, 1.9148203685e-01, 2.1017996197e-01, 1.8473051034e-01,
                         1.3507592954e-01, 8.3934873426e-02, 4.4576152096e-02, 2.0199674414e-02])),
                    np.array([6.0361736941e+00, 2.6886533526e+00, 1.3648152211e+00, 7.1835811171e-01,
                              3.8118123021e-01, 2.0042645010e-01, 1.0274359203e-01, 5.0215392865e-02]),
                ),
                4.0: (
                    mpfutil.normalize(np.array(
                        [1.3050468325e-01, 1.9044951688e-01, 2.0874610747e-01, 1.8397333270e-01,
                         1.3525663822e-01, 8.4708868885e-02, 4.5461937689e-02, 2.0898914905e-02])),
                    np.array([6.0721045194e+00, 2.7016061284e+00, 1.3714545129e+00, 7.2211812610e-01,
                              3.8348404027e-01, 2.0193608252e-01, 1.0377947788e-01, 5.0933501664e-02]),
                ),
                4.1: (
                    mpfutil.normalize(np.array(
                        [1.3184825454e-01, 1.8845150671e-01, 2.0596125095e-01, 1.8247090737e-01,
                         1.3556538880e-01, 8.6194805579e-02, 4.7203140322e-02, 2.2304745724e-02])),
                    np.array([6.1422983042e+00, 2.7270382985e+00, 1.3846329619e+00, 7.2964380225e-01,
                              3.8811250115e-01, 2.0497049539e-01, 1.0585627643e-01, 5.2367819802e-02]),
                ),
                4.2: (
                    mpfutil.normalize(np.array(
                        [1.3301513494e-01, 1.8631141709e-01, 2.0311396247e-01, 1.8096505533e-01,
                         1.3591730858e-01, 8.7769491409e-02, 4.9066063914e-02, 2.3841566259e-02])),
                    np.array([6.2146435327e+00, 2.7550582491e+00, 1.4000653887e+00, 7.3884173113e-01,
                              3.9388979167e-01, 2.0875869812e-01, 1.0840933854e-01, 5.4090081334e-02]),
                ),
                4.3: (
                    mpfutil.normalize(np.array(
                        [1.3430431293e-01, 1.8449156691e-01, 2.0054270431e-01, 1.7949957768e-01,
                         1.3609498671e-01, 8.9091499317e-02, 5.0718536359e-02, 2.5256815789e-02])),
                    np.array([6.2806592599e+00, 2.7792508586e+00, 1.4129488972e+00, 7.4635052679e-01,
                              3.9855381588e-01, 2.1181821820e-01, 1.1049171900e-01, 5.5516568425e-02]),
                ),
                4.4: (
                    mpfutil.normalize(np.array(
                        [1.3556410874e-01, 1.8274805186e-01, 1.9806987002e-01, 1.7805825360e-01,
                         1.3622032776e-01, 9.0340786925e-02, 5.2327199805e-02, 2.6671401292e-02])),
                    np.array([6.3447965811e+00, 2.8029000648e+00, 1.4257072603e+00, 7.5385502234e-01,
                              4.0323626364e-01, 2.1489091680e-01, 1.1257796183e-01, 5.6940376121e-02]),
                ),
                4.5: (
                    mpfutil.normalize(np.array(
                        [1.3679594823e-01, 1.8107788964e-01, 1.9569083703e-01, 1.7664236084e-01,
                         1.3629837769e-01, 9.1520760271e-02, 5.3891466965e-02, 2.8082359335e-02])),
                    np.array([6.4071473021e+00, 2.8260258462e+00, 1.4383381698e+00, 7.6135081278e-01,
                              4.0793349707e-01, 2.1797470518e-01, 1.1466727484e-01, 5.8361643837e-02]),
                ),
                4.6: (
                    mpfutil.normalize(np.array(
                        [1.3800108456e-01, 1.7947759431e-01, 1.9340174659e-01, 1.7525331410e-01,
                         1.3633363569e-01, 9.2634597524e-02, 5.5411072816e-02, 2.9486954411e-02])),
                    np.array([6.4678088704e+00, 2.8486506844e+00, 1.4508423168e+00, 7.6883372946e-01,
                              4.1264180393e-01, 2.2106747670e-01, 1.1675891531e-01, 5.9780434911e-02]),
                ),
                4.7: (
                    mpfutil.normalize(np.array(
                        [1.3918127169e-01, 1.7794501050e-01, 1.9119871681e-01, 1.7389082026e-01,
                         1.3633023670e-01, 9.3685459716e-02, 5.6885876583e-02, 3.0882607748e-02])),
                    np.array([6.5268491379e+00, 2.8707816551e+00, 1.4632109827e+00, 7.7629656263e-01,
                              4.1735689916e-01, 2.2416663456e-01, 1.1885159191e-01, 6.1196311899e-02]),
                ),
                4.8: (
                    mpfutil.normalize(np.array(
                        [1.4033716723e-01, 1.7647598941e-01, 1.8907813850e-01, 1.7255669992e-01,
                         1.3629206863e-01, 9.4676513678e-02, 5.8316176201e-02, 3.2267246434e-02])),
                    np.array([6.5843648296e+00, 2.8924513040e+00, 1.4754531109e+00, 7.8373918855e-01,
                              4.2207679887e-01, 2.2727115737e-01, 1.2094518804e-01, 6.2609681569e-02]),
                ),
                4.9: (
                    mpfutil.normalize(np.array(
                        [1.4157125392e-01, 1.7522838121e-01, 1.8716966594e-01, 1.7128851232e-01,
                         1.3616253195e-01, 9.5491108829e-02, 5.9568398417e-02, 3.3520147420e-02])),
                    np.array([6.6366410301e+00, 2.9108913690e+00, 1.4854449240e+00, 7.8964888920e-01,
                              4.2577282028e-01, 2.2970121174e-01, 1.2260089345e-01, 6.3746109656e-02]),
                ),
                5.0: (
                    mpfutil.normalize(np.array(
                        [1.4267733172e-01, 1.7387113205e-01, 1.8519870416e-01, 1.7001196979e-01,
                         1.3606987408e-01, 9.6377393452e-02, 6.0915101900e-02, 3.4878492853e-02])),
                    np.array([6.6913690235e+00, 2.9317311731e+00, 1.4974620436e+00, 7.9706074560e-01,
                              4.3050678750e-01, 2.3281907675e-01, 1.2469829023e-01, 6.5156094457e-02]),
                ),
                5.1: (
                    mpfutil.normalize(np.array(
                        [1.4385476816e-01, 1.7271653749e-01, 1.8342561300e-01, 1.6880344270e-01,
                         1.3590032299e-01, 9.7102754325e-02, 6.2092023240e-02, 3.6104538095e-02])),
                    np.array([6.7410850083e+00, 2.9494624758e+00, 1.5072860987e+00, 8.0296697109e-01,
                              4.3423169553e-01, 2.3527145513e-01, 1.2636356164e-01, 6.6292616115e-02]),
                ),
                5.2: (
                    mpfutil.normalize(np.array(
                        [1.4491621392e-01, 1.7146216671e-01, 1.8159203583e-01, 1.6758451266e-01,
                         1.3576315676e-01, 9.7893935269e-02, 6.3357091479e-02, 3.7430887362e-02])),
                    np.array([6.7932718323e+00, 2.9695276563e+00, 1.5190746582e+00, 8.1033303262e-01,
                              4.3896750675e-01, 2.3839580208e-01, 1.2846199275e-01, 6.7698849263e-02]),
                ),
                5.3: (
                    mpfutil.normalize(np.array(
                        [1.4604271678e-01, 1.7039389811e-01, 1.7994226969e-01, 1.6643347488e-01,
                         1.3556178198e-01, 9.8539227951e-02, 6.4461234748e-02, 3.8625395859e-02])),
                    np.array([6.8406552922e+00, 2.9865933131e+00, 1.5287243865e+00, 8.1622345086e-01,
                              4.4271157529e-01, 2.4086452609e-01, 1.3013382272e-01, 6.8834656271e-02]),
                ),
                5.4: (
                    mpfutil.normalize(np.array(
                        [1.4722412010e-01, 1.6949366122e-01, 1.7846318982e-01, 1.6535139442e-01,
                         1.3530810191e-01, 9.9053342293e-02, 6.5414944160e-02, 3.9691246071e-02])),
                    np.array([6.8834009876e+00, 3.0007780552e+00, 1.5363056270e+00, 8.2067796810e-01,
                              4.4548686862e-01, 2.4269140525e-01, 1.3138700961e-01, 6.9704202530e-02]),
                ),
                5.5: (
                    mpfutil.normalize(np.array(
                        [1.4829637111e-01, 1.6849368091e-01, 1.7691939685e-01, 1.6425685308e-01,
                         1.3508817294e-01, 9.9635017023e-02, 6.6456106191e-02, 4.0854401893e-02])),
                    np.array([6.9287658032e+00, 3.0173188729e+00, 1.5458415305e+00, 8.2657886747e-01,
                              4.4926293579e-01, 2.4518413066e-01, 1.3307059556e-01, 7.0842731983e-02]),
                ),
                5.6: (
                    mpfutil.normalize(np.array(
                        [1.4942165771e-01, 1.6765121824e-01, 1.7553685105e-01, 1.6322992755e-01,
                         1.3482308095e-01, 1.0009495176e-01, 6.7353118274e-02, 4.1889194464e-02])),
                    np.array([6.9695675064e+00, 3.0310137040e+00, 1.5533208942e+00, 8.3104642017e-01,
                              4.5207105571e-01, 2.4703541710e-01, 1.3433575994e-01, 7.1715139327e-02]),
                ),
                5.7: (
                    mpfutil.normalize(np.array(
                        [1.5051799722e-01, 1.6683328855e-01, 1.7419762841e-01, 1.6222969356e-01,
                         1.3455525798e-01, 1.0053213477e-01, 6.8224693805e-02, 4.2909305717e-02])),
                    np.array([7.0095276785e+00, 3.0445210399e+00, 1.5607750677e+00, 8.3553303435e-01,
                              4.5490191327e-01, 2.4890256681e-01, 1.3560934858e-01, 7.2590558062e-02]),
                ),
                5.8: (
                    mpfutil.normalize(np.array(
                        [1.5158792725e-01, 1.6604005473e-01, 1.7289989309e-01, 1.6125472516e-01,
                         1.3428484264e-01, 1.0094748236e-01, 6.9071232434e-02, 4.3913842346e-02])),
                    np.array([7.0486425661e+00, 3.0578222685e+00, 1.5681891740e+00, 8.4002957177e-01,
                              4.5775019026e-01, 2.5078226758e-01, 1.3688916818e-01, 7.3467835723e-02]),
                ),
                5.9: (
                    mpfutil.normalize(np.array(
                        [1.5263279582e-01, 1.6527061144e-01, 1.7164248857e-01, 1.6030482148e-01,
                         1.3401191311e-01, 1.0134133028e-01, 6.9893311577e-02, 4.4902727718e-02])),
                    np.array([7.0869326327e+00, 3.0709231376e+00, 1.5755609077e+00, 8.4453112005e-01,
                              4.6061222379e-01, 2.5267309951e-01, 1.3817487885e-01, 7.4346947017e-02]),
                ),
                6.0: (
                    mpfutil.normalize(np.array(
                        [1.5371713898e-01, 1.6463100011e-01, 1.7052243811e-01, 1.5941948048e-01,
                         1.3371154242e-01, 1.0163821022e-01, 7.0589635597e-02, 4.5770554090e-02])),
                    np.array([7.1210215057e+00, 3.0813908883e+00, 1.5809925533e+00, 8.4766436584e-01,
                              4.6254287480e-01, 2.5394296737e-01, 1.3905432302e-01, 7.4966879418e-02]),
                ),
                6.15: (
                    mpfutil.normalize(np.array(
                        [1.5532199260e-01, 1.6375229101e-01, 1.6895141678e-01, 1.5815510695e-01,
                         1.3325313744e-01, 1.0202071497e-01, 7.1550315388e-02, 4.6995024867e-02])),
                    np.array([7.1691377015e+00, 3.0956215831e+00, 1.5881735110e+00, 8.5171833318e-01,
                              4.6500955889e-01, 2.5556294912e-01, 1.4018414564e-01, 7.5772718074e-02]),
                ),
                6.3: (
                    mpfutil.normalize(np.array(
                        [1.5686603179e-01, 1.6290521063e-01, 1.6744659939e-01, 1.5693971192e-01,
                         1.3280243138e-01, 1.0237690308e-01, 7.2472727947e-02, 4.8190383863e-02])),
                    np.array([7.2157687319e+00, 3.1095524059e+00, 1.5953244210e+00, 8.5581352855e-01,
                              4.6752173591e-01, 2.5721456866e-01, 1.4133137743e-01, 7.6585577304e-02]),
                ),
                6.45: (
                    mpfutil.normalize(np.array(
                        [1.5840765803e-01, 1.6218346059e-01, 1.6609566205e-01, 1.5581108055e-01,
                         1.3233942017e-01, 1.0264089699e-01, 7.3265856087e-02, 4.9255965537e-02])),
                    np.array([7.2576107819e+00, 3.1208221651e+00, 1.6005851953e+00, 8.5859482922e-01,
                              4.6914535235e-01, 2.5827458822e-01, 1.4208894217e-01, 7.7146971445e-02]),
                ),
                6.6: (
                    mpfutil.normalize(np.array(
                        [1.5994306187e-01, 1.6157753990e-01, 1.6488859503e-01, 1.5476619314e-01,
                         1.3186912057e-01, 1.0282192383e-01, 7.3937826800e-02, 5.0195738872e-02])),
                    np.array([7.2947660041e+00, 3.1294755712e+00, 1.6039725302e+00, 8.6006890360e-01,
                              4.6988388751e-01, 2.5874522320e-01, 1.4245864655e-01, 7.7458236566e-02]),
                ),
                6.75: (
                    mpfutil.normalize(np.array(
                        [1.6141910294e-01, 1.6098767282e-01, 1.6372898648e-01, 1.5376188229e-01,
                         1.3141305425e-01, 1.0299136793e-01, 7.4585487837e-02, 5.1112445437e-02])),
                    np.array([7.3306902820e+00, 3.1379271793e+00, 1.6073614821e+00, 8.6159121219e-01,
                              4.7066869743e-01, 2.5924786377e-01, 1.4284641961e-01, 7.7777411618e-02]),
                ),
                7.0: (
                    mpfutil.normalize(np.array(
                        [1.6386662083e-01, 1.6024541094e-01, 1.6209433699e-01, 1.5226568372e-01,
                         1.3064631376e-01, 1.0310845885e-01, 7.5410830021e-02, 5.2362344881e-02])),
                    np.array([7.3800711240e+00, 3.1460298465e+00, 1.6086521566e+00, 8.6106440904e-01,
                              4.6989867480e-01, 2.5870391810e-01, 1.4258610237e-01, 7.7726701690e-02]),
                ),
                7.25: (
                    mpfutil.normalize(np.array(
                        [1.6625045892e-01, 1.5970370924e-01, 1.6073156290e-01, 1.5094575689e-01,
                         1.2989424076e-01, 1.0309376513e-01, 7.6017051878e-02, 5.3363454282e-02])),
                    np.array([7.4195180989e+00, 3.1488186642e+00, 1.6061580697e+00, 8.5790733902e-01,
                              4.6735872259e-01, 2.5698876004e-01, 1.4155871604e-01, 7.7182431792e-02]),
                ),
                7.5: (
                    mpfutil.normalize(np.array(
                        [1.6857173662e-01, 1.5935270491e-01, 1.5962467309e-01, 1.4979276842e-01,
                         1.2916003978e-01, 1.0295932451e-01, 7.6416704059e-02, 5.4122048623e-02])),
                    np.array([7.4490163730e+00, 3.1461993534e+00, 1.5997758459e+00, 8.5204749480e-01,
                              4.6300731542e-01, 2.5408172029e-01, 1.3975555905e-01, 7.6141918540e-02]),
                ),
                7.75: (
                    mpfutil.normalize(np.array(
                        [1.7083435424e-01, 1.5918958798e-01, 1.5876424273e-01, 1.4879934469e-01,
                         1.2844440326e-01, 1.0271170289e-01, 7.6616636570e-02, 5.4639727635e-02])),
                    np.array([7.4684071981e+00, 3.1379717367e+00, 1.5893304661e+00, 8.4337165764e-01,
                              4.5678037973e-01, 2.4994977872e-01, 1.3716165853e-01, 7.4599734900e-02]),
                ),
                8.0: (
                    mpfutil.normalize(np.array(
                        [1.7304825187e-01, 1.5921892478e-01, 1.5814669452e-01, 1.4796027259e-01,
                         1.2774431184e-01, 1.0235044003e-01, 7.6617131731e-02, 5.4913972642e-02])),
                    np.array([7.4772419651e+00, 3.1237557818e+00, 1.5745497550e+00, 8.3171078484e-01,
                              4.4858426833e-01, 2.4454696379e-01, 1.3375692440e-01, 7.2548667757e-02]),
                ),
            }
        }
    }

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    def isgaussian(self):
        return True

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)] + \
            [value for value in self.parameters.values() if \
                (value.fixed and fixed) or (not value.fixed and free)]

    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        self._checkengine(engine)

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Asked for EllipticalProfile (profile={:s}, name={:s}) model for band={:s} not in "
                    "bands with fluxes {}".format(self.profile, self.name, band, fluxesbands))

        profile = {param.name: param.getvalue(transformed=False) for param in
                   self.parameters.values()}
        slope = profile["nser"] if self.profile == "sersic" else None
        if slope is None:
            raise RuntimeError("Can't get multigaussian profiles for profile {}".format(profile))
        if self.profile == "sersic" and slope <= 0.5:
            weights = [1]
            sigmas = [1]
            profiles = [{}]
        else:
            if slope in MultiGaussianApproximationProfile.weights[self.profile][self.order]:
                weights, sigmas = MultiGaussianApproximationProfile.weights[self.profile][self.order][slope]
                weights[weights < 0] = 0
            else:
                slope = np.log10(slope)
                weights = np.array([max([f(slope), 0]) for f in self.weightsplines])
                sigmas = np.array([f(slope) for f in self.sigmasplines])
            if not all([0 <= w <= 1 and (s > 0 or s > 0) for w, s in zip(weights, sigmas)]):
                raise RuntimeError('Weights {} not all >= 0 and <= 1 and/or sigmas {} not all >=0 for slope '
                                   '{:.4e}'.format(weights, sigmas, slope))
            weights = mpfutil.normalize(weights)
            profiles = [{} for _ in range(self.order)]

        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            if fluxesbands[band].isfluxratio:
                fluxratio = copy.copy(flux)
                if not 0 <= fluxratio <= 1:
                    raise ValueError("flux ratio not 0 <= {} <= 1".format(fluxratio))
                flux *= bandfluxes[band]
                # TODO: Is subtracting as above best? Should be more accurate, but mightn't guarantee flux>=0
                bandfluxes[band] *= (1.0-fluxratio)
            if not 0 < profile["axrat"] <= 1:
                if profile["axrat"] > 1:
                    profile["axrat"] = 1
                elif profile["axrat"] <= 0:
                    profile["axrat"] = 1e-15
                else:
                    raise ValueError("axrat {} ! >0 and <=1".format(profile["axrat"]))

            cens = {"cenx": cenx, "ceny": ceny}
            for key, value in cens.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = copy.copy(value)
            if engine == "galsim":
                axrat = profile["axrat"]
                axratsqrt = np.sqrt(axrat)
                gsparams = getgsparams(engineopts)
            elif engine == "libprofit":
                profile["profile"] = "sersic"
            else:
                raise ValueError("Unimplemented rendering engine {:s}".format(engine))
            profile["pointsource"] = False
            profile["resolved"] = True

            for subcomp, (weight, sigma) in enumerate(zip(weights, sigmas)):
                weightprofile = copy.copy(profile)
                re = profile["re"]*sigma
                fluxsub = weight*flux
                if not fluxsub >= 0:
                    print(np.array([f(slope) for f in self.weightsplines]))
                    print(weights)
                    print(sigmas)
                    print(weight, sigma, slope, weightprofile)
                    raise RuntimeError('wtf2 fluxsub !>=0')
                if engine == "galsim":
                    profilegs = gs.Gaussian(flux=weight*flux, fwhm=2.0*re*axratsqrt, gsparams=gsparams)
                    weightprofile.update({
                        "profile": profilegs,
                        "shear": gs.Shear(q=axrat, beta=(profile["ang"] + 90.)*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    })
                elif engine == "libprofit":
                    weightprofile["nser"] = 0.5
                    weightprofile["mag"] = mpfutil.fluxtomag(weight*flux)
                    weightprofile["re"] = re
                profiles[subcomp][band] = weightprofile

        return profiles

    @classmethod
    def _checkparameters(cls, parameters, profile, order):
        mandatory = {param: False for param in EllipticalProfile.mandatory[profile]}
        paramnamesneeded = mandatory.keys()
        paramnames = [param.name for param in parameters]
        errors = []
        if len(paramnames) > len(set(paramnames)):
            errors.append("Parameters array not unique")
        # Check if these parameters are known (in mandatory)

        if profile != "sersic" and order not in MultiGaussianApproximationProfile.weights[profile]:
            raise ValueError("{} profile={} order={} not in supported {}".format(
                cls.__name__, profile, order,
                MultiGaussianApproximationProfile.weights[profile].keys()))

        for param in parameters:
            if isinstance(param, FluxParameter):
                errors.append("Param {:s} is {:s}, not {:s}".format(param.name, type(FluxParameter),
                                                                    type(Parameter)))
            if param.name in paramnamesneeded:
                mandatory[param.name] = True
                if param.name == "nser":
                    nser = param.getvalue(transformed=False)
                    nsers = [x for x in MultiGaussianApproximationProfile.weights[profile][order]]
                    nsermin = min(nsers)
                    nsermax = max(nsers)
                    if nser < nsermin or nser > nsermax:
                        raise RuntimeError("Asked for Multigaussiansersic with n={} not {}<n<{}".format(
                            nser, nsermin, nsermax
                        ))

            elif param.name not in Component.optional:
                errors.append("Unknown param {:s}".format(param.name))

        for paramname, found in mandatory.items():
            if not found:
                errors.append("Missing mandatory param {:s}".format(paramname))
        if errors:
            errorstr = "Errors validating params of component (profile={:s}):\n" + \
                       "\n".join(errors) + "\nPassed params:" + str(parameters)
            raise ValueError(errorstr)

    def __init__(self, fluxes, name="", profile="sersic", parameters=None, order=8, weightvars=None):
        if profile not in MultiGaussianApproximationProfile.profilesavailable:
            raise ValueError("Profile type={:s} not in available: ".format(profile) + str(
                MultiGaussianApproximationProfile.profilesavailable))
        self._checkparameters(parameters, profile, order)
        self.profile = profile
        self.order = order
        Component.__init__(self, fluxes, name)
        self.parameters = {param.name: param for param in parameters}

        # Also shamelessly modified from Tractor
        # TODO: Update this to raise errors instead of asserting
        if weightvars is None:
            weightvars = MultiGaussianApproximationProfile.weights[profile][order]
        self.weightsplines = []
        self.sigmasplines = []
        for index, (weights, variances) in weightvars.items():
            assert (len(weights) == order), 'len(n={})={}'.format(index, len(weights))
            assert (len(variances) == order), 'len(n={})={}'.format(index, len(variances))
        indices = np.log10(np.array(list(weightvars.keys())))
        weightvalues = np.array(list(weightvars.values()))
        for i in range(order):
            # Weights we want to ignore are flagged by negative radii
            # you might want a spline knot at r=0 and weight=0, although there is a danger of getting r < 0
            isweight = np.array([value[1][i] >= 0 for value in weightvalues])
            weightvaluestouse = weightvalues[isweight]
            for j, (splines, ext, splineorder) in enumerate(
                    [(self.weightsplines, 'zeros', 3), (self.sigmasplines, 'const', 5)]):
                splines.append(spinterp.InterpolatedUnivariateSpline(
                    indices[isweight], [values[j][i] for values in weightvaluestouse],
                    ext=ext, k=splineorder))



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

        :param bandfluxes:
        :param engine:
        :param cenx:
        :param ceny:
        :param psf: A PSF (required, despite the default).
        :return:
        """
        self._checkengine(engine)
        if not isinstance(psf, PSF):
            raise TypeError("")

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
                 fixed=False):
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
        self.inheritors = []


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
