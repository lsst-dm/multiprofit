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
import copy
import galsim as gs
import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg
import pyprofit as pyp
import scipy.stats as spstats
import scipy.optimize as spopt
import scipy.interpolate as spinterp
import seaborn as sns
import sys
import time

# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
# Can get better performance but this isn't critical as it's just being used to check if bands are identical
def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

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


def fluxtomag(x):
    return -2.5*np.log10(x)


def magtoflux(x):
    return 10**(-0.4*x)


class Exposure:
    """
        A class to hold an image, sigma map, bad pixel mask and reference to a PSF model/image
    """
    def __init__(self, band, image, maskinverse=None, sigmainverse=None, psf=None, calcinvmask=None, meta={}):
        if psf is not None and not isinstance(psf, PSF):
            raise TypeError("Exposure (band={}) PSF type={:s} not instanceof({:s})".format(
                band, type(psf), type(PSF)))
        self.band = band
        self.image = image
        self.maskinverse = maskinverse
        self.sigmainverse = sigmainverse
        self.psf = psf
        self.calcinvmask = calcinvmask
        self.meta = meta


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
        if engine not in Model.ENGINES:
            raise ValueError("Unknown Model rendering engine {:s}".format(engine))

    def evaluate(self, params=None, data=None, bands=None, engine=None, engineopts=None,
                 paramstransformed=True, getlikelihood=True, likelihoodlog=True, keeplikelihood=False,
                 keepimages=False, keepmodels=False, plot=False, figure=None, axes=None, figurerow=None,
                 modelname="Model", modeldesc=None, drawimage=True, scale=1, clock=False, flipplot=False):
        """
            Get the likelihood and/or model images
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
            if figure is None or axes is None or figurerow is None:
                for band in bands:
                    plotslen += len(data.exposures[band])
                nrows = plotslen + (plotslen == 3)
                figure, axes = plt.subplots(nrows=nrows, ncols=5,
                                            figsize=(10, 2*nrows), dpi=100)
                if plotslen == 1:
                    axes.shape = (1, 5)
                figurerow = 0
        else:
            figaxes = None
        chis = []
        chiclips = []
        chiimgs = []
        imgclips = []
        modelclips = []
        if clock:
            times["setup"] = time.time() - timenow
            timenow = time.time()

        for band in bands:
            # TODO: Check band
            for idxexposure, exposure in enumerate(data.exposures[band]):
                image, model, timesmodel = self.getexposuremodel(
                    exposure, engine=engine, engineopts=engineopts, drawimage=drawimage, scale=scale,
                    clock=clock)
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
                        figaxes = (figure, axes[figurerow])
                    likelihoodexposure, chi, chiimg, chiclip, imgclip, modelclip = \
                        self.getexposurelikelihood(
                            exposure, image, log=likelihoodlog, figaxes=figaxes, modelname=modelname,
                            modeldesc=modeldesc, istoprow=figurerow is None or figurerow == 0,
                            isbottomrow=figurerow is None or axes is None or (figurerow+1) == axes.shape[0],
                            flipplot=flipplot
                        )
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
                        figurerow += 1
                        if plotslen == 3:
                            chiclips.append(chiclip)
                            chiimgs.append(chiimg)
                            imgclips.append(imgclip)
                            modelclips.append(modelclip)
        # Color images! whooo
        if plot:
            # TODO:
            if plotslen == 3:
                self.plotexposurescolor(imgclips, modelclips, chis, chiimgs,
                                        chiclips, (figure, axes[figurerow]))
        if clock:
            print(','.join(['{}={:.2e}'.format(name, value) for name, value in times.items()]))
        return likelihood, params, chis, times

    def plotexposurescolor(self, images, modelimages, chis, chiimgs, chiclips, figaxes):
        # TODO: verify lengths
        axes = figaxes[1]
        shapeimg = images[0].shape
        for i, imagesbytype in enumerate([images, modelimages]):
            rgb = np.zeros(shapeimg + (0,))
            for image in imagesbytype:
                rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
            axes[i].imshow(np.flip(rgb, axis=2), origin="bottom")
        rgb = np.zeros(shapeimg + (0,))
        for image in chiimgs:
            rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
        # The difference map
        axes[2].imshow(np.flip(rgb, axis=2), origin="bottom")
        # axes[2].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
        # The chi (data-model)/error map clipped at +/- 5 sigma
        rgb = np.zeros(shapeimg + (0,))
        for image in chiclips:
            rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
        axes[3].imshow(np.flip(rgb, axis=2), origin="bottom")
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
        chisqred = (np.sum(chi ** 2)/len(chi))
        axes[4].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
        for i in range(1, 5):
            axes[i].set_yticklabels([])

    def getexposurelikelihood(self, exposure, modelimage, log=True, likefunc=None,
                              figaxes=None, maximg=None, minimg=None, modelname="Model",
                              modeldesc=None, istoprow=True, isbottomrow=True, flipplot=False):
        if likefunc is None:
            likefunc = self.likefunc
        hasmask = exposure.maskinverse is not None
        if figaxes is not None:
            axes = figaxes[1]
            xlist = np.arange(0, modelimage.shape[1])
            ylist = np.arange(0, modelimage.shape[0])
            x, y = np.meshgrid(xlist, ylist)
            chi = (exposure.image - modelimage)
            if maximg is None:
                if hasmask:
                    maximg = np.max([np.max(exposure.image[exposure.maskinverse]),
                                     np.max(modelimage[exposure.maskinverse])])
                else:
                    maximg = np.max([np.max(exposure.image), np.max(modelimage)])
            if minimg is None:
                minimg = maximg/1e4
            # The original image and model image
            imgclips = []
            for i, img in enumerate([exposure.image, modelimage]):
                imgclip = (np.log10(np.clip(img, minimg, maximg)) - np.log10(minimg))/np.log10(maximg/minimg)
                imgclips.append(imgclip)
                axes[i].imshow(imgclip, cmap='gray', origin="bottom")
                if hasmask:
                    z = exposure.maskinverse
                    axes[i].contour(x, y, z)
            (axes[0].set_title if flipplot else axes[0].set_ylabel)('Band={}'.format(exposure.band))
            # Check if the modelname is informative as it's redundant otherwise
            if modelname != "Model":
                (axes[1].set_title if flipplot else axes[1].set_ylabel)(modelname)
            if modeldesc is not None:
                (axes[2].set_title if flipplot else axes[2].set_ylabel)(modeldesc)
            # The (logged) difference map
            chilog = np.log10(np.clip(np.abs(chi), minimg, np.inf)/minimg)*np.sign(chi)
            chilog /= np.log10(maximg/minimg)
            chilog = np.clip((chilog+1.)/2., 0, 1)
            axes[2].imshow(chilog, cmap='gray', origin='bottom')
            if hasmask:
                axes[2].contour(x, y, z)
            # The chi (data-model)/error map clipped at +/- 5 sigma
            chi *= exposure.sigmainverse
            chiclip = np.copy(chi)
            chiclip[chi < -5] = -5
            chiclip[chi > 5] = 5
            if hasmask:
                chi = chi[exposure.maskinverse]
                chisqred = (np.sum(chi**2) /
                            np.sum(exposure.maskinverse))
            else:
                chisqred = np.sum(chi**2) / np.prod(chi.shape)
            axes[3].imshow(chiclip, cmap='RdYlBu_r', origin="bottom")
            chiclip += 5.
            chiclip /= 10.
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
            axes[3].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
            axes[4].yaxis.tick_right()
            if flipplot:
                # TODO: What to do here?
                pass
            else:
                for i in range(1, 5):
                    if i != 4:
                        axes[i].set_yticklabels([])
                    axes[i].yaxis.set_label_position("right")
                    if not isbottomrow:
                        axes[i].set_xticklabels([])
            if istoprow:
                labels = ["Data", "Model", "Residual", "Residual/\sigma"]
                for axis, label in enumerate(labels):
                    (axes[axis].set_ylabel if flipplot else axes[axis].set_title)(label)
        else:
            chilog = None
            chiclip = None
            imgclips = [None, None]
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
            likelihood = np.sum(spstats.norm.logpdf(chi))
        else:
            raise ValueError("Unknown likelihood function {:s}".format(self.likefunc))

        if not log:
            likelihood = np.exp(likelihood)

        return likelihood, chi, chilog, chiclip, imgclips[0], imgclips[1]


    def getexposuremodel(self, exposure, engine=None, engineopts=None, drawimage=True, scale=1, clock=False):
        """
            Draw model image for one exposure with one PSF

            Returns the image and the engine-dependent model used to draw it
        """
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        Model._checkengine(engine)
        if engine == "galsim":
            gsparams = getgsparams(engineopts)
        ny, nx = exposure.image.shape
        band = exposure.band
        if clock:
            times = {}
            timenow = time.time()
        else:
            times = None
        profiles = self.getprofiles([band], engine=engine)
        if clock:
            times['getprofiles'] = time.time() - timenow
            timenow = time.time()
        if profiles:
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
                                profile["mag"] += fluxtomag(fluxfrac)
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

            # TODO: Do this in a smarter way
            if usefastgauss:
                profilesrunning = []
                image = None
                if clock:
                    times['modelsetup'] = time.time() - timenow
                    timenow = time.time()
                for profile in profiles:
                    params = profile[band]
                    profilesrunning.append(profile)
                    if len(profilesrunning) == 8:
                        paramsall = [x[band] for x in profilesrunning]
                        imgprofile = np.array(pyp.make_gaussian_mix_8_pixel(
                            params['cenx'], params['ceny'],
                            magtoflux(paramsall[0]['mag']), magtoflux(paramsall[1]['mag']),
                            magtoflux(paramsall[2]['mag']), magtoflux(paramsall[3]['mag']),
                            magtoflux(paramsall[4]['mag']), magtoflux(paramsall[5]['mag']),
                            magtoflux(paramsall[6]['mag']), magtoflux(paramsall[7]['mag']),
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
                        imgprofile = np.array(pyp.make_gaussian_pixel(
                            params['cenx'], params['ceny'], 10**(-0.4*params['mag']), params['re'],
                            params['ang'], params['axrat'], 0, nx, 0, ny, nx, ny))
                        if image is None:
                            image = imgprofile
                        else:
                            image += imgprofile
                profiles = []
                model = None

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
                    profiletype = ("all" if not convolve else
                        ("big" if profilegs.original.half_light_radius > 1 else "small"))
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
                if psfispixelated:
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
                # TODO: Determine why this is necessary. Should we keep an imageD per exposure?
                if drawimage:
                    image = np.copy(imagegs.array)

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
        '''
        :param bands: List of bands
        :param engine: Valid rendering engine
        :param engineopts: Dict of engine options
        :return: List of profiles
        '''
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
        return [-self.modeller.evaluate(x, returnlponly=True, timing=self.timing)]

    def get_bounds(self):
        return self.boundslower, self.boundsupper

    def gradient(self, x):
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
    def evaluate(self, paramsfree=None, timing=False, returnlponly=False, returnlog=True, plot=False):

        if timing:
            tinit = time.time()
        # TODO: Attempt to prevent/detect defeating this by modifying fixed/free params?
        prior = self.fitinfo["priorLogfixed"] + self.model.getpriorvalue(free=True, fixed=False)
        likelihood = self.model.getlikelihood(paramsfree, plot=plot)
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
                return -modeller.evaluate(params, timing=timing, returnlponly=True)

            tinit = time.time()
            result = spopt.minimize(neg_like_model, paramsinit, method=algo, bounds=np.array(limits),
                                    options={} if 'options' not in self.modellibopts else
                                    self.modellibopts['options'], args=(self, ))
            timerun += time.time() - tinit
            paramsbest = result.x

        elif self.modellib == "pygmo":
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
            print("Parameter names:        " + ",".join(["{:10s}".format(i) for i in paramnames]))
            print("Transformed parameters: " + ",".join(["{:.4e}".format(i) for i in paramsbest]))
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
        '''

        :param engine: Valid rendering engine
        :param bands: List of bands
        :param cenx: X coordinate
        :param ceny:
        :param time: A time for variable sources. Should actually be a duration for very long
        exposures/highly variable sources.
        :param engineopts: Dict of engine options
        :return: List of dicts by band
        '''
        # TODO: Check if this should skip entirely instead of adding a None for non-included bands
        if bands is None:
            bands = self.fluxes.keys()
        bandfluxes = {band: self.fluxes[band].getvalue(transformed=False) if
                      band in self.fluxes else None for band in bands}
        profiles = []
        for comp in self.components:
            profiles += comp.getprofiles(bandfluxes, engine, cenx, ceny, engineopts=engineopts)
        return profiles

    def __init__(self, components, fluxes=[]):
        for i, comp in enumerate(components):
            if not isinstance(comp, Component):
                raise TypeError("PhotometricModel component[{:s}](type={:s}) "
                                "is not an instance of {:s}".format(
                    i, type(comp), type(Component)))
        for i, flux in enumerate(fluxes):
            if not isinstance(flux, FluxParameter):
                raise TypeError("PhotometricModel flux[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(flux), type(FluxParameter)))
        bandscomps = [[flux.band for flux in comp.fluxes] for comp in components]
        # TODO: Check if component has a redundant mag or no specified flux ratio
        if not allequal(bandscomps):
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
        for key, value in {"x":x, "y":y}:
            if not isinstance(value, Parameter):
                raise TypeError("Position[{:s}](type={:s}) is not an instance of {:s}".format(
                    key, type(param), type(Parameter)))
        self.x = x
        self.y = y


class AstrometricModel:
    """
        The astrometric model for this source.
        TODO: Implement moving models, or at least think about how to do it
    """

    def getparameters(self, free=True, fixed=True, time=None):
        return [value for value in self.params.values() if \
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
                        i, type(param), type(Parameter))
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

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    # TODO: Should the parameters be stored as a dict? This method is the only reason why it's useful now
    def isgaussian(self):
        return (self.profile == "sersic" and self.parameters["nser"].getvalue() == 0.5) \
            or (self.profile == "moffat" and np.isinf(self.parameters["con"].getvalue()))

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)] + \
            [value for value in self.parameters.values() if \
                (value.fixed and fixed) or (not value.fixed and free)]

    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        '''

        :param bandfluxes: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param engineopts: Dict of engine options
        :return: Dict by band with list of profiles
        '''
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
                    profile["mag"] = -2.5 * np.log10(flux)
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


def normalize(array):
    array /= np.sum(array)
    return array


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

    # Shamelessly copied from Tractor
    # TODO: review these; could they be improved? Clearly large deV needs more components regardles
    weights = {
        "sersic": {
            8: {
                0.5: (
                    np.array([0, 0, 0, 0, 0, 0, 0.47, 0.53]),
                    np.array([8.5e-04, 4.9e-03, 1.8e-02,
                              5.2e-02, 1.35e-01, 3.0e-01,
                              1, 1]),
                 ),
                0.6: (
                     normalize(np.array([
                         2.35059121e-05, 4.13721322e-04, 3.92293893e-03,
                         2.85625019e-02, 1.89838613e-01, 1.20615614e+00,
                         4.74797981e+00, 3.52402557e+00])),
                     np.array([9.56466036e-04, 5.63033141e-03, 2.09789252e-02,
                               6.26359534e-02, 1.62128157e-01, 3.69124775e-01,
                               6.99199094e-01, 1.06945187e+00]),
                 ),
                0.65: (
                    normalize(np.array([
                        6.33289982e-05, 9.92144846e-04, 8.80546187e-03,
                        6.04526939e-02, 3.64161094e-01, 1.84433400e+00,
                        5.01041449e+00, 2.71713117e+00])),
                    np.array([1.02431077e-03, 6.00267283e-03, 2.24606615e-02,
                              6.75504786e-02, 1.75591563e-01, 3.99764693e-01,
                              7.73156172e-01, 1.26419221e+00]),
                ),
                0.7: (
                    normalize(np.array([
                        1.39910412e-04, 2.11974313e-03, 1.77871639e-02,
                        1.13073467e-01, 5.99838314e-01, 2.43606518e+00,
                        4.97726634e+00, 2.15764611e+00])),
                    np.array([1.07167590e-03, 6.54686686e-03, 2.48658528e-02,
                              7.49393553e-02, 1.93700754e-01, 4.38556714e-01,
                              8.61967334e-01, 1.48450726e+00]),
                ),
                0.8: (
                    normalize(np.array([
                        3.11928667e-04, 4.47378538e-03, 3.54873170e-02,
                        2.07033725e-01, 9.45282820e-01, 3.03897766e+00,
                        4.83305346e+00, 1.81226322e+00])),
                    np.array([8.90900573e-04, 5.83282884e-03, 2.33187424e-02,
                              7.33352158e-02, 1.97225551e-01, 4.68406904e-01,
                              9.93007283e-01, 1.91959493e+00]),
                ),
                0.9: (
                    normalize(np.array([
                        5.26094326e-04, 7.19992667e-03, 5.42573298e-02,
                        2.93808638e-01, 1.20034838e+00, 3.35614909e+00,
                        4.75813890e+00, 1.75240066e+00])),
                    np.array([7.14984597e-04, 4.97740520e-03, 2.08638701e-02,
                              6.84402817e-02, 1.92119676e-01, 4.80831073e-01,
                              1.09767934e+00, 2.35783460e+00]),
                ),
                # exp
                1.: (
                    normalize(np.array([
                        7.73835603e-04, 1.01672452e-02, 7.31297606e-02,
                        3.71875005e-01, 1.39727069e+00, 3.56054423e+00,
                        4.74340409e+00, 1.78731853e+00])),
                    np.array([5.72481639e-04, 4.21236311e-03, 1.84425003e-02,
                              6.29785639e-02, 1.84402973e-01, 4.85424877e-01,
                              1.18547337e+00, 2.79872887e+00]),
                ),
                1.25: (
                    normalize(np.array([
                        1.43424042e-03, 1.73362596e-02, 1.13799622e-01,
                        5.17202414e-01, 1.70456683e+00, 3.84122107e+00,
                        4.87413759e+00, 2.08569105e+00])),
                    np.array([3.26997106e-04, 2.70835745e-03, 1.30785763e-02,
                              4.90588258e-02, 1.58683880e-01, 4.68953025e-01,
                              1.32631667e+00, 3.83737061e+00]),
                ),
                1.5: (
                    normalize(np.array([
                        2.03745495e-03, 2.31813045e-02, 1.42838322e-01,
                        6.05393876e-01, 1.85993681e+00, 3.98203612e+00,
                        5.10207126e+00, 2.53254513e+00])),
                    np.array([1.88236828e-04, 1.72537665e-03, 9.09041026e-03,
                              3.71208318e-02, 1.31303364e-01, 4.29173028e-01,
                              1.37227840e+00, 4.70057547e+00]),
                ),
                1.75: (
                    normalize(np.array([
                        2.50657937e-03, 2.72749636e-02, 1.60825323e-01,
                        6.52207158e-01, 1.92821692e+00, 4.05148405e+00,
                        5.35173671e+00, 3.06654746e+00])),
                    np.array([1.09326774e-04, 1.09659966e-03, 6.25155085e-03,
                              2.75753740e-02, 1.05729535e-01, 3.77827360e-01,
                              1.34325363e+00, 5.31805274e+00]),
                ),
                # ser2
                2.: (
                    normalize(np.array([
                        2.83066070e-03, 2.98109751e-02, 1.70462302e-01,
                        6.72109095e-01, 1.94637497e+00, 4.07818245e+00,
                        5.58981857e+00, 3.64571339e+00])),
                    np.array([6.41326241e-05, 6.98618884e-04, 4.28218364e-03,
                              2.02745634e-02, 8.36658982e-02, 3.24006007e-01,
                              1.26549998e+00, 5.68924078e+00]),
                ),
                2.25: (
                    normalize(np.array([
                        3.02233733e-03, 3.10959566e-02, 1.74091827e-01,
                        6.74457937e-01, 1.93387183e+00, 4.07555480e+00,
                        5.80412767e+00, 4.24327026e+00])),
                    np.array([3.79516055e-05, 4.46695835e-04, 2.92969367e-03,
                              1.48143362e-02, 6.54274109e-02, 2.72741926e-01,
                              1.16012436e+00, 5.84499592e+00]),
                ),
                2.5: (
                    normalize(np.array([
                        3.09907888e-03, 3.13969645e-02, 1.73360850e-01,
                        6.64847427e-01, 1.90082698e+00, 4.04984377e+00,
                        5.99057823e+00, 4.84416683e+00])),
                    np.array([2.25913531e-05, 2.86414090e-04, 2.00271733e-03,
                              1.07730420e-02, 5.06946307e-02, 2.26291195e-01,
                              1.04135407e+00, 5.82166367e+00]),
                ),
                2.75: (
                    normalize(np.array([
                        3.07759263e-03, 3.09199432e-02, 1.69375193e-01,
                        6.46610533e-01, 1.85258212e+00, 4.00373109e+00,
                        6.14743945e+00, 5.44062854e+00])),
                    np.array([1.34771532e-05, 1.83790379e-04, 1.36657861e-03,
                              7.79600019e-03, 3.89487163e-02, 1.85392485e-01,
                              9.18220664e-01, 5.65190045e+00]),
                ),
                # ser3
                3.: (
                    normalize(np.array([
                        2.97478081e-03, 2.98325539e-02, 1.62926966e-01,
                        6.21897569e-01, 1.79221947e+00, 3.93826776e+00,
                        6.27309371e+00, 6.02826557e+00])),
                    np.array([8.02949133e-06, 1.17776376e-04, 9.29524545e-04,
                              5.60991573e-03, 2.96692431e-02, 1.50068210e-01,
                              7.96528251e-01, 5.36403456e+00]),
                ),
                3.25: (
                    normalize(np.array([
                        2.81333543e-03, 2.83103276e-02, 1.54743106e-01,
                        5.92538218e-01, 1.72231584e+00, 3.85446072e+00,
                        6.36549870e+00, 6.60246632e+00])),
                    np.array([4.77515101e-06, 7.53310436e-05, 6.30003331e-04,
                              4.01365507e-03, 2.24120138e-02, 1.20086835e-01,
                              6.80450508e-01, 4.98555042e+00]),
                ),
                3.5: (
                    normalize(np.array([
                        2.63493918e-03, 2.66202873e-02, 1.45833127e-01,
                        5.61055473e-01, 1.64694115e+00, 3.75564199e+00,
                        6.42306039e+00, 7.15406756e+00])),
                    np.array([2.86364388e-06, 4.83717889e-05, 4.27246310e-04,
                              2.86453738e-03, 1.68362578e-02, 9.52427526e-02,
                              5.73853421e-01, 4.54960434e+00]),
                ),
                3.75: (
                    normalize(np.array([
                        2.52556233e-03, 2.52687568e-02, 1.38061528e-01,
                        5.32259513e-01, 1.57489025e+00, 3.65196012e+00,
                        6.44759766e+00, 7.66322744e+00])),
                    np.array([1.79898320e-06, 3.19025602e-05, 2.94738112e-04,
                              2.06601434e-03, 1.27125806e-02, 7.55475779e-02,
                              4.81498066e-01, 4.10421637e+00]),
                ),
                # dev
                4.: (
                    normalize(np.array([
                        2.62202676e-03, 2.50014044e-02, 1.34130119e-01,
                        5.13259912e-01, 1.52004848e+00, 3.56204592e+00,
                        6.44844889e+00, 8.10104944e+00])),
                    np.array([1.26864655e-06, 2.25833632e-05, 2.13622743e-04,
                              1.54481548e-03, 9.85336661e-03, 6.10053309e-02,
                              4.08099539e-01, 3.70794983e+00]),
                ),
                4.25: (
                    normalize(np.array([
                        2.98703553e-03, 2.60418901e-02, 1.34745429e-01,
                        5.05981783e-01, 1.48704427e+00, 3.49526076e+00,
                        6.43784889e+00, 8.46064115e+00])),
                    np.array([1.02024747e-06, 1.74340853e-05, 1.64846771e-04,
                              1.21125378e-03, 7.91888730e-03, 5.06072396e-02,
                              3.52330049e-01, 3.38157214e+00]),
                ),
                4.5: (
                    normalize(np.array([
                        3.57010614e-03, 2.79496099e-02, 1.38169983e-01,
                        5.05879847e-01, 1.46787842e+00, 3.44443589e+00,
                        6.42125506e+00, 8.76168208e+00])),
                    np.array([8.86446183e-07, 1.42626489e-05, 1.32908651e-04,
                              9.82479942e-04, 6.53278969e-03, 4.28068927e-02,
                              3.08213788e-01, 3.10322461e+00]),
                ),
                4.75: (
                    normalize(np.array([
                        4.34147576e-03, 3.04293019e-02, 1.43230140e-01,
                        5.09832167e-01, 1.45679015e+00, 3.40356818e+00,
                        6.40074908e+00, 9.01902624e+00])),
                    np.array([8.01531774e-07, 1.20948120e-05, 1.10300128e-04,
                              8.15434233e-04, 5.48651484e-03, 3.66906220e-02,
                              2.71953278e-01, 2.85731362e+00]),
                ),
                # ser5
                5.: (
                    normalize(np.array([
                        5.30069413e-03, 3.33623146e-02, 1.49418074e-01,
                        5.16448916e-01, 1.45115226e+00, 3.36990018e+00,
                        6.37772131e+00, 9.24101590e+00])),
                    np.array([7.41574279e-07, 1.05154771e-05, 9.35192405e-05,
                              6.88777943e-04, 4.67219862e-03, 3.17741406e-02,
                              2.41556167e-01, 2.63694124e+00]),
                ),
                5.25: (
                    normalize(np.array([
                        6.45944550e-03, 3.67009077e-02, 1.56495371e-01,
                        5.25048515e-01, 1.44962975e+00, 3.34201845e+00,
                        6.35327017e+00, 9.43317911e+00])),
                    np.array([6.96302951e-07, 9.31687929e-06, 8.06697436e-05,
                              5.90325057e-04, 4.02564583e-03, 2.77601343e-02,
                              2.15789342e-01, 2.43845348e+00])
                ),
                5.5: (
                    normalize(np.array([
                        7.83422239e-03, 4.04238492e-02, 1.64329516e-01,
                        5.35236245e-01, 1.45142179e+00, 3.31906077e+00,
                        6.32826172e+00, 9.59975321e+00])),
                    np.array([6.60557943e-07, 8.38015660e-06, 7.05996176e-05,
                              5.12344075e-04, 3.50453676e-03, 2.44453624e-02,
                              1.93782688e-01, 2.25936724e+00]),
                ),
                5.75: (
                    normalize(np.array([
                        9.44354234e-03, 4.45212136e-02, 1.72835877e-01,
                        5.46749762e-01, 1.45597815e+00, 3.30040905e+00,
                        6.30333260e+00, 9.74419729e+00])),
                    np.array([6.31427920e-07, 7.63131191e-06, 6.25591461e-05,
                              4.49619447e-04, 3.07929986e-03, 2.16823076e-02,
                              1.74874928e-01, 2.09764087e+00]),
                ),
                6.: (
                    normalize(np.array([
                        0.0113067, 0.04898785, 0.18195408, 0.55939775,
                        1.46288372, 3.28556791, 6.27896305, 9.86946446])),
                    np.array([6.07125356e-07, 7.02153046e-06, 5.60375312e-05,
                              3.98494081e-04, 2.72853912e-03, 1.93601976e-02,
                              1.58544866e-01, 1.95149972e+00]),
                ),
                6.25: (
                    normalize(np.array([
                        0.01344308, 0.05382052, 0.19163668, 0.57302986,
                        1.47180585, 3.2741163, 6.25548875, 9.97808294])),
                    np.array([5.86478729e-07, 6.51723629e-06, 5.06751401e-05,
                              3.56331345e-04, 2.43639735e-03, 1.73940780e-02,
                              1.44372912e-01, 1.81933298e+00]),
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
                sigmas = np.sqrt(sigmas)
            else:
                slope = np.log10(slope)
                weights = np.array([f(slope) for f in self.weightsplines])
                sigmas = np.array([f(slope) for f in self.sigmasplines])
            profiles = [{} for _ in range(self.order)]

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
                if engine == "galsim":
                    profilegs = gs.Gaussian(flux=weight*flux, fwhm=2.0*re*axratsqrt, gsparams=gsparams)
                    weightprofile.update({
                        "profile": profilegs,
                        "shear": gs.Shear(q=axrat, beta=(profile["ang"] + 90.)*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    })
                elif engine == "libprofit":
                    weightprofile["nser"] = 0.5
                    weightprofile["mag"] = -2.5 * np.log10(weight * flux)
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
                        raise RuntimeError("Asked for Multigaussiansersic with n=plt{} not {}<n<{}".format(
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

    def __init__(self, fluxes, name="", profile="exp", parameters=None, order=8):
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
        weightvars = MultiGaussianApproximationProfile.weights[profile][order]
        self.weightsplines = []
        self.sigmasplines = []
        for index, (weights, variances) in weightvars.items():
            assert (len(weights) == order)
            assert (len(variances) == order)
        indices = [np.log10(x) for x in weightvars.keys()]
        for i in range(order):
            self.weightsplines.append(spinterp.InterpolatedUnivariateSpline(
                indices, [values[0][i] for values in weightvars.values()]))
            self.sigmasplines.append(spinterp.InterpolatedUnivariateSpline(
                indices, [np.sqrt(values[1][i]) for values in weightvars.values()]))



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
                    "both or neither must be".format(type(transform, type(reverse), type(None)))
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
