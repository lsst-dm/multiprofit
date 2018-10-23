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
import multiprofit as mpf
import numpy as np
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
                        imgprofile = np.array(mpf.make_gaussian_mix_8_pixel(
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
                        imgprofile = np.array(mpf.make_gaussian_pixel(
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
    # TODO: review these; could they be improved? Clearly large deV needs more components regardless
    weights = {
        "sersic": {
            8: {
                0.5: (
                    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([1, 0.65, -1, -1, -1, -1, -1, -1])
                ),
                0.5005: (
                    normalize(np.array([9.9871175186e-01, 1.2882481414e-03, 0, 0, 0, 0, 0, 0])),
                    np.array([1.0005231705e+00, 6.7321516832e-01, 0.32, -1, -1, -1, -1, -1]),
                ),
                0.5028: (
                    normalize(np.array([9.8270256947e-01, 1.7020106076e-02, 2.7732445383e-04, 0, 0, 0, 0, 0])),
                    np.array([1.0044281046e+00, 7.9218715234e-01, 3.6593937491e-01, 0.18, -1, -1, -1, -1]),
                ),
                0.511: (
                    normalize(np.array([9.0330097315e-01, 9.3436557426e-02, 3.1425474017e-03,
                                        1.1992201809e-04, 0, 0, 0, 0])),
                    np.array([1.0206604682e+00, 8.4567931172e-01, 4.9197887448e-01, 2.1734597888e-01, 0.1, -1, -1, -1]),
                ),
                0.531: (
                    normalize(np.array([7.0646412867e-01, 2.7247940220e-01, 1.9608916377e-02,
                                        1.3787325848e-03, 6.8820166681e-05, 0, 0, 0])),
                    np.array([1.0646030793e+00, 8.9219454342e-01, 5.8990036552e-01, 3.2393980529e-01,
                              1.4133952436e-01, 0.15, -1, -1]),
                ),
                0.565: (
                    normalize(np.array([5.2464629998e-01, 4.1300335711e-01, 5.6129846233e-02,
                                        5.6647263546e-03, 5.2597950829e-04, 2.9790811417e-05, 0, 0])),
                    np.array([1.1347495014e+00, 9.2444742235e-01, 6.3685014753e-01, 3.8272757637e-01,
                              2.0451531046e-01, 8.7609279094e-02, 0.02, -1]),
                ),
                0.6: (
                    normalize(np.array([3.9407708877e-01, 4.8296130046e-01, 1.0642939715e-01,
                                        1.4480371404e-02, 1.8449691480e-03, 1.9502433610e-04,
                                        1.1848730340e-05, 0])),
                    np.array([1.2105298619e+00, 9.6681604434e-01, 6.8229540886e-01, 4.3289218231e-01,
                              2.5376774083e-01, 1.3364413181e-01, 5.6330199940e-02, 3.75e-2]),
                ),
                0.65: (
                    normalize(np.array([2.7644879355e-01, 5.0462678830e-01, 1.7857882264e-01, 3.3883248674e-02, 5.5436240936e-03, 8.1826149710e-04, 9.4415992069e-05, 6.0452472996e-06])),
                    np.array([1.3225167915e+00, 1.0303813107e+00, 7.3537412895e-01, 4.8379705734e-01, 2.9958136394e-01, 1.7344129641e-01, 9.0074210836e-02, 3.7136066936e-02]),
                ),
                0.7: (
                    normalize(np.array([2.1913524761e-01, 4.9061322757e-01, 2.2633481392e-01, 5.2650323786e-02, 9.5887014589e-03, 1.4888283964e-03, 1.7714117052e-04, 1.1716094895e-05])),
                    np.array([1.4295033668e+00, 1.0823193970e+00, 7.6400463606e-01, 5.0215514298e-01, 3.1001601719e-01, 1.7798862821e-01, 9.1251418345e-02, 3.6960762932e-02]),
                ),
                0.75: (
                    normalize(np.array([1.8301185980e-01, 4.6875271952e-01, 2.6043113688e-01, 7.0988049797e-02, 1.4200518979e-02, 2.3135032807e-03, 2.8301925921e-04, 1.9192488236e-05])),
                    np.array([1.5361855788e+00, 1.1324232947e+00, 7.8818653776e-01, 5.1478319329e-01, 3.1582869745e-01, 1.7964804795e-01, 9.0909435872e-02, 3.6108284232e-02]),
                ),
                0.8: (
                    normalize(np.array([1.6007109305e-01, 4.4613662047e-01, 2.8329587707e-01, 8.7699161810e-02, 1.9094981011e-02, 3.2628814132e-03, 4.1078431491e-04, 2.8600864999e-05])),
                    np.array([1.6409713492e+00, 1.1794102555e+00, 8.0870675305e-01, 5.2379135872e-01, 3.1899865673e-01, 1.7976843908e-01, 8.9878043418e-02, 3.5112187379e-02]),
                ),
                0.85: (
                    normalize(np.array([1.4983125801e-01, 4.2915577983e-01, 2.9408635117e-01, 9.9212989192e-02, 2.3043392169e-02, 4.1016832156e-03, 5.3066585159e-04, 3.7880561194e-05])),
                    np.array([1.7365111034e+00, 1.2153911167e+00, 8.1863344627e-01, 5.2389206546e-01, 3.1590136290e-01, 1.7621315342e-01, 8.7066322929e-02, 3.3482800793e-02]),
                ),
                0.9: (
                    normalize(np.array([1.4281515977e-01, 4.1537607783e-01, 3.0104105297e-01, 1.0855492865e-01, 2.6602063685e-02, 4.9112416485e-03, 6.5186723676e-04, 4.7608201833e-05])),
                    np.array([1.8300744982e+00, 1.2486614142e+00, 8.2612606033e-01, 5.2192304646e-01, 3.1138823786e-01, 1.7190477112e-01, 8.3968914838e-02, 3.1817771621e-02]),
                ),
                0.95: (
                    normalize(np.array([1.3951138367e-01, 4.0427405944e-01, 3.0427487387e-01, 1.1571864827e-01, 2.9715241024e-02, 5.6762314310e-03, 7.7195959056e-04, 5.7602707981e-05])),
                    np.array([1.9182788436e+00, 1.2770276092e+00, 8.3043970443e-01, 5.1808234202e-01, 3.0589408234e-01, 1.6718495837e-01, 8.0764481990e-02, 3.0177397537e-02]),
                ),
                1.0: (
                    normalize(np.array([1.3690689179e-01, 3.9460865185e-01, 3.0650093753e-01, 1.2196285410e-01, 3.2629627568e-02, 6.4288331405e-03, 8.9418477376e-04, 6.8019252669e-05])),
                    np.array([2.0056078299e+00, 1.3045129968e+00, 8.3448218895e-01, 5.1419721377e-01, 3.0045697594e-01, 1.6257972270e-01, 7.7690601585e-02, 2.8636084563e-02]),
                ),
                1.05: (
                    normalize(np.array([1.3709630298e-01, 3.8754523486e-01, 3.0626670428e-01, 1.2606165949e-01, 3.4884816940e-02, 7.0635062893e-03, 1.0038782682e-03, 7.7896896326e-05])),
                    np.array([2.0861061998e+00, 1.3260040591e+00, 8.3448149479e-01, 5.0784100513e-01, 2.9365592565e-01, 1.5735320132e-01, 7.4429029035e-02, 2.7091198274e-02]),
                ),
                1.1: (
                    normalize(np.array([1.3743983167e-01, 3.8123589573e-01, 3.0579513664e-01, 1.2966528355e-01, 3.6984275300e-02, 7.6785668299e-03, 1.1130899456e-03, 8.7920327175e-05])),
                    np.array([2.1653593267e+00, 1.3466726893e+00, 8.3445808404e-01, 5.0176289278e-01, 2.8718612143e-01, 1.5240911033e-01, 7.1365358174e-02, 2.5658950943e-02]),
                ),
                1.15: (
                    normalize(np.array([1.3849050471e-01, 3.7594926852e-01, 3.0472696301e-01, 1.3247403415e-01, 3.8802811760e-02, 8.2417101884e-03, 1.2169664870e-03, 9.7741164545e-05])),
                    np.array([2.2415976927e+00, 1.3650395044e+00, 8.3330130881e-01, 4.9523979835e-01, 2.8062015732e-01, 1.4752312761e-01, 6.8401071005e-02, 2.4303751951e-02]),
                ),
                1.2: (
                    normalize(np.array([1.4009589733e-01, 3.7149773667e-01, 3.0324275718e-01, 1.3462373094e-01, 4.0365057596e-02, 8.7527374456e-03, 1.3148157512e-03, 1.0726709375e-04])),
                    np.array([2.3148716728e+00, 1.3812547496e+00, 8.3112936953e-01, 4.8834988301e-01, 2.7400295078e-01, 1.4271481377e-01, 6.5539781991e-02, 2.3022402726e-02]),
                ),
                1.25: (
                    normalize(np.array([1.4217205339e-01, 3.6773580762e-01, 3.0144979957e-01, 1.3621525356e-01, 4.1692498175e-02, 9.2120991012e-03, 1.4060751321e-03, 1.1641344756e-04])),
                    np.array([2.3851285147e+00, 1.3953887270e+00, 8.2801339669e-01, 4.8114303348e-01, 2.6736359893e-01, 1.3799515258e-01, 6.2781139975e-02, 2.1810745498e-02]),
                ),
                1.3: (
                    normalize(np.array([1.4462732459e-01, 3.6454268875e-01, 2.9943929388e-01, 1.3734214248e-01, 4.2810457939e-02, 9.6223001640e-03, 1.4906561065e-03, 1.2513608993e-04])),
                    np.array([2.4524352834e+00, 1.4075794134e+00, 8.2405307308e-01, 4.7368398549e-01, 2.6073844880e-01, 1.3338008277e-01, 6.0128280843e-02, 2.0666121114e-02]),
                ),
                1.35: (
                    normalize(np.array([1.4740403388e-01, 3.6180274350e-01, 2.9727137681e-01, 1.3808959080e-01, 4.3744213085e-02, 9.9861502797e-03, 1.5684906785e-03, 1.3340096798e-04])),
                    np.array([2.5167911330e+00, 1.4179433680e+00, 8.1935346823e-01, 4.6603877146e-01, 2.5415734651e-01, 1.2887880812e-01, 5.7580757063e-02, 1.9585365668e-02]),
                ),
                1.4: (
                    normalize(np.array([1.5044168718e-01, 3.5948092489e-01, 2.9500095594e-01, 1.3848748313e-01, 4.4502579391e-02, 1.0305438155e-02, 1.6397593630e-03, 1.4117196178e-04])),
                    np.array([2.5783347275e+00, 1.4265653404e+00, 8.1391996488e-01, 4.5819475132e-01, 2.4762308233e-01, 1.2449566761e-01, 5.5137669503e-02, 1.8564199068e-02]),
                ),
                1.45: (
                    normalize(np.array([1.5375733370e-01, 3.5748985738e-01, 2.9261691584e-01, 1.3859478373e-01, 4.5107587233e-02, 1.0581372171e-02, 1.7037643436e-03, 1.4838560551e-04])),
                    np.array([2.6367899264e+00, 1.4334185398e+00, 8.0780299530e-01, 4.5020328575e-01, 2.4114613075e-01, 1.2021965103e-01, 5.2787284171e-02, 1.7598291132e-02]),
                ),
                1.5: (
                    normalize(np.array([1.5725192210e-01, 3.5575521917e-01, 2.9019539578e-01, 1.3847799770e-01, 4.5582669663e-02, 1.0819992568e-02, 1.7616978372e-03, 1.5510518339e-04])),
                    np.array([2.6925321126e+00, 1.4387905675e+00, 8.0116104168e-01, 4.4213591344e-01, 2.3476512752e-01, 1.1607425174e-01, 5.0539516126e-02, 1.6686817425e-02]),
                ),
                1.55: (
                    normalize(np.array([1.6092021991e-01, 3.5424621756e-01, 2.8773880831e-01, 1.3815908731e-01, 4.5937895980e-02, 1.1023043946e-02, 1.8134282210e-03, 1.6129875519e-04])),
                    np.array([2.7455159472e+00, 1.4427005768e+00, 7.9400579555e-01, 4.3399691360e-01, 2.2847897620e-01, 1.1205233111e-01, 4.8386645760e-02, 1.5825610525e-02]),
                ),
                1.6: (
                    normalize(np.array([1.6472751862e-01, 3.5291720773e-01, 2.8526838324e-01, 1.3767644255e-01, 4.6190121772e-02, 1.1193930942e-02, 1.8594080552e-03, 1.6698709876e-04])),
                    np.array([2.7958587998e+00, 1.4452798413e+00, 7.8641787789e-01, 4.2582771818e-01, 2.2230308018e-01, 1.0815726951e-01, 4.6327242881e-02, 1.5012397064e-02]),
                ),
                1.65: (
                    normalize(np.array([1.6865761265e-01, 3.5175248371e-01, 2.8278724459e-01, 1.3704852129e-01, 4.6347963341e-02, 1.1334304402e-02, 1.8997104981e-03, 1.7215952634e-04])),
                    np.array([2.8436280951e+00, 1.4465852482e+00, 7.7841942401e-01, 4.1763567469e-01, 2.1623555424e-01, 1.0438329972e-01, 4.4355478119e-02, 1.4243771341e-02]),
                ),
                1.7: (
                    normalize(np.array([1.7268705820e-01, 3.5071087509e-01, 2.8031239368e-01, 1.3630466369e-01, 4.6425424642e-02, 1.1447851747e-02, 1.9348835115e-03, 1.7684943721e-04])),
                    np.array([2.8889186734e+00, 1.4467357236e+00, 7.7008666703e-01, 4.0945625969e-01, 2.1029081175e-01, 1.0073405930e-01, 4.2470615376e-02, 1.3517674998e-02]),
                ),
                1.75: (
                    normalize(np.array([1.7680056649e-01, 3.4977533823e-01, 2.7784827840e-01, 1.3546179167e-01, 4.6431156454e-02, 1.1536629730e-02, 1.9651724783e-03, 1.8106654858e-04])),
                    np.array([2.9318162535e+00, 1.4458055015e+00, 7.6145552404e-01, 4.0130323518e-01, 2.0447030335e-01, 9.7206071429e-02, 4.0668434024e-02, 1.2831455759e-02]),
                ),
                1.8: (
                    normalize(np.array([1.8098266031e-01, 3.4892902928e-01, 2.7540046948e-01, 1.3453568086e-01, 4.6373623562e-02, 1.1602834016e-02, 1.9908752644e-03, 1.8482722077e-04])),
                    np.array([2.9724159386e+00, 1.4438748567e+00, 7.5256568805e-01, 3.9319190759e-01, 1.9877679270e-01, 9.3796807353e-02, 3.8945432190e-02, 1.2182736396e-02]),
                ),
                1.85: (
                    normalize(np.array([1.8522258343e-01, 3.4815633705e-01, 2.7297176201e-01, 1.3353986677e-01, 4.6260386577e-02, 1.1648584439e-02, 2.0123267053e-03, 1.8815301237e-04])),
                    np.array([3.0107946886e+00, 1.4410127014e+00, 7.4345293582e-01, 3.8513612382e-01, 1.9321259670e-01, 9.0504041659e-02, 3.7298553253e-02, 1.1569375698e-02]),
                ),
                1.9: (
                    normalize(np.array([1.8950971057e-01, 3.4744481331e-01, 2.7056523366e-01, 1.3248576859e-01, 4.6097960656e-02, 1.1675670879e-02, 2.0297838637e-03, 1.9105846906e-04])),
                    np.array([3.0470361822e+00, 1.4372870329e+00, 7.3414972010e-01, 3.7714682204e-01, 1.8777815448e-01, 8.7324364690e-02, 3.5724182261e-02, 1.0989138517e-02]),
                ),
                1.95: (
                    normalize(np.array([1.9383556720e-01, 3.4678433327e-01, 2.6818228405e-01, 1.3138284445e-01, 4.5892048758e-02, 1.1685817791e-02, 2.0435373395e-03, 1.9356713656e-04])),
                    np.array([3.0812166081e+00, 1.4327583697e+00, 7.2468435389e-01, 3.6923334148e-01, 1.8247368167e-01, 8.4254868801e-02, 3.4219267629e-02, 1.0440244755e-02]),
                ),
                2.0: (
                    normalize(np.array([1.9819445790e-01, 3.4616632310e-01, 2.6582315113e-01, 1.3023848560e-01, 4.5647583353e-02, 1.1680490332e-02, 2.0538172735e-03, 1.9569131623e-04])),
                    np.array([3.1133995025e+00, 1.4274775617e+00, 7.1508006584e-01, 3.6140250217e-01, 1.7729816439e-01, 8.1291857346e-02, 3.2780402060e-02, 9.9206032077e-03]),
                ),
                2.05: (
                    normalize(np.array([1.9841639842e-01, 3.4552850257e-01, 2.6547667345e-01, 1.3049530201e-01, 4.5963295608e-02, 1.1827549903e-02, 2.0917735253e-03, 2.0050451445e-04])),
                    np.array([3.1754954144e+00, 1.4351000951e+00, 7.1133487007e-01, 3.5640066063e-01, 1.7350408555e-01, 7.8977718682e-02, 3.1615889036e-02, 9.4905311872e-03]),
                ),
                2.1: (
                    normalize(np.array([2.0268907011e-01, 3.4504543687e-01, 2.6320514184e-01, 1.2930289058e-01, 4.5661139744e-02, 1.1797867544e-02, 2.0964537453e-03, 2.0199956974e-04])),
                    np.array([3.2056650982e+00, 1.4289278568e+00, 7.0163831706e-01, 3.4877966705e-01, 1.6858385430e-01, 7.6213736661e-02, 3.0297052150e-02, 9.0229979139e-03]),
                ),
                2.15: (
                    normalize(np.array([2.0699941245e-01, 3.4458653604e-01, 2.6094800959e-01, 1.2807982902e-01, 4.5329287109e-02, 1.1755617034e-02, 2.0981598049e-03, 2.0314896346e-04])),
                    np.array([3.2338917092e+00, 1.4220733890e+00, 6.9183262619e-01, 3.4124348586e-01, 1.6378207301e-01, 7.3543652634e-02, 2.9034571257e-02, 8.5796768381e-03]),
                ),
                2.2: (
                    normalize(np.array([2.1134588764e-01, 3.4414657478e-01, 2.5870473892e-01, 1.2682968533e-01, 4.4970324671e-02, 1.1701760454e-02, 2.0970667805e-03, 2.0396142059e-04])),
                    np.array([3.2602089709e+00, 1.4145697215e+00, 6.8193175827e-01, 3.3379386746e-01, 1.5909609525e-01, 7.0963835163e-02, 2.7825649093e-02, 8.1590363116e-03]),
                ),
                2.25: (
                    normalize(np.array([2.1572257902e-01, 3.4372053532e-01, 2.5647605244e-01, 1.2555772486e-01, 4.4587805915e-02, 1.1637436303e-02, 2.0934043196e-03, 2.0446183005e-04])),
                    np.array([3.2846870547e+00, 1.4064645896e+00, 6.7195657294e-01, 3.2643731029e-01, 1.5452546426e-01, 6.8471913697e-02, 2.6668143489e-02, 7.7599078382e-03]),
                ),
                2.3: (
                    normalize(np.array([2.2012684040e-01, 3.4330418016e-01, 2.5426168977e-01, 1.2426730894e-01, 4.4184361666e-02, 1.1563606949e-02, 2.0873473113e-03, 2.0466480581e-04])),
                    np.array([3.3073691211e+00, 1.3977942685e+00, 6.6192194384e-01, 3.1917700756e-01, 1.5006851237e-01, 6.6064935023e-02, 2.5559651152e-02, 7.3810508698e-03]),
                ),
                2.35: (
                    normalize(np.array([2.2455552189e-01, 3.4289382308e-01, 2.5206161623e-01, 1.2296186334e-01, 4.3762390024e-02, 1.1481121651e-02, 2.0790762328e-03, 2.0458755482e-04])),
                    np.array([3.3283069651e+00, 1.3885952088e+00, 6.5184304309e-01, 3.1201600916e-01, 1.4572339363e-01, 6.3740092995e-02, 2.4498004043e-02, 7.0213427774e-03]),
                ),
                2.4: (
                    normalize(np.array([2.2900586985e-01, 3.4248595494e-01, 2.4987575278e-01, 1.2164437015e-01, 4.3324236931e-02, 1.1390822691e-02, 2.0687496422e-03, 2.0424300720e-04])),
                    np.array([3.3475475595e+00, 1.3789025838e+00, 6.4173411573e-01, 3.0495739399e-01, 1.4148850404e-01, 6.1494669880e-02, 2.3481025095e-02, 6.6796663859e-03]),
                ),
                2.45: (
                    normalize(np.array([2.3347496578e-01, 3.4207855055e-01, 2.4770412764e-01, 1.2031716963e-01, 4.2871629393e-02, 1.1293387754e-02, 2.0565218844e-03, 2.0364736830e-04])),
                    np.array([3.3651455515e+00, 1.3687496985e+00, 6.3160691384e-01, 2.9800243567e-01, 1.3736151103e-01, 5.9325874523e-02, 2.2506744581e-02, 6.3550516071e-03]),
                ),
                2.5: (
                    normalize(np.array([2.3796102595e-01, 3.4166843601e-01, 2.4554623701e-01, 1.1898275274e-01, 4.2406617244e-02, 1.1189568399e-02, 2.0425465014e-03, 2.0281614100e-04])),
                    np.array([3.3811415596e+00, 1.3581670278e+00, 6.2147412524e-01, 2.9115387618e-01, 1.3334087895e-01, 5.7231305378e-02, 2.1573263420e-02, 6.0465663102e-03]),
                ),
                2.55: (
                    normalize(np.array([2.4246162101e-01, 3.4125370515e-01, 2.4340200229e-01, 1.1764313836e-01, 4.1930824374e-02, 1.1079993358e-02, 2.0269530637e-03, 2.0176239935e-04])),
                    np.array([3.3955834126e+00, 1.3471854599e+00, 6.1134672427e-01, 2.8441312355e-01, 1.2942456929e-01, 5.5208377732e-02, 2.0678734587e-02, 5.7533138198e-03]),
                ),
                2.6: (
                    normalize(np.array([2.4697537071e-01, 3.4083271342e-01, 2.4127118800e-01, 1.1629974565e-01, 4.1445439657e-02, 1.0965165935e-02, 2.0098765696e-03, 2.0050006168e-04])),
                    np.array([3.4085183023e+00, 1.3358308262e+00, 6.0123337322e-01, 2.7778036797e-01, 1.2561005526e-01, 5.3254586917e-02, 1.9821446037e-02, 5.4744904933e-03]),
                ),
                2.65: (
                    normalize(np.array([2.5150125502e-01, 3.4040354206e-01, 2.3915297336e-01, 1.1495416750e-01, 4.0951947441e-02, 1.0845652581e-02, 1.9914213914e-03, 1.9904063796e-04])),
                    np.array([3.4199785411e+00, 1.3241283309e+00, 5.9114319803e-01, 2.7125683019e-01, 1.2189545594e-01, 5.1367508504e-02, 1.8999679545e-02, 5.2092882024e-03]),
                ),
                2.7: (
                    normalize(np.array([2.5603635787e-01, 3.3996473964e-01, 2.3704770402e-01, 1.1360829009e-01, 4.0451772215e-02, 1.0722014901e-02, 1.9717219976e-03, 1.9739926358e-04])),
                    np.array([3.4300215256e+00, 1.3121081912e+00, 5.8108687115e-01, 2.6484406904e-01, 1.1827914408e-01, 4.9545087625e-02, 1.8211968285e-02, 4.9570196098e-03]),
                ),
                2.75: (
                    normalize(np.array([2.6058055622e-01, 3.3951550822e-01, 2.3495502141e-01, 1.1226246312e-01, 3.9945453604e-02, 1.0594555244e-02, 1.9508583723e-03, 1.9558381619e-04])),
                    np.array([3.4386754546e+00, 1.2997893723e+00, 5.7106903224e-01, 2.5854037843e-01, 1.1475797813e-01, 4.7784746611e-02, 1.7456677796e-02, 4.7169510444e-03]),
                ),
                2.8: (
                    normalize(np.array([2.6513259999e-01, 3.3905467563e-01, 2.3287400379e-01, 1.1091802298e-01, 3.9434331333e-02, 1.0463808431e-02, 1.9289483895e-03, 1.9360945295e-04])),
                    np.array([3.4459842336e+00, 1.2871949902e+00, 5.6109772902e-01, 2.5234703294e-01, 1.1133059246e-01, 4.6084604163e-02, 1.6732497591e-02, 4.4884902517e-03]),
                ),
                2.85: (
                    normalize(np.array([2.6969046164e-01, 3.3858074462e-01, 2.3080519717e-01, 1.0957651282e-01, 3.8919361991e-02, 1.0330146194e-02, 1.9060890486e-03, 1.9148651317e-04])),
                    np.array([3.4519932590e+00, 1.2743508483e+00, 5.5118171368e-01, 2.4626456036e-01, 1.0799489041e-01, 4.4442595960e-02, 1.6038068356e-02, 4.2710211735e-03]),
                ),
                2.9: (
                    normalize(np.array([2.7425455973e-01, 3.3809311678e-01, 2.2874775179e-01, 1.0823794027e-01, 3.8401141195e-02, 1.0193901728e-02, 1.8823625847e-03, 1.8922592838e-04])),
                    np.array([3.4567293994e+00, 1.2612722805e+00, 5.4132427926e-01, 2.4029157350e-01, 1.0474854495e-01, 4.2856681852e-02, 1.5372096187e-02, 4.0639904491e-03]),
                ),
                2.95: (
                    normalize(np.array([2.7882402022e-01, 3.3759113938e-01, 2.2670098911e-01, 1.0690329594e-01, 3.7880462638e-02, 1.0055397780e-02, 1.8578536876e-03, 1.8684125728e-04])),
                    np.array([3.4602294597e+00, 1.2479783179e+00, 5.3153132819e-01, 2.3442807269e-01, 1.0158953793e-01, 4.1324927816e-02, 1.4733392817e-02, 3.8668974085e-03]),
                ),
                3.0: (
                    normalize(np.array([2.8339586721e-01, 3.3707382683e-01, 2.2466590244e-01, 1.0557395714e-01, 3.7358355307e-02, 9.9150750079e-03, 1.8326728491e-03, 1.8434322082e-04])),
                    np.array([3.4625470418e+00, 1.2344960814e+00, 5.2181161501e-01, 2.2867507595e-01, 9.8516567191e-02, 3.9845824782e-02, 1.4120896453e-02, 3.6792477108e-03]),
                ),
                3.05: (
                    normalize(np.array([2.8797063610e-01, 3.3654056068e-01, 2.2264153368e-01, 1.0425014388e-01, 3.6835314290e-02, 9.7731843407e-03, 1.8068849854e-03, 1.8174203946e-04])),
                    np.array([3.4637074366e+00, 1.2208387762e+00, 5.1216816120e-01, 2.2303139862e-01, 9.5527424557e-02, 3.8417492098e-02, 1.3533470346e-02, 3.5005785419e-03]),
                ),
                3.1: (
                    normalize(np.array([2.9254661174e-01, 3.3599065900e-01, 2.2062808534e-01, 1.0293280440e-01, 3.6312121142e-02, 9.6300781164e-03, 1.7805879374e-03, 1.7905231576e-04])),
                    np.array([3.4637534498e+00, 1.2070273071e+00, 5.0260725098e-01, 2.1749729115e-01, 9.2620612345e-02, 3.7038463129e-02, 1.2970191150e-02, 3.3304985702e-03]),
                ),
                3.15: (
                    normalize(np.array([2.9712233820e-01, 3.3542361376e-01, 2.1862559335e-01, 1.0162270438e-01, 3.5789497541e-02, 9.4860955330e-03, 1.7538706286e-03, 1.7628660147e-04])),
                    np.array([3.4627258999e+00, 1.1930808223e+00, 4.9313434752e-01, 2.1207278131e-01, 8.9794646307e-02, 3.5707295904e-02, 1.2430139604e-02, 3.1686214374e-03]),
                ),
                3.2: (
                    normalize(np.array([3.0169688519e-01, 3.3483854653e-01, 2.1663410979e-01, 1.0032063193e-01, 3.5268055238e-02, 9.3414963734e-03, 1.7268165388e-03, 1.7345841133e-04])),
                    np.array([3.4606597606e+00, 1.1790168374e+00, 4.8375446703e-01, 2.0675761340e-01, 8.7047836659e-02, 3.4422521140e-02, 1.1912453076e-02, 3.0145954913e-03]),
                ),
                3.25: (
                    normalize(np.array([3.0626809208e-01, 3.3423508706e-01, 2.1465388918e-01, 9.9027498301e-02, 3.4748633509e-02, 9.1966873502e-03, 1.6995290573e-03, 1.7058345629e-04])),
                    np.array([3.4576014755e+00, 1.1648556893e+00, 4.7447369015e-01, 2.0155241315e-01, 8.4379213809e-02, 3.3183020985e-02, 1.1416381727e-02, 2.8681095080e-03]),
                ),
                3.3: (
                    normalize(np.array([3.1083369328e-01, 3.3361255518e-01, 2.1268550711e-01, 9.7744401689e-02, 3.4232034639e-02, 9.0520192779e-03, 1.6721108692e-03, 1.6767796469e-04])),
                    np.array([3.4535970392e+00, 1.1506182300e+00, 4.6529842354e-01, 1.9645779603e-01, 8.1787703962e-02, 3.1987680634e-02, 1.0941226986e-02, 2.7288760233e-03]),
                ),
                3.35: (
                    normalize(np.array([3.1539152937e-01, 3.3297050294e-01, 2.1072934020e-01, 9.6472317744e-02, 3.3719031752e-02, 8.9078501192e-03, 1.6446670823e-03, 1.6476078473e-04])),
                    np.array([3.4486916592e+00, 1.1363238854e+00, 4.5623443735e-01, 1.9147427636e-01, 7.9272312479e-02, 3.0835471314e-02, 1.0486356649e-02, 2.5966476063e-03]),
                ),
                3.4: (
                    normalize(np.array([3.1993790035e-01, 3.3230824011e-01, 2.0878655222e-01, 9.5212806365e-02, 3.3210685259e-02, 8.7646371599e-03, 1.6173264381e-03, 1.6185210731e-04])),
                    np.array([3.4429426970e+00, 1.1219974658e+00, 4.4729000854e-01, 1.8660354880e-01, 7.6832626561e-02, 2.9725640501e-02, 1.0051240666e-02, 2.4712002682e-03]),
                ),
                3.45: (
                    normalize(np.array([3.2446913177e-01, 3.3162520838e-01, 2.0685820980e-01, 9.3967380474e-02, 3.2708034927e-02, 8.6228375957e-03, 1.5902218076e-03, 1.5897524344e-04])),
                    np.array([3.4364072342e+00, 1.1076627941e+00, 4.3847298558e-01, 1.8184722461e-01, 7.4468267882e-02, 2.8657498875e-02, 9.6354114276e-03, 2.3523476800e-03]),
                ),
                3.5: (
                    normalize(np.array([3.2897898167e-01, 3.3092065750e-01, 2.0494647255e-01, 9.2738506027e-02, 3.2212614535e-02, 8.4830824226e-03, 1.5635269541e-03, 1.5615834308e-04])),
                    np.array([3.4291635398e+00, 1.0933524055e+00, 4.2979524057e-01, 1.7720884649e-01, 7.2179782437e-02, 2.7630769251e-02, 9.2385605793e-03, 2.2399485348e-03]),
                ),
                3.55: (
                    normalize(np.array([3.3346455664e-01, 3.3019466828e-01, 2.0305212012e-01, 9.1527079847e-02, 3.1725084518e-02, 8.3457088335e-03, 1.5373565788e-03, 1.5342518552e-04])),
                    np.array([3.4212616720e+00, 1.0790849831e+00, 4.2126159976e-01, 1.7268853712e-01, 6.9966194963e-02, 2.6644595947e-02, 8.8602087189e-03, 2.1338343965e-03]),
                ),
                3.6: (
                    normalize(np.array([3.3791916511e-01, 3.2944621157e-01, 2.0117732665e-01, 9.0335880830e-02, 3.1247229483e-02, 8.2114550705e-03, 1.5119182160e-03, 1.5081308021e-04])),
                    np.array([3.4127827944e+00, 1.0648933420e+00, 4.1288477372e-01, 1.6829063272e-01, 6.7828585347e-02, 2.5699027864e-02, 8.5002230849e-03, 2.0339426497e-03]),
                ),
                3.65: (
                    normalize(np.array([3.4233201243e-01, 3.2867494041e-01, 1.9932622408e-01, 8.9168683477e-02, 3.0781168129e-02, 8.0811677358e-03, 1.4874443643e-03, 1.4835937968e-04])),
                    np.array([3.4038426761e+00, 1.0508231816e+00, 4.0468172283e-01, 1.6402086922e-01, 6.5768548938e-02, 2.4794288628e-02, 8.1585174926e-03, 1.9402023601e-03]),
                ),
                3.7: (
                    normalize(np.array([3.4669490956e-01, 3.2788067414e-01, 1.9750165380e-01, 8.8028330369e-02, 3.0328612134e-02, 7.9555671697e-03, 1.4641483893e-03, 1.4610443941e-04])),
                    np.array([3.3945350753e+00, 1.0369093815e+00, 3.9666490684e-01, 1.5988316528e-01, 6.3786966962e-02, 2.3930384487e-02, 7.8349720517e-03, 1.8525606413e-03]),
                ),
                3.75: (
                    normalize(np.array([3.5099770418e-01, 3.2706323245e-01, 1.9570729508e-01, 8.6918317000e-02, 2.9891597014e-02, 7.8354892928e-03, 1.4422719503e-03, 1.4409303061e-04])),
                    np.array([3.3849696881e+00, 1.0231928001e+00, 3.8884931451e-01, 1.5588257316e-01, 6.1885234812e-02, 2.3107536326e-02, 7.5295462986e-03, 1.7709835829e-03]),
                ),
                3.8: (
                    normalize(np.array([3.5522825591e-01, 3.2622265288e-01, 1.9394757035e-01, 8.5842740683e-02, 2.9472448599e-02, 7.7218756822e-03, 1.4220818016e-03, 1.4237409853e-04])),
                    np.array([3.3752733799e+00, 1.0097197129e+00, 3.8125211412e-01, 1.5202509730e-01, 6.0065170871e-02, 2.2326137079e-02, 7.2422591887e-03, 1.6954485777e-03]),
                ),
                3.85: (
                    normalize(np.array([3.5937430835e-01, 3.2535928773e-01, 1.9222699088e-01, 8.4805551036e-02, 2.9073391524e-02, 7.6156299085e-03, 1.4038423123e-03, 1.4099826224e-04])),
                    np.array([3.3655733854e+00, 9.9653518196e-01, 3.7388948928e-01, 1.4831617823e-01, 5.8328320983e-02, 2.1586478077e-02, 6.9731005594e-03, 1.6259233162e-03]),
                ),
                3.9: (
                    normalize(np.array([3.6342255989e-01, 3.2447381326e-01, 1.9055045360e-01, 8.3810924591e-02, 2.8696727620e-02, 7.5176794132e-03, 1.3878235118e-03, 1.4001811405e-04])),
                    np.array([3.3560062995e+00, 9.8368622030e-01, 3.6677812385e-01, 1.4476137118e-01, 5.6676245235e-02, 2.0888838472e-02, 6.7220467970e-03, 1.5623622847e-03]),
                ),
                3.95: (
                    normalize(np.array([3.6736029485e-01, 3.2356716781e-01, 1.8892258858e-01, 8.2862731665e-02, 2.8344577300e-02, 7.4288794429e-03, 1.3742759798e-03, 1.3948435813e-04])),
                    np.array([3.3467036633e+00, 9.7121640498e-01, 3.5993291645e-01, 1.4136533674e-01, 5.5110058718e-02, 2.0233281490e-02, 6.4889755692e-03, 1.5046793325e-03]),
                ),
                4.0: (
                    normalize(np.array([3.7075114139e-01, 3.2280409587e-01, 1.8749664618e-01, 8.2039906106e-02, 2.8046425671e-02, 7.3574230960e-03, 1.3647898365e-03, 1.3957184585e-04])),
                    np.array([3.3427245912e+00, 9.6027387847e-01, 3.5369147053e-01, 1.3824007297e-01, 5.3667520574e-02, 1.9631772487e-02, 6.2770878073e-03, 1.4533998846e-03]),
                ),
                4.05: (
                    normalize(np.array([3.7445656938e-01, 3.2185515841e-01, 1.8596950419e-01, 8.1188165971e-02, 2.7745794188e-02, 7.2881537358e-03, 1.3566151457e-03, 1.4003898138e-04])),
                    np.array([3.3340911908e+00, 9.4860150163e-01, 3.4738628815e-01, 1.3515952073e-01, 5.2269461596e-02, 1.9057612065e-02, 6.0784498327e-03, 1.4068235113e-03]),
                ),
                4.1: (
                    normalize(np.array([3.7802592729e-01, 3.2089117424e-01, 1.8450011362e-01, 8.0388655955e-02, 2.7472586278e-02, 7.2292097384e-03, 1.3512866907e-03, 1.4104619596e-04])),
                    np.array([3.3260165390e+00, 9.3738701203e-01, 3.4136515179e-01, 1.3223822233e-01, 5.0954065117e-02, 1.8522671022e-02, 5.8961119905e-03, 1.3653822077e-03]),
                ),
                4.15: (
                    normalize(np.array([3.8146107852e-01, 3.1991426850e-01, 1.8308742205e-01, 7.9639818025e-02, 2.7225849910e-02, 7.1802485153e-03, 1.3487238185e-03, 1.4259066387e-04])),
                    np.array([3.3185057170e+00, 9.2662087928e-01, 3.3561848988e-01, 1.2946923058e-01, 4.9716990139e-02, 1.8024537543e-02, 5.7289047034e-03, 1.3286242672e-03]),
                ),
                4.2: (
                    normalize(np.array([3.8476547025e-01, 3.1892754804e-01, 1.8172989388e-01, 7.8939062572e-02, 2.7003975112e-02, 7.1406270139e-03, 1.3487690640e-03, 1.4465407426e-04])),
                    np.array([3.3115559160e+00, 9.1628649545e-01, 3.3013247502e-01, 1.2684337291e-01, 4.8552819987e-02, 1.7560337379e-02, 5.5754788497e-03, 1.2960401204e-03]),
                ),
                4.25: (
                    normalize(np.array([3.8794716721e-01, 3.1793327020e-01, 1.8042410506e-01, 7.8282639651e-02, 2.6804858949e-02, 7.1095416425e-03, 1.3512083625e-03, 1.4720892203e-04])),
                    np.array([3.3051189305e+00, 9.0635360468e-01, 3.2488857731e-01, 1.2434984074e-01, 4.7455586727e-02, 1.7126987973e-02, 5.4344109230e-03, 1.2671098940e-03]),
                ),
                4.3: (
                    normalize(np.array([3.9101493188e-01, 3.1693396059e-01, 1.7916636663e-01, 7.7666442608e-02, 2.6626172110e-02, 7.0860969423e-03, 1.3558065184e-03, 1.5022271389e-04])),
                    np.array([3.2991418353e+00, 8.9678928304e-01, 3.1986700304e-01, 1.2197733459e-01, 4.6419184463e-02, 1.6721413433e-02, 5.3043092772e-03, 1.2413351321e-03]),
                ),
                4.35: (
                    normalize(np.array([3.9397806244e-01, 3.1593052636e-01, 1.7795337071e-01, 7.7086986132e-02, 2.6465687627e-02, 7.0693711477e-03, 1.3623322775e-03, 1.5366331616e-04])),
                    np.array([3.2935633710e+00, 8.8756212976e-01, 3.1504997640e-01, 1.1971538462e-01, 4.5437768511e-02, 1.6340699753e-02, 5.1838956773e-03, 1.2182668432e-03]),
                ),
                4.4: (
                    normalize(np.array([3.9684417484e-01, 3.1492718945e-01, 1.7678097581e-01, 7.6539687754e-02, 2.6321269336e-02, 7.0586298714e-03, 1.3705725737e-03, 1.5750036370e-04])),
                    np.array([3.2883414951e+00, 8.7863995573e-01, 3.1041754727e-01, 1.1755362406e-01, 4.4506241607e-02, 1.5982328795e-02, 5.0720241243e-03, 1.1975106587e-03]),
                ),
                4.45: (
                    normalize(np.array([3.9962258474e-01, 3.1392307184e-01, 1.7564621739e-01, 7.6021878220e-02, 2.6191091680e-02, 7.0531094489e-03, 1.3803388523e-03, 1.6170782157e-04])),
                    np.array([3.2834096464e+00, 8.6999488367e-01, 3.0595543938e-01, 1.1548362187e-01, 4.3619867362e-02, 1.5643984387e-02, 4.9677058892e-03, 1.1787324253e-03]),
                ),
                4.5: (
                    normalize(np.array([4.0231986224e-01, 3.1292007635e-01, 1.7454611713e-01, 7.5530467351e-02, 2.6073542188e-02, 7.0522035487e-03, 1.3914677993e-03, 1.6626338847e-04])),
                    np.array([3.2787295207e+00, 8.6160274983e-01, 3.0164928471e-01, 1.1349756469e-01, 4.2774618904e-02, 1.5323703481e-02, 4.8700875728e-03, 1.1616500244e-03]),
                ),
                4.55: (
                    normalize(np.array([4.0494230072e-01, 3.1191903425e-01, 1.7347809896e-01, 7.5062962127e-02, 2.5967248837e-02, 7.0553846121e-03, 1.4038216149e-03, 1.7114887897e-04])),
                    np.array([3.2742624213e+00, 8.5344238085e-01, 2.9748701573e-01, 1.1158876484e-01, 4.1966984017e-02, 1.5019786520e-02, 4.7784437807e-03, 1.1460284653e-03]),
                ),
                4.6: (
                    normalize(np.array([4.0749525128e-01, 3.1092079594e-01, 1.7243991258e-01, 7.4617197764e-02, 2.5871023015e-02, 7.0621892530e-03, 1.4172810200e-03, 1.7634915273e-04])),
                    np.array([3.2699771215e+00, 8.4549564095e-01, 2.9345804628e-01, 1.0975133191e-01, 4.1193866674e-02, 1.4730745770e-02, 4.6921498815e-03, 1.1316706830e-03]),
                ),
                4.65: (
                    normalize(np.array([4.0998295531e-01, 3.0992653818e-01, 1.7142966400e-01, 7.4191130115e-02, 2.5783825752e-02, 7.0722810542e-03, 1.4317523853e-03, 1.8185319614e-04])),
                    np.array([3.2658522987e+00, 8.3774743388e-01, 2.8955294505e-01, 1.0798007588e-01, 4.0452685467e-02, 1.4455362928e-02, 4.6106856334e-03, 1.1184153297e-03]),
                ),
                4.7: (
                    normalize(np.array([4.1241012183e-01, 3.0893607548e-01, 1.7044551948e-01, 7.3783353506e-02, 2.5704839289e-02, 7.0852931781e-03, 1.4471464497e-03, 1.8765079478e-04])),
                    np.array([3.2618589504e+00, 8.3018361341e-01, 2.8576403875e-01, 1.0627064420e-01, 3.9741037071e-02, 1.4192478295e-02, 4.5335841891e-03, 1.1061229455e-03]),
                ),
                4.75: (
                    normalize(np.array([4.1477981057e-01, 3.0795020055e-01, 1.6948614381e-01, 7.3392448934e-02, 2.5633303002e-02, 7.1009608176e-03, 1.4633970804e-03, 1.9373523881e-04])),
                    np.array([3.2579843513e+00, 8.2279361504e-01, 2.8208437842e-01, 1.0461906769e-01, 3.9056890505e-02, 1.3941148500e-02, 4.4604607745e-03, 1.0946797670e-03]),
                ),
                4.8: (
                    normalize(np.array([4.1709520245e-01, 3.0696926415e-01, 1.6855014548e-01, 7.3017180477e-02, 2.5568605690e-02, 7.1190565858e-03, 1.4804440984e-03, 2.0010106694e-04])),
                    np.array([3.2542128306e+00, 8.1556723415e-01, 2.7850774815e-01, 1.0302189417e-01, 3.8398464268e-02, 1.3700529797e-02, 4.3909736699e-03, 1.0839886711e-03]),
                ),
                4.85: (
                    normalize(np.array([4.1935897995e-01, 3.0599367950e-01, 1.6763632211e-01, 7.2656484050e-02, 2.5510178505e-02, 7.1393748794e-03, 1.4982367371e-03, 2.0674426867e-04])),
                    np.array([3.2505330655e+00, 8.0849572955e-01, 2.7502860840e-01, 1.0147597883e-01, 3.7764130992e-02, 1.3469870328e-02, 4.3248255411e-03, 1.0739672151e-03]),
                ),
                4.9: (
                    normalize(np.array([4.2157352248e-01, 3.0502361475e-01, 1.6674370619e-01, 7.2309459585e-02, 2.5457559302e-02, 7.1617427232e-03, 1.5167330444e-03, 2.1366192404e-04])),
                    np.array([3.2469361064e+00, 8.0157181113e-01, 2.7164222287e-01, 9.9978580591e-02, 3.7152463919e-02, 1.3248508200e-02, 4.2617568891e-03, 1.0645448001e-03]),
                ),
                4.95: (
                    normalize(np.array([4.2374097585e-01, 3.0405946201e-01, 1.6587119350e-01, 7.1975275539e-02, 2.5410335864e-02, 7.1860098280e-03, 1.5358947741e-03, 2.2085262795e-04])),
                    np.array([3.2434122867e+00, 7.9478839103e-01, 2.6834406304e-01, 9.8527189357e-02, 3.6562166891e-02, 1.3035845398e-02, 4.2015374768e-03, 1.0556621305e-03]),
                ),
                5.0: (
                    normalize(np.array([4.2586355142e-01, 3.0310109739e-01, 1.6501806082e-01, 7.1653148647e-02, 2.5368102797e-02, 7.2120379290e-03, 1.5556865740e-03, 2.2831442319e-04])),
                    np.array([3.2399533775e+00, 7.8813922787e-01, 2.6513007678e-01, 9.7119415910e-02, 3.5992029478e-02, 1.2831333426e-02, 4.1439553175e-03, 1.0472652441e-03]),
                ),
                5.05: (
                    normalize(np.array([4.2794219396e-01, 3.0214934928e-01, 1.6418365993e-01, 7.1342394290e-02, 2.5330544338e-02, 7.2397262941e-03, 1.5760846583e-03, 2.3604724746e-04])),
                    np.array([3.2365627051e+00, 7.8162031669e-01, 2.6199671139e-01, 9.5753208340e-02, 3.5441025605e-02, 1.2634509834e-02, 4.0888341135e-03, 1.0393096572e-03]),
                ),
                5.1: (
                    normalize(np.array([4.2997947923e-01, 3.0120352127e-01, 1.6336701358e-01, 7.1042563475e-02, 2.5297374021e-02, 7.2689336404e-03, 1.5970630489e-03, 2.4405173186e-04])),
                    np.array([3.2332250943e+00, 7.7522515218e-01, 2.5894071231e-01, 9.4426696793e-02, 3.4908109025e-02, 1.2444904417e-02, 4.0360108614e-03, 1.0317568416e-03]),
                ),
                5.15: (
                    normalize(np.array([4.3197625819e-01, 3.0026425872e-01, 1.6256758119e-01, 7.0753009998e-02, 2.5268352962e-02, 7.2996107490e-03, 1.6186002271e-03, 2.5232795617e-04])),
                    np.array([3.2299434156e+00, 7.6895025264e-01, 2.5595896624e-01, 9.3138136628e-02, 3.4392464598e-02, 1.2262143270e-02, 3.9853345934e-03, 1.0245714271e-03]),
                ),
                5.2: (
                    normalize(np.array([4.3393388225e-01, 2.9933162961e-01, 1.6178478654e-01, 7.0473275859e-02, 2.5243216817e-02, 7.3316538192e-03, 1.6406779454e-03, 2.6087716085e-04])),
                    np.array([3.2267147046e+00, 7.6279161154e-01, 2.5304864444e-01, 9.1885841345e-02, 3.3893211691e-02, 1.2085845940e-02, 3.9366738951e-03, 1.0177234634e-03]),
                ),
                5.25: (
                    normalize(np.array([4.3585381726e-01, 2.9840557642e-01, 1.6101798264e-01, 7.0202884907e-02, 2.5221764785e-02, 7.3649963434e-03, 1.6632772866e-03, 2.6970035075e-04])),
                    np.array([3.2235342426e+00, 7.5674504676e-01, 2.5020703432e-01, 9.0668303637e-02, 3.3409598339e-02, 1.1915673555e-02, 3.8899039102e-03, 1.0111856743e-03]),
                ),
                5.3: (
                    normalize(np.array([4.3773710221e-01, 2.9748623321e-01, 1.6026667956e-01, 6.9941435195e-02, 2.5203798233e-02, 7.3995687465e-03, 1.6863837517e-03, 2.7879909051e-04])),
                    np.array([3.2204013700e+00, 7.5080736041e-01, 2.4743170771e-01, 8.9484102186e-02, 3.2940904490e-02, 1.1751313304e-02, 3.8449145980e-03, 1.0049342780e-03]),
                ),
                5.35: (
                    normalize(np.array([4.3958502559e-01, 2.9657350944e-01, 1.5953028771e-01, 6.9688538360e-02, 2.5189163365e-02, 7.4353187401e-03, 1.7099819193e-03, 2.8817488130e-04])),
                    np.array([3.2173122002e+00, 7.4497493055e-01, 2.4472033206e-01, 8.8331951315e-02, 3.2486487648e-02, 1.1592476590e-02, 3.8016017616e-03, 9.9894771730e-04]),
                ),
                5.4: (
                    normalize(np.array([4.4139797973e-01, 2.9566789288e-01, 1.5880859877e-01, 6.9443847055e-02, 2.5177622080e-02, 7.4721655747e-03, 1.7340636828e-03, 2.9783023104e-04])),
                    np.array([3.2142724311e+00, 7.3924594159e-01, 2.4207075875e-01, 8.7210470162e-02, 3.2045654759e-02, 1.1438890212e-02, 3.7598786330e-03, 9.9320778609e-04]),
                ),
                5.45: (
                    normalize(np.array([4.4317775771e-01, 2.9476884305e-01, 1.5810081378e-01, 6.9207001832e-02, 2.5169115761e-02, 7.5100901992e-03, 1.7586113763e-03, 3.0776628486e-04])),
                    np.array([3.2112717995e+00, 7.3361591769e-01, 2.3948084639e-01, 8.6118642096e-02, 3.1617910714e-02, 1.1290307547e-02, 3.7196498109e-03, 9.8769621717e-04]),
                ),
                5.5: (
                    normalize(np.array([4.4492444584e-01, 2.9387696581e-01, 1.5740684729e-01, 6.8977704806e-02, 2.5163414800e-02, 7.5490153499e-03, 1.7836202506e-03, 3.1798585650e-04])),
                    np.array([3.2083185770e+00, 7.2808376575e-01, 2.3694870039e-01, 8.5055202016e-02, 3.1202624908e-02, 1.1146490901e-02, 3.6808437398e-03, 9.8239846511e-04]),
                ),
                5.55: (
                    normalize(np.array([4.4663960964e-01, 2.9299164903e-01, 1.5672603000e-01, 6.8755736363e-02, 2.5160484596e-02, 7.5889221333e-03, 1.8090772261e-03, 3.2849100362e-04])),
                    np.array([3.2054040733e+00, 7.2264582163e-01, 2.3447263084e-01, 8.4019307998e-02, 3.0799361084e-02, 1.1007227875e-02, 3.6433826986e-03, 9.7730041700e-04]),
                ),
                5.6: (
                    normalize(np.array([4.4832341239e-01, 2.9211364569e-01, 1.5605820457e-01, 6.8540638409e-02, 2.5160087487e-02, 7.6297534515e-03, 1.8349743612e-03, 3.3928363298e-04])),
                    np.array([3.2025355619e+00, 7.1730043022e-01, 2.3205046295e-01, 8.3009683395e-02, 3.0407556153e-02, 1.0872307172e-02, 3.6071982984e-03, 9.7238891612e-04]),
                ),
                5.65: (
                    normalize(np.array([4.4997710070e-01, 2.9124227405e-01, 1.5540293452e-01, 6.8332340719e-02, 2.5162197725e-02, 7.6714817041e-03, 1.8613038740e-03, 3.5036669888e-04])),
                    np.array([3.1997068423e+00, 7.1204508297e-01, 2.2968099914e-01, 8.2025625450e-02, 3.0026808179e-02, 1.0741543376e-02, 3.5722291511e-03, 9.6765323943e-04]),
                ),
                5.7: (
                    normalize(np.array([4.5160133761e-01, 2.9037784689e-01, 1.5475972177e-01, 6.8130521005e-02, 2.5166702572e-02, 7.7140713511e-03, 1.8880566729e-03, 3.6174212875e-04])),
                    np.array([3.1969189557e+00, 7.0687720177e-01, 2.2736238303e-01, 8.1066192081e-02, 2.9656678723e-02, 1.0614752632e-02, 3.5384136769e-03, 9.6308235068e-04]),
                ),
                5.75: (
                    normalize(np.array([4.5319665561e-01, 2.8952038556e-01, 1.5412843493e-01, 6.7934942846e-02, 2.5173461144e-02, 7.7574807646e-03, 1.9152267256e-03, 3.7341241360e-04])),
                    np.array([3.1941735435e+00, 7.0179527586e-01, 2.2509313662e-01, 8.0130459928e-02, 2.9296727868e-02, 1.0491763383e-02, 3.5056979899e-03, 9.5866674385e-04]),
                ),
                5.8: (
                    normalize(np.array([4.5476387706e-01, 2.8866977456e-01, 1.5350867438e-01, 6.7745399970e-02, 2.5182402395e-02, 7.8016850377e-03, 1.9428066275e-03, 3.8537997809e-04])),
                    np.array([3.1914688451e+00, 6.9679702927e-01, 2.2287182299e-01, 7.9217667252e-02, 2.8946585950e-02, 1.0372417673e-02, 3.4740299334e-03, 9.5439764886e-04]),
                ),
                5.85: (
                    normalize(np.array([4.5630386116e-01, 2.8782568389e-01, 1.5290016196e-01, 6.7561749202e-02, 2.5193454423e-02, 7.8466525439e-03, 1.9707896621e-03, 3.9764715636e-04])),
                    np.array([3.1888019953e+00, 6.9188049858e-01, 2.2069721046e-01, 7.8327093061e-02, 2.8605887025e-02, 1.0256562450e-02, 3.4433608696e-03, 9.5026688141e-04]),
                ),
                5.9: (
                    normalize(np.array([4.5781649470e-01, 2.8698915567e-01, 1.5230255509e-01, 6.7383576533e-02, 2.5206468549e-02, 7.8923620204e-03, 1.9991706797e-03, 4.1021676668e-04])),
                    np.array([3.1861828615e+00, 6.8704414356e-01, 2.1856743760e-01, 7.7457820431e-02, 2.8274266457e-02, 1.0144061717e-02, 3.4136469265e-03, 9.4626744406e-04]),
                ),
                5.95: (
                    normalize(np.array([4.5930324364e-01, 2.8615906153e-01, 1.5171551484e-01, 6.7210914818e-02, 2.5221453476e-02, 7.9387809516e-03, 2.0279406193e-03, 4.2309012027e-04])),
                    np.array([3.1836001067e+00, 6.8228554861e-01, 2.1648171730e-01, 7.6609357247e-02, 2.7951411353e-02, 1.0034768534e-02, 3.3848408399e-03, 9.4239133579e-04]),
                ),
                6.0: (
                    normalize(np.array([4.6076465182e-01, 2.8533508326e-01, 1.5113909248e-01, 6.7043648527e-02, 2.5238279496e-02, 7.9858754299e-03, 2.0570981544e-03, 4.3627083654e-04])),
                    np.array([3.1810537803e+00, 6.7760377793e-01, 2.1443906259e-01, 7.5780978890e-02, 2.7636975616e-02, 9.9285608879e-03, 3.3569087172e-03, 9.3863341561e-04]),
                ),
                6.05: (
                    normalize(np.array([4.6219996876e-01, 2.8451948266e-01, 1.5057253668e-01, 6.6881141914e-02, 2.5256828261e-02, 8.0336458631e-03, 2.0866351565e-03, 4.4976070595e-04])),
                    np.array([3.1785623912e+00, 6.7299678633e-01, 2.1243706233e-01, 7.4971796592e-02, 2.7330672365e-02, 9.8253246410e-03, 3.3298090513e-03, 9.3498728989e-04]),
                ),
                6.1: (
                    normalize(np.array([4.6361138860e-01, 2.8370953117e-01, 1.5001602533e-01, 6.6723768411e-02, 2.5277125625e-02, 8.0820518683e-03, 2.1165465814e-03, 4.6356241905e-04])),
                    np.array([3.1761031112e+00, 6.6846305430e-01, 2.1047598049e-01, 7.4181541191e-02, 2.7032223463e-02, 9.7249356352e-03, 3.3035078193e-03, 9.3144774207e-04]),
                ),
                6.15: (
                    normalize(np.array([4.6499876450e-01, 2.8290639827e-01, 1.4946907238e-01, 6.6571116246e-02, 2.5299066876e-02, 8.1310764079e-03, 2.1468275554e-03, 4.7767776157e-04])),
                    np.array([3.1736854161e+00, 6.6400078730e-01, 2.0855403268e-01, 7.3409460263e-02, 2.6741345692e-02, 9.6272869030e-03, 3.2779702240e-03, 9.2800925049e-04]),
                ),
                6.2: (
                    normalize(np.array([4.6636278998e-01, 2.8210985558e-01, 1.4893141354e-01, 6.6423057367e-02, 2.5322601603e-02, 8.1807003545e-03, 2.1774727870e-03, 4.9210878096e-04])),
                    np.array([3.1713067172e+00, 6.5960826278e-01, 2.0667021015e-01, 7.2655014682e-02, 2.6457782619e-02, 9.5322744157e-03, 3.2531638359e-03, 9.2466692970e-04]),
                ),
                6.25: (
                    normalize(np.array([4.6770384669e-01, 2.8131982853e-01, 1.4840293441e-01, 6.6279489128e-02, 2.5347662163e-02, 8.2309030641e-03, 2.2084777382e-03, 5.0685826895e-04])),
                    np.array([3.1689687633e+00, 6.5528446091e-01, 2.0482362265e-01, 7.1917665108e-02, 2.6181279181e-02, 9.4398003182e-03, 3.2290602798e-03, 9.2141681224e-04]),
                ),
                6.3: (
                    normalize(np.array([4.6902224858e-01, 2.8053649511e-01, 1.4788343152e-01, 6.6140225396e-02, 2.5374170433e-02, 8.2816623114e-03, 2.2398383058e-03, 5.2192833656e-04])),
                    np.array([3.1666736978e+00, 6.5102807773e-01, 2.0301317370e-01, 7.1196838516e-02, 2.5911584488e-02, 9.3497703144e-03, 3.2056314065e-03, 9.1825470159e-04]),
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
            weights = normalize(weights)
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
                fluxsub = weight*flux
                if not fluxsub >= 0:
                    print(np.array([f(slope) for f in self.weightsplines]))
                    print(weights)
                    print(sigmas)
                    print('wtf')
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

    def __init__(self, fluxes, name="", profile="sersic", parameters=None, order=8):
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
        indices = np.log10(np.array(list(weightvars.keys())))
        weightvalues = np.array(list(weightvars.values()))
        for i in range(order):
            # Weights we want to ignore are flagged by negative radii
            # you might want a spline knot at r=0 and weight=0, although there is a danger of getting r < 0
            isweight = np.array([value[1][i] >= 0 for value in weightvalues])
            weightvaluestouse = weightvalues[isweight]
            for j, (splines, ext) in enumerate([(self.weightsplines, 'zeros'), (self.sigmasplines, 'const')]):
                splines.append(spinterp.InterpolatedUnivariateSpline(
                    indices[isweight], [values[j][i] for values in weightvaluestouse], ext=ext))



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
