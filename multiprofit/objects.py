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

    # These weights are derived here:
    # https://github.com/lsst-dm/modelling_research/blob/master/jupyternotebooks/multigaussian_sersic1d.ipynb
    weights = {
        "sersic": {
            8: {
                0.5: (
                    normalize(np.array([1., 0, 0, 0, 0, 0, 0, 0])),
                    np.array([1., 9.7033878782e-01, 8.0499602574e-01, 6.0616317021e-01, 4.0699848261e-01, -1,
                              1.5812253132e-01, -1]),
                ),
                0.50025: (
                    normalize(np.array(
                        [9.9258524947e-01, 5.8385143016e-03, 1.5234429522e-03, 2.1261896999e-05,
                         3.0745465695e-05, 0, 7.8591615771e-07, 0])),
                    np.array([1.0005590181e+00, 9.7033878782e-01, 8.0499602574e-01, 6.0616317021e-01,
                              4.0699848261e-01, -1, 1.5812253132e-01, -1]),
                ),
                0.5005: (
                    normalize(np.array(
                        [9.8528675848e-01, 1.1567547958e-02, 3.0390507910e-03, 4.3569056476e-05,
                         6.1496686636e-05, 0, 1.5770281935e-06, 0])),
                    np.array([1.0011180669e+00, 9.7063279087e-01, 8.0511904898e-01, 6.0619406386e-01,
                              4.0699848261e-01, -1, 1.5809691740e-01, -1]),
                ),
                0.50075: (
                    normalize(np.array(
                        [9.7810150557e-01, 1.7190108332e-02, 4.5468330732e-03, 6.6928312775e-05,
                         9.2251791448e-05, 0, 2.3729162228e-06, 0])),
                    np.array([1.0016771485e+00, 9.7092673610e-01, 8.0524202958e-01, 6.0622494365e-01,
                              4.0699848261e-01, -1, 1.5807132042e-01, -1]),
                ),
                0.501: (
                    normalize(np.array(
                        [9.7102693825e-01, 2.2708650700e-02, 6.0469123650e-03, 9.1310676658e-05,
                         1.2301517115e-04, 0, 3.1728380741e-06, 0])),
                    np.array([1.0022362632e+00, 9.7122062358e-01, 8.0536496756e-01, 6.0625580960e-01,
                              4.0699848261e-01, -1, 1.5804574035e-01, -1]),
                ),
                0.50125: (
                    normalize(np.array(
                        [9.6406070359e-01, 2.8125397132e-02, 7.5394552197e-03, 1.1667491985e-04,
                         1.5379106885e-04, 0, 3.9780718085e-06, 0])),
                    np.array([1.0027954107e+00, 9.7151445332e-01, 8.0548786296e-01, 6.0628666172e-01,
                              4.0699848261e-01, -1, 1.5802017719e-01, -1]),
                ),
                0.5015: (
                    normalize(np.array(
                        [9.5719951901e-01, 3.3443775703e-02, 9.0242590301e-03, 1.4309078183e-04,
                         1.8456681397e-04, 0, 4.7886653030e-06, 0])),
                    np.array([1.0033545912e+00, 9.7180822539e-01, 8.0561071582e-01, 6.0631750003e-01,
                              4.0699848261e-01, -1, 1.5799463090e-01, -1]),
                ),
                0.50175: (
                    normalize(np.array(
                        [9.5044167315e-01, 3.8665241033e-02, 1.0501665007e-02, 1.7045589058e-04,
                         2.1536225616e-04, 0, 5.6026612152e-06, 0])),
                    np.array([1.0039138046e+00, 9.7210193981e-01, 8.0573352619e-01, 6.0634832454e-01,
                              4.0699848261e-01, -1, 1.5796910147e-01, -1]),
                ),
                0.502: (
                    normalize(np.array(
                        [9.4378370192e-01, 4.3793502536e-02, 1.1971329414e-02, 1.9889292684e-04,
                         2.4615017830e-04, 0, 6.4230204461e-06, 0])),
                    np.array([1.0044730508e+00, 9.7239559662e-01, 8.0585629408e-01, 6.0637913526e-01,
                              4.0699848261e-01, -1, 1.5794358888e-01, -1]),
                ),
                0.5025: (
                    normalize(np.array(
                        [9.3076065151e-01, 5.3776282189e-02, 1.4888572800e-02, 2.5865006268e-04,
                         3.0776683297e-04, 0, 8.0766060164e-06, 0])),
                    np.array([1.0055916418e+00, 9.7298273760e-01, 8.0610170260e-01, 6.0644071538e-01,
                              4.0699848261e-01, -1, 1.5789261413e-01, -1]),
                ),
                0.503: (
                    normalize(np.array(
                        [9.1811275183e-01, 6.3409318231e-02, 1.7776491037e-02, 3.2226172817e-04,
                         3.6942857960e-04, 0, 9.7485978260e-06, 0])),
                    np.array([1.0067103642e+00, 9.7356964864e-01, 8.0634694168e-01, 6.0650224050e-01,
                              4.0699848261e-01, -1, 1.5784170651e-01, -1]),
                ),
                0.504: (
                    normalize(np.array(
                        [8.9387488929e-01, 8.1692798042e-02, 2.3465287782e-02, 4.6101834118e-04,
                         4.9285502011e-04, 0, 1.3151521938e-05, 0])),
                    np.array([1.0089482026e+00, 9.7474278219e-01, 8.0683691265e-01, 6.0662512616e-01,
                              4.0699848261e-01, -1, 1.5774009202e-01, -1]),
                ),
                0.505: (
                    normalize(np.array(
                        [8.7094817921e-01, 9.8763191556e-02, 2.9040875888e-02, 6.1466424085e-04,
                         6.1646035192e-04, 0, 1.6628750699e-05, 0])),
                    np.array([1.0111865650e+00, 9.7591499980e-01, 8.0732620927e-01, 6.0674779307e-01,
                              4.0699848261e-01, -1, 1.5763874417e-01, -1]),
                ),
                0.506: (
                    normalize(np.array(
                        [8.4922361300e-01, 1.1472715920e-01, 3.4505976467e-02, 7.8282140475e-04,
                         7.4024670887e-04, 0, 2.0183220452e-05, 0])),
                    np.array([1.0134254504e+00, 9.7708630399e-01, 8.0781483378e-01, 6.0687024204e-01,
                              4.0699848261e-01, -1, 1.5753766173e-01, -1]),
                ),
                0.508: (
                    normalize(np.array(
                        [8.0900645764e-01, 1.4370029461e-01, 4.5116492400e-02, 1.1607621959e-03,
                         9.8847826132e-04, 0, 2.7514887767e-05, 0])),
                    np.array([1.0179047869e+00, 9.7942618220e-01, 8.0879007550e-01, 6.0711448945e-01,
                              4.0699848261e-01, -1, 1.5733628822e-01, -1]),
                ),
                0.51: (
                    normalize(np.array(
                        [7.7256600098e-01, 1.6925160100e-01, 5.5317942711e-02, 1.5916461383e-03,
                         1.2376647787e-03, 0, 3.5144399178e-05, 0])),
                    np.array([1.0223862047e+00, 9.8176243679e-01, 8.0976265565e-01, 6.0735787486e-01,
                              4.0699848261e-01, -1, 1.5713596187e-01, -1]),
                ),
                0.512: (
                    normalize(np.array(
                        [7.3937251076e-01, 1.9189361212e-01, 6.5130560820e-02, 2.0723123381e-03,
                         1.4879346589e-03, 0, 4.3069303048e-05, 0])),
                    np.array([1.0268696968e+00, 9.8409508755e-01, 8.1073259190e-01, 6.0760040469e-01,
                              4.0699848261e-01, -1, 1.5693667314e-01, -1]),
                ),
                0.514: (
                    normalize(np.array(
                        [7.0899326326e-01, 2.1204298031e-01, 7.4573290394e-02, 2.5997711160e-03,
                         1.7394088808e-03, 0, 5.1286039046e-05, 0])),
                    np.array([1.0313552560e+00, 9.8642415408e-01, 8.1169990172e-01, 6.0784208526e-01,
                              4.0699848261e-01, -1, 1.5673841264e-01, -1]),
                ),
                0.516: (
                    normalize(np.array(
                        [6.8107033079e-01, 2.3004285983e-01, 8.3663531331e-02, 3.1713079895e-03,
                         1.9921767319e-03, 0, 5.9793322102e-05, 0])),
                    np.array([1.0358428751e+00, 9.8874965581e-01, 8.1266460239e-01, 6.0808292284e-01,
                              4.0699848261e-01, -1, 1.5654117109e-01, -1]),
                ),
                0.518: (
                    normalize(np.array(
                        [6.5530668666e-01, 2.4617599374e-01, 9.2418200701e-02, 3.7841814100e-03,
                         2.2463497669e-03, 0, 6.8587721196e-05, 0])),
                    np.array([1.0403325471e+00, 9.9107161197e-01, 8.1362671103e-01, 6.0832292360e-01,
                              4.0699848261e-01, -1, 1.5634493934e-01, -1]),
                ),
                0.52: (
                    normalize(np.array(
                        [6.3145202196e-01, 2.6067953885e-01, 1.0085286736e-01, 4.4358892704e-03,
                         2.5020147771e-03, 0, 7.7667773694e-05, 0])),
                    np.array([1.0448242651e+00, 9.9339004164e-01, 8.1458624458e-01, 6.0856209369e-01,
                              4.0699848261e-01, -1, 1.5614970835e-01, -1]),
                ),
                0.522: (
                    normalize(np.array(
                        [6.0929454610e-01, 2.7375285462e-01, 1.0898227068e-01, 5.1240458981e-03,
                         2.7592504655e-03, 0, 8.7032233832e-05, 0])),
                    np.array([1.0493180221e+00, 9.9570496370e-01, 8.1554321978e-01, 6.0880043915e-01,
                              4.0699848261e-01, 2.6870993089e-01, 1.5595546920e-01, -1]),
                ),
                0.524: (
                    normalize(np.array(
                        [5.8845997598e-01, 2.8585393500e-01, 1.1667685508e-01, 5.9145867582e-03,
                         2.9879821157e-03, 1.2685912871e-05, 9.3979143424e-05, 0])),
                    np.array([1.0538138112e+00, 9.9801639686e-01, 8.1649765322e-01, 6.0903796596e-01,
                              4.0699848261e-01, 2.6852414309e-01, 1.5576221307e-01, -1]),
                ),
                0.526: (
                    normalize(np.array(
                        [5.6903299697e-01, 2.9677452229e-01, 1.2411808553e-01, 6.7263848761e-03,
                         3.2228628908e-03, 2.3572657295e-05, 1.0157478454e-04, 0])),
                    np.array([1.0583116255e+00, 1.0003243597e+00, 8.1744956132e-01, 6.0927468005e-01,
                              4.0699848261e-01, 2.6833835529e-01, 1.5556993127e-01, -1]),
                ),
                0.528: (
                    normalize(np.array(
                        [5.5083062326e-01, 3.0671060379e-01, 1.3128316436e-01, 7.5741268531e-03,
                         3.4565580298e-03, 3.5764628819e-05, 1.0915907943e-04, 0])),
                    np.array([1.0628114583e+00, 1.0026288705e+00, 8.1839896032e-01, 6.0951058728e-01,
                              4.0699848261e-01, 2.6815340039e-01, 1.5537861521e-01, -1]),
                ),
                0.53: (
                    normalize(np.array(
                        [5.3373764497e-01, 3.1576690589e-01, 1.3818478104e-01, 8.4554302195e-03,
                         3.6892937230e-03, 4.9200819606e-05, 1.1674334056e-04, 0])),
                    np.array([1.0673133028e+00, 1.0049299476e+00, 8.1934586630e-01, 6.0974569342e-01,
                              4.0699848261e-01, 2.6796927151e-01, 1.5518825641e-01, -1]),
                ),
                0.532: (
                    normalize(np.array(
                        [5.1765437436e-01, 3.2403277496e-01, 1.4483553471e-01, 9.3678339714e-03,
                         3.9213425170e-03, 6.3805211044e-05, 1.2433427376e-04, 0])),
                    np.array([1.0718171524e+00, 1.0072276089e+00, 8.2029029517e-01, 6.0998000422e-01,
                              4.0699848261e-01, 2.6778596187e-01, 1.5499884649e-01, -1]),
                ),
                0.535: (
                    normalize(np.array(
                        [4.9522880846e-01, 3.3512490107e-01, 1.5436308688e-01, 1.0791562352e-02,
                         4.2678833803e-03, 8.8056127394e-05, 1.3570172893e-04, 0])),
                    np.array([1.0785766716e+00, 1.0106677356e+00, 8.2170232835e-01, 6.1032999152e-01,
                              4.0699848261e-01, 2.6751251884e-01, 1.5471649270e-01, -1]),
                ),
                0.54: (
                    normalize(np.array(
                        [4.6178375897e-01, 3.5065460886e-01, 1.6913577387e-01, 1.3293695525e-02,
                         4.8433704286e-03, 1.3408542013e-04, 1.5470693357e-04, 0])),
                    np.array([1.0898524684e+00, 1.0163844524e+00, 8.2404355242e-01, 6.1090940632e-01,
                              4.0699848261e-01, 2.6706078280e-01, 1.5425053175e-01, -1]),
                ),
                0.545: (
                    normalize(np.array(
                        [4.3245721135e-01, 3.6317999558e-01, 1.8264991837e-01, 1.5934584072e-02,
                         5.4176591161e-03, 1.8681538912e-04, 1.7381613173e-04, 0])),
                    np.array([1.1011405919e+00, 1.0220803696e+00, 8.2636977680e-01, 6.1148402330e-01,
                              4.0699848261e-01, 2.6661396260e-01, 1.5379024945e-01, -1]),
                ),
                0.55: (
                    normalize(np.array(
                        [4.0653129980e-01, 3.7331299348e-01, 1.9503419029e-01, 1.8690245197e-02,
                         5.9923658145e-03, 2.4582501349e-04, 1.9308040378e-04, 0])),
                    np.array([1.1124409422e+00, 1.0277557525e+00, 8.2868123386e-01, 6.1205392567e-01,
                              4.0699848261e-01, 2.6617196038e-01, 1.5333552543e-01, -1]),
                ),
                0.56: (
                    normalize(np.array(
                        [3.6277168471e-01, 3.8815274536e-01, 2.1684908855e-01, 2.4465013395e-02,
                         7.1479186018e-03, 3.8131011683e-04, 2.3223926263e-04, 0])),
                    np.array([1.1350779325e+00, 1.0390459488e+00, 8.3326074749e-01, 6.1317990851e-01,
                              4.0699848261e-01, 2.6530203239e-01, 1.5244228873e-01, -1]),
                ),
                0.57: (
                    normalize(np.array(
                        [3.2729217229e-01, 3.9776989744e-01, 2.3533044530e-01, 3.0478819346e-02,
                         8.3181872063e-03, 5.3797521451e-04, 2.7250320482e-04, 0])),
                    np.array([1.1577626743e+00, 1.0502570529e+00, 8.3778384238e-01, 6.1428797815e-01,
                              4.0699848261e-01, 2.6445027035e-01, 1.5156992865e-01, -1]),
                ),
                0.58: (
                    normalize(np.array(
                        [2.9798967625e-01, 4.0378386983e-01, 2.5106559963e-01, 3.6624491405e-02,
                         9.5084651482e-03, 7.1377805658e-04, 3.1411967561e-04, 0])),
                    np.array([1.1804944278e+00, 1.0613909922e+00, 8.4225218538e-01, 6.1537872646e-01,
                              4.0699848261e-01, 2.6361598606e-01, 1.5071760363e-01, -1]),
                ),
                0.59: (
                    normalize(np.array(
                        [2.7342311737e-01, 4.0725114082e-01, 2.6451982308e-01, 4.2819672717e-02,
                         1.0721897659e-02, 9.0706123730e-04, 3.5728712119e-04, 0])),
                    np.array([1.2032724773e+00, 1.0724496147e+00, 8.4666736639e-01, 6.1645271595e-01,
                              4.0699848261e-01, 2.6279852866e-01, 1.4988451982e-01, 7.0410196163e-02]),
                ),
                0.6: (
                    normalize(np.array(
                        [2.5245834662e-01, 4.0909467761e-01, 2.7589974443e-01, 4.9102878429e-02,
                         1.1898113870e-02, 1.1568647668e-03, 3.7908445807e-04, 1.0289812892e-05])),
                    np.array([1.2260961300e+00, 1.0834346938e+00, 8.5103090314e-01, 6.1751048171e-01,
                              4.0699848261e-01, 2.6199728208e-01, 1.4906992761e-01, 6.9844011123e-02]),
                ),
                0.61: (
                    normalize(np.array(
                        [2.3455073326e-01, 4.0942935137e-01, 2.8580229988e-01, 5.5247954001e-02,
                         1.3144485969e-02, 1.3928554440e-03, 4.1862834918e-04, 1.3691725032e-05])),
                    np.array([1.2489647149e+00, 1.0943479322e+00, 8.5534424555e-01, 6.1855253318e-01,
                              4.0699848261e-01, 2.6121166258e-01, 1.4827311847e-01, 6.9267380016e-02]),
                ),
                0.625: (
                    normalize(np.array(
                        [2.1205989120e-01, 4.0816339559e-01, 2.9817126605e-01, 6.4272437465e-02,
                         1.5059125604e-02, 1.7739169622e-03, 4.8074083507e-04, 1.9226301341e-05])),
                    np.array([1.2833504226e+00, 1.1105866505e+00, 8.6172316078e-01, 6.2008720171e-01,
                              4.0699848261e-01, 2.6006133089e-01, 1.4710979142e-01, 6.8428548522e-02]),
                ),
                0.65: (
                    normalize(np.array(
                        [1.8312307677e-01, 4.0314634073e-01, 3.1366616322e-01, 7.8607934512e-02,
                         1.8360131813e-02, 2.4743336827e-03, 5.9202712728e-04, 2.9992147618e-05])),
                    np.array([1.3408756591e+00, 1.1373142787e+00, 8.7212251927e-01, 6.2257297452e-01,
                              4.0699848261e-01, 2.5821479496e-01, 1.4525082497e-01, 6.7095636048e-02]),
                ),
                0.675: (
                    normalize(np.array(
                        [1.6158143373e-01, 3.9623156376e-01, 3.2451354464e-01, 9.1895150739e-02,
                         2.1774578466e-02, 3.2472193970e-03, 7.1388662155e-04, 4.2622648253e-05])),
                    np.array([1.3986626227e+00, 1.1636401634e+00, 8.8224782055e-01, 6.2497432916e-01,
                              4.0699848261e-01, 2.5645033574e-01, 1.4348420644e-01, 6.5837552531e-02]),
                ),
                0.7: (
                    normalize(np.array(
                        [1.4506237306e-01, 3.8851911422e-01, 3.3209310629e-01, 1.0406686999e-01,
                         2.5270467213e-02, 4.0841463033e-03, 8.4681512148e-04, 5.7107798106e-05])),
                    np.array([1.4567027409e+00, 1.1895849005e+00, 8.9211604928e-01, 6.2729710496e-01,
                              4.0699848261e-01, 2.5476146129e-01, 1.4180217454e-01, 6.4647551298e-02]),
                ),
                0.75: (
                    normalize(np.array(
                        [1.2171057651e-01, 3.7274118798e-01, 3.4085132120e-01, 1.2515373710e-01,
                         3.2382257539e-02, 5.9228317009e-03, 1.1463765424e-03, 9.1711443625e-05])),
                    np.array([1.5735109829e+00, 1.2404047712e+00, 9.1114135570e-01, 6.3172738063e-01,
                              4.0699848261e-01, 2.5158801022e-01, 1.3866519632e-01, 6.2448793978e-02]),
                ),
                0.8: (
                    normalize(np.array(
                        [1.0630591798e-01, 3.5788680408e-01, 3.4441936291e-01, 1.4234048678e-01,
                         3.9480676666e-02, 7.9425290289e-03, 1.4902269164e-03, 1.3399563559e-04])),
                    np.array([1.6912423757e+00, 1.2899071293e+00, 9.2930546099e-01, 6.3589994631e-01,
                              4.0699848261e-01, 2.4865523904e-01, 1.3579360014e-01, 6.0459740412e-02]),
                ),
                0.85: (
                    normalize(np.array(
                        [9.5613880921e-02, 3.4448300732e-01, 3.4506665471e-01, 1.5626062552e-01,
                         4.6412493305e-02, 1.0102620577e-02, 1.8765512077e-03, 1.8416643867e-04])),
                    np.array([1.8098457733e+00, 1.3382054855e+00, 9.4669772023e-01, 6.3984456942e-01,
                              4.0699848261e-01, 2.4593147172e-01, 1.3315034021e-01, 5.8649055981e-02]),
                ),
                0.9: (
                    normalize(np.array(
                        [8.7912500959e-02, 3.3256300095e-01, 3.4402212500e-01, 1.6750993074e-01,
                         5.3079318197e-02, 1.2367723481e-02, 2.3030188843e-03, 2.4238179772e-04])),
                    np.array([1.9292757222e+00, 1.3853975245e+00, 9.6339359071e-01, 6.4358606716e-01,
                              4.0699848261e-01, 2.4339076737e-01, 1.3070534928e-01, 5.6991591281e-02]),
                ),
                0.95: (
                    normalize(np.array(
                        [8.2210242979e-02, 3.2200297925e-01, 3.4198569568e-01, 1.7659406647e-01,
                         5.9423949568e-02, 1.4707229494e-02, 2.7670925902e-03, 3.0874397257e-04])),
                    np.array([2.0494915452e+00, 1.4315680624e+00, 9.7945743495e-01, 6.4714535391e-01,
                              4.0699848261e-01, 2.4101162873e-01, 1.2843392006e-01, 5.5466896117e-02]),
                ),
                1.0: (
                    normalize(np.array(
                        [7.7900190678e-02, 3.1264016273e-01, 3.3937178782e-01, 1.8392530470e-01,
                         6.5418042323e-02, 1.7095071454e-02, 3.2661231097e-03, 3.8331719642e-04])),
                    np.array([2.1704566125e+00, 1.4767913255e+00, 9.9494464239e-01, 6.5054022453e-01,
                              4.0699848261e-01, 2.3877605265e-01, 1.2631552536e-01, 5.4058147753e-02])
                ),
                1.0208: (
                    normalize(np.array(
                        [7.7138475282e-02, 3.0818535260e-01, 3.3825478461e-01, 1.8679993070e-01,
                         6.7734549427e-02, 1.7990900883e-02, 3.4825489934e-03, 4.1345750103e-04])),
                    np.array([2.2153679082e+00, 1.4949545487e+00, 1.0013395116e+00, 6.5188806266e-01,
                              4.0638960450e-01, 2.3764329136e-01, 1.2529686354e-01, 5.3413407127e-02]),
                ),
                1.0339: (
                    normalize(np.array(
                        [7.6573505491e-02, 3.0539862416e-01, 3.3763799570e-01, 1.8859515284e-01,
                         6.9180789806e-02, 1.8558596987e-02, 3.6220663318e-03, 4.3326869433e-04])),
                    np.array([2.2441692454e+00, 1.5067199213e+00, 1.0055576659e+00, 6.5282298307e-01,
                              4.0604890765e-01, 2.3696146653e-01, 1.2468573455e-01, 5.3029660147e-02]),
                ),
                1.0471: (
                    normalize(np.array(
                        [7.6053162516e-02, 3.0272295147e-01, 3.3697895323e-01, 1.9030239132e-01,
                         7.0599936741e-02, 1.9125813218e-02, 3.7632616835e-03, 4.5352982803e-04])),
                    np.array([2.2731357730e+00, 1.5184587908e+00, 1.0097191952e+00, 6.5371094966e-01,
                              4.0567725975e-01, 2.3626134318e-01, 1.2406522435e-01, 5.2643040860e-02]),
                ),
                1.0605: (
                    normalize(np.array(
                        [7.5582612820e-02, 3.0014313423e-01, 3.3626842842e-01, 1.9192978042e-01,
                         7.1998047963e-02, 1.9696191313e-02, 3.9073717259e-03, 4.7443311236e-04])),
                    np.array([2.3024332888e+00, 1.5302221424e+00, 1.0138348185e+00, 6.5454747444e-01,
                              4.0526809835e-01, 2.3554051315e-01, 1.2343455219e-01, 5.2252462156e-02]),
                ),
                1.0741: (
                    normalize(np.array(
                        [7.4921178156e-02, 2.9725203704e-01, 3.3560478671e-01, 1.9378152567e-01,
                         7.3551447516e-02, 2.0325901342e-02, 4.0657231095e-03, 4.9740046403e-04])),
                    np.array([2.3331232030e+00, 1.5429937435e+00, 1.0187244356e+00, 6.5594492474e-01,
                              4.0522791042e-01, 2.3503558020e-01, 1.2291420595e-01, 5.1911250416e-02]),
                ),
                1.0879: (
                    normalize(np.array(
                        [7.4484976679e-02, 2.9481776987e-01, 3.3486680427e-01, 1.9529295567e-01,
                         7.4909881827e-02, 2.0895313543e-02, 4.2129992098e-03, 5.1929892551e-04])),
                    np.array([2.3633971132e+00, 1.5550176139e+00, 1.0228449349e+00, 6.5669876603e-01,
                              4.0471818070e-01, 2.3423235775e-01, 1.2223515431e-01, 5.1501372909e-02]),
                ),
                1.1: (
                    normalize(np.array(
                        [7.4046916070e-02, 2.9260743724e-01, 3.3426305588e-01, 1.9665715131e-01,
                         7.6129875058e-02, 2.1409545549e-02, 4.3464770110e-03, 5.3954188880e-04])),
                    np.array([2.3903024760e+00, 1.5658692516e+00, 1.0267124646e+00, 6.5752831963e-01,
                              4.0439214015e-01, 2.3360045717e-01, 1.2168733415e-01, 5.1175164452e-02]),
                ),
                1.1159: (
                    normalize(np.array(
                        [7.4131552030e-02, 2.9067006997e-01, 3.3306073569e-01, 1.9772975055e-01,
                         7.7363842150e-02, 2.1978318855e-02, 4.5024836875e-03, 5.6324705943e-04])),
                    np.array([2.4225145769e+00, 1.5774563790e+00, 1.0299005083e+00, 6.5742060807e-01,
                              4.0323267430e-01, 2.3237744611e-01, 1.2075894169e-01, 5.0636813400e-02]),
                ),
                1.1302: (
                    normalize(np.array(
                        [7.3614917710e-02, 2.8813218656e-01, 3.3229363767e-01, 1.9929856614e-01,
                         7.8808589987e-02, 2.2597314020e-02, 4.6665231363e-03, 5.8826477412e-04])),
                    np.array([2.4545412961e+00, 1.5903914550e+00, 1.0346334018e+00, 6.5858818948e-01,
                              4.0297597154e-01, 2.3171947294e-01, 1.2016126220e-01, 5.0265767304e-02]),
                ),
                1.1447: (
                    normalize(np.array(
                        [7.3393885410e-02, 2.8597066522e-01, 3.3136262579e-01, 2.0053240289e-01,
                         8.0101653196e-02, 2.3193998393e-02, 4.8313779830e-03, 6.1339112431e-04])),
                    np.array([2.4855559637e+00, 1.6023275062e+00, 1.0386281721e+00, 6.5927180091e-01,
                              4.0248252469e-01, 2.3097757786e-01, 1.1953081764e-01, 4.9865555008e-02]),
                ),
                1.1593: (
                    normalize(np.array(
                        [7.3429807614e-02, 2.8437865383e-01, 3.3036389065e-01, 2.0139307697e-01,
                         8.1143962376e-02, 2.3686223708e-02, 4.9687988928e-03, 6.3558597136e-04])),
                    np.array([2.5155115733e+00, 1.6130154051e+00, 1.0415373913e+00, 6.5912879754e-01,
                              4.0136128152e-01, 2.2978934663e-01, 1.1864590392e-01, 4.9392343840e-02]),
                ),
                1.1742: (
                    normalize(np.array(
                        [7.3260987795e-02, 2.8241246428e-01, 3.2942067519e-01, 2.0249993603e-01,
                         8.2361576148e-02, 2.4255355680e-02, 5.1281488322e-03, 6.6085605313e-04])),
                    np.array([2.5472170790e+00, 1.6249438052e+00, 1.0453534562e+00, 6.5961468968e-01,
                              4.0066866998e-01, 2.2888122703e-01, 1.1791715127e-01, 4.8969319413e-02]),
                ),
                1.1892: (
                    normalize(np.array(
                        [7.3050548682e-02, 2.8039109583e-01, 3.2848800806e-01, 2.0363253427e-01,
                         8.3610953492e-02, 2.4845177798e-02, 5.2942891845e-03, 6.8739268938e-04])),
                    np.array([2.5794315001e+00, 1.6371926375e+00, 1.0494045526e+00, 6.6027360267e-01,
                              4.0010131738e-01, 2.2805650291e-01, 1.1723559144e-01, 4.8567537324e-02]),
                ),
                1.2: (
                    normalize(np.array(
                        [7.3048653779e-02, 2.7922351062e-01, 3.2777252085e-01, 2.0424833408e-01,
                         8.4378435121e-02, 2.5220001607e-02, 5.4033475871e-03, 7.0519635232e-04])),
                    np.array([2.6018224911e+00, 1.6452483650e+00, 1.0517418278e+00, 6.6034096444e-01,
                              3.9941800417e-01, 2.2729405861e-01, 1.1666016042e-01, 4.8245234139e-02]),
                ),
                1.2199: (
                    normalize(np.array(
                        [7.2831301211e-02, 2.7672755724e-01, 3.2649273438e-01, 2.0562654540e-01,
                         8.5969077874e-02, 2.5987242572e-02, 5.6244815294e-03, 7.4105978065e-04])),
                    np.array([2.6444064733e+00, 1.6612302363e+00, 1.0570022912e+00, 6.6119143733e-01,
                              3.9866012666e-01, 2.2620448030e-01, 1.1576840595e-01, 4.7719203132e-02]),
                ),
                1.2355: (
                    normalize(np.array(
                        [7.2895458412e-02, 2.7523463076e-01, 3.2550718192e-01, 2.0634523514e-01,
                         8.6970094347e-02, 2.6501949991e-02, 5.7784401913e-03, 7.6700923059e-04])),
                    np.array([2.6763900903e+00, 1.6725138449e+00, 1.0601050262e+00, 6.6110667189e-01,
                              3.9758769431e-01, 2.2507206397e-01, 1.1492867297e-01, 4.7261356033e-02]),
                ),
                1.2513: (
                    normalize(np.array(
                        [7.2879350158e-02, 2.7359629424e-01, 3.2449595986e-01, 2.0717354291e-01,
                         8.8059410777e-02, 2.7056446076e-02, 5.9442011693e-03, 7.9479481798e-04])),
                    np.array([2.7093228335e+00, 1.6843795391e+00, 1.0636682728e+00, 6.6136808150e-01,
                              3.9673961076e-01, 2.2407548122e-01, 1.1416114391e-01, 4.6828383298e-02]),
                ),
                1.2674: (
                    normalize(np.array(
                        [7.2887509582e-02, 2.7199995827e-01, 3.2346922742e-01, 2.0796088413e-01,
                         8.9134222255e-02, 2.7612361664e-02, 6.1124706563e-03, 8.2336602485e-04])),
                    np.array([2.7427525833e+00, 1.6963300297e+00, 1.0672186188e+00, 6.6159742789e-01,
                              3.9586318524e-01, 2.2305935415e-01, 1.1338411751e-01, 4.6393580863e-02]),
                ),
                1.2836: (
                    normalize(np.array(
                        [7.2928379510e-02, 2.7047384289e-01, 3.2244197816e-01, 2.0868758723e-01,
                         9.0174520341e-02, 2.8160757359e-02, 6.2808016702e-03, 8.5213284359e-04])),
                    np.array([2.7761857285e+00, 1.7081700867e+00, 1.0706784315e+00, 6.6176934596e-01,
                              3.9495767619e-01, 2.2203160759e-01, 1.1260281166e-01, 4.5956101726e-02]),
                ),
                1.3: (
                    normalize(np.array(
                        [7.2986715356e-02, 2.6900627516e-01, 3.2141814637e-01, 2.0937195771e-01,
                         9.1185762595e-02, 2.8701242855e-02, 6.4485514181e-03, 8.8134853811e-04])),
                    np.array([2.8099447197e+00, 1.7200174479e+00, 1.0740835988e+00, 6.6188255446e-01,
                              3.9400355168e-01, 2.2097139885e-01, 1.1180664264e-01, 4.5517678092e-02]),
                ),
                1.3167: (
                    normalize(np.array(
                        [7.3075858396e-02, 2.6758024242e-01, 3.2037677954e-01, 2.1001311415e-01,
                         9.2180755945e-02, 2.9242986022e-02, 6.6189962865e-03, 9.1126724430e-04])),
                    np.array([2.8441098053e+00, 1.7319024643e+00, 1.0774569217e+00, 6.6195991509e-01,
                              3.9302508080e-01, 2.1989804378e-01, 1.1100409044e-01, 4.5075966304e-02]),
                ),
                1.3335: (
                    normalize(np.array(
                        [7.3191454494e-02, 2.6621559268e-01, 3.1933662629e-01, 2.1060486780e-01,
                         9.3144168483e-02, 2.9776721289e-02, 6.7891087671e-03, 9.4146020039e-04])),
                    np.array([2.8782858443e+00, 1.7436812905e+00, 1.0807488458e+00, 6.6198880321e-01,
                              3.9202215361e-01, 2.1881512760e-01, 1.1019995016e-01, 4.4635682832e-02]),
                ),
                1.3506: (
                    normalize(np.array(
                        [7.3327465336e-02, 2.6489042356e-01, 3.1829121363e-01, 2.1116122255e-01,
                         9.4088743401e-02, 3.0307874896e-02, 6.9607486725e-03, 9.7230795837e-04])),
                    np.array([2.9129279815e+00, 1.7555206468e+00, 1.0840120846e+00, 6.6197422982e-01,
                              3.9098270102e-01, 2.1770866870e-01, 1.0938551647e-01, 4.4191769936e-02]),
                ),
                1.3679: (
                    normalize(np.array(
                        [7.3498621743e-02, 2.6362459633e-01, 3.1723969626e-01, 2.1166407214e-01,
                         9.5003481711e-02, 3.0833652052e-02, 7.1324269046e-03, 1.0034528579e-03])),
                    np.array([2.9476862090e+00, 1.7672676448e+00, 1.0871800058e+00, 6.6189597004e-01,
                              3.8990864383e-01, 2.1658541272e-01, 1.0856117716e-01, 4.3745939280e-02]),
                ),
                1.3854: (
                    normalize(np.array(
                        [7.3691154093e-02, 2.6240605915e-01, 3.1618982170e-01, 2.1212848398e-01,
                         9.5892355892e-02, 3.1352730591e-02, 7.3043757775e-03, 1.0350188137e-03])),
                    np.array([2.9826621081e+00, 1.7789810667e+00, 1.0902880854e+00, 6.6176912862e-01,
                              3.8880267888e-01, 2.1544537189e-01, 1.0773190876e-01, 4.3299282871e-02]),
                ),
                1.4: (
                    normalize(np.array(
                        [7.3869941287e-02, 2.6143682530e-01, 3.1532172512e-01, 2.1248055482e-01,
                         9.6606373474e-02, 3.1776730628e-02, 7.4464709706e-03, 1.0613783964e-03])),
                    np.array([3.0116606365e+00, 1.7886048170e+00, 1.0927992044e+00, 6.6162555245e-01,
                              3.8786679067e-01, 2.1449308962e-01, 1.0704325132e-01, 4.2930110489e-02]),
                ),
                1.4212: (
                    normalize(np.array(
                        [7.4145593206e-02, 2.6008856977e-01, 3.1408297172e-01, 2.1294759195e-01,
                         9.7605774324e-02, 3.2379148589e-02, 7.6506944367e-03, 1.0996560063e-03])),
                    np.array([3.0535767618e+00, 1.8024102703e+00, 1.0963534202e+00, 6.6137626891e-01,
                              3.8649602558e-01, 2.1311126184e-01, 1.0604918867e-01, 4.2399204820e-02]),
                ),
                1.4394: (
                    normalize(np.array(
                        [7.4425972967e-02, 2.5902153449e-01, 3.1302346020e-01, 2.1328244776e-01,
                         9.8411944954e-02, 3.2879054842e-02, 7.8231031996e-03, 1.1324815819e-03])),
                    np.array([3.0891020609e+00, 1.8139202764e+00, 1.0992097673e+00, 6.6106485303e-01,
                              3.8527865411e-01, 2.1191512804e-01, 1.0519732699e-01, 4.1949264391e-02]),
                ),
                1.4578: (
                    normalize(np.array(
                        [7.4720552906e-02, 2.5798720226e-01, 3.1197176034e-01, 2.1358821187e-01,
                         9.9198577432e-02, 3.3372537414e-02, 7.9955433391e-03, 1.1656144376e-03])),
                    np.array([3.1248448991e+00, 1.8254136919e+00, 1.1020222605e+00, 6.6072030869e-01,
                              3.8403984670e-01, 2.1070708481e-01, 1.0434264401e-01, 4.1497809959e-02]),
                ),
                1.4765: (
                    normalize(np.array(
                        [7.5044423796e-02, 2.5699155620e-01, 3.1091500224e-01, 2.1385908069e-01,
                         9.9961119154e-02, 3.3861387045e-02, 8.1681633580e-03, 1.1992675242e-03])),
                    np.array([3.1608527765e+00, 1.8368670895e+00, 1.1047646433e+00, 6.6031605444e-01,
                              3.8276260633e-01, 2.0947792493e-01, 1.0347756236e-01, 4.1045038646e-02]),
                ),
                1.5: (
                    normalize(np.array(
                        [7.5482256892e-02, 2.5582564909e-01, 3.0960965742e-01, 2.1413835957e-01,
                         1.0086659438e-01, 3.4454800427e-02, 8.3813687604e-03, 1.2413134719e-03])),
                    np.array([3.2056719117e+00, 1.8509333167e+00, 1.1080232617e+00, 6.5971901713e-01,
                              3.8112282828e-01, 2.0792649738e-01, 1.0239511916e-01, 4.0481128013e-02]),
                ),
                1.5145: (
                    normalize(np.array(
                        [7.5741245215e-02, 2.5511159320e-01, 3.0882151080e-01, 2.1430640152e-01,
                         1.0141734598e-01, 3.4820247158e-02, 8.5141365353e-03, 1.2675195896e-03])),
                    np.array([3.2333463291e+00, 1.8596213955e+00, 1.1100571555e+00, 6.5938182160e-01,
                              3.8014382271e-01, 2.0699798247e-01, 1.0174634768e-01, 4.0141659341e-02]),
                ),
                1.5339: (
                    normalize(np.array(
                        [7.6172206144e-02, 2.5428500711e-01, 3.0776482521e-01, 2.1443968482e-01,
                         1.0207931506e-01, 3.5275174884e-02, 8.6821753007e-03, 1.3016114709e-03])),
                    np.array([3.2694233239e+00, 1.8706014789e+00, 1.1123972403e+00, 6.5871207626e-01,
                              3.7871083379e-01, 2.0568800497e-01, 1.0084810673e-01, 3.9680554893e-02]),
                ),
                1.5536: (
                    normalize(np.array(
                        [7.6603917282e-02, 2.5346309649e-01, 3.0671436945e-01, 2.1456250964e-01,
                         1.0273478057e-01, 3.5732014361e-02, 8.8528826742e-03, 1.3364295408e-03])),
                    np.array([3.3059628211e+00, 1.8816867563e+00, 1.1147576024e+00, 6.5804517377e-01,
                              3.7728122693e-01, 2.0438452178e-01, 9.9955586473e-02, 3.9222376658e-02]),
                ),
                1.5735: (
                    normalize(np.array(
                        [7.7062571863e-02, 2.5268329953e-01, 3.0566818281e-01, 2.1465055722e-01,
                         1.0336243436e-01, 3.6179267227e-02, 9.0222875573e-03, 1.3713994427e-03])),
                    np.array([3.3424754976e+00, 1.8926255220e+00, 1.1170096937e+00, 6.5731399127e-01,
                              3.7581862020e-01, 2.0306707113e-01, 9.9059122039e-02, 3.8764682482e-02]),
                ),
                1.6: (
                    normalize(np.array(
                        [7.7706935428e-02, 2.5172135794e-01, 3.0429904015e-01, 2.1471431704e-01,
                         1.0414517019e-01, 3.6752518705e-02, 9.2430533522e-03, 1.4176071993e-03])),
                    np.array([3.3904646586e+00, 1.9067861555e+00, 1.1197994013e+00, 6.5624798889e-01,
                              3.7384063102e-01, 2.0131098853e-01, 9.7873370434e-02, 3.8162862052e-02]),
                ),
                1.6141: (
                    normalize(np.array(
                        [7.8065153137e-02, 2.5124344009e-01, 3.0358181824e-01, 2.1472461056e-01,
                         1.0453757447e-01, 3.7047170980e-02, 9.3582184746e-03, 1.4420140481e-03])),
                    np.array([3.4156949924e+00, 1.9141301109e+00, 1.1211862670e+00, 6.5563770263e-01,
                              3.7277395059e-01, 2.0037574954e-01, 9.7246180262e-02, 3.7846247993e-02]),
                ),
                1.6347: (
                    normalize(np.array(
                        [7.8607320963e-02, 2.5058532915e-01, 3.0254702770e-01, 2.1471178862e-01,
                         1.0508232443e-01, 3.7465173867e-02, 9.5236100540e-03, 1.4774252201e-03])),
                    np.array([3.4521663968e+00, 1.9246204746e+00, 1.1230922729e+00, 6.5469447367e-01,
                              3.7119930435e-01, 1.9900898205e-01, 9.6334541331e-02, 3.7388041501e-02]),
                ),
                1.6557: (
                    normalize(np.array(
                        [7.9182621648e-02, 2.4996164062e-01, 3.0150814193e-01, 2.1466615984e-01,
                         1.0560349431e-01, 3.7876081459e-02, 9.6886533181e-03, 1.5132068776e-03])),
                    np.array([3.4888584808e+00, 1.9350202800e+00, 1.1248880002e+00, 6.5366948137e-01,
                              3.6957392590e-01, 1.9761503544e-01, 9.5410900580e-02, 3.6926274555e-02]),
                ),
                1.6769: (
                    normalize(np.array(
                        [7.9785703666e-02, 2.4937764679e-01, 3.0047553642e-01, 2.1458880065e-01,
                         1.0609602933e-01, 3.8275669481e-02, 9.8516289244e-03, 1.5489847420e-03])),
                    np.array([3.5253925152e+00, 1.9452190745e+00, 1.1265521379e+00, 6.5257132112e-01,
                              3.6791338986e-01, 1.9620761795e-01, 9.4484431556e-02, 3.6465508508e-02]),
                ),
                1.7: (
                    normalize(np.array(
                        [8.0467267885e-02, 2.4879072214e-01, 2.9936728240e-01, 2.1447164047e-01,
                         1.0659652272e-01, 3.8694059860e-02, 1.0024954178e-02, 1.5875503489e-03])),
                    np.array([3.5646061425e+00, 1.9559904053e+00, 1.1281989065e+00, 6.5130579302e-01,
                              3.6608297491e-01, 1.9467379108e-01, 9.3481526863e-02, 3.5969464252e-02]),
                ),
                1.7201: (
                    normalize(np.array(
                        [8.1082310353e-02, 2.4832090130e-01, 2.9841984480e-01, 2.1434051002e-01,
                         1.0699987191e-01, 3.9043550983e-02, 1.0172278860e-02, 1.6207317752e-03])),
                    np.array([3.5982117574e+00, 1.9650619669e+00, 1.1294829861e+00, 6.5013925744e-01,
                              3.6447013497e-01, 1.9333960407e-01, 9.2614944609e-02, 3.5543054814e-02]),
                ),
                1.7421: (
                    normalize(np.array(
                        [8.1776200016e-02, 2.4784655279e-01, 2.9739704873e-01, 2.1417085278e-01,
                         1.0741145765e-01, 3.9411498564e-02, 1.0329729017e-02, 1.6566604450e-03])),
                    np.array([3.6344367100e+00, 1.9746837790e+00, 1.1307421523e+00, 6.4880383173e-01,
                              3.6268811591e-01, 1.9188015462e-01, 9.1672892445e-02, 3.5081926950e-02]),
                ),
                1.7644: (
                    normalize(np.array(
                        [8.2501790747e-02, 2.4740667110e-01, 2.9637650087e-01, 2.1397153743e-01,
                         1.0779671776e-01, 3.9768914385e-02, 1.0485257019e-02, 1.6926106934e-03])),
                    np.array([3.6705540589e+00, 1.9841048808e+00, 1.1318600005e+00, 6.4738487764e-01,
                              3.6086270706e-01, 1.9040160835e-01, 9.0724594588e-02, 3.4620100048e-02]),
                ),
                1.787: (
                    normalize(np.array(
                        [8.3259159234e-02, 2.4700047456e-01, 2.9535837657e-01, 2.1374321207e-01,
                         1.0815582700e-01, 4.0115630494e-02, 1.0638751895e-02, 1.7285681735e-03])),
                    np.array([3.7065321788e+00, 1.9933140954e+00, 1.1328329244e+00, 6.4588188172e-01,
                              3.5899443320e-01, 1.8890465210e-01, 8.9770642144e-02, 3.4157952552e-02]),
                ),
                1.8: (
                    normalize(np.array(
                        [8.3704523705e-02, 2.4678395802e-01, 2.9477995558e-01, 2.1360058371e-01,
                         1.0834862391e-01, 4.0308163899e-02, 1.0725167824e-02, 1.7490233375e-03])),
                    np.array([3.7269405360e+00, 1.9984581432e+00, 1.1333207803e+00, 6.4498843734e-01,
                              3.5791177886e-01, 1.8804437955e-01, 8.9225111526e-02, 3.3894728868e-02]),
                ),
                1.8331: (
                    normalize(np.array(
                        [8.4869651673e-02, 2.4628626926e-01, 2.9332981476e-01, 2.1320213948e-01,
                         1.0879613330e-01, 4.0776382715e-02, 1.0939228622e-02, 1.8003801984e-03])),
                    np.array([3.7779509919e+00, 2.0110507719e+00, 1.1343296607e+00, 6.4262133633e-01,
                              3.5513122753e-01, 1.8585816441e-01, 8.7847658593e-02, 3.3233513253e-02]),
                ),
                1.8566: (
                    normalize(np.array(
                        [8.5722936267e-02, 2.4597627912e-01, 2.9232062610e-01, 2.1289096758e-01,
                         1.0907758347e-01, 4.1089716677e-02, 1.1085736914e-02, 1.8361538837e-03])),
                    np.array([3.8133218117e+00, 2.0195588436e+00, 1.1348462759e+00, 6.4086170006e-01,
                              3.5313555772e-01, 1.8430838110e-01, 8.6878648164e-02, 3.2771408729e-02]),
                ),
                1.8804: (
                    normalize(np.array(
                        [8.6607379131e-02, 2.4569608628e-01, 2.9131433977e-01, 2.1255431948e-01,
                         1.0933408209e-01, 4.1392090677e-02, 1.1229875425e-02, 1.8718271386e-03])),
                    np.array([3.8484465167e+00, 2.0278175949e+00, 1.1352085374e+00, 6.3901817950e-01,
                              3.5110014209e-01, 1.8274334355e-01, 8.5906040452e-02, 3.2309823682e-02]),
                ),
                1.9: (
                    normalize(np.array(
                        [8.7351528447e-02, 2.4549040479e-01, 2.9049694605e-01, 2.1226066140e-01,
                         1.0952437126e-01, 4.1629960305e-02, 1.1345361900e-02, 1.9007658418e-03])),
                    np.array([3.8768283561e+00, 2.0343430928e+00, 1.1353830920e+00, 6.3745333151e-01,
                              3.4941314343e-01, 1.8145792055e-01, 8.5111645146e-02, 3.1934615802e-02]),
                ),
                1.9289: (
                    normalize(np.array(
                        [8.8472243328e-02, 2.4522403526e-01, 2.8931197797e-01, 2.1180466832e-01,
                         1.0977183731e-01, 4.1962371674e-02, 1.1510173230e-02, 1.9426929010e-03])),
                    np.array([3.9177948688e+00, 2.0435299454e+00, 1.1354461805e+00, 6.3507106882e-01,
                              3.4690740789e-01, 1.7956683996e-01, 8.3950320081e-02, 3.1388899808e-02]),
                ),
                1.9536: (
                    normalize(np.array(
                        [8.9450948728e-02, 2.4502884544e-01, 2.8831643666e-01, 2.1139464766e-01,
                         1.0995453260e-01, 4.2230512496e-02, 1.1646228698e-02, 1.9778477166e-03])),
                    np.array([3.9519823449e+00, 2.0509780039e+00, 1.1353237020e+00, 6.3297087959e-01,
                              3.4475292067e-01, 1.7795730952e-01, 8.2968248665e-02, 3.0929923345e-02]),
                ),
                1.9786: (
                    normalize(np.array(
                        [9.0460086982e-02, 2.4485947695e-01, 2.8732483816e-01, 2.1096227802e-01,
                         1.1011372944e-01, 4.2487339759e-02, 1.1779465326e-02, 2.0127853603e-03])),
                    np.array([3.9858126691e+00, 2.0581446707e+00, 1.1350387025e+00, 6.3078702363e-01,
                              3.4256092258e-01, 1.7633461586e-01, 8.1984004033e-02, 3.0472230569e-02]),
                ),
                2.0: (
                    normalize(np.array(
                        [9.1336700722e-02, 2.4473428959e-01, 2.8648805781e-01, 2.1058004155e-01,
                         1.1023150253e-01, 4.2696830476e-02, 1.1890355861e-02, 2.0422214609e-03])),
                    np.array([4.0141718737e+00, 2.0639981833e+00, 1.1346760659e+00, 6.2887801648e-01,
                              3.4067984903e-01, 1.7495327288e-01, 8.1150372160e-02, 3.0086149755e-02]),
                ),
                2.0296: (
                    normalize(np.array(
                        [9.2565186299e-02, 2.4458561094e-01, 2.8535092134e-01, 2.1003759678e-01,
                         1.1036926302e-01, 4.2970699542e-02, 1.2038516715e-02, 2.0822053624e-03])),
                    np.array([4.0525359146e+00, 2.0716967508e+00, 1.1340111904e+00, 6.2618313219e-01,
                              3.3807115880e-01, 1.7305210934e-01, 8.0009653611e-02, 2.9560752172e-02]),
                ),
                2.0556: (
                    normalize(np.array(
                        [9.3672769936e-02, 2.4449283515e-01, 2.8436729697e-01, 2.0953807474e-01,
                         1.1045715793e-01, 4.3192645809e-02, 1.2162865231e-02, 2.1163542279e-03])),
                    np.array([4.0852121366e+00, 2.0779609612e+00, 1.1332051178e+00, 6.2372779784e-01,
                              3.3575461389e-01, 1.7138390671e-01, 7.9015809792e-02, 2.9105483440e-02]),
                ),
                2.0819: (
                    normalize(np.array(
                        [9.4802223896e-02, 2.4441561425e-01, 2.8338976962e-01, 2.0902305143e-01,
                         1.1052820782e-01, 4.3405940983e-02, 1.2284838446e-02, 2.1503535515e-03])),
                    np.array([4.1175341978e+00, 2.0839773010e+00, 1.1322629697e+00, 6.2121005630e-01,
                              3.3341548458e-01, 1.6971083575e-01, 7.8024192925e-02, 2.8653346648e-02]),
                ),
                2.1: (
                    normalize(np.array(
                        [9.5588452830e-02, 2.4437405817e-01, 2.8272649151e-01, 2.0866260615e-01,
                         1.1056442345e-01, 4.3544598673e-02, 1.2366051990e-02, 2.1733172275e-03])),
                    np.array([4.1393007416e+00, 2.0879027160e+00, 1.1315278259e+00, 6.1944768322e-01,
                              3.3180114551e-01, 1.6856406909e-01, 7.7347766185e-02, 2.8346291183e-02]),
                ),
                2.1356: (
                    normalize(np.array(
                        [9.7151570386e-02, 2.4431785694e-01, 2.8144231305e-01, 2.0794000573e-01,
                         1.1060944604e-01, 4.3800937091e-02, 1.2520297208e-02, 2.2175735505e-03])),
                    np.array([4.1810303432e+00, 2.0951433003e+00, 1.1298904291e+00, 6.1592426343e-01,
                              3.2862479625e-01, 1.6632529879e-01, 7.6034181460e-02, 2.7752665329e-02]),
                ),
                2.1629: (
                    normalize(np.array(
                        [9.8365371503e-02, 2.4429608804e-01, 2.8047632480e-01, 2.0737440673e-01,
                         1.1062079312e-01, 4.3982741083e-02, 1.2633574947e-02, 2.2506997765e-03])),
                    np.array([4.2120565571e+00, 2.1002672983e+00, 1.1284643591e+00, 6.1316982725e-01,
                              3.2618671166e-01, 1.6462247601e-01, 7.5041527574e-02, 2.7306548438e-02]),
                ),
                2.2: (
                    normalize(np.array(
                        [1.0003247911e-01, 2.4429235562e-01, 2.7918865556e-01, 2.0659336650e-01,
                         1.1060728293e-01, 4.4210556221e-02, 1.2780741545e-02, 2.2945625138e-03])),
                    np.array([4.2529089442e+00, 2.1066639971e+00, 1.1263061006e+00, 6.0936241079e-01,
                              3.2287351670e-01, 1.6232853922e-01, 7.3712510692e-02, 2.6712554345e-02]),
                ),
                2.2187: (
                    normalize(np.array(
                        [1.0087973267e-01, 2.4430065817e-01, 2.7855036370e-01, 2.0619493395e-01,
                         1.1058863682e-01, 4.4317404281e-02, 1.2852086038e-02, 2.3161843783e-03])),
                    np.array([4.2729377144e+00, 2.1096482477e+00, 1.1251261564e+00, 6.0741728183e-01,
                              3.2120456395e-01, 1.6118149588e-01, 7.3051475112e-02, 2.6418496036e-02]),
                ),
                2.2471: (
                    normalize(np.array(
                        [1.0217408449e-01, 2.4432481651e-01, 2.7759442395e-01, 2.0558490668e-01,
                         1.1054640012e-01, 4.4470013873e-02, 1.2956948890e-02, 2.3484054917e-03])),
                    np.array([4.3026537669e+00, 2.1138855725e+00, 1.1232229030e+00, 6.0443327078e-01,
                              3.1867262224e-01, 1.5945167163e-01, 7.2058898644e-02, 2.5978642814e-02]),
                ),
                2.2759: (
                    normalize(np.array(
                        [1.0349530500e-01, 2.4436265475e-01, 2.7664119865e-01, 2.0496097482e-01,
                         1.1048719266e-01, 4.4613264574e-02, 1.3059065845e-02, 2.3803437048e-03])),
                    np.array([4.3319377437e+00, 2.1178266231e+00, 1.1211598141e+00, 6.0137193022e-01,
                              3.1610899875e-01, 1.5771261679e-01, 7.1066227928e-02, 2.5540837423e-02]),
                ),
                2.3: (
                    normalize(np.array(
                        [1.0460644397e-01, 2.4440315342e-01, 2.7585578930e-01, 2.0443569395e-01,
                         1.1042614397e-01, 4.4724833236e-02, 1.3141433494e-02, 2.4065086584e-03])),
                    np.array([4.3558014324e+00, 2.1208617431e+00, 1.1193372232e+00, 5.9878646050e-01,
                              3.1396840979e-01, 1.5626965686e-01, 7.0246403186e-02, 2.5180764735e-02]),
                ),
                2.3346: (
                    normalize(np.array(
                        [1.0620864747e-01, 2.4447372014e-01, 2.7474743020e-01, 2.0367782535e-01,
                         1.1032169514e-01, 4.4872518623e-02, 1.3254962216e-02, 2.4432008649e-03])),
                    np.array([4.3890725911e+00, 2.1248191665e+00, 1.1165764102e+00, 5.9504099994e-01,
                              3.1090430056e-01, 1.5421800682e-01, 6.9086635255e-02, 2.4673695436e-02]),
                ),
                2.3645: (
                    normalize(np.array(
                        [1.0759992325e-01, 2.4454529878e-01, 2.7380661881e-01, 2.0302024960e-01,
                         1.1021638635e-01, 4.4988723838e-02, 1.3348691007e-02, 2.4741083542e-03])),
                    np.array([4.4168883655e+00, 2.1278631366e+00, 1.1140581221e+00, 5.9177451161e-01,
                              3.0826567740e-01, 1.5246403395e-01, 6.8100576249e-02, 2.4244752198e-02]),
                ),
                2.4: (
                    normalize(np.array(
                        [1.0925612940e-01, 2.4464186553e-01, 2.7271119666e-01, 2.0223730420e-01,
                         1.1007474974e-01, 4.5113938960e-02, 1.3454974246e-02, 2.5098412598e-03])),
                    np.array([4.4488513891e+00, 2.1310565450e+00, 1.1109198766e+00, 5.8786460143e-01,
                              3.0514597631e-01, 1.5040502925e-01, 6.6949175999e-02, 2.3746314503e-02]),
                ),
                2.4254: (
                    normalize(np.array(
                        [1.1044364813e-01, 2.4471692579e-01, 2.7194065750e-01, 2.0167684168e-01,
                         1.0996368075e-01, 4.5195586481e-02, 1.3527858812e-02, 2.5348008578e-03])),
                    np.array([4.4710226759e+00, 2.1330709902e+00, 1.1085832111e+00, 5.8505029765e-01,
                              3.0292409597e-01, 1.4894780547e-01, 6.6138295302e-02, 2.3396865627e-02]),
                ),
                2.4565: (
                    normalize(np.array(
                        [1.1189955951e-01, 2.4481493309e-01, 2.7101202438e-01, 2.0099071794e-01,
                         1.0981736694e-01, 4.5287051269e-02, 1.3613660499e-02, 2.5646863653e-03])),
                    np.array([4.4973982336e+00, 2.1352414992e+00, 1.1056233903e+00, 5.8158733649e-01,
                              3.0021608330e-01, 1.4718191044e-01, 6.5159997375e-02, 2.2976983311e-02]),
                ),
                2.488: (
                    normalize(np.array(
                        [1.1337454443e-01, 2.4491999742e-01, 2.7008801615e-01, 2.0029703500e-01,
                         1.0965877965e-01, 4.5370621290e-02, 1.3696804851e-02, 2.5942011953e-03])),
                    np.array([4.5232833642e+00, 2.1371260436e+00, 1.1025228745e+00, 5.7806398460e-01,
                              2.9748814562e-01, 1.4541380014e-01, 6.4185140118e-02, 2.2560446535e-02]),
                ),
                2.5: (
                    normalize(np.array(
                        [1.1393682411e-01, 2.4496166070e-01, 2.6974008514e-01, 2.0003298105e-01,
                         1.0959556995e-01, 4.5400101797e-02, 1.3727520343e-02, 2.6052569129e-03])),
                    np.array([4.5329210503e+00, 2.1377594258e+00, 1.1013141672e+00, 5.7671731571e-01,
                              2.9645278942e-01, 1.4474568021e-01, 6.3818022390e-02, 2.2404077615e-02]),
                ),
                2.5198: (
                    normalize(np.array(
                        [1.1484707355e-01, 2.4500191274e-01, 2.6915819817e-01, 1.9960865205e-01,
                         1.0950881068e-01, 4.5462905078e-02, 1.3786013768e-02, 2.6264339693e-03])),
                    np.array([4.5486712843e+00, 2.1388860808e+00, 1.0994322855e+00, 5.7459561613e-01,
                              2.9482217096e-01, 1.4369839017e-01, 6.3248372364e-02, 2.2167743613e-02]),
                ),
                2.5521: (
                    normalize(np.array(
                        [1.1623466931e-01, 2.4493362761e-01, 2.6817375455e-01, 1.9897597900e-01,
                         1.0945559811e-01, 4.5634055706e-02, 1.3918068008e-02, 2.6742477069e-03])),
                    np.array([4.5750812669e+00, 2.1415056681e+00, 1.0970073050e+00, 5.7161891339e-01,
                              2.9250556401e-01, 1.4222095826e-01, 6.2465512771e-02, 2.1868729790e-02]),
                ),
                2.5848: (
                    normalize(np.array(
                        [1.1728493851e-01, 2.4436614070e-01, 2.6701397780e-01, 1.9854554849e-01,
                         1.0974235121e-01, 4.6077452928e-02, 1.4195390464e-02, 2.7741998868e-03])),
                    np.array([4.6061355571e+00, 2.1476651188e+00, 1.0971292049e+00, 5.7041538466e-01,
                              2.9138944580e-01, 1.4153012638e-01, 6.2176380706e-02, 2.1864310899e-02]),
                ),
                2.6179: (
                    normalize(np.array(
                        [1.1829756677e-01, 2.4374184522e-01, 2.6583780249e-01, 1.9813281827e-01,
                         1.1006167074e-01, 4.6551962407e-02, 1.4493097404e-02, 2.8832366982e-03])),
                    np.array([4.6377658208e+00, 2.1541952074e+00, 1.0975355006e+00, 5.6942138805e-01,
                              2.9042060755e-01, 1.4093721261e-01, 6.1947822376e-02, 2.1893590765e-02]),
                ),
                2.6514: (
                    normalize(np.array(
                        [1.1917911455e-01, 2.4292504685e-01, 2.6459291236e-01, 1.9778998640e-01,
                         1.1050742372e-01, 4.7134863398e-02, 1.4853500931e-02, 3.0171517854e-03])),
                    np.array([4.6712660658e+00, 2.1621275732e+00, 1.0989616491e+00, 5.6914074783e-01,
                              2.8993454233e-01, 1.4065891418e-01, 6.1914444941e-02, 2.2036348518e-02]),
                ),
                2.6854: (
                    normalize(np.array(
                        [1.1958804712e-01, 2.4142487399e-01, 2.6308852505e-01, 1.9770309924e-01,
                         1.1141852633e-01, 4.8108432477e-02, 1.5433135698e-02, 3.2353600942e-03])),
                    np.array([4.7120809579e+00, 2.1753887418e+00, 1.1041342828e+00, 5.7141710056e-01,
                              2.9114972925e-01, 1.4147879523e-01, 6.2561771253e-02, 2.2579776388e-02]),
                ),
                2.7198: (
                    normalize(np.array(
                        [1.2026567343e-01, 2.4030325868e-01, 2.6171953322e-01, 1.9745041976e-01,
                         1.1205858374e-01, 4.8874015746e-02, 1.5908780439e-02, 3.4197349817e-03])),
                    np.array([4.7487821653e+00, 2.1857072684e+00, 1.1072937168e+00, 5.7234109372e-01,
                              2.9147242893e-01, 1.4172316661e-01, 6.2849743063e-02, 2.2906313481e-02]),
                ),
                2.7546: (
                    normalize(np.array(
                        [1.2089462012e-01, 2.3910817486e-01, 2.6031760058e-01, 1.9721049768e-01,
                         1.1274057241e-01, 4.9687149982e-02, 1.6419476508e-02, 3.6219078699e-03])),
                    np.array([4.7861776818e+00, 2.1966135301e+00, 1.1109086248e+00, 5.7359050525e-01,
                              2.9201590278e-01, 1.4211003698e-01, 6.3224223602e-02, 2.3280472174e-02]),
                ),
                2.7899: (
                    normalize(np.array(
                        [1.2107448641e-01, 2.3726229713e-01, 2.5864530963e-01, 1.9718674634e-01,
                         1.1386001225e-01, 5.0888805392e-02, 1.7161700977e-02, 3.9206418589e-03])),
                    np.array([4.8309041552e+00, 2.2128894444e+00, 1.1183023643e+00, 5.7741029269e-01,
                              2.9426135985e-01, 1.4359047071e-01, 6.4274039056e-02, 2.4049969997e-02]),
                ),
                2.8257: (
                    normalize(np.array(
                        [1.2165265454e-01, 2.3597590773e-01, 2.5718557839e-01, 1.9693710830e-01,
                         1.1458366307e-01, 5.1771399611e-02, 1.7734496765e-02, 4.1591915973e-03])),
                    np.array([4.8693249818e+00, 2.2246495464e+00, 1.1225875694e+00, 5.7913806291e-01,
                              2.9512749870e-01, 1.4418409983e-01, 6.4772906624e-02, 2.4490941258e-02]),
                ),
                2.8618: (
                    normalize(np.array(
                        [1.2221570977e-01, 2.3467131783e-01, 2.5571517510e-01, 1.9667853002e-01,
                         1.1531003327e-01, 5.2670421300e-02, 1.8327194606e-02, 4.4116180964e-03])),
                    np.array([4.9077691980e+00, 2.2365865671e+00, 1.1270589128e+00, 5.8101178592e-01,
                              2.9609533924e-01, 1.4484273192e-01, 6.5309641434e-02, 2.4950466741e-02]),
                ),
                2.8985: (
                    normalize(np.array(
                        [1.2236790454e-01, 2.3276270945e-01, 2.5397157844e-01, 1.9659409689e-01,
                         1.1643491849e-01, 5.3943378105e-02, 1.9155512439e-02, 4.7699016625e-03])),
                    np.array([4.9533602640e+00, 2.2537560322e+00, 1.1352152489e+00, 5.8538709286e-01,
                              2.9871536355e-01, 1.4656089364e-01, 6.6500404412e-02, 2.5792903003e-02]),
                ),
                2.9356: (
                    normalize(np.array(
                        [1.2293418470e-01, 2.3144917404e-01, 2.5247685829e-01, 1.9629447139e-01,
                         1.1714357244e-01, 5.4863234079e-02, 1.9785989036e-02, 5.0525160396e-03])),
                    np.array([4.9918912504e+00, 2.2659183852e+00, 1.1399296790e+00, 5.8745061992e-01,
                              2.9981399991e-01, 1.4730210511e-01, 6.7084384178e-02, 2.6274335197e-02]),
                ),
                2.9732: (
                    normalize(np.array(
                        [1.2350582877e-01, 2.3013795617e-01, 2.5097582126e-01, 1.9597500193e-01,
                         1.1783949972e-01, 5.5787466348e-02, 2.0430520231e-02, 5.3479055812e-03])),
                    np.array([5.0303254842e+00, 2.2781005997e+00, 1.1447042513e+00, 5.8957016007e-01,
                              3.0095377473e-01, 1.4806938735e-01, 6.7682736131e-02, 2.6761556644e-02]),
                ),
                3.0: (
                    normalize(np.array(
                        [1.2368203840e-01, 2.2888233678e-01, 2.4976101917e-01, 1.9583519262e-01,
                         1.1854412501e-01, 5.6648853384e-02, 2.1023744125e-02, 5.6226905121e-03])),
                    np.array([5.0614202438e+00, 2.2896717313e+00, 1.1501878688e+00, 5.9250366523e-01,
                              3.0270746190e-01, 1.4921970665e-01, 6.8481067360e-02, 2.7325555377e-02]),
                ),
                3.0499: (
                    normalize(np.array(
                        [1.2428801201e-01, 2.2697939587e-01, 2.4770168136e-01, 1.9543020416e-01,
                         1.1955113652e-01, 5.7989248684e-02, 2.1980796004e-02, 6.0795253882e-03])),
                    np.array([5.1136464873e+00, 2.3074767608e+00, 1.1579057226e+00, 5.9631783497e-01,
                              3.0489479667e-01, 1.5066813850e-01, 6.9533425064e-02, 2.8114090972e-02]),
                ),
                3.0889: (
                    normalize(np.array(
                        [1.2488459293e-01, 2.2570024746e-01, 2.4619702991e-01, 1.9504562960e-01,
                         1.2018839273e-01, 5.8909276491e-02, 2.2660753369e-02, 6.4140775125e-03])),
                    np.array([5.1514936731e+00, 2.3195732489e+00, 1.1627723503e+00, 5.9854349508e-01,
                              3.0611458435e-01, 1.5148540737e-01, 7.0157446271e-02, 2.8608659150e-02]),
                ),
                3.1285: (
                    normalize(np.array(
                        [1.2549229955e-01, 2.2443111168e-01, 2.4469102068e-01, 1.9464068407e-01,
                         1.2080656509e-01, 5.9826637039e-02, 2.3350842194e-02, 6.7608397113e-03])),
                    np.array([5.1891896398e+00, 2.3316254369e+00, 1.1676468429e+00, 6.0078598961e-01,
                              3.0734816700e-01, 1.5231110202e-01, 7.0785230098e-02, 2.9103418693e-02]),
                ),
                3.1686: (
                    normalize(np.array(
                        [1.2575325780e-01, 2.2265556282e-01, 2.4292635315e-01, 1.9434053427e-01,
                         1.2174261735e-01, 6.1076501222e-02, 2.4275329895e-02, 7.2298434906e-03])),
                    np.array([5.2333546130e+00, 2.3485380628e+00, 1.1759897024e+00, 6.0538724244e-01,
                              3.1013614553e-01, 1.5413126943e-01, 7.2026867523e-02, 2.9958355610e-02]),
                ),
                3.2092: (
                    normalize(np.array(
                        [1.2638348251e-01, 2.2142053231e-01, 2.4142738242e-01, 1.9389182378e-01,
                         1.2231235301e-01, 6.1979809204e-02, 2.4982518186e-02, 7.6020985704e-03])),
                    np.array([5.2705237713e+00, 2.3604319165e+00, 1.1808543061e+00, 6.0765155538e-01,
                              3.1139008174e-01, 1.5496868523e-01, 7.2658477969e-02, 3.0450989490e-02]),
                ),
                3.25: (
                    normalize(np.array(
                        [1.2701060417e-01, 2.2019913652e-01, 2.3994006212e-01, 1.9343201753e-01,
                         1.2286302528e-01, 6.2875418673e-02, 2.5695118634e-02, 7.9846170721e-03])),
                    np.array([5.3072750214e+00, 2.3722444636e+00, 1.1857359413e+00, 6.0994845127e-01,
                              3.1267011371e-01, 1.5582184315e-01, 7.3297207276e-02, 3.0944804287e-02]),
                ),
                3.2919: (
                    normalize(np.array(
                        [1.2733648229e-01, 2.1850691216e-01, 2.3818830817e-01, 1.9304233249e-01,
                         1.2369839778e-01, 6.4087674935e-02, 2.6642716860e-02, 8.4971753106e-03])),
                    np.array([5.3505690278e+00, 2.3887751929e+00, 1.1939511087e+00, 6.1450124512e-01,
                              3.1543335755e-01, 1.5762452301e-01, 7.4524409464e-02, 3.1786513956e-02]),
                ),
                3.3341: (
                    normalize(np.array(
                        [1.2799643719e-01, 2.1731935917e-01, 2.3670472353e-01, 1.9253786842e-01,
                         1.2419937761e-01, 6.4966184608e-02, 2.7370560432e-02, 8.9054890445e-03])),
                    np.array([5.3869449565e+00, 2.4004119651e+00, 1.1987824572e+00, 6.1678421393e-01,
                              3.1670782412e-01, 1.5847329059e-01, 7.5158212398e-02, 3.2274417877e-02]),
                ),
                3.3768: (
                    normalize(np.array(
                        [1.2866159921e-01, 2.1614688232e-01, 2.3523027000e-01, 1.9201888477e-01,
                         1.2467917363e-01, 6.5835745541e-02, 2.8102986218e-02, 9.3244583032e-03])),
                    np.array([5.4230520760e+00, 2.4119608692e+00, 1.2035967823e+00, 6.1906960554e-01,
                              3.1798796705e-01, 1.5932524183e-01, 7.5792493503e-02, 3.2761070349e-02]),
                ),
                3.42: (
                    normalize(np.array(
                        [1.2902805054e-01, 2.1453902575e-01, 2.3351320573e-01, 1.9156101126e-01,
                         1.2541802907e-01, 6.7001375408e-02, 2.9063202822e-02, 9.8760994217e-03])),
                    np.array([5.4651499014e+00, 2.4280664044e+00, 1.2116959163e+00, 6.2359337095e-01,
                              3.2074158263e-01, 1.6111931880e-01, 7.7009246231e-02, 3.3591009013e-02]),
                ),
                3.4638: (
                    normalize(np.array(
                        [1.2971175144e-01, 2.1340109685e-01, 2.3205445759e-01, 1.9100915108e-01,
                         1.2585069068e-01, 6.7849153549e-02, 2.9805221271e-02, 1.0318477543e-02])),
                    np.array([5.5006869582e+00, 2.4394191885e+00, 1.2164719178e+00, 6.2587992317e-01,
                              3.2202745177e-01, 1.6197362082e-01, 7.7641916389e-02, 3.4073117319e-02]),
                ),
                3.5: (
                    normalize(np.array(
                        [1.3016826459e-01, 2.1232432558e-01, 2.3077889993e-01, 1.9057170043e-01,
                         1.2628553254e-01, 6.8642618597e-02, 3.0495284596e-02, 1.0733373732e-02])),
                    np.array([5.5317811787e+00, 2.4502987254e+00, 1.2215492029e+00, 6.2855399011e-01,
                              3.2360965857e-01, 1.6301135939e-01, 7.8368528425e-02, 3.4590307876e-02]),
                ),
                3.5531: (
                    normalize(np.array(
                        [1.3081355100e-01, 2.1074761577e-01, 2.2891922896e-01, 1.8992351936e-01,
                         1.2690961251e-01, 6.9804614033e-02, 3.1520639610e-02, 1.1361218753e-02])),
                    np.array([5.5770749140e+00, 2.4663773779e+00, 1.2291912268e+00, 6.3263857600e-01,
                              3.2604361167e-01, 1.6460482672e-01, 7.9475501922e-02, 3.5370162417e-02]),
                ),
                3.5986: (
                    normalize(np.array(
                        [1.3151766132e-01, 2.0965606176e-01, 2.2748713931e-01, 1.8933207227e-01,
                         1.2727829754e-01, 7.0619296406e-02, 3.2273206570e-02, 1.1836264825e-02])),
                    np.array([5.6117895850e+00, 2.4774492059e+00, 1.2339144389e+00, 6.3492975906e-01,
                              3.2733996944e-01, 1.6546380107e-01, 8.0106621192e-02, 3.5846419312e-02]),
                ),
                3.6447: (
                    normalize(np.array(
                        [1.3222580377e-01, 2.0857796798e-01, 2.2606587608e-01, 1.8873025150e-01,
                         1.2762755793e-01, 7.1423411331e-02, 3.3027928362e-02, 1.2321203059e-02])),
                    np.array([5.6462673455e+00, 2.4884390403e+00, 1.2386176037e+00, 6.3721840849e-01,
                              3.2863793760e-01, 1.6632342573e-01, 8.0736912620e-02, 3.6321059310e-02]),
                ),
                3.6914: (
                    normalize(np.array(
                        [1.3267758456e-01, 2.0712564957e-01, 2.2442017772e-01, 1.8815598473e-01,
                         1.2818508342e-01, 7.2488530455e-02, 3.4000611299e-02, 1.2946378238e-02])),
                    np.array([5.6863799027e+00, 2.5037117586e+00, 1.2464284794e+00, 6.4162810931e-01,
                              3.3133251485e-01, 1.6807680423e-01, 8.1921371278e-02, 3.7124418056e-02]),
                ),
                3.7387: (
                    normalize(np.array(
                        [1.3339840897e-01, 2.0607980360e-01, 2.2302026803e-01, 1.8753144114e-01,
                         1.2849175596e-01, 7.3267238041e-02, 3.4758950112e-02, 1.3452134142e-02])),
                    np.array([5.7203078318e+00, 2.5145055450e+00, 1.2510872959e+00, 6.4391253102e-01,
                              3.3263265568e-01, 1.6893680760e-01, 8.2549240519e-02, 3.7594715517e-02]),
                ),
                3.7866: (
                    normalize(np.array(
                        [1.3412178418e-01, 2.0504596525e-01, 2.2163082144e-01, 1.8689909833e-01,
                         1.2878115121e-01, 7.4035216067e-02, 3.5518757046e-02, 1.3967206474e-02])),
                    np.array([5.7539816090e+00, 2.5252178208e+00, 1.2557331372e+00, 6.4620073826e-01,
                              3.3393800120e-01, 1.6979963927e-01, 8.3177488082e-02, 3.8063820243e-02]),
                ),
                3.8351: (
                    normalize(np.array(
                        [1.3460390431e-01, 2.0366197801e-01, 2.2002450686e-01, 1.8628321788e-01,
                         1.2926006384e-01, 7.5049623156e-02, 3.6491304263e-02, 1.4625401683e-02])),
                    np.array([5.7931547698e+00, 2.5401206480e+00, 1.2634348228e+00, 6.5057712547e-01,
                              3.3661647319e-01, 1.7153978813e-01, 8.4349658956e-02, 3.8856134685e-02]),
                ),
                3.8842: (
                    normalize(np.array(
                        [1.3533702971e-01, 2.0266205126e-01, 2.1866150286e-01, 1.8563269939e-01,
                         1.2950784355e-01, 7.5789367869e-02, 3.7250540544e-02, 1.5158964816e-02])),
                    np.array([5.8262552007e+00, 2.5506186627e+00, 1.2680182839e+00, 6.5284896618e-01,
                              3.3791752114e-01, 1.7239967629e-01, 8.4974062843e-02, 3.9320781435e-02]),
                ),
                3.934: (
                    normalize(np.array(
                        [1.3607215315e-01, 2.0167065554e-01, 2.1730631112e-01, 1.8497601363e-01,
                         1.2974132436e-01, 7.6519953644e-02, 3.8011570389e-02, 1.5702018164e-02])),
                    np.array([5.8591672667e+00, 2.5610652165e+00, 1.2726068331e+00, 6.5513472252e-01,
                              3.3922844740e-01, 1.7326447046e-01, 8.5599813763e-02, 3.9784644711e-02]),
                ),
                4.0: (
                    normalize(np.array(
                        [1.3694710039e-01, 2.0026363685e-01, 2.1546940075e-01, 1.8411409448e-01,
                         1.3009353257e-01, 7.7550170771e-02, 3.9085175186e-02, 1.6476889008e-02])),
                    np.array([5.9039474596e+00, 2.5761950499e+00, 1.2797365614e+00, 6.5891135101e-01,
                              3.4146425325e-01, 1.7472839096e-01, 8.6623475263e-02, 4.0512203337e-02]),
                ),
                4.05: (
                    normalize(np.array(
                        [1.3743950476e-01, 1.9898422617e-01, 2.1395037097e-01, 1.8346711521e-01,
                         1.3046864904e-01, 7.8481374567e-02, 4.0039593851e-02, 1.7169165422e-02])),
                    np.array([5.9412691724e+00, 2.5904839075e+00, 1.2872711569e+00, 6.6325072090e-01,
                              3.4413544123e-01, 1.7646247656e-01, 8.7786083530e-02, 4.1293126645e-02]),
                ),
                4.1: (
                    normalize(np.array(
                        [1.3814131915e-01, 1.9805042716e-01, 2.1266858195e-01, 1.8281503598e-01,
                         1.3065039974e-01, 7.9160512254e-02, 4.0784484191e-02, 1.7729239569e-02])),
                    np.array([5.9725637488e+00, 2.6005638338e+00, 1.2918465979e+00, 6.6559718348e-01,
                              3.4550086819e-01, 1.7735970924e-01, 8.8425129083e-02, 4.1757924234e-02]),
                ),
                4.15: (
                    normalize(np.array(
                        [1.3882999623e-01, 1.9713205616e-01, 2.1140947850e-01, 1.8216739252e-01,
                         1.3081970225e-01, 7.9825570057e-02, 4.1523290028e-02, 1.8292514263e-02])),
                    np.array([6.0033957235e+00, 2.6105484564e+00, 1.2964199751e+00, 6.6796112562e-01,
                              3.4688206829e-01, 1.7826618959e-01, 8.9067950104e-02, 4.2223338235e-02]),
                ),
                4.2: (
                    normalize(np.array(
                        [1.3950658612e-01, 1.9622781819e-01, 2.1017106965e-01, 1.8152393997e-01,
                         1.3097756596e-01, 8.0477127567e-02, 4.2256740284e-02, 1.8859152272e-02])),
                    np.array([6.0337703820e+00, 2.6204375407e+00, 1.3009963488e+00, 6.7034711996e-01,
                              3.4828197221e-01, 1.7918406963e-01, 8.9715939716e-02, 4.2689753775e-02]),
                ),
                4.25: (
                    normalize(np.array(
                        [1.4017122740e-01, 1.9533795046e-01, 2.0895353597e-01, 1.8088496665e-01,
                         1.3112420572e-01, 8.1115188128e-02, 4.2984289685e-02, 1.9428635998e-02])),
                    np.array([6.0637029897e+00, 2.6302338315e+00, 1.3055729160e+00, 6.7275202416e-01,
                              3.4969833584e-01, 1.8011185298e-01, 9.0368258830e-02, 4.3156944632e-02]),
                ),
                4.3: (
                    normalize(np.array(
                        [1.4063004265e-01, 1.9417129368e-01, 2.0755664940e-01, 1.8024164582e-01,
                         1.3140766034e-01, 8.1951893613e-02, 4.3900423228e-02, 2.0140391261e-02])),
                    np.array([6.0985050124e+00, 2.6438602450e+00, 1.3129990707e+00, 6.7712028743e-01,
                              3.5241104884e-01, 1.8187014046e-01, 9.1537193370e-02, 4.3933148878e-02]),
                ),
                4.35: (
                    normalize(np.array(
                        [1.4127613791e-01, 1.9331626970e-01, 2.0638252545e-01, 1.7960995091e-01,
                         1.3152854061e-01, 8.2558865236e-02, 4.4613107558e-02, 2.0714602629e-02])),
                    np.array([6.1275466314e+00, 2.6534418776e+00, 1.3175514965e+00, 6.7954571272e-01,
                              3.5384803887e-01, 1.8280972020e-01, 9.2193848990e-02, 4.4400088457e-02]),
                ),
                4.4: (
                    normalize(np.array(
                        [1.4191108091e-01, 1.9247432817e-01, 2.0522791147e-01, 1.7898294225e-01,
                         1.3163926377e-01, 8.3153470661e-02, 4.5320052670e-02, 2.1290950107e-02])),
                    np.array([6.1561876051e+00, 2.6629418326e+00, 1.3221065389e+00, 6.8198995559e-01,
                              3.5530187053e-01, 1.8375991837e-01, 9.2855259694e-02, 4.4867969482e-02]),
                ),
                4.45: (
                    normalize(np.array(
                        [1.4253521569e-01, 1.9164600728e-01, 2.0409288793e-01, 1.7836105059e-01,
                         1.3174043098e-01, 8.3735312884e-02, 4.6020326633e-02, 2.1868768030e-02])),
                    np.array([6.1844361756e+00, 2.6723578935e+00, 1.3266590243e+00, 6.8444875649e-01,
                              3.5676882766e-01, 1.8471799718e-01, 9.3520037749e-02, 4.5336396183e-02]),
                ),
                4.5: (
                    normalize(np.array(
                        [1.4314911194e-01, 1.9083111773e-01, 2.0297686990e-01, 1.7774426074e-01,
                         1.3183239565e-01, 8.4304582286e-02, 4.6713844304e-02, 2.2447817442e-02])),
                    np.array([6.2122968365e+00, 2.6816887224e+00, 1.3312077455e+00, 6.8692132558e-01,
                              3.5824836903e-01, 1.8568368595e-01, 9.4188107662e-02, 4.5805364407e-02]),
                ),
                4.55: (
                    normalize(np.array(
                        [1.4375247357e-01, 1.9002879414e-01, 2.0187981109e-01, 1.7713289096e-01,
                         1.3191579104e-01, 8.4861885438e-02, 4.7400572989e-02, 2.3027780780e-02])),
                    np.array([6.2397955765e+00, 2.6909475208e+00, 1.3357577159e+00, 6.8940904135e-01,
                              3.5974103450e-01, 1.8665694400e-01, 9.4859269505e-02, 4.6274845222e-02]),
                ),
                4.6: (
                    normalize(np.array(
                        [1.4434726407e-01, 1.8924091822e-01, 2.0080061437e-01, 1.7652635766e-01,
                         1.3199010761e-01, 8.5406157681e-02, 4.8080125047e-02, 2.3608455343e-02])),
                    np.array([6.2668976804e+00, 2.7001028174e+00, 1.3402918635e+00, 6.9190417294e-01,
                              3.6124239088e-01, 1.8763619437e-01, 9.5533311273e-02, 4.6744711397e-02]),
                ),
                4.65: (
                    normalize(np.array(
                        [1.4493209962e-01, 1.8846526130e-01, 1.9973970968e-01, 1.7592536106e-01,
                         1.3205651551e-01, 8.5938858771e-02, 4.8752687460e-02, 2.4189506595e-02])),
                    np.array([6.2936571746e+00, 2.7091891886e+00, 1.3448260611e+00, 6.9441276088e-01,
                              3.6275560157e-01, 1.8862225705e-01, 9.6210094782e-02, 4.7215008544e-02]),
                ),
                4.7: (
                    normalize(np.array(
                        [1.4550782913e-01, 1.8770237251e-01, 1.9869651560e-01, 1.7532972000e-01,
                         1.3211500208e-01, 8.6459766553e-02, 4.9418082274e-02, 2.4770711849e-02])),
                    np.array([6.3200678213e+00, 2.7181966089e+00, 1.3493539676e+00, 6.9693157531e-01,
                              3.6427878302e-01, 1.8961438874e-01, 9.6889453185e-02, 4.7685681810e-02]),
                ),
                4.75: (
                    normalize(np.array(
                        [1.4607472338e-01, 1.8695199240e-01, 1.9767073904e-01, 1.7473952448e-01,
                         1.3216591235e-01, 8.6969075322e-02, 5.0076233986e-02, 2.5351799056e-02])),
                    np.array([6.3461387605e+00, 2.7271270965e+00, 1.3538754350e+00, 6.9945994087e-01,
                              3.6581130849e-01, 1.9061220814e-01, 9.7571198164e-02, 4.8156655154e-02]),
                ),
                4.8: (
                    normalize(np.array(
                        [1.4663306270e-01, 1.8621398966e-01, 1.9666197817e-01, 1.7415474984e-01,
                         1.3220954079e-01, 8.7467004128e-02, 5.0727101268e-02, 2.5932573434e-02])),
                    np.array([6.3718775442e+00, 2.7359812710e+00, 1.3583897346e+00, 7.0199724035e-01,
                              3.6735279247e-01, 1.9161549216e-01, 9.8255252369e-02, 4.8627945118e-02]),
                ),
                4.85: (
                    normalize(np.array(
                        [1.4718305876e-01, 1.8548812951e-01, 1.9566994657e-01, 1.7357538619e-01,
                         1.3224620421e-01, 8.7953786617e-02, 5.1370654158e-02, 2.6512833994e-02])),
                    np.array([6.3972938359e+00, 2.7447614366e+00, 1.3628968054e+00, 7.0454303445e-01,
                              3.6890289526e-01, 1.9262403383e-01, 9.8941538633e-02, 4.9099540972e-02]),
                ),
                4.9: (
                    normalize(np.array(
                        [1.4787578477e-01, 1.8500273201e-01, 1.9486499743e-01, 1.7303065857e-01,
                         1.3217604168e-01, 8.8262094229e-02, 5.1835142265e-02, 2.6952549049e-02])),
                    np.array([6.4174926090e+00, 2.7498736510e+00, 1.3647478957e+00, 7.0527616211e-01,
                              3.6925970629e-01, 1.9286561074e-01, 9.9145529778e-02, 4.9277244190e-02]),
                ),
                4.95: (
                    normalize(np.array(
                        [1.4840686810e-01, 1.8429627763e-01, 1.9390321581e-01, 1.7246344990e-01,
                         1.3220269725e-01, 8.8730341727e-02, 5.2465803900e-02, 2.7531345691e-02])),
                    np.array([6.4423117526e+00, 2.7585318801e+00, 1.3692536528e+00, 7.0784565392e-01,
                              3.7083081157e-01, 1.9388701866e-01, 9.9837682060e-02, 4.9750267331e-02]),
                ),
                5.0: (
                    normalize(np.array(
                        [1.4893040718e-01, 1.8360156508e-01, 1.9295722327e-01, 1.7190168791e-01,
                         1.3222308080e-01, 8.9187978707e-02, 5.3089042683e-02, 2.8109014380e-02])),
                    np.array([6.4668301219e+00, 2.7671189253e+00, 1.3737505350e+00, 7.1042151625e-01,
                              3.7240890404e-01, 1.9491271849e-01, 1.0053159257e-01, 5.0223457245e-02]),
                ),
                5.05: (
                    normalize(np.array(
                        [1.4944667722e-01, 1.8291839601e-01, 1.9202673455e-01, 1.7134536994e-01,
                         1.3223741709e-01, 8.9635190072e-02, 5.3704841970e-02, 2.8685373143e-02])),
                    np.array([6.4910531952e+00, 2.7756352970e+00, 1.3782379012e+00, 7.1300303203e-01,
                              3.7399350710e-01, 1.9594246055e-01, 1.0122715926e-01, 5.0696793576e-02]),
                ),
                5.1: (
                    normalize(np.array(
                        [1.4995588720e-01, 1.8224648988e-01, 1.9111144970e-01, 1.7079451308e-01,
                         1.3224596599e-01, 9.0072205091e-02, 5.4313227022e-02, 2.9260262037e-02])),
                    np.array([6.5149877749e+00, 2.7840827366e+00, 1.3827159415e+00, 7.1558994388e-01,
                              3.7558438513e-01, 1.9697611841e-01, 1.0192433813e-01, 5.1170274304e-02]),
                ),
                5.15: (
                    normalize(np.array(
                        [1.5045826231e-01, 1.8158592107e-01, 1.9021108561e-01, 1.7024907885e-01,
                         1.3224894267e-01, 9.0499157163e-02, 5.4914097519e-02, 2.9833454806e-02])),
                    np.array([6.5386405092e+00, 2.7924605817e+00, 1.3871827851e+00, 7.1818098001e-01,
                              3.7718070380e-01, 1.9801317081e-01, 1.0262286927e-01, 5.1643827102e-02]),
                ),
                5.2: (
                    normalize(np.array(
                        [1.5108783039e-01, 1.8114109431e-01, 1.8948365541e-01, 1.6974375886e-01,
                         1.3216361208e-01, 9.0766518895e-02, 5.5346570808e-02, 3.0266959244e-02])),
                    np.array([6.5572729654e+00, 2.7973108806e+00, 1.3890742745e+00, 7.1900656403e-01,
                              3.7761244691e-01, 1.9830064325e-01, 1.0284916031e-01, 5.1828707061e-02]),
                ),
                5.25: (
                    normalize(np.array(
                        [1.5157434023e-01, 1.8049838719e-01, 1.8861022513e-01, 1.6920985580e-01,
                         1.3215881634e-01, 9.1176966088e-02, 5.5934525971e-02, 3.0836883244e-02])),
                    np.array([6.5804079016e+00, 2.8055790877e+00, 1.3935348204e+00, 7.2161454794e-01,
                              3.7922483454e-01, 1.9934766477e-01, 1.0355224469e-01, 5.2303348888e-02]),
                ),
                5.3: (
                    normalize(np.array(
                        [1.5205461702e-01, 1.7986632663e-01, 1.8775080586e-01, 1.6868142788e-01,
                         1.3214907986e-01, 9.1577962112e-02, 5.6515067618e-02, 3.1404713015e-02])),
                    np.array([6.6032802197e+00, 2.8137822840e+00, 1.3979848181e+00, 7.2422599713e-01,
                              3.8084193154e-01, 2.0039765363e-01, 1.0425649022e-01, 5.2778009299e-02]),
                ),
                5.35: (
                    normalize(np.array(
                        [1.5252885137e-01, 1.7924481658e-01, 1.8690518818e-01, 1.6815835823e-01,
                         1.3213455371e-01, 9.1969708145e-02, 5.7088218744e-02, 3.1970305039e-02])),
                    np.array([6.6258952196e+00, 2.8219208452e+00, 1.4024232940e+00, 7.2684017409e-01,
                              3.8246339234e-01, 2.0145043338e-01, 1.0496182129e-01, 5.3252680313e-02]),
                ),
                5.4: (
                    normalize(np.array(
                        [1.5312129380e-01, 1.7882485359e-01, 1.8622406652e-01, 1.6767774567e-01,
                         1.3204200155e-01, 9.2212894297e-02, 5.7499800422e-02, 3.2397344152e-02])),
                    np.array([6.6435984787e+00, 2.8266125726e+00, 1.4043328669e+00, 7.2771617751e-01,
                              3.8293683423e-01, 2.0176363988e-01, 1.0520070466e-01, 5.3441699780e-02]),
                ),
                5.45: (
                    normalize(np.array(
                        [1.5358143976e-01, 1.7822023923e-01, 1.8540329531e-01, 1.6716588171e-01,
                         1.3202094969e-01, 9.2589350309e-02, 5.8060144426e-02, 3.2958699564e-02])),
                    np.array([6.6657398728e+00, 2.8346476330e+00, 1.4087622690e+00, 7.3034345739e-01,
                              3.8457135871e-01, 2.0282459229e-01, 1.0590974583e-01, 5.3917218099e-02]),
                ),
                5.5: (
                    normalize(np.array(
                        [1.5403615657e-01, 1.7762565618e-01, 1.8459557299e-01, 1.6665925302e-01,
                         1.3199564410e-01, 9.2957052182e-02, 5.8613193419e-02, 3.3517471530e-02])),
                    np.array([6.6876376222e+00, 2.8426201990e+00, 1.4131792827e+00, 7.3297225753e-01,
                              3.8620932308e-01, 2.0388781062e-01, 1.0661965494e-01, 5.4392699543e-02]),
                ),
                5.55: (
                    normalize(np.array(
                        [1.5460269517e-01, 1.7722317759e-01, 1.8394640283e-01, 1.6619651369e-01,
                         1.3189902125e-01, 9.3183846940e-02, 5.9009584008e-02, 3.3938758520e-02])),
                    np.array([6.7047056313e+00, 2.8472033093e+00, 1.4151000240e+00, 7.3388179399e-01,
                              3.8671071567e-01, 2.0421831939e-01, 1.0686702515e-01, 5.4584545449e-02]),
                ),
                5.6: (
                    normalize(np.array(
                        [1.5504445301e-01, 1.7664486193e-01, 1.8316210915e-01, 1.6570075472e-01,
                         1.3186794308e-01, 9.3537156529e-02, 5.9550007813e-02, 3.4492713772e-02])),
                    np.array([6.7261618345e+00, 2.8550771414e+00, 1.4195061473e+00, 7.3652118718e-01,
                              3.8835990482e-01, 2.0528862371e-01, 1.0758013100e-01, 5.5060725960e-02]),
                ),
                5.65: (
                    normalize(np.array(
                        [1.5548129491e-01, 1.7607613982e-01, 1.8239017051e-01, 1.6521011337e-01,
                         1.3183305414e-01, 9.3882206189e-02, 6.0083245037e-02, 3.5043776023e-02])),
                    np.array([6.7473883550e+00, 2.8628905379e+00, 1.4238989423e+00, 7.3916085110e-01,
                              3.9001164796e-01, 2.0636069009e-01, 1.0829387537e-01, 5.5536811428e-02]),
                ),
                5.7: (
                    normalize(np.array(
                        [1.5602402215e-01, 1.7569031595e-01, 1.8177104973e-01, 1.6476447528e-01,
                         1.3173309469e-01, 9.4093601554e-02, 6.0464751904e-02, 3.5458688740e-02])),
                    np.array([6.7638577215e+00, 2.8673700532e+00, 1.4258283889e+00, 7.4010042203e-01,
                              3.9053824448e-01, 2.0670683390e-01, 1.0854892099e-01, 5.5731215856e-02]),
                ),
                5.75: (
                    normalize(np.array(
                        [1.5644895111e-01, 1.7513717816e-01, 1.8102118084e-01, 1.6428435785e-01,
                         1.3169313242e-01, 9.4425120522e-02, 6.0985616608e-02, 3.6004462494e-02])),
                    np.array([6.7846707643e+00, 2.8750891342e+00, 1.4302089362e+00, 7.4274865381e-01,
                              3.9219968454e-01, 2.0778509582e-01, 1.0926545994e-01, 5.6207897757e-02]),
                ),
                5.8: (
                    normalize(np.array(
                        [1.5697631729e-01, 1.7476140846e-01, 1.8042071569e-01, 1.6384964454e-01,
                         1.3159189647e-01, 9.4627365618e-02, 6.1357733872e-02, 3.6414918051e-02])),
                    np.array([6.8007653656e+00, 2.8795048699e+00, 1.4321443042e+00, 7.4370762188e-01,
                              3.9274251425e-01, 2.0814120269e-01, 1.0952544023e-01, 5.6404017685e-02]),
                ),
                5.85: (
                    normalize(np.array(
                        [1.5739022615e-01, 1.7422343068e-01, 1.7969219487e-01, 1.6337990362e-01,
                         1.3154705001e-01, 9.4945776540e-02, 6.1866360171e-02, 3.6955057961e-02])),
                    np.array([6.8211710081e+00, 2.8871288624e+00, 1.4365100224e+00, 7.4636184241e-01,
                              3.9441192685e-01, 2.0922478862e-01, 1.1024432130e-01, 5.6881096885e-02]),
                ),
                5.9: (
                    normalize(np.array(
                        [1.5790296976e-01, 1.7385764451e-01, 1.7910966823e-01, 1.6295596744e-01,
                         1.3144448516e-01, 9.5139146782e-02, 6.2229236912e-02, 3.7360881205e-02])),
                    np.array([6.8369064228e+00, 2.8914824132e+00, 1.4384497624e+00, 7.4733808490e-01,
                              3.9496955108e-01, 2.0959024738e-01, 1.1050900082e-01, 5.7078875859e-02]),
                ),
                5.95: (
                    normalize(np.array(
                        [1.5830628284e-01, 1.7333403922e-01, 1.7840120614e-01, 1.6249657343e-01,
                         1.3139536393e-01, 9.5445167028e-02, 6.2726053960e-02, 3.7895313461e-02])),
                    np.array([6.8569277780e+00, 2.8990188290e+00, 1.4428044356e+00, 7.5000006238e-01,
                              3.9664753279e-01, 2.1067939312e-01, 1.1123035490e-01, 5.7556412573e-02]),
                ),
                6.0: (
                    normalize(np.array(
                        [1.5880521368e-01, 1.7297821995e-01, 1.7783632507e-01, 1.6208278191e-01,
                         1.3129171313e-01, 9.5629918791e-02, 6.3079629293e-02, 3.8296198180e-02])),
                    np.array([6.8723161958e+00, 2.9033098640e+00, 1.4447443522e+00, 7.5099038564e-01,
                              3.9721783682e-01, 2.1105277388e-01, 1.1149891664e-01, 5.7755487715e-02]),
                ),
                6.05: (
                    normalize(np.array(
                        [1.5939559775e-01, 1.7278407263e-01, 1.7741040478e-01, 1.6171615928e-01,
                         1.3113690204e-01, 9.5698231024e-02, 6.3293866242e-02, 3.8564766241e-02])),
                    np.array([6.8830870746e+00, 2.9043823509e+00, 1.4442895579e+00, 7.5032092675e-01,
                              3.9668764320e-01, 2.1071541506e-01, 1.1131753335e-01, 5.7677360642e-02]),
                ),
                6.1: (
                    normalize(np.array(
                        [1.5997609374e-01, 1.7259040825e-01, 1.7699041544e-01, 1.6135533493e-01,
                         1.3098445878e-01, 9.5765845080e-02, 6.3506001321e-02, 3.8831442459e-02])),
                    np.array([6.8937094968e+00, 2.9054453777e+00, 1.4438482832e+00, 7.4966538550e-01,
                              3.9616776297e-01, 2.1038462988e-01, 1.1113970107e-01, 5.7600759349e-02]),
                ),
                6.15: (
                    normalize(np.array(
                        [1.6054907710e-01, 1.7239981886e-01, 1.7657699373e-01, 1.6099931987e-01,
                         1.3083326873e-01, 9.5831162046e-02, 6.3714756590e-02, 3.9095603066e-02])),
                    np.array([6.9041120896e+00, 2.9064473470e+00, 1.4433857740e+00, 7.4900586575e-01,
                              3.9564891406e-01, 2.1005557501e-01, 1.1096320980e-01, 5.7524850510e-02]),
                ),
                6.2: (
                    normalize(np.array(
                        [1.6111251741e-01, 1.7220980414e-01, 1.7616925702e-01, 1.6064872055e-01,
                         1.3068425360e-01, 9.5895920088e-02, 6.3921585515e-02, 3.9357941685e-02])),
                    np.array([6.9143786380e+00, 2.9074418623e+00, 1.4429367361e+00, 7.4836007803e-01,
                              3.9514041887e-01, 2.0973312872e-01, 1.1079025219e-01, 5.7450432616e-02]),
                ),
                6.25: (
                    normalize(np.array(
                        [1.6166734733e-01, 1.7202118758e-01, 1.7576732438e-01, 1.6030320406e-01,
                         1.3053710318e-01, 9.5959565440e-02, 6.4126031103e-02, 3.9618236920e-02])),
                    np.array([6.9244836718e+00, 2.9084116340e+00, 1.4424895480e+00, 7.4772189075e-01,
                              3.9463888435e-01, 2.0941544071e-01, 1.1061996530e-01, 5.7377190439e-02]),
                ),
                6.3: (
                    normalize(np.array(
                        [1.6221360762e-01, 1.7183334920e-01, 1.7537078997e-01, 1.5996309766e-01,
                         1.3039193090e-01, 9.6022373759e-02, 6.4328343365e-02, 3.9876507521e-02])),
                    np.array([6.9344335765e+00, 2.9093630458e+00, 1.4420501149e+00, 7.4709385141e-01,
                              3.9414526861e-01, 2.0910297790e-01, 1.1045247049e-01, 5.7305137553e-02]),
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
