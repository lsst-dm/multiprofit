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
import galsim as gs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize as spopt
import time
import traceback

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


# Take a source (HST) image, convolve with a target (HSC) PSF, shift, rotate, and scale in amplitude until
# they sort of match.
# TODO: Would any attempt to incorporate the HST PSF improve things?
# This obviously won't work well if imgsrc's PSF isn't much smaller than imgtarget's
def imgoffsetchisq(x, args, returnimg=False):
    img = gs.Convolve(
        args['imgsrc']*10**x[0],
        args['psf']).shift(x[1], x[2]).drawImage(nx=args['nx'], ny=args['ny'], scale=args['scale'])
    chisq = np.sum((img.array - args['imgtarget'])**2/args['vartarget'])
    if returnimg:
        return img
    return chisq


# Fit the transform for a single COSMOS F814W image to match the HSC-I band image
def fitcosmosgalaxytransform(ra, dec, imghst, imgpsfgs, cutouthsc, varhsc, scalehsc, scalehst, plot=False):
    # Use Sophie's code to make our own cutout for comparison to the catalog
    # TODO: Double check the origin of these images; I assume they're the rotated and rescaled v2.0 mosaic
    anglehst = None
    pathhst = os.path.join(os.path.sep, 'project', 'sr525', 'hstCosmosImages')
    corners = os.path.join(pathhst, 'tiles', 'corners.txt')
    with open(corners, 'r') as f:
        for line in f:
            itemsline = line.split(',')
            ra_min = float(itemsline[2])
            ra_max = float(itemsline[0])
            dec_min = float(itemsline[1])
            dec_max = float(itemsline[3])
            filename = os.path.join(pathhst, itemsline[4][38:])
            if ra_min < ra < ra_max and dec_min < dec < dec_max:
                with ap.io.fits.open(filename[:-1]) as h:
                    anglehst = h[0].header['ORIENTAT']*gs.degrees
                    break
    if anglehst is None:
        raise RuntimeError('Failed to find COSMOS-HST image orientation from corner file {}'.format(corners))

    scalefluxhst2hsc = np.sum(cutouthsc)/np.sum(imghst)
    args = {
        "imgsrc": gs.InterpolatedImage(gs.Image(imghst, scale=scalehst)).rotate(anglehst),
        "imgtarget": cutouthsc,
        "vartarget": varhsc,
        "psf": imgpsfgs,
        "nx": cutouthsc.shape[1],
        "ny": cutouthsc.shape[0],
        "scale": scalehsc,
    }
    result = spopt.minimize(imgoffsetchisq, [np.log10(scalefluxhst2hsc), 0, 0], method="Nelder-Mead",
                            args=args)
    print("Offsetchisq fit params:", result.x)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(np.log10(imgoffsetchisq(result.x, args, returnimg=True).array))
        ax[0].set_title("COSMOS-HST GalSim matched")
        ax[1].imshow(np.log10(args['imgtarget']))
        ax[1].set_title("COSMOS-HSC target")
        ax[2].imshow(np.log10(args['imgsrc'].drawImage().array))
        ax[2].set_title("COSMOS-HST GalSim original")

    return result, anglehst


def gethstexposures(idcosmosgs, rgcat):
    """
    Get HST exposures and PSF images from the GalSim catalog cutouts.

    :param idcosmosgs: Galaxy GalSim catalog ID
    :param rgcat: A COSMOS galsim.RealGalaxyCatalog
    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    band = rgcat.band[idcosmosgs]
    exposurespsfs = [
        (
            mpfobj.Exposure(
                band, np.float64(rgcat.getGalImage(idcosmosgs).array),
                sigmainverse=np.power(np.float64(rgcat.getNoiseProperties(idcosmosgs)[2]), -0.5)),
            rgcat.getPSF(idcosmosgs)
        ),
    ]
    return exposurespsfs


def gethst2hscexposures(
        cutouts, scalehsc, radec, imghst, psfhst, scalehst, results, realgalaxy,
        bands=None, typecutout='deblended', bandhst=None, plot=False, hst2hscmodel=None
    ):
    """
    Get HST exposures and PSF image from an HSC image-based model, convolved to the HSC PSF for this galaxy.

    :param cutouts: Dict; key=band: value=dict; key=cutout type: value=dict; key=image type: value=image
        As returned by multiprofit.datautils.gethsc.gethsccutout
    :param scalehsc: Float; HSC pixel scale in arcseconds (0.168)
    :param radec: Iterable/tuple; [0]=right ascension, [1]=declination in degrees
    :param imghst: ndarray; HST galaxy cutout
    :param psfhst: ndarray; HST PSF image
    :param scalehst: Float; HST pixel scale in arcseconds (0.03)
    :param results: Dict; fit results for HST. Must contain keys as such: ['hst']['fits']['galsim']
    :param realgalaxy: galsim.RealGalaxy; a pre-built RealGalaxy based on the HST image of this galaxy.
    :param bands: List of bands; currently strings.
    :param typecutout: String cutout type; one of 'blended' or 'deblended'.
        'Deblended' should contain only a single galaxy with neighbours subtracted.
    :param bandhst: HST filter (should be F814W)
    :param plot: Boolean; plot summary figure after matching?
    :param hst2hscmodel: String; the model specification (as in modelspecs) to use for src=='hst2hsc'.

    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    exposurespsfs = []
    metadatas = {}
    for band in bands:
        cutoutsband = cutouts[band][typecutout]
        imgpsf = cutoutsband['psf']
        imgpsfgs = gs.InterpolatedImage(gs.Image(imgpsf, scale=scalehsc))
        ny, nx = cutoutsband['img'].shape

        # The COSMOS GalSim catalog is in the original HST frame, which is rotated by
        # 10-12 degrees from RA/Dec axes; fit for this
        result, anglehst = fitcosmosgalaxytransform(
            radec[0], radec[1], imghst, imgpsfgs, cutoutsband['img'],
            cutoutsband['var'], scalehsc, scalehst, plot=plot
        )
        fluxscale = (10 ** result.x[0])
        metadata = {
            "lenhst2hsc": scalehst/scalehsc,
            "fluxscalehst2hsc": fluxscale,
            "anglehst2hsc": anglehst,
        }

        # Assuming that these images match, add HSC noise back in
        if hst2hscmodel is None:
            # TODO: Fix this as it's not working by default
            img = imgoffsetchisq(result.x, returnimg=True, imgref=cutoutsband['img'],
                                 psf=imgpsfgs, nx=nx, ny=ny, scale=scalehsc)
            img = gs.Convolve(img, imgpsfgs).drawImage(nx=nx, ny=ny, scale=scalehsc)*fluxscale
            # The PSF is now HSTPSF*HSCPSF, and "truth" is the deconvolved HST image/model
            psf = gs.Convolve(imgpsfgs, psfhst.rotate(anglehst*gs.degrees)).drawImage(
                nx=nx, ny=ny, scale=scalehsc
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
            # In fact there wasn't really any need to store a model since we
            # can reconstruct it, but let's go ahead and use the unpickled one
            paramsbest = fits[modeltouse]['fits'][-1]['paramsbestalltransformed']
            # Apply all of the same rotations and shifts directly to the model
            modeltouse = results['hst']['models'][modeltype]
            metadata["hst2hscmodel"] = modeltouse
            imghstshape = imghst.shape
            # I'm pretty sure that these should all be converted to arcsec units
            # TODO: Verify above
            for param, value in zip(modeltouse.getparameters(fixed=True), paramsbest):
                param.setvalue(value, transformed=True)
                valueset = param.getvalue(transformed=False)
                if param.name == "cenx":
                    valueset = (scalehst*(valueset - imghstshape[1]/2) + result.x[1] + nx/2)
                elif param.name == "ceny":
                    valueset = (scalehst*(valueset - imghstshape[0]/2) + result.x[2] + ny/2)
                elif param.name == "ang":
                    valueset += anglehst/gs.degrees
                elif param.name == "re":
                    valueset *= scalehst
                param.setvalue(valueset, transformed=False)
            exposuremodel = modeltouse.data.exposures[bandhst][0]
            exposuremodel.image = mpffit.ImageEmpty((ny, nx))
            # Save the GalSim model object
            modeltouse.evaluate(keepmodels=True, getlikelihood=False, drawimage=False)
            img = np.float64(gs.Convolve(exposuremodel.meta['model']['galsim'], imgpsfgs).drawImage(
                nx=nx, ny=ny, scale=scalehsc, method='no_pixel').array)*fluxscale
            psf = imgpsfgs

        noisetoadd = np.random.normal(scale=np.sqrt(cutoutsband['var']))
        img += noisetoadd

        if plot:
            fig2, ax2 = plt.subplots(nrows=2, ncols=3)
            ax2[0, 0].imshow(np.log10(cutoutsband['img']))
            ax2[0, 0].set_title("HSC {}".format(band))
            imghst2hsc = gs.Convolve(
                realgalaxy.rotate(anglehst * gs.radians).shift(
                    result.x[1], result.x[2]
                ), imgpsfgs).drawImage(
                nx=nx, ny=ny, scale=scalehsc)
            imghst2hsc += noisetoadd
            imgsplot = (img.array, "my naive"), (imghst2hsc.array, "GS RealGal")
            descpre = "HST {} - {}"
            for imgidx, (imgit, desc) in enumerate(imgsplot):
                ax2[1, 1 + imgidx].imshow(np.log10(imgit))
                ax2[1, 1 + imgidx].set_title(descpre.format(bandhst, desc))
                ax2[0, 1 + imgidx].imshow(np.log10(imgit))
                ax2[0, 1 + imgidx].set_title((descpre + " + noise").format(
                    bandhst, desc))
        # TODO: Use the mask in cutoutsband['mask'] (how?)
        exposurespsfs.append(
            (mpfobj.Exposure(band, img, sigmainverse=1.0/np.sqrt(cutoutsband['var'])), psf)
        )
        metadatas[band] = metadata
    return exposurespsfs, metadatas


def fitcosmosgalaxy(
        radec, idcosmosgs, srcs, rgcat=None, ccat=None, butler=None, skymap=None, hst2hscmodel=None,
        hscbands=['HSC-I'], scalehst=0.03, **kwargs
):
    """
    Fit a COSMOS galaxy using HST/HSC data.

    :param radec: RA/dec in degrees of the galaxy.
    :param idcosmosgs: ID of the COSMOS galaxy (in the GalSim catalog)
    :param srcs: Collection of strings; data sources to fit. Allowed values: 'hst', 'hsc', 'hst2hsc'.
    :param rgcat: A COSMOS galsim.RealGalaxyCatalog.
    :param ccat: A COSMOS galsim.scene.COSMOSGalaxyCatalog.
    :param butler: lsst.daf.persistence.butler (Gen2) pointed at an HSC repo.
    :param skymap: The appropriate HSC skymap from the butler.
    :param hscbands: Iterable of strings; list of HSC bands.
    :param scalehst: Float; HST image pixel scale in arcsec
    :param kwargs: Dict of key string argname: value arg to pass to fitgalaxyexposures
    :return: dict; key=src: value=dict of fit results, model object, etc.
    """
    needhsc = "hsc" in srcs or "hst2hsc" in srcs
    if 'hst' in srcs or needhsc:
        exposurespsfshst = gethstexposures(idcosmosgs, rgcat)
        imghst = exposurespsfshst[0][0].image
    if needhsc:
        if butler is None or skymap is None:
            raise ValueError('Must provide butler and skymap if fitting HSC or HST2HSC')
        import multiprofit.datautils.gethsc as gethsc
        # Determine the approximate HSC cutout size (larger than HST due to bigger PSF)
        # scalehsc should always be ~0.168
        sizeCutoutHSC = np.int(4 + np.ceil(np.max(imghst.shape)*scalehst/0.168))
        sizeCutoutHSC += np.int(sizeCutoutHSC % 2)
        cutouts, spherePoint, scalehsc, _ = gethsc.gethsccutout(
            butler, skymap, hscbands, radec, sizeinpix=sizeCutoutHSC, deblend=True, bandmatch='HSC-I',
            distmatchinasec=1.0)

    results = kwargs['results'] if 'results' in kwargs and kwargs['results'] is not None else {}
    for src in srcs:
        metadatas = None
        if src == 'hst':
            exposurespsfs = exposurespsfshst
        elif src == 'hsc' or src == 'hst2hsc':
            argsexposures = {
                'cutouts': cutouts,
                'scalehsc': scalehsc,
                'bands': hscbands,
            }

            if src == 'hsc':
                exposurespsfs = gethsc.gethscexposures(**argsexposures)
            elif src == 'hst2hsc':
                argsexposures['radec'] = radec
                argsexposures['imghst'] = imghst
                argsexposures['psfhst'] = exposurespsfshst[0][1]
                argsexposures['scalehst'] = scalehst
                argsexposures['results'] = results
                argsexposures['realgalaxy'] = ccat.makeGalaxy(index=idcosmosgs, gal_type="real")
                argsexposures['bandhst'] = exposurespsfshst[0][0].band
                argsexposures['hst2hscmodel'] = hst2hscmodel
                exposurespsfs, metadatas = gethst2hscexposures(**argsexposures)
            else:
                raise RuntimeError('Unknown HSC galaxy data source {}'.format(src))
        else:
            raise RuntimeError('Unknown galaxy data source {}'.format(src))
        bands = [rgcat.band[idcosmosgs]] if src == "hst" else hscbands
        fitname = 'COSMOS #{}'.format(idcosmosgs)
        kwargs['results'] = results[src] if src in results else None
        redo = {}
        for suffix in ['', 'psfs']:
            key = 'redo' + suffix
            if key in kwargs:
                redo[suffix] = kwargs[key]
                del kwargs[key]
        results[src] = mpffit.fitgalaxyexposures(
            exposurespsfs, bands, redo=redo[''], redopsfs=redo['psfs'], fitname=fitname, **kwargs)
        if metadatas is not None:
            if 'metadata' not in results[src]:
                results[src]['metadata'] = metadatas
            else:
                results[src]['metadata'].update(metadatas)
    return results


def main():
    parser = argparse.ArgumentParser(description='MultiProFit HST COSMOS galaxy modelling test')

    flags = {
        'catalogpath': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog path'},
        'catalogfile': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog filename'},
        'file': {'type': str, 'nargs': '?', 'default': None, 'help': 'Filename for input/output'},
        'fitfluxfracs': {'type': mpfutil.str2bool, 'default': False,
                         'help': 'Fit component flux fractions for galaxies instead of fluxes'},
        'fithsc': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HSC I band image'},
        'fithst': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image'},
        'fithst2hsc': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image convolved '
                                                                           'to HSC seeing'},
        'hscbands': {'type': str, 'nargs': '*', 'default': ['HSC-I'], 'help': 'HSC Bands to fit'},
        'hscrepo': {'type': str, 'default': '/datasets/hsc/repo/rerun/RC/w_2019_02/DM-16110/',
                    'help': 'Path to HSC processing repository'},
        'hst2hscmodel': {'type': str, 'default': None, 'help': 'HST model fit to use for mock HSC image'},
        'imgplotmaxs': {'type': float, 'nargs': '*', 'default': None,
                        'help': 'Max. flux for scaling single-band images. F814W first if fitting HST, '
                                'then HSC bands.'},
        'imgplotmaxmulti': {'type': float, 'default': None, 'help': 'Max. flux for scaling color images'},
        'indices': {'type': str, 'nargs': '*', 'default': None, 'help': 'Galaxy catalog index'},
        'modelspecfile': {'type': str, 'default': None, 'help': 'Model specification file'},
        'modellib': {'type': str, 'nargs': '?', 'default': 'scipy', 'help': 'Optimization libraries'},
        'modellibopts': {'type': str, 'nargs': '?', 'default': None, 'help': 'Model fitting options'},
        'nwrite': {'type': int, 'default': 5, 'help': 'Number of galaxies to fit before writing file'},
        #        'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
        'plot': {'type': mpfutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
        #        'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
        'redo': {'type': mpfutil.str2bool, 'default': True, 'help': 'Redo existing fits'},
        'redopsfs': {'type': mpfutil.str2bool, 'default': False, 'help': 'Redo existing PSF fits'},
        'weightsband': {'type': float, 'nargs': '*', 'default': None,
                        'help': 'Multiplicative weights for scaling images in multi-band RGB'},
        'write': {'type': mpfutil.str2bool, 'default': True, 'help': 'Write file?'},
    }

    for key, value in flags.items():
        if key in options:
            default = options[key]["default"]
        else:
            default = value['default']
        if 'help' in value:
            value['help'] += ' (default: ' + str(default) + ')'
        value["default"] = default
        parser.add_argument('--' + key, **value)

    args = parser.parse_args()
    args.catalogpath = os.path.expanduser(args.catalogpath)

    modelspecs = mpffit.getmodelspecs(
        None if args.modelspecfile is None else os.path.expanduser(args.modelspecfile))

    if args.file is not None:
        args.file = os.path.expanduser(args.file)
    if args.file is not None and os.path.isfile(args.file):
            with open(args.file, 'rb') as f:
                data = pickle.load(f)
    else:
        data = {}

    if args.plot:
        mpl.rcParams['image.origin'] = 'lower'

    srcs = ['hst'] if args.fithst else []
    if args.fithsc or args.fithst2hsc:
        import lsst.daf.persistence as dafPersist
        butler = dafPersist.Butler(args.hscrepo)
        skymap = butler.get("deepCoadd_skyMap", dataId={"tract": 9813})
    else:
        butler = None
        skymap = None
    if args.fithsc:
        srcs += ["hsc"]
    if args.fithst2hsc:
        srcs += ["hst2hsc"]
    bands = (['F814W'] if args.fithst else []) + (args.hscbands if (args.fithsc or args.fithst2hsc) else [])
    for nameattr in ['imgplotmaxs', 'weightsband']:
        attr = getattr(args, nameattr)
        if attr is not None:
            if len(bands) != len(attr):
                raise ValueError('len({}={})={} != len(bands={})={}'.format(
                    nameattr, attr, len(attr), bands, len(bands)))
            setattr(args, nameattr, {key: value for key, value in zip(bands, attr)})

    print('Loading COSMOS catalog at ' + os.path.join(args.catalogpath, args.catalogfile))
    try:
        rgcat = gs.RealGalaxyCatalog(args.catalogfile, dir=args.catalogpath)
    except RuntimeError as err:
        print("Failed to load RealGalaxyCatalog {} in directory {} due to {}".format(
            args.catalogfile, args.catalogpath, err))
        raise err
    if 'hst2hsc' in srcs:
        try:
            ccat = gs.COSMOSCatalog(args.catalogfile, dir=args.catalogpath)
        except RuntimeError as err:
            print("Failed to load COSMOSCatalog {} in directory {} due to {}".format(
                args.catalogfile, args.catalogpath, err))
            raise err
    else:
        ccat = None
    rgcfits = ap.io.fits.open(os.path.join(args.catalogpath, args.catalogfile))[1].data
    nfit = 0
    for index in args.indices:
        idrange = [np.int(x) for x in index.split(",")]
        for idnum in range(idrange[0], idrange[0 + (len(idrange) > 1)] + 1):
            print("Fitting COSMOS galaxy with ID: {}".format(idnum))
            timenow = time.time()
            try:
                np.random.seed(idnum)
                radec = rgcfits[idnum][1:3]
                scalehst = rgcfits[idnum]['PIXEL_SCALE']
                fits = fitcosmosgalaxy(
                    radec=radec, idcosmosgs=idnum, srcs=srcs, modelspecs=modelspecs, rgcat=rgcat, ccat=ccat,
                    butler=butler, skymap=skymap, plot=args.plot,
                    redo=args.redo, redopsfs=args.redopsfs, resetimages=True,
                    hst2hscmodel=args.hst2hscmodel, hscbands=args.hscbands, scalehst=scalehst,
                    modellib=args.modellib, results=data[idnum] if idnum in data else None,
                    imgplotmaxs=args.imgplotmaxs, imgplotmaxmulti=args.imgplotmaxmulti,
                    weightsband=args.weightsband, fitfluxfracs=args.fitfluxfracs)
                data[idnum] = fits
            except Exception as e:
                print("Error fitting id={}:".format(idnum))
                print(e)
                trace = traceback.format_exc()
                print(trace)
                if idnum not in data:
                    data[idnum] = {'error': e, 'trace': trace}
            print("Finished fitting COSMOS galaxy with ID: {} in total time of {:.2f} seconds".format(
                idnum, time.time() - timenow
            ))
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
    main()
