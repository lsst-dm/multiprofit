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
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize as spopt
import sys
import time
import traceback

import multiprofit.fitutils as mpffit
import multiprofit.gaussutils as mpfgauss
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
    "size_image":   {"default": [60]},
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
def offset_img_chisq(x, args, return_img=False):
    img = gs.Convolve(
        args['imgsrc']*10**x[0],
        args['psf']).shift(x[1], x[2]).drawImage(nx=args['nx'], ny=args['ny'], scale=args['scale'])
    chisq = np.sum((img.array - args['imgtarget'])**2/args['vartarget'])
    if return_img:
        return img
    return chisq


# Fit the transform for a single COSMOS F814W image to match the HSC-I band image
def fit_cosmos_galaxy_transform(
        ra, dec, img_hst, imgpsf_gs, cutout_hsc, varhsc, scale_pixel_hsc, scale_hst, plot=False):
    # Use Sophie's code to make our own cutout for comparison to the catalog
    # TODO: Double check the origin of these images; I assume they're the rotated and rescaled v2.0 mosaic
    angle_hst = None
    path_hst = os.path.join(os.path.sep, 'project', 'sr525', 'hstCosmosImages')
    corners = os.path.join(path_hst, 'tiles', 'corners.txt')
    with open(corners, 'r') as f:
        for line in f:
            itemsline = line.split(',')
            ra_min = float(itemsline[2])
            ra_max = float(itemsline[0])
            dec_min = float(itemsline[1])
            dec_max = float(itemsline[3])
            filename = os.path.join(path_hst, itemsline[4][38:])
            if ra_min < ra < ra_max and dec_min < dec < dec_max:
                with ap.io.fits.open(filename[:-1]) as h:
                    angle_hst = h[0].header['ORIENTAT']*gs.degrees
                    break
    if angle_hst is None:
        raise RuntimeError('Failed to find COSMOS-HST image orientation from corner file {}'.format(corners))

    scale_flux_hst2hsc = np.sum(cutout_hsc)/np.sum(img_hst)
    args = {
        "imgsrc": gs.InterpolatedImage(gs.Image(img_hst, scale=scale_hst)).rotate(angle_hst),
        "imgtarget": cutout_hsc,
        "vartarget": varhsc,
        "psf": imgpsf_gs,
        "nx": cutout_hsc.shape[1],
        "ny": cutout_hsc.shape[0],
        "scale": scale_pixel_hsc,
    }
    result = spopt.minimize(offset_img_chisq, [np.log10(scale_flux_hst2hsc), 0, 0], method="Nelder-Mead",
                            args=args)
    print("Offsetchisq fit params:", result.x)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(np.log10(offset_img_chisq(result.x, args, return_img=True).array))
        ax[0].set_title("COSMOS-HST GalSim matched")
        ax[1].imshow(np.log10(args['imgtarget']))
        ax[1].set_title("COSMOS-HSC target")
        ax[2].imshow(np.log10(args['imgsrc'].drawImage().array))
        ax[2].set_title("COSMOS-HST GalSim original")

    return result, angle_hst


def get_exposures_hst(id_cosmos_gs, rgcat):
    """
    Get HST exposures and PSF images from the GalSim catalog cutouts.

    :param id_cosmos_gs: Galaxy GalSim catalog ID
    :param rgcat: A COSMOS galsim.RealGalaxyCatalog
    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    band = rgcat.band[id_cosmos_gs]
    exposures_psfs = [
        (
            mpfobj.Exposure(
                band, np.float64(rgcat.getGalImage(id_cosmos_gs).array),
                error_inverse=np.array([[np.power(np.float64(
                    rgcat.getNoiseProperties(id_cosmos_gs)[2]), -0.5)]])),
            rgcat.getPSF(id_cosmos_gs)
        ),
    ]
    return exposures_psfs


def get_exposures_hst2hsc(
        cutouts, scale_pixel_hsc, radec, img_hst, psf_hst, scale_hst, results, realgalaxy,
        bands=None, typecutout='deblended', bandhst=None, plot=False, model_name_hst2hsc=None
    ):
    """
    Get HST exposures and PSF image from an HSC image-based model, convolved to the HSC PSF for this galaxy.

    :param cutouts: Dict; key=band: value=dict; key=cutout type: value=dict; key=image type: value=image
        As returned by multiprofit.datautils.gethsc.get_cutout_hsc
    :param scale_pixel_hsc: Float; HSC pixel scale in arcseconds (0.168)
    :param radec: Iterable/tuple; [0]=right ascension, [1]=declination in degrees
    :param img_hst: ndarray; HST galaxy cutout
    :param psf_hst: ndarray; HST PSF image
    :param scale_hst: Float; HST pixel scale in arcseconds (0.03)
    :param results: Dict; fit results for HST. Must contain keys as such: ['hst']['fits']['galsim']
    :param realgalaxy: galsim.RealGalaxy; a pre-built RealGalaxy based on the HST image of this galaxy.
    :param bands: List of bands; currently strings.
    :param typecutout: String cutout type; one of 'blended' or 'deblended'.
        'Deblended' should contain only a single galaxy with neighbours subtracted.
    :param bandhst: HST filter (should be F814W)
    :param plot: Boolean; plot summary figure after matching?
    :param model_name_hst2hsc: String; the model specification (as in modelspecs) to use for src=='hst2hsc'.

    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    exposures_psfs = []
    metadatas = {}
    for band in bands:
        cutouts_band = cutouts[band][typecutout]
        imgpsf = cutouts_band['psf']
        imgpsf_gs = gs.InterpolatedImage(gs.Image(imgpsf, scale=scale_pixel_hsc))
        ny, nx = cutouts_band['img'].shape

        # The COSMOS GalSim catalog is in the original HST frame, which is rotated by
        # 10-12 degrees from RA/Dec axes; fit for this
        result, angle_hst = fit_cosmos_galaxy_transform(
            radec[0], radec[1], img_hst, imgpsf_gs, cutouts_band['img'],
            cutouts_band['var'], scale_pixel_hsc, scale_hst, plot=plot
        )
        fluxscale = (10 ** result.x[0])
        metadata = {
            "lenhst2hsc": scale_hst/scale_pixel_hsc,
            "fluxscale_hst2hsc": fluxscale,
            "angle_hst2hsc": angle_hst,
        }

        # Assuming that these images match, add HSC noise back in
        if model_name_hst2hsc is None:
            # TODO: Fix this as it's not working by default
            img = offset_img_chisq(
                result.x, return_img=True, imgref=cutouts_band['img'], psf=imgpsf_gs,
                nx=nx, ny=ny, scale=scale_pixel_hsc)
            img = gs.Convolve(img, imgpsf_gs).drawImage(nx=nx, ny=ny, scale=scale_pixel_hsc)*fluxscale
            # The PSF is now HSTPSF*HSCPSF, and "truth" is the deconvolved HST image/model
            psf = gs.Convolve(imgpsf_gs, psf_hst.rotate(angle_hst*gs.degrees)).drawImage(
                nx=nx, ny=ny, scale=scale_pixel_hsc
            )
            psf /= np.sum(psf.array)
            psf = gs.InterpolatedImage(psf)
        else:
            fits = results['hst']['fits']['galsim']
            if model_name_hst2hsc == 'best':
                chisqredsmodel = {model: fit['fits'][-1]['chisqred'] for model, fit in fits.items()}
                model_to_use = min(chisqredsmodel, key=chisqredsmodel.get)
            else:
                model_to_use = model_name_hst2hsc
            modeltype = fits[model_to_use]['modeltype']
            # In fact there wasn't really any need to store a model since we
            # can reconstruct it, but let's go ahead and use the unpickled one
            params_best = fits[model_to_use]['fits'][-1]['params_bestalltransformed']
            # Apply all of the same rotations and shifts directly to the model
            model_to_use = results['hst']['models'][modeltype]
            metadata["model_name_hst2hsc"] = model_to_use
            img_hst_shape = img_hst.shape
            # I'm pretty sure that these should all be converted to arcsec units
            # TODO: Verify above
            params_ell = {}
            names_param_ell = ["sigma_x", "sigma_y", "rho"]
            for param, value in zip(model_to_use.get_parameters(fixed=True), params_best):
                param.set_value_transformed(value)
                value_to_set = param.get_value()
                if param.name == "cenx":
                    value_to_set = (scale_hst*(value_to_set - img_hst_shape[1]/2) + result.x[1] + nx/2)
                elif param.name == "ceny":
                    value_to_set = (scale_hst*(value_to_set - img_hst_shape[0]/2) + result.x[2] + ny/2)
                elif param.name == "rho":
                    params_ell[param.name] = param
                elif param.name == "sigma_x" or param.name == "sigma_y":
                    params_ell[param.name] = param
                    value_to_set *= scale_hst
                param.set_value(value_to_set)
            covar = np.array([
                p.get_value() for p in [params_ell[x] for x in names_param_ell]
            ])
            covar[2] *= covar[0]*covar[1]
            covar[1] *= covar[1]
            covar[0] *= covar[0]
            sigma, axrat, ang = mpfgauss.covar_to_ellipse(covar)
            ang += angle_hst/gs.degrees
            sigma_x, sigma_y, rho = mpfgauss.ellipse_to_covar(
                sigma, axrat, ang, return_as_matrix=False, return_as_params=True)
            for param, value in zip([params_ell[x] for x in names_param_ell], [sigma_x, sigma_y, rho]):
                param.set_value(value)
            image_model_exposure = model_to_use.data.exposures[bandhst][0]
            image_model_exposure.image = mpffit.ImageEmpty((ny, nx))
            # Save the GalSim model object
            # We pass engineopts to ensure that use_fast_gauss=False and null gsparams since those are
            # irrelevant here (they're used in the next line to actually draw the convolved model)
            model_to_use.evaluate(
                keep_models=True, get_likelihood=False, do_draw_image=False, engine='galsim',
                engineopts={'use_fast_gauss': False, "gsparams": None})
            img = np.float64(gs.Convolve(image_model_exposure.meta['model']['galsim'], imgpsf_gs).drawImage(
                nx=nx, ny=ny, scale=scale_pixel_hsc, method='no_pixel').array)*fluxscale
            psf = imgpsf_gs

        noisetoadd = np.random.normal(scale=np.sqrt(cutouts_band['var']))
        img += noisetoadd

        if plot:
            fig2, ax2 = plt.subplots(nrows=2, ncols=3)
            ax2[0, 0].imshow(np.log10(cutouts_band['img']))
            ax2[0, 0].set_title("HSC {}".format(band))
            img_hst2hsc = gs.Convolve(
                realgalaxy.rotate(angle_hst * gs.radians).shift(
                    result.x[1], result.x[2]
                ), imgpsf_gs).drawImage(
                nx=nx, ny=ny, scale=scale_pixel_hsc)
            img_hst2hsc += noisetoadd
            imgs_to_plot = (img.array, "my naive"), (img_hst2hsc.array, "GS RealGal")
            descpre = "HST {} - {}"
            for imgidx, (imgit, desc) in enumerate(imgs_to_plot):
                ax2[1, 1 + imgidx].imshow(np.log10(imgit))
                ax2[1, 1 + imgidx].set_title(descpre.format(bandhst, desc))
                ax2[0, 1 + imgidx].imshow(np.log10(imgit))
                ax2[0, 1 + imgidx].set_title((descpre + " + noise").format(
                    bandhst, desc))
        # TODO: Use the mask in cutouts_band['mask'] (how?)
        exposures_psfs.append(
            (mpfobj.Exposure(band, img, error_inverse=1.0/np.sqrt(cutouts_band['var'])), psf)
        )
        metadatas[band] = metadata
    return exposures_psfs, metadatas


def fit_galaxy_cosmos(
        radec, id_cosmos_gs, srcs, rgcat=None, ccat=None, butler=None, skymap=None, model_name_hst2hsc=None,
        bands_hsc=None, scale_hst=0.03, **kwargs
):
    """
    Fit a COSMOS galaxy using HST/HSC data.

    :param radec: RA/dec in degrees of the galaxy.
    :param id_cosmos_gs: ID of the COSMOS galaxy (in the GalSim catalog)
    :param srcs: Collection of strings; data sources to fit. Allowed values: 'hst', 'hsc', 'hst2hsc'.
    :param rgcat: A COSMOS galsim.RealGalaxyCatalog.
    :param ccat: A COSMOS galsim.scene.COSMOSGalaxyCatalog.
    :param butler: lsst.daf.persistence.butler (Gen2) pointed at an HSC repo.
    :param skymap: The appropriate HSC skymap from the butler.
    :param model_name_hst2hsc: String; name of the model to use for fitting mock HSC image
    :param bands_hsc: Iterable of strings; list of HSC bands. Default ['HSC-I'].
    :param scale_hst: Float; HST image pixel scale in arcsec
    :param kwargs: Dict of key string argname: value arg to pass to fit_galaxy_exposures
    :return: dict; key=src: value=dict of fit results, model object, etc.
    """
    need_hsc = "hsc" in srcs or "hst2hsc" in srcs
    if 'hst' in srcs or need_hsc:
        exposures_psfs_hst = get_exposures_hst(id_cosmos_gs, rgcat)
        img_hst = exposures_psfs_hst[0][0].image
    if need_hsc:
        if bands_hsc is None:
            bands_hsc = ['HSC-I']
        if butler is None or skymap is None:
            raise ValueError('Must provide butler and skymap if fitting HSC or HST2HSC')
        import multiprofit.datautils.gethsc as gethsc
        # Determine the approximate HSC cutout size (larger than HST due to bigger PSF)
        # scale_pixel_hsc should always be ~0.168
        sizeCutoutHSC = np.int(4 + np.ceil(np.max(img_hst.shape)*scale_hst/0.168))
        sizeCutoutHSC += np.int(sizeCutoutHSC % 2)
        cutouts, spherePoint, scale_pixel_hsc, _ = gethsc.get_cutout_hsc(
            butler, skymap, bands_hsc, radec, size_in_pix=sizeCutoutHSC, do_deblend=True, band_match='HSC-I',
            dist_match_in_asec=1.0)

    results = kwargs['results'] if 'results' in kwargs and kwargs['results'] is not None else {}
    fluxes = {}
    sizes = {}
    for src in srcs:
        metadatas = None
        to_fit = True
        if src == 'hst':
            exposures_psfs = exposures_psfs_hst
            flux = rgcat.stamp_flux[id_cosmos_gs]
            to_fit = flux > 0
            flux = np.log10(flux)
            size = np.log10(np.sqrt(img_hst.shape[0]*img_hst.shape[1]))
            if to_fit:
                # Shrink the cutout if it's way too big for its flux
                # This is based on outliers in the flux-size relation in the COSMOS 25.2 rgcat
                if (size > 2.33) & (size > (1.75 + 0.5*flux)):
                    print('Exposure with log flux={} & log size={} ({}x{}) too big; cropping to 30%'.format(
                        flux, size, img_hst.shape[0], img_hst.shape[1]))
                    for exposure_psf in exposures_psfs:
                        shape = exposure_psf[0].image.shape
                        exposure_psf[0].image = exposure_psf[0].image[
                                                   int(np.floor(shape[0]*0.35)):int(np.floor(shape[0]*0.65)),
                                                   int(np.floor(shape[1]*0.35)):int(np.floor(shape[1]*0.65))
                                                ]
                    img_hst = exposures_psfs_hst[0][0].image
            fluxes['F814W'] = flux
            sizes['F814W'] = size
        elif src == 'hsc' or src == 'hst2hsc':
            args_exposures = {
                'cutouts': cutouts,
                'scale_pixel_hsc': scale_pixel_hsc,
                'bands': bands_hsc,
            }

            if src == 'hsc':
                exposures_psfs = gethsc.get_exposures_hsc(**args_exposures)
            elif src == 'hst2hsc':
                args_exposures['radec'] = radec
                args_exposures['img_hst'] = img_hst
                args_exposures['psf_hst'] = exposures_psfs_hst[0][1]
                args_exposures['scale_hst'] = scale_hst
                args_exposures['results'] = results
                args_exposures['realgalaxy'] = ccat.makeGalaxy(index=id_cosmos_gs, gal_type="real")
                args_exposures['bandhst'] = exposures_psfs_hst[0][0].band
                args_exposures['model_name_hst2hsc'] = model_name_hst2hsc
                exposures_psfs, metadatas = get_exposures_hst2hsc(**args_exposures)
            else:
                raise RuntimeError('Unknown HSC galaxy data source {}'.format(src))
            for exposure, _ in exposures_psfs:
                flux = np.sum(exposure.image)
                to_fit = to_fit and flux > 0
                fluxes[exposure.band] = flux
                sizes[exposure.band] = np.log10(np.sqrt(exposure.image.shape[0]*exposure.image.shape[1]))
        else:
            raise RuntimeError('Unknown galaxy data source {}'.format(src))
        if to_fit:
            bands = [rgcat.band[id_cosmos_gs]] if src == "hst" else bands_hsc
            name_fit = 'COSMOS #{}'.format(id_cosmos_gs)
            kwargs['results'] = results[src] if src in results else None
            redo = {}
            for suffix in ['', 'psfs']:
                key = '_'.join(['redo'] + ([suffix] if suffix is not '' else []))
                if key in kwargs:
                    redo[suffix] = kwargs[key]
                    del kwargs[key]
            results[src] = mpffit.fit_galaxy_exposures(
                exposures_psfs, bands, redo=redo[''], redo_psfs=redo['psfs'], name_fit=name_fit, **kwargs)
            if metadatas is not None:
                if 'metadata' not in results[src]:
                    results[src]['metadata'] = metadatas
                else:
                    results[src]['metadata'].update(metadatas)
        else:
            if src not in results:
                results[src] = {}
            results[src]['error'] = 'Skipping {} fit for {} with unreasonable fluxes={} sizes={}'.format(
                src, id_cosmos_gs, fluxes, sizes)
    return results


def main():
    parser = argparse.ArgumentParser(description='MultiProFit HST COSMOS galaxy modelling test')

    flags = {
        'catalogpath': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog path'},
        'catalogfile': {'type': str, 'nargs': '?', 'default': None, 'help': 'GalSim catalog filename'},
        'file': {'type': str, 'nargs': '?', 'default': None, 'help': 'Filename for input/output'},
        'do_fit_fluxfracs': {'type': mpfutil.str2bool, 'default': False,
                             'help': 'Fit component flux fractions for galaxies instead of fluxes'},
        'fithsc': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HSC I band image'},
        'fithst': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image'},
        'fithst2hsc': {'type': mpfutil.str2bool, 'default': False, 'help': 'Fit HST F814W image convolved '
                                                                           'to HSC seeing'},
        'bands_hsc': {'type': str, 'nargs': '*', 'default': ['HSC-I'], 'help': 'HSC Bands to fit'},
        'hscrepo': {'type': str, 'default': '/datasets/hsc/repo/rerun/RC/w_2019_26/DM-19560/',
                    'help': 'Path to HSC processing repository'},
        'model_name_hst2hsc': {'type': str, 'default': None,
                               'help': 'HST model fit to use for mock HSC image'},
        'img_plot_maxs': {'type': float, 'nargs': '*', 'default': None,
                          'help': 'Max. flux for scaling single-band images. F814W first if fitting HST, '
                                  'then HSC bands.'},
        'img_multi_plot_max': {'type': float, 'default': None, 'help': 'Max. flux for scaling color images'},
        'indices': {'type': str, 'nargs': '*', 'default': None, 'help': 'Galaxy catalog index'},
        'loglevel': {'type': int, 'nargs': '?', 'default': 20, 'help': 'logging.Logger default level'},
        'modelspecfile': {'type': str, 'default': None, 'help': 'Model specification file'},
        'modellib': {'type': str, 'nargs': '?', 'default': 'scipy', 'help': 'Optimization libraries'},
        'modellibopts': {'type': str, 'nargs': '?', 'default': None, 'help': 'Model fitting options'},
        'nwrite': {'type': int, 'default': 5, 'help': 'Number of galaxies to fit before writing file'},
        #'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
        'plot': {'type': mpfutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
        'print_step': {'type': int, 'default': 100, 'help': 'Number of fitting steps before printing'},
        #'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
        'redo': {'type': mpfutil.str2bool, 'default': True, 'help': 'Redo existing fits'},
        'redo_psfs': {'type': mpfutil.str2bool, 'default': False, 'help': 'Redo existing PSF fits'},
        'weights_band': {'type': float, 'nargs': '*', 'default': None,
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
    logging.basicConfig(stream=sys.stdout, level=args.loglevel)

    modelspecs = mpffit.get_modelspecs(
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
    bands = (['F814W'] if args.fithst else []) + (args.bands_hsc if (args.fithsc or args.fithst2hsc) else [])
    for nameattr in ['img_plot_maxs', 'weights_band']:
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
            time_now = time.time()
            try:
                np.random.seed(idnum)
                radec = rgcfits[idnum][1:3]
                scale_hst = rgcfits[idnum]['PIXEL_SCALE']
                fits = fit_galaxy_cosmos(
                    radec=radec, id_cosmos_gs=idnum, srcs=srcs, modelspecs=modelspecs, rgcat=rgcat, ccat=ccat,
                    butler=butler, skymap=skymap, plot=args.plot,
                    redo=args.redo, redo_psfs=args.redo_psfs, reset_images=True,
                    model_name_hst2hsc=args.model_name_hst2hsc, bands_hsc=args.bands_hsc, scale_hst=scale_hst,
                    modellib=args.modellib, results=data[idnum] if idnum in data else None,
                    img_plot_maxs=args.img_plot_maxs, img_multi_plot_max=args.img_multi_plot_max,
                    weights_band=args.weights_band, do_fit_fluxfracs=args.do_fit_fluxfracs,
                    print_step_interval=args.print_step)
                data[idnum] = fits
            except Exception as e:
                print(f"Error fitting id={idnum}: {e}")
                trace = traceback.format_exc()
                print(trace)
                if idnum not in data:
                    try:
                        pickle.loads(pickle.dumps(e))
                    except Exception as te:
                        e = RuntimeError(str(type(e)) + str(e) + "; pickling error:" + str(te))
                    data[idnum] = {'error': e, 'trace': trace}
            print("Finished fitting COSMOS galaxy with ID: {} in total time of {:.2f} seconds".format(
                idnum, time.time() - time_now
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
