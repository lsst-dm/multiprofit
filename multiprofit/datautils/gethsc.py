import galsim as gs
import lsst.afw.geom as geom
import multiprofit.objects as mpfobj
import numpy as np

from modelling_research import make_cutout


def find_hsc_match(spherePoint, measCat, dist_match_in_asec=None, radius_nearby_objects=0):
    if dist_match_in_asec is None:
        dist_match_in_asec = 1.0
    # Get and verify match
    # TODO: Verify coord units?
    dists = np.array([
        spherePoint.separation(geom.SpherePoint(ra, dec, geom.radians)).asArcseconds()
        for ra, dec in zip(measCat["coord_ra"], measCat["coord_dec"])
    ])
    nearest = np.argmin(dists)
    dist = dists[nearest]
    if radius_nearby_objects > 0:
        rows_nearby = np.where(np.array(dists <= radius_nearby_objects))[0]
        # TODO: There must be a better way to do this
        rows_nearby = rows_nearby[np.argsort(dists[rows_nearby])][1:]
    else:
        rows_nearby = []

    print('Source distance={:.2e}"'.format(dist))
    if dist > dist_match_in_asec:
        raise RuntimeError("Nearest HSC source at distance {:.3e}>1; aborting".format(dist))
    return measCat["id"][nearest], rows_nearby


def get_cutout_hsc(butler, skymap, bands, radec, tract=9813, size_in_pix=60,
                   do_deblend=False, band_match=None, dist_match_in_asec=None, radius_nearby_objects=0,
                   do_keep_wcs=False):
    if do_deblend:
        from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
    spherePoint = geom.SpherePoint(radec[0], radec[1], geom.degrees)
    patch = skymap[tract].findPatch(spherePoint).getIndex()
    patch = ",".join([str(x) for x in patch])
    dataId = {"tract": tract, "patch": patch}
    if band_match is not None:
        dataId.update({"filter": band_match})
        dataRef = butler.dataRef("deepCoadd", dataId=dataId)
        measCat = dataRef.get("deepCoadd_meas")
        idmatch, rows_nearby = find_hsc_match(spherePoint, measCat, dist_match_in_asec=dist_match_in_asec,
                                              radius_nearby_objects=radius_nearby_objects)
    elif do_deblend or radius_nearby_objects > 0:
        raise RuntimeError('Cannot deblend without a bandmatch to match on')
    cutouts = {
        band: {
            key: {} for key in [pre + 'blended' for pre in [''] + (['de'] if do_deblend else [])]
        } for band in bands
    }
    scale_pixel = None
    for band in bands:
        dataId.update({"filter": band})
        dataRef = butler.dataRef("deepCoadd", dataId=dataId)
        # Get the coadd
        coadd = dataRef.get("deepCoadd_calexp")
        pixel = coadd.getWcs().skyToPixel(spherePoint)
        cutout = make_cutout.make_cutout_lsst(spherePoint, coadd, size=np.floor_divide(size_in_pix, 2))
        cutouts[band]['blended']['img'] = np.copy(cutout[0])
        scale_pixel_band = coadd.getWcs().getPixelScale().asArcseconds()
        if scale_pixel is None:
            scale_pixel = scale_pixel_band
        elif np.abs(scale_pixel-scale_pixel_band)/scale_pixel > 1e-3:
            raise RuntimeError('Inconsistent pixel scale for band {} ({} != {})'.format(
                band, scale_pixel_band, scale_pixel
            ))
        ids_hsc = cutout[4]
        cutouts[band]['blended']['psf'] = coadd.getPsf().computeKernelImage(pixel).array
        cutouts[band]['blended']['mask'] = coadd.getMaskedImage().getMask().array[
                                            ids_hsc[3]: ids_hsc[2], ids_hsc[1]: ids_hsc[0]]
        cutouts[band]['blended']['var'] = np.copy(coadd.getMaskedImage().getVariance().array[
                                            ids_hsc[3]: ids_hsc[2], ids_hsc[1]: ids_hsc[0]])

        if do_deblend:
            measCat = dataRef.get("deepCoadd_meas")
            noiseReplacer = rebuildNoiseReplacer(coadd, measCat)
            noiseReplacer.insertSource(idmatch)
            cutouts[band]['deblended']['psf'] = cutouts[band]['blended']['psf']
            cutouts[band]['deblended']['mask'] = cutouts[band]['blended']['mask']
            maskedimage = coadd.getMaskedImage()
            cutouts[band]['deblended']['img'] = np.copy(maskedimage.getImage().array[
                                                    ids_hsc[3]: ids_hsc[2], ids_hsc[1]: ids_hsc[0]])
            cutouts[band]['deblended']['var'] = np.copy(maskedimage.getVariance().array[
                                                    ids_hsc[3]: ids_hsc[2], ids_hsc[1]: ids_hsc[0]])
        for blend, imgs in cutouts[band].items():
            for imgtype in imgs:
                cutouts[band][blend][imgtype] = np.float64(cutouts[band][blend][imgtype])

        if band_match is not None:
            cutouts[band]['blended']['id'] = idmatch
        cutouts[band]['blended']['coordpix'] = ids_hsc
        if do_keep_wcs:
            cutouts[band]['blended']['WCS'] = coadd.getWcs()
            cutouts[band]['blended']['bbox'] = coadd.getBBox()

    radecs_nearby = None if band_match is None else (measCat["coord_ra"][rows_nearby], measCat["coord_dec"][
        rows_nearby])
    return cutouts, spherePoint, scale_pixel, radecs_nearby


def get_exposures_hsc(cutouts, scale_pixel_hsc, bands=None, typecutout='deblended'):
    """
    Get HSC exposures and PSF images from the given cutouts.

    :param cutouts: Dict; key=band: value=dict; key=cutout type: value=dict; key=image type: value=image
        As returned by multiprofit.datautils.gethsc.gethsccutout
    :param scale_pixel_hsc: Float; HSC pixel scale in arcseconds (0.168)
    :param bands: List of bands; currently strings.
    :param typecutout: String cutout type; one of 'blended' or 'deblended'.
        'Deblended' should contain only a single galaxy with neighbours subtracted.
    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    exposures_psfs = []
    if bands is None:
        bands = cutouts.keys()
    for band in bands:
        cutouts_band = cutouts[band][typecutout]
        exposures_psfs.append(
            (
                mpfobj.Exposure(
                    band, cutouts_band['img'], error_inverse=1.0/cutouts_band['var'], is_error_sigma=False),
                gs.InterpolatedImage(gs.Image(cutouts_band['psf'], scale=scale_pixel_hsc))
            )
        )
    return exposures_psfs
