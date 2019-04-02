import galsim as gs
import lsst.afw.geom as geom
import multiprofit.objects as mpfobj
import numpy as np

from modelling_research import make_cutout


def findhscmatch(spherePoint, measCat, distmatchinasec=None, radiusNearbyObjects=0):
    if distmatchinasec is None:
        distmatchinasec = 1.0
    # Get and verify match
    # TODO: Verify coord units?
    dists = np.array([
        spherePoint.separation(geom.SpherePoint(ra, dec, geom.radians)).asArcseconds()
        for ra, dec in zip(measCat["coord_ra"], measCat["coord_dec"])
    ])
    nearest = np.argmin(dists)
    dist = dists[nearest]
    if radiusNearbyObjects > 0:
        rowsnearby = np.where(np.array(dists <= radiusNearbyObjects))[0]
        # TODO: There must be a better way to do this
        rowsnearby = rowsnearby[np.argsort(dists[rowsnearby])][1:]
    else:
        rowsnearby = []

    print('Source distance={:.2e}"'.format(dist))
    if dist > distmatchinasec:
        raise RuntimeError("Nearest HSC source at distance {:.3e}>1; aborting".format(dist))
    return measCat["id"][nearest], rowsnearby


def gethsccutout(butler, skymap, bands, radec, tract=9813, sizeinpix=60,
                 deblend=False, bandmatch=None, distmatchinasec=None, radiusNearbyObjects=0, keepwcs=False):
    if deblend:
        from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
    spherePoint = geom.SpherePoint(radec[0], radec[1], geom.degrees)
    patch = skymap[tract].findPatch(spherePoint).getIndex()
    patch = ",".join([str(x) for x in patch])
    dataId = {"tract": tract, "patch": patch}
    if bandmatch is not None:
        dataId.update({"filter": bandmatch})
        dataRef = butler.dataRef("deepCoadd", dataId=dataId)
        measCat = dataRef.get("deepCoadd_meas")
        idmatch, rowsnearby = findhscmatch(spherePoint, measCat, distmatchinasec=distmatchinasec,
                                           radiusNearbyObjects=radiusNearbyObjects)
    elif deblend or radiusNearbyObjects > 0:
        raise RuntimeError('Cannot deblend without a bandmatch to match on')
    cutouts = {
        band: {
            key: {} for key in [pre + 'blended' for pre in [''] + (['de'] if deblend else [])]
        } for band in bands
    }
    scalepixel = None
    for band in bands:
        dataId.update({"filter": band})
        dataRef = butler.dataRef("deepCoadd", dataId=dataId)
        # Get the coadd
        coadd = dataRef.get("deepCoadd_calexp")
        pixel = coadd.getWcs().skyToPixel(spherePoint)
        cutout = make_cutout.make_cutout_lsst(spherePoint, coadd, size=np.floor_divide(sizeinpix, 2))
        cutouts[band]['blended']['img'] = np.copy(cutout[0])
        scalepixelband = coadd.getWcs().getPixelScale().asArcseconds()
        if scalepixel is None:
            scalepixel = scalepixelband
        elif np.abs(scalepixel-scalepixelband)/scalepixel > 1e-3:
            raise RuntimeError('Inconsistent pixel scale for band {} ({} != {})'.format(
                band, scalepixelband, scalepixel
            ))
        idshsc = cutout[4]
        cutouts[band]['blended']['psf'] = coadd.getPsf().computeKernelImage(pixel).array
        cutouts[band]['blended']['mask'] = coadd.getMaskedImage().getMask().array[
                                            idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]]
        cutouts[band]['blended']['var'] = np.copy(coadd.getMaskedImage().getVariance().array[
                                            idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]])

        if deblend :
            measCat = dataRef.get("deepCoadd_meas")
            noiseReplacer = rebuildNoiseReplacer(coadd, measCat)
            noiseReplacer.insertSource(idmatch)
            cutouts[band]['deblended']['psf'] = cutouts[band]['blended']['psf']
            cutouts[band]['deblended']['mask'] = cutouts[band]['blended']['mask']
            maskedimage = coadd.getMaskedImage()
            cutouts[band]['deblended']['img'] = np.copy(maskedimage.getImage().array[
                                                    idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]])
            cutouts[band]['deblended']['var'] = np.copy(maskedimage.getVariance().array[
                                                    idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]])
        for blend, imgs in cutouts[band].items():
            for imgtype in imgs:
                cutouts[band][blend][imgtype] = np.float64(cutouts[band][blend][imgtype])

        cutouts[band]['blended']['coordpix'] = idshsc
        if keepwcs:
            cutouts[band]['blended']['WCS'] = coadd.getWcs()
            cutouts[band]['blended']['bbox'] = coadd.getBBox()

    radecsnearby = (measCat["coord_ra"][rowsnearby], measCat["coord_dec"][rowsnearby])
    return cutouts, spherePoint, scalepixel, radecsnearby


def gethscexposures(cutouts, scalehsc, bands=None, typecutout='deblended'):
    """
    Get HSC exposures and PSF images from the given cutouts.

    :param cutouts: Dict; key=band: value=dict; key=cutout type: value=dict; key=image type: value=image
        As returned by multiprofit.datautils.gethsc.gethsccutout
    :param scalehsc: Float; HSC pixel scale in arcseconds (0.168)
    :param bands: List of bands; currently strings.
    :param typecutout: String cutout type; one of 'blended' or 'deblended'.
        'Deblended' should contain only a single galaxy with neighbours subtracted.
    :return: List of tuples; [0]: multiprofit.objects.Exposure, [1]: PSF image (galsim object)
    """
    exposurespsfs = []
    if bands is None:
        bands = cutouts.keys()
    for band in bands:
        cutoutsband = cutouts[band][typecutout]
        exposurespsfs.append(
            (
                mpfobj.Exposure(band, cutoutsband['img'], sigmainverse=1.0/np.sqrt(cutoutsband['var'])),
                gs.InterpolatedImage(gs.Image(cutoutsband['psf'], scale=scalehsc))
            )
        )
    return exposurespsfs
