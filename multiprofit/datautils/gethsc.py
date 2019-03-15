import lsst.afw.geom as geom
import numpy as np

from modelling_research import make_cutout


def findhscmatch(radec, measCat, distmatchinasec=None):
    if distmatchinasec is None:
        distmatchinasec = 1.0
    # Get and verify match
    distsq = ((radec[0] - np.degrees(measCat["coord_ra"])) ** 2 +
              (radec[1] - np.degrees(measCat["coord_dec"])) ** 2)
    row = np.int(np.argmin(distsq))
    dist = np.sqrt(distsq[row]) * 3600
    print('Source distance={:.2e}"'.format(dist))
    if dist > distmatchinasec:
        raise RuntimeError("Nearest HSC source at distance {:.3e}>1; aborting".format(dist))
    return measCat["id"][row]


def gethsccutout(butler, skymap, bands, radec, tract=9813, sizeinpix=60,
                 deblend=False, bandmatch=None, distmatchinasec=None):
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
        idmatch = findhscmatch(radec, measCat, distmatchinasec=distmatchinasec)
    elif deblend:
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

        if deblend:
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
    return cutouts, spherePoint, scalepixel