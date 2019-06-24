#!/usr/bin/env python
# coding: utf-8

# # Fitting HSC data in multiband mode using MultiProFit

# In[1]:


import os
import astropy.visualization as apvis
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprofit.fitutils as mpffit
import multiprofit.objects as mpfobj
import numpy as np
import pickle

mpl.rcParams['image.origin'] = 'bottom'
mpl.rcParams['figure.dpi'] = 120


# In[2]:


path = os.path.expanduser("~/raid/lsst/hsc/")
haslsst = True
try:
    import lsst.daf.persistence as dafPersist
    import lsst.afw.geom as afwGeom
except:
    haslsst = False
    print("Failed to import LSST packages; using HSC PDR1 data instead")

if haslsst:
    butler = dafPersist.Butler('/datasets/hsc/repo/rerun/DM-10404/WIDE/')

# Define model specs
modelspecfile = os.path.expanduser('~/src/mine/multiprofit/examples/modelspecs-mg-base-psfg2.csv')
modelspecs = mpffit.get_modelspecs(modelspecfile)


# In[3]:


# Pick your favourite galaxy
# GAMA 79635
# ra, dec = 222.51551376, 0.09749601
# Song's favourite
ra, dec, z = 30.1198469, -2.5947567, 0.24422
bands = ['I', 'R', 'G']
weightsfilters = [1., 1.19124201, 2.24905461]
filters = ['HSC-' + x for x in bands]
filterref = 'HSC-I'
filtermulti = 'HSC-' + ''.join(bands)
filesave = os.path.join(path, "fit-" + str(ra) + '-' + str(dec) + '_pickle.dat')


# In[4]:


if haslsst:
    pos = afwGeom.SpherePoint(ra, dec, afwGeom.degrees)
    skymap = butler.get("deepCoadd_skyMap")
    tractInfo = skymap.findTract(pos)
    tract = tractInfo.getId()
#    patch = ','.join([str(x) for x in tractInfo.findPatch(pos).getIndex()])
    import multiprofit.datautils.gethsc as gethsc
    cutouts, spherePoint, scalepix, nearby = gethsc.get_cutout_hsc(
        butler, skymap, filters, [ra, dec], tract=tract, size_in_pix=256, do_deblend=True, radius_nearby_objects=21,
        do_keep_wcs=True, band_match=filterref)


# In[5]:


fig, axes = plt.subplots(1, 2, figsize=(20, 20))
if haslsst:
    bbox = cutouts[filterref]['blended']['bbox']
    wcs = cutouts[filterref]['blended']['WCS']
    coordscutout = cutouts[filterref]['blended']['coordpix']
    offset = -(afwGeom.Extent2D(bbox.getMin()) + afwGeom.Extent2D(coordscutout[1], coordscutout[3]))
    target = wcs.skyToPixel(spherePoint)
    target.shift(offset)
    objectsnearby = [wcs.skyToPixel(afwGeom.SpherePoint(ra, dec, afwGeom.radians)) for ra, dec in zip(nearby[0], nearby[1])]
    for objectnearby in objectsnearby:
        objectnearby.shift(offset)
    print(objectsnearby)
for i, (key, value) in enumerate(cutouts[filterref].items()):
    norm = apvis.ImageNormalize(vmin=-0.1, vmax=20, stretch=apvis.AsinhStretch(1e-2))
    axes[i].imshow(value['img'], cmap='gray', norm=norm)
    if haslsst:
        axes[i].scatter([target[0]], [target[1]], s=32, marker='+', c='cyan', linewidth=0.5)
        axes[i].scatter([x[0] for x in objectsnearby], [x[1] for x in objectsnearby], s=32, marker='+', c='red', linewidth=0.5)
    axes[i].set_title(filterref + ' ' + key)
fig, axes = plt.subplots(1, 2, figsize=(20, 20))
for i, typeimg in enumerate(['blended', 'deblended']):
    imgs = [weight*cutouts[band][typeimg]['img'] for band, weight in zip(filters, weightsfilters)]
    img = apvis.make_lupton_rgb(imgs[0], imgs[1], imgs[2], stretch=1, Q=16)
    axes[i].imshow(img)
    axes[i].set_title(filtermulti + ' ' + typeimg)
plt.show()


# In[6]:


exposures = gethsc.get_exposures_hsc(cutouts=cutouts, scale_pixel_hsc=scalepix, bands=filters)


# In[7]:


import pickle
write = True
if os.path.isfile(filesave):
    with open(filesave, 'rb') as f:
        results = pickle.load(f)
    write = False
else:
    results = {}
    for band in filters:
        print("Fitting in band={}".format(band))
        results[band] = mpffit.fit_galaxy_exposures(exposures, [band], modelspecs=modelspecs)
if filtermulti not in results:
    # A rather hacky way to recast the results of single-band fits into what the multi-band version would have returned
    # This would break if we were multifitting multiple exposures per band, for example
    results_psfs = {'psfs': {idx: results[band]['psfs'][idx] for idx, band in enumerate(filters)}}
    results[filtermulti] = mpffit.fit_galaxy_exposures(exposures, filters, modelspecs=modelspecs, results=results_psfs)
    write = True
if write:
    with open(filesave, 'wb') as f:
        pickle.dump(results, f)


# In[8]:


for fittype, resultfull in results.items():
    fits = resultfull['fits']['galsim']
    for model, result in fits.items():
        print(fittype, model, -result['fits'][-1]['result'].cost, np.sum([x['time'] for x in result['fits']]))


# In[9]:


figuresize = mpl.rcParams['figure.figsize']
mpl.rcParams['figure.figsize'] = [32.0, 18.0]
_ = mpffit.fit_galaxy_exposures(
    exposures, filters, modelspecs=modelspecs, results=results[filtermulti], redo=False, plot=True,
    img_plot_maxs=[40., 20., 10.], img_multi_plot_max=2,
    weights_band={band: weight for band, weight in zip(filters, weightsfilters)}
)
plt.show();
mpl.rcParams['figure.figsize'] = figuresize


# In[10]:


plt.show()


# In[ ]:




