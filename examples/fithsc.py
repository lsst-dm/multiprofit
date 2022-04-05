#!/usr/bin/env python
# coding: utf-8

# # Fitting HSC data in multiband mode using MultiProFit

# In[1]:


import os
from astropy.io.ascii import Csv
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import astropy.visualization as apvis
from astropy.wcs import WCS
import galsim as gs
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprofit.fitutils as mpffit
import multiprofit.objects as mpfobj
import numpy as np

mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['figure.dpi'] = 120


# In[2]:


# Define model specs
modelspecs = mpffit.get_modelspecs('modelspecs-mg8-base-psfg2.csv')


# In[3]:


# Define settings
band_ref = 'i'
bands = {'i': 0.87108833, 'r': 0.97288654, 'g': 1.44564678}
band_multi = ''.join(bands)

# This is in the WCS, but may as well keep full precision
scale_pixel_hsc = 0.168

# Common to all FITS
hdu_img, hdu_mask, hdu_var = 1, 2, 3
maskbits = tuple(f'MP_{b}' for b in ('BAD', 'SAT', 'INTRP', 'CR', 'EDGE', 'CLIPPED', 'NO_DATA', 'CROSSTALK',
                                     'NO_DATA', 'UNMASKEDNAN', 'SUSPECT', 'REJECTED', 'SENSOR_EDGE'))


# In[4]:


# Define source to fit
id_gama, z = 79635, 0.0403
# Acquired from https://hsc-release.mtk.nao.ac.jp/datasearch/catalog_jobs with query:
# SELECT object_id, ra, dec, g_cmodel_mag, g_cmodel_magerr, r_cmodel_mag, 
#     r_cmodel_magerr, i_cmodel_mag, i_cmodel_magerr
# FROM pdr3_wide.forced
# WHERE isprimary AND conesearch(coord, 222.51551376, 0.09749601, 35.64)
# AND r_cmodel_mag < 26 AND NOT i_cmodel_flag AND NOT r_cmodel_flag;
cat = Csv().read('fithsc_src.csv')
prefix = '222.51551376,0.09749601_'

# Read data, acquired with:
# https://github.com/taranu/astro_imaging/blob/4d5a8e095e6a3944f1fbc19318b1dbc22b22d9ca/examples/HSC.ipynb
# (with get_mask=True, get_variance=True,)
images, psfs = {}, {}
for band in bands:
    images[band] = fits.open(f'{prefix}300x300_{band}.fits')
    psfs[band] = fits.open(f'{prefix}{band}_psf.fits')

wcs = WCS(images[band_ref][hdu_img])
cat['x'], cat['y'] = wcs.world_to_pixel(SkyCoord(cat['ra'], cat['dec'], unit='deg'))


# In[5]:


# Plot image
img_rgb = apvis.make_lupton_rgb(*[img[1].data*bands[band] for band, img in images.items()])
plt.scatter(cat['x'], cat['y'], s=10, c='g', marker='x')
plt.imshow(img_rgb)
plt.show()


# In[6]:


# Generate a rough mask around other sources
bright = cat['i_cmodel_mag'] < 21
# Should depend on mag, or source moments, but for now keep it constant
radius_mask = 20

img_ref = images[band_ref][hdu_img].data

mask_inverse = np.ones(img_ref.shape, dtype=bool)
y_cen, x_cen = (x/2. for x in img_ref.shape)
y, x = np.indices(img_ref.shape)

# There are two saturated stars that are missing from the catalog
# (probably because their cmodel flags are set; lesson: don't query 
#  with those flags if you want sources you'd need to mask)
sources = (cat[bright], ({'x': 257.5, 'y': 247.9}, {'x': 12.8, 'y': 295.6}))

for srcs in sources:
    for src in srcs:
        x_src, y_src = src['x'], src['y']
        dist = np.hypot(x_src - x_cen, y_src - y_cen)
        if dist > 2:
            print(f'Masking src=({x_src}, {y_src}) dist={dist}')
            dist = np.hypot(y - y_src, x - x_src)
            mask_inverse[dist < radius_mask] = 0

plt.imshow(mask_inverse)


# In[7]:


# Setup exposure with band-specific image, mask and variance
exposures_psfs = []
for band in bands:
    data = images[band]
    # There are better ways to use bitmasks, but this will do
    header = data[hdu_mask].header
    bitmask = data[hdu_mask].data
    mask = np.zeros_like(bitmask, dtype='bool')
    for bit in maskbits:
        mask |= ((bitmask & 2**header[bit]) != 0)
        
    mask = (mask == 0) & mask_inverse
    variance_inv = 1.0/data[hdu_var].data
    variance_inv[mask != 1] = 0
    
    exposures_psfs.append(
        (
            mpfobj.Exposure(band, data[hdu_img].data, error_inverse=variance_inv, mask_inverse=mask, is_error_sigma=False),
            gs.InterpolatedImage(gs.Image(psfs[band][0].data, scale=scale_pixel_hsc))
        )
    )


# In[8]:


# Fit in each band separately
results = {}
for band in bands:
    print(f"Fitting in band={band}")
    results[band] = mpffit.fit_galaxy_exposures(exposures_psfs, [band], modelspecs=modelspecs)


# In[9]:


# A rather hacky way to recast the results of single-band fits into what the multi-band version would have returned
# This would break if we were multifitting multiple exposures per band, for example
results_psfs = {'psfs': {idx: results[band]['psfs'][idx] for idx, band in enumerate(bands)}}
# Do the multi-band fit using the single-band parameters as initial params
# (this uses structural parameters from either a single reference band or some weighted average)
results[band_multi] = mpffit.fit_galaxy_exposures(exposures_psfs, bands, modelspecs=modelspecs, results=results_psfs)


# In[10]:


# Print model LLs and fitting time
for fittype, resultfull in results.items():
    fits_mpf = resultfull['fits']['galsim']
    for model, result in fits_mpf.items():
        print(fittype, model, -result.fits[-1]['result'].cost, np.sum([x['time'] for x in result.fits]))


# In[11]:


# Plot every model fit
figuresize = mpl.rcParams['figure.figsize']
mpl.rcParams['figure.figsize'] = [32.0, 18.0]
_ = mpffit.fit_galaxy_exposures(
    exposures_psfs, bands, modelspecs=modelspecs, results=results[band_multi], redo=False, plot=True,
    img_plot_maxs=[40., 20., 10.], img_multi_plot_max=2,
    weights_band=bands,
)
plt.show();
mpl.rcParams['figure.figsize'] = figuresize


# In[ ]:




