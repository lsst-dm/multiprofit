#!/usr/bin/env python
# coding: utf-8

# # Fitting HSC data in multiband mode using MultiProFit

# In[1]:


# Import required packages
import time
from typing import Any, Iterable, Mapping

from astropy.coordinates import SkyCoord
from astropy.io.ascii import Csv
import astropy.io.fits as fits
import astropy.table as apTab
import astropy.visualization as apVis
from astropy.wcs import WCS
import gauss2d as g2
import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import SersicComponentConfig, SersicIndexParameterConfig
from lsst.multiprofit.fit_psf import (
    CatalogExposurePsfABC,
    CatalogPsfFitter,
    CatalogPsfFitterConfig,
    CatalogPsfFitterConfigData,
)
from lsst.multiprofit.fit_source import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterABC,
    CatalogSourceFitterConfig,
    CatalogSourceFitterConfigData,
)
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.plots import plot_model_rgb
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.multiprofit.utils import ArbitraryAllowedConfig, get_params_uniq
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass


# In[2]:


# Define settings
band_ref = 'i'
bands = {'i': 0.87108833, 'r': 0.97288654, 'g': 1.44564678}
band_multi = ''.join(bands)
channels = {band: g2f.Channel.get(band) for band in bands}

# This is in the WCS, but may as well keep full precision
scale_pixel_hsc = 0.168

# Common to all FITS
hdu_img, hdu_mask, hdu_var = 1, 2, 3

# Masks
bad_masks = (
    'BAD', 'SAT', 'INTRP', 'CR', 'EDGE', 'CLIPPED', 'NO_DATA', 'CROSSTALK',
    'NO_DATA', 'UNMASKEDNAN', 'SUSPECT', 'REJECTED', 'SENSOR_EDGE',
)
maskbits = tuple(f'MP_{b}' for b in bad_masks)

# A pre-defined bitmask to exclude regions with low SN
read_mask_highsn = True
write_mask_highsn = False

# matplotlib settings
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['figure.dpi'] = 120


# In[3]:


# Define source to fit
id_gama, z = 79635, 0.0403
"""
Acquired from https://hsc-release.mtk.nao.ac.jp/datasearch/catalog_jobs with query:

SELECT object_id, ra, dec,
    g_cmodel_mag, g_cmodel_magerr, r_cmodel_mag, r_cmodel_magerr, i_cmodel_mag, i_cmodel_magerr,
    g_psfflux_mag, g_psfflux_magerr, r_psfflux_mag, r_psfflux_magerr, i_psfflux_mag, i_psfflux_magerr,
    g_kronflux_mag, g_kronflux_magerr, r_kronflux_mag, r_kronflux_magerr, i_kronflux_mag, i_kronflux_magerr,
    g_sdssshape_shape11, g_sdssshape_shape11err, g_sdssshape_shape22, g_sdssshape_shape22err,
    g_sdssshape_shape12, g_sdssshape_shape12err,
    r_sdssshape_shape11, r_sdssshape_shape11err, r_sdssshape_shape22, r_sdssshape_shape22err,
    r_sdssshape_shape12, r_sdssshape_shape12err,
    i_sdssshape_shape11, i_sdssshape_shape11err, i_sdssshape_shape22, i_sdssshape_shape22err,
    i_sdssshape_shape12, i_sdssshape_shape12err
FROM pdr3_wide.forced
LEFT JOIN pdr3_wide.forced2 USING (object_id)
WHERE isprimary AND conesearch(coord, 222.51551376, 0.09749601, 35.64)
AND (r_kronflux_mag < 26 OR i_kronflux_mag < 26) AND NOT i_kronflux_flag AND NOT r_kronflux_flag;
"""
cat = Csv()
cat.header.splitter.escapechar = '#'
cat = cat.read('fithsc_src.csv')

prefix = '222.51551376,0.09749601_'
prefix_img = f'{prefix}300x300_'

# Read data, acquired with:
# https://github.com/taranu/astro_imaging/blob/4d5a8e095e6a3944f1fbc19318b1dbc22b22d9ca/examples/HSC.ipynb
# (with get_mask=True, get_variance=True,)
images, psfs = {}, {}
for band in bands:
    images[band] = fits.open(f'{prefix_img}{band}.fits')
    psfs[band] = fits.open(f'{prefix}{band}_psf.fits')

wcs = WCS(images[band_ref][hdu_img])
cat['x'], cat['y'] = wcs.world_to_pixel(SkyCoord(cat['ra'], cat['dec'], unit='deg'))


# In[4]:


# Plot image
img_rgb = apVis.make_lupton_rgb(*[img[1].data*bands[band] for band, img in images.items()])
plt.scatter(cat['x'], cat['y'], s=10, c='g', marker='x')
plt.imshow(img_rgb)
plt.title("gri image with detected objects")
plt.show()


# In[5]:


# Generate a rough mask around other sources
bright = (cat['i_cmodel_mag'] < 23) | (cat['i_psfflux_mag'] < 23)

img_ref = images[band_ref][hdu_img].data

mask_inverse = np.ones(img_ref.shape, dtype=bool)
y_cen, x_cen = (x/2. for x in img_ref.shape)
y, x = np.indices(img_ref.shape)

idx_src_main, row_main = None, None

sizes_override = {
    42305088563206480: 8.,
}

for src in cat[bright]:
    id_src, x_src, y_src = (src[col] for col in ['object_id', 'x', 'y'])
    dist = np.hypot(x_src - x_cen, y_src - y_cen)
    if dist > 20:
        dists = np.hypot(y - y_src, x - x_src)
        mag = np.nanmin([src['i_cmodel_mag'], src['r_cmodel_mag'], src['i_psfflux_mag'], src['r_psfflux_mag']])
        if (radius_mask := sizes_override.get(id_src)) is None:
            radius_mask = 2*np.sqrt(
                src[f'{band_ref}_sdssshape_shape11'] + src[f'{band_ref}_sdssshape_shape22']
            )/scale_pixel_hsc
            if (radius_mask > 10) and (mag > 21):
                radius_mask = 5
        mask_inverse[dists < radius_mask] = 0
        print(f'Masking src=({id_src} at {x_src:.3f}, {y_src:.3f}) dist={dist:.3f}'
              f', mag={mag:.3f}, radius_mask={radius_mask:.3f}')
    elif dist < 2:
        idx_src_main = id_src
        row_main = src
        print(f"{idx_src_main=} {dict(src)=}")

tab_row_main = apTab.Table(row_main)

if read_mask_highsn:
    mask_highsn = np.load(f'{prefix_img}mask_inv_highsn.npz')['mask_inv']
    mask_inverse *= mask_highsn

plt.imshow(mask_inverse)
plt.title("Fitting mask")
plt.show()


# In[6]:


# Fit PSF
@dataclass(frozen=True, config=ArbitraryAllowedConfig)
class CatalogExposurePsf(CatalogExposurePsfABC):
    catalog: apTab.Table = pydantic.Field(title="The detected object catalog")
    img: np.ndarray = pydantic.Field(title="The PSF image")

    def get_catalog(self) -> Iterable:
        return self.catalog

    def get_psf_image(self, source: apTab.Row | Mapping[str, Any]) -> np.array:
        return self.img

config_psf = CatalogPsfFitterConfig(column_id='object_id')
fitter_psf = CatalogPsfFitter()
catalog_psf = apTab.Table({'object_id': [tab_row_main['object_id']]})
results_psf = {}

# Keep a separate configdata_psf per band, because it has a cached PSF model
# those should not be shared!
config_data_psfs = {}
for band, psf_file in psfs.items():
    config_data_psf = CatalogPsfFitterConfigData(config=config_psf)
    catexp = CatalogExposurePsf(catalog=catalog_psf, img=psf_file[0].data)
    t_start = time.time()
    result = fitter_psf.fit(config_data=config_data_psf, catexp=catexp)
    t_end = time.time()
    results_psf[band] = result
    config_data_psfs[band] = config_data_psf
    print(f"Fit {band}-band PSF in {t_end - t_start:.2e}s; result:")
    print(dict(result[0]))


# In[7]:


# Set fit configs
config_source = CatalogSourceFitterConfig(
    column_id='object_id',
    config_model=ModelConfig(
        sources={
            "src": SourceConfig(
                component_groups={
                    "": ComponentGroupConfig(
                        components_sersic={
                            'disk': SersicComponentConfig(
                                sersic_index=SersicIndexParameterConfig(value_initial=1., fixed=True),
                                prior_size_stddev=0.5,
                                prior_axrat_stddev=0.2,
                            ),
                            'bulge': SersicComponentConfig(
                                sersic_index=SersicIndexParameterConfig(value_initial=4., fixed=True),
                                prior_size_stddev=0.1,
                                prior_axrat_stddev=0.2,
                            ),
                        },
                    ),
                }
            ),
        },
    ),
)
config_data_source = CatalogSourceFitterConfigData(
    channels=list(channels.values()),
    config=config_source,
)


# In[8]:


# Setup exposure with band-specific image, mask and variance
@dataclass(frozen=True, config=ArbitraryAllowedConfig)
class CatalogExposureSources(CatalogExposureSourcesABC):
    config_data_psf: CatalogPsfFitterConfigData = pydantic.Field(title="The PSF fit config")
    observation: g2f.Observation = pydantic.Field(title="The observation to fit")
    table_psf_fits: apTab.Table = pydantic.Field(title="The table of PSF fit parameters")

    @property
    def channel(self) -> g2f.Channel:
        return self.observation.channel

    def get_catalog(self) -> Iterable:
        return self.table_psf_fits

    def get_psf_model(self, params: Mapping[str, Any]) -> g2f.PsfModel:
        self.config_data_psf.init_psf_model(params)
        return self.config_data_psf.psf_model

    def get_source_observation(self, source: Mapping[str, Any]) -> g2f.Observation:
        return self.observation


@dataclass(frozen=True, config=ArbitraryAllowedConfig)
class CatalogSourceFitter(CatalogSourceFitterABC):
    band: str = pydantic.Field(title="The reference band for initialization and priors")
    scale_pixel: float = pydantic.Field(title="The pixel scale in arcsec")
    wcs_ref: WCS = pydantic.Field(title="The WCS for the coadded image")

    def initialize_model(
        self,
        model: g2f.Model,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        centroid_pixel_offset: float = 0,
        **kwargs
    ):
        if values_init is None:
            values_init = {}
        x, y = source['x'], source['y']
        scale_sq = self.scale_pixel**(-2)
        ellipse = g2.Ellipse(g2.Covariance(
            sigma_x_sq=source[f'{band}_sdssshape_shape11']*scale_sq,
            sigma_y_sq=source[f'{band}_sdssshape_shape22']*scale_sq,
            cov_xy=source[f'{band}_sdssshape_shape12']*scale_sq,
        ))
        size_major = g2.EllipseMajor(ellipse).r_major
        limits_size = g2f.LimitsD(1e-5, np.sqrt(x*x + y*y))
        # An R_eff larger than the box size is problematic
        # Also should stop unreasonable size proposals; log10 transform isn't enough
        # TODO: Try logit for r_eff?
        params_limits_init = {
            # Should set limits based on image size, but this shortcut is fine
            # for this particular object
            g2f.CentroidXParameterD: (x, g2f.LimitsD(0, 2*x)),
            g2f.CentroidYParameterD: (x, g2f.LimitsD(0, 2*y)),
            g2f.ReffXParameterD: (ellipse.sigma_x, limits_size),
            g2f.ReffYParameterD: (ellipse.sigma_y, limits_size),
            # There is a sign convention difference
            g2f.RhoParameterD: (-ellipse.rho, None),
            g2f.IntegralParameterD: (1.0, g2f.LimitsD(1e-10, 1e10)),
        }
        params_free = get_params_uniq(model, fixed=False)
        for param in params_free:
            type_param = type(param)
            value_init, limits_new = params_limits_init.get(
                type_param,
                (values_init.get(param), None)
            )
            if value_init is not None:
                param.value = value_init
            if limits_new:
                # For slightly arcane reasons, we must set a new limits object
                # Changing limits values is unreliable
                param.limits = limits_new
        for prior in model.priors:
            if isinstance(prior, g2f.ShapePrior):
                prior.prior_size.mean_parameter.value = size_major


    def validate_fit_inputs(
        self,
        catalog_multi,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData = None,
        logger = None,
        **kwargs: Any,
    ) -> None:
        super().validate_fit_inputs(
            catalog_multi=catalog_multi, catexps=catexps, config_data=config_data,
            logger=logger, **kwargs
        )


# In[9]:


# Set up Fitter, Observations and CatalogExposureSources
fitter = CatalogSourceFitter(band=band, scale_pixel=scale_pixel_hsc, wcs_ref=wcs)

observations = {}
catexps = {}

for band in bands:
    data = images[band]
    # There are better ways to use bitmasks, but this will do
    header = data[hdu_mask].header
    bitmask = data[hdu_mask].data
    mask = np.zeros_like(bitmask, dtype='bool')
    for bit in maskbits:
        mask |= ((bitmask & 2**header[bit]) != 0)

    mask = (mask == 0) & mask_inverse
    sigma_inv = 1.0/np.sqrt(data[hdu_var].data)
    sigma_inv[mask != 1] = 0

    observation = g2f.Observation(
        image=g2.ImageD(data[hdu_img].data),
        sigma_inv=g2.ImageD(sigma_inv),
        mask_inv=g2.ImageB(mask),
        channel=g2f.Channel.get(band),
    )
    observations[band] = observation
    catexps[band] = CatalogExposureSources(
        config_data_psf=config_data_psfs[band],
        observation=observation,
        table_psf_fits=results_psf[band],
    )


# In[10]:


# Now do the multi-band fit
t_start = time.time()
result_multi = fitter.fit(
    catalog_multi=tab_row_main,
    catexps=list(catexps.values()),
    config_data=config_data_source,
)
t_end = time.time()
print(f"Fit {','.join(bands.keys())}-band bulge-disk model in {t_end - t_start:.2e}s; result:")
print(dict(result_multi[0]))


# In[11]:


# Fit in each band separately
results = {}
for band, observation in bands.items():
    config_data_source_band = CatalogSourceFitterConfigData(
        channels=[channels[band]],
        config=config_source,
    )
    t_start = time.time()
    result = fitter.fit(
        catalog_multi=tab_row_main,
        catexps=[catexps[band]],
        config_data=config_data_source_band,
    )
    t_end = time.time()
    results[band] = result
    print(f"Fit {band}-band bulge-disk model in {t_end - t_start:.2f}s; result:")
    print(dict(result[0]))


# In[12]:


# Make a model for the best-fit params
data, psf_models = config_source.make_model_data(idx_row=0, catexps=list(catexps.values()))
model = g2f.Model(data=data, psfmodels=psf_models, sources=config_data_source.sources_priors[0], priors=config_data_source.sources_priors[1])
params = get_params_uniq(model, fixed=False)
result_multi_row = dict(result_multi[0])
# This is the last column before fit params
idx_last = next(idx for idx, column in enumerate(result_multi_row.keys()) if column == 'mpf_unknown_flag')
# Set params to best fit values
for param, (column, value) in zip(params, list(result_multi_row.items())[idx_last+1:]):
    param.value = value
model.setup_evaluators(model.EvaluatorMode.loglike_image)
# Print the loglikelihoods, which are from the data and end with the (sum of all) priors
loglikes = model.evaluate()
print(f"{loglikes=}")


# ### Multiband Residuals
# 
# What's with the structure in the residuals? Most broadly, a point source + exponential disk + deVauc bulge model is totally inadequate for this galaxy for several possible reasons:
# 
# 1. The disk isn't exactly exponential (n=1)
# 2. The disk has colour gradients not accounted for in this model*
# 3. If the galaxy even has a bulge, it's very weak and def. not a deVaucouleurs (n=4) profile; it may be an exponential "pseudobulge"
# 
# \*MultiProFit can do more general Gaussian mixture models (linear or non-linear), which may be explored in a future iteration of this notebook, but these are generally do not improve the accuracy of photometry for smaller/fainter galaxies.
# 
# Note that the two scalings of the residual plots (98%ile and +/- 20 sigma) end up looking very similar.
# 

# In[13]:


# Make some basic plots
_, _, _, _, mask_inv_highsn = plot_model_rgb(
    model, weights=bands, high_sn_threshold=0.2 if write_mask_highsn else None,
)
plt.show()

# Write the high SN bitmask to a compressed, bitpacked file
if write_mask_highsn:
    plt.figure()
    plt.imshow(mask_highsn, cmap='gray')
    plt.show()
    packed = np.packbits(mask_inv_highsn, bitorder='little')
    np.savez_compressed(f'{prefix_img}mask_inv_highsn.npz', mask_inv=mask_highsn)

# TODO: Some features still missing from plot_model_rgb
# residual histograms, param values, better labels, etc


# ### More exercises for the reader
# 
# These are of the sort that the author hasn't gotten around to yet because they're far from trivial. Try:
# 
# 0. Use the WCS to compute ra, dec and errors thereof.
# Hint: override CatalogSourceFitter.get_model_radec
# 
# 1. Replace the real data with simulated data.
# Make new observations using model.evaluate and add noise based on the variance maps.
# Try fitting again and see how well results converge depending on the initialization scheme.
# 
# 2. Fit every other source individually.
# Try subtracting the best-fit galaxy model from above first.
# Hint: get_source_observation should be redefined to return a smaller postage stamp around the nominal centroid.
# Pass the full catalog (excluding the central galaxy) to catalog_multi.
# 
# 3. Fit all sources simultaneously.
# Redefine CatalogFitterConfig.make_model_data to make a model with multiple sources, using the catexp catalogs
# initialize_model will no longer need to do anything
# catalog_multi should still be a single row
