from dataclasses import dataclass
import gauss2d as g2
import gauss2d.fit as g2f
from itertools import chain
import multiprofit.fitutils as mpffit
from multiprofit.multigaussianapproxprofile import MultiGaussianApproximationComponent
import multiprofit.objects as mpfobj
import numpy as np
import pytest
import timeit


@dataclass
class ComponentConfig:
    n_comp: int = 2
    rho_base: float = -0.2
    rho_increment: float = 0.4
    size_base: float = 1.5
    size_increment: float = 1.


@dataclass
class Config:
    comps_psf: ComponentConfig
    comps_src: ComponentConfig
    n_x: int = 35
    n_y: int = 33
    src_n: int = 1
    src_flux_base: float = 1.0
    src_flux_increment: float = 1.0


@dataclass
class Limits:
    x: g2f.LimitsD
    y: g2f.LimitsD
    rho: g2f.LimitsD


@pytest.fixture(scope='module')
def config():
    return Config(
        comps_psf=ComponentConfig(size_base=2.5),
        comps_src=ComponentConfig(size_base=2.0),
    )


@pytest.fixture(scope='module')
def limits(config):
    return Limits(
        x=g2f.LimitsD(min=0, max=config.n_y),
        y=g2f.LimitsD(min=0, max=config.n_x),
        rho=g2f.LimitsD(min=-0.99, max=0.99),
    )


@pytest.fixture(scope='module')
def bands():
    return tuple(('i', 'r', 'g'))


@pytest.fixture(scope='module')
def channels(bands):
    return {band: g2f.Channel(band) for band in bands}


@pytest.fixture(scope='module')
def images(bands, config):
    mask_inv = g2.ImageB(np.ones((config.n_y, config.n_x), dtype=bool))
    return {
        band: (
            g2.ImageD(config.n_y, config.n_x),
            g2.ImageD(config.n_y, config.n_x),
            mask_inv,
        )
        for band in bands
    }


@pytest.fixture(scope='module')
def data(channels, images):
    return g2f.Data([
        g2f.Observation(
            channel=channels[band],
            image=imagelist[0],
            sigma_inv=imagelist[1],
            mask_inv=imagelist[2],
        )
        for band, imagelist in images.items()
    ])


@pytest.fixture(scope='module')
def data_old(bands, images, psfmodels_old):
    exposures = [
        mpfobj.Exposure(
            band=band,
            image=imagelist[0],
            error_inverse=imagelist[1],
            mask_inverse=imagelist[2],
            psf=mpfobj.PSF(band=band, model=psfmodels_old[band], use_model=True, is_model_pixelated=True),
        )
        for band, imagelist in images.items()
    ]
    assert len(exposures[0].psf.model.get_parameters(free=True, fixed=True)) > 0
    return mpfobj.Data(exposures)


@pytest.fixture(scope='module')
def psfmodels(channels, config, data, limits):
    compconf = config.comps_psf
    n_comps = compconf.n_comp
    n_last = n_comps - 1
    psfmodels = [None]*len(data)
    for i in range(len(psfmodels)):
        components = [None]*n_comps
        centroid = g2f.CentroidParameters(
            g2f.CentroidXParameterD(config.n_y/2., limits=limits.x),
            g2f.CentroidYParameterD(config.n_x/2., limits=limits.y),
        )
        for c in range(n_comps):
            last = g2f.FractionalIntegralModel(
                { g2f.Channel.NONE: g2f.ProperFractionParameterD(0.5 + 0.5*(c > 0), fixed=(c == n_last)) },
                g2f.LinearIntegralModel({
                    g2f.Channel.NONE: g2f.IntegralParameterD(1.0)
                }) if (c == 0) else last,
            )
            components[c] = g2f.GaussianComponent(
                g2f.GaussianParametricEllipse(
                    g2f.SigmaXParameterD(compconf.size_base + c*compconf.size_increment),
                    g2f.SigmaYParameterD(compconf.size_base + c*compconf.size_increment),
                    g2f.RhoParameterD(compconf.rho_base + c*compconf.rho_increment, limits=limits.rho),
                ),
                centroid,
                last,
            )
        psfmodels[i] = g2f.PsfModel(components)
    return psfmodels


@pytest.fixture(scope='module')
def psfmodels_old(bands, config):
    compconf = config.comps_psf
    sigmas = compconf.size_base + compconf.size_increment*np.arange(compconf.n_comp)
    sources = {
        band: mpffit.get_model(
            {band: 1},
            "gaussian:2",
            (config.n_y, config.n_x),
            sigma_xs=sigmas,
            sigma_ys=sigmas,
            rhos=compconf.rho_base + compconf.rho_increment*np.arange(compconf.n_comp),
            fluxfracs=(0.5, 1),
            engine='galsim',
        ).sources[0]
        for band in bands
    }
    return sources


def get_sources(channels, config, limits, old: bool = False):
    compconf = config.comps_src
    n_components = compconf.n_comp
    sources = [None]*config.src_n

    for i in range(len(sources)):
        flux = (config.src_flux_base + i*config.src_flux_increment)/n_components
        components = [None]*n_components
        centroid = (mpfobj.AstrometricModel if old else g2f.CentroidParameters)(
            g2f.CentroidXParameterD(config.n_y/2., limits=limits.x),
            g2f.CentroidYParameterD(config.n_x/2., limits=limits.y),
        )
        for c in range(n_components):
            fluxes = {
                channel.name if old else channel: g2f.IntegralParameterD(flux, label=channel.name)
                for channel in channels.values()
            }
            size = compconf.size_base + c*compconf.size_increment
            sersicindex = g2f.SersicMixComponentIndexParameterD(1.0 + 3*c)
            ellipse = (mpfobj.EllipseParameters if old else g2f.SersicParametricEllipse)(
                (g2f.SigmaXParameterD if old else g2f.ReffXParameterD)(size),
                (g2f.SigmaYParameterD if old else g2f.ReffYParameterD)(size),
                g2f.RhoParameterD(compconf.rho_base + c*compconf.rho_increment, limits=limits.rho)
            )
            if old:
                component = MultiGaussianApproximationComponent(
                    fluxes=list(fluxes.values()),
                    params_ellipse=ellipse,
                    parameters=[sersicindex],
                    order=4,
                )
            else:
                component = g2f.SersicMixComponent(
                    ellipse,
                    centroid,
                    g2f.LinearIntegralModel(fluxes),
                    sersicindex,
                )
            components[c] = component
        if old:
            sources[i] = mpfobj.Source(
                centroid,
                mpfobj.PhotometricModel(components),
            )
        else:
            sources[i] = g2f.Source(components)
            gaussians = sources[i].gaussians(list(channels.values())[0])
            assert len(gaussians) == 4*n_components
    return sources


@pytest.fixture(scope='module')
def sources(channels, config, limits):
    return get_sources(channels, config, limits, old=False)


@pytest.fixture(scope='module')
def sources_old(channels, config, limits):
    return get_sources(channels, config, limits, old=True)


@pytest.fixture(scope='module')
def model(data, psfmodels, sources):
    return g2f.Model(data, list(psfmodels), list(sources))


@pytest.fixture(scope='module')
def model_old(data_old, sources_old):
    return mpfobj.Model(sources_old, data_old)


def _get_exposure_params(exposure):
    params_all = exposure[0].psf.model.get_parameters(free=True, fixed=True)
    param_integral = None
    params = []
    for param in params_all:
        str_param = str(param)
        if str_param.startswith("gauss2d::fit::Integral"):
            if param_integral is None:
                param_integral = param
                continue
            else:
                raise RuntimeError("Got second Exposure IntegralParameter before entering first")
        elif str_param.startswith("gauss2d::fit::ProperFraction"):
            if param_integral is not None:
                params.append(param_integral)
                param_integral = None
        params.append(param)
    return params


def test_model_evaluation(channels, model, model_old):
    with pytest.raises(RuntimeError):
        model.evaluate()

    printout = False
    plot = True

    params_new_all = model.parameters()
    # Cheat a little and remove duplicate CentroidParameters without explicitly converting to a set
    param_cenx_found = None
    param_ceny_found = None
    param_frac_found = None
    param_integral_found = None

    params_new_str = []
    for param in params_new_all:
        str_param = str(param)
        if str_param.startswith("gauss2d::fit::Centroid"):
            is_x = str_param[22] == 'X'
            if not is_x and (str_param[22] != 'Y'):
                raise ValueError(f"Unexpected str_param={str_param}")
            param_found = param_cenx_found if is_x else param_ceny_found
            if param_found:
                if param != param_found:
                    raise RuntimeError(f"{param} != {param_found}")
                if is_x:
                    param_cenx_found = None
                else:
                    param_ceny_found = None
                continue
            else:
                if is_x:
                    param_cenx_found = param
                else:
                    param_ceny_found = param
        elif str_param.startswith("gauss2d::fit::Integral"):
            if param_integral_found:
                if param == param_integral_found:
                    param_integral_found = None
                    continue
                else:
                    param_integral_found = param
            else:
                param_integral_found = param
        elif str_param.startswith("gauss2d::fit::ProperFraction"):
            if not param.fixed:
                if param_frac_found:
                    if param != param_frac_found:
                        raise RuntimeError(f"{param} != {param_frac_found}")
                    param_frac_found = None
                    continue
                else:
                    param_frac_found = param
        elif str_param.startswith("gauss2d::fit::Reff"):
            str_param = str_param.replace("Reff", "Sigma")
        params_new_str.append(str_param)

    params_old = list(chain(
        model_old.get_parameters(),
        chain.from_iterable(_get_exposure_params(exp)
                            for exp in model_old.data.exposures.values()),
    ))

    params_new_str = '\n'.join(params_new_str)
    params_old_str = '\n'.join(f'{x}' for x in params_old)

    assert params_new_str == params_old_str

    model.setup_evaluators(print=printout)
    models = {'new': model, 'old': model_old}
    model.evaluate()
    model_old.evaluate(do_draw_image=True, keep_images=True)
    outputs = [model.outputs[0].data, list(model_old.data.exposures.values())[0][0].meta["img_model"]]

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(np.log10(outputs[0]))
        ax[1].imshow(np.log10(outputs[1]))
        ax[2].imshow(outputs[0] - outputs[1])
        plt.show()

    assert np.allclose(outputs[0], outputs[1], rtol=1e-3, atol=1e-5)

    for name, object in models.items():
        result = timeit.repeat('model.evaluate()', repeat=20, number=5, globals={'model': object})
        print(f'{name}: {np.min(result, axis=0)}')
