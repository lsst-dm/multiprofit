from dataclasses import dataclass
import gauss2d as g2
import gauss2d.fit as g2f
from itertools import chain
import multiprofit.fitutils as mpffit
from multiprofit.multigaussianapproxprofile import MultiGaussianApproximationComponent
import multiprofit.objects as mpfobj
from multiprofit.transforms import transforms_ref
import numpy as np
import pytest
import scipy.optimize as spopt
import time
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
    n_x: int = 41
    n_y: int = 27
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
    images = {}
    for band in bands:
        image = g2.ImageD(config.n_y, config.n_x)
        sigma_inv = g2.ImageD(config.n_y, config.n_x)
        sigma_inv.data.flat = 1/1e-3
        mask_inv = g2.ImageB(np.ones((config.n_y, config.n_x), dtype=bool))
        images[band] = (image, sigma_inv, mask_inv)
    return images


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
    translog = transforms_ref['log10']
    transrho = transforms_ref['logit_rho']
    for i in range(len(psfmodels)):
        components = [None]*n_comps
        centroid = g2f.CentroidParameters(
            g2f.CentroidXParameterD(config.n_y/2., limits=limits.x),
            g2f.CentroidYParameterD(config.n_x/2., limits=limits.y),
        )
        for c in range(n_comps):
            is_last = c == n_last
            last = g2f.FractionalIntegralModel(
                {
                    g2f.Channel.NONE: g2f.ProperFractionParameterD(
                        (is_last == 1) or (0.5 + 0.5*(c > 0)), fixed=is_last,
                        transform=transforms_ref['logit']
                    )
                },
                g2f.LinearIntegralModel({
                    g2f.Channel.NONE: g2f.IntegralParameterD(1.0, fixed=True)
                }) if (c == 0) else last,
                is_last,
            )
            components[c] = g2f.GaussianComponent(
                g2f.GaussianParametricEllipse(
                    g2f.SigmaXParameterD(compconf.size_base + c*compconf.size_increment, transform=translog),
                    g2f.SigmaYParameterD(compconf.size_base + c*compconf.size_increment, transform=translog),
                    g2f.RhoParameterD(compconf.rho_base + c*compconf.rho_increment, limits=limits.rho,
                                      transform=transrho),
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
                (g2f.SigmaXParameterD if old else g2f.ReffXParameterD)(size, transform=transforms_ref['log10']),
                (g2f.SigmaYParameterD if old else g2f.ReffYParameterD)(size, transform=transforms_ref['log10']),
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
def model_jac(data, psfmodels, sources):
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


def fit_model(model: g2f.Model, jacobian, residuals):
    def residual_func(params_new, model, params, jacobian, residuals):
        for param, value in zip(params, params_new):
            param.value_transformed = value
        model.evaluate()
        return residuals

    def jacobian_func(params_new, model, params, jacobian, residuals):
        return -jacobian

    params = list({x: None for x in model.parameters(paramfilter=g2f.ParamFilter(fixed=False))})
    n_params = len(params)
    bounds = ([None]*n_params, [None]*n_params)
    params_init = [None]*n_params

    for idx, param in enumerate(params):
        limits = param.limits
        bounds[0][idx] = limits.min
        bounds[1][idx] = limits.max
        params_init[idx] = param.value_transformed + 0.02

    time_init = time.process_time()
    result = spopt.least_squares(
        residual_func, params_init, jac=jacobian_func, bounds=bounds,
        args=(model, params, jacobian, residuals), x_scale='jac',
        ftol=2e-4, xtol=2e-4
    )
    time_run = time.process_time() - time_init
    return result, time_run


def test_model_evaluation(channels, model, model_jac, model_old, images):
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

    # to be deprecated
    # assert params_new_str == params_old_str

    # Freeze the PSF params - they can't be fit anyway
    for m in (model, model_jac):
        for psfmodel in m.psfmodels:
            params = psfmodel.parameters()
            for param in params:
                param.fixed = True

    model.setup_evaluators(print=printout)
    models = {
        'new': (model, ''),
        'new-jac': (model_jac, ''),
        'old': (model_old, ''),
        'old-jac': (model_old, 'do_jacobian=True'),
    }
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

    n_priors = 0
    n_obs = len(model.data)
    n_rows = np.zeros(n_obs, dtype=int)
    n_cols = np.zeros(n_obs, dtype=int)
    datasizes = np.zeros(n_obs, dtype=int)
    ranges_params = [None]*n_obs
    params_free = list({x: None for x in model_jac.parameters(paramfilter=g2f.ParamFilter(fixed=False))})

    # There's one extra validation array
    n_params_jac = len(params_free) + 1
    assert n_params_jac > 1

    for idx_obs in range(n_obs):
        observation = model.data[idx_obs]
        output = model.outputs[idx_obs]
        observation.image.data.flat = output.data.flat + np.random.normal(
            loc=0, scale=(1/observation.sigma_inv.data).flat)
        n_rows[idx_obs] = observation.image.n_rows
        n_cols[idx_obs] = observation.image.n_cols
        datasizes[idx_obs] = n_rows[idx_obs]*n_cols[idx_obs]
        params = list({
            x: None
            for x in model_jac.parameters(paramfilter=g2f.ParamFilter(fixed=False, channel=observation.channel))
        })
        n_params_obs = len(params)
        ranges_params_obs = [0]*(n_params_obs + 1)
        for idx_param in range(n_params_obs):
            ranges_params_obs[idx_param + 1] = params_free.index(params[idx_param]) + 1
        ranges_params[idx_obs] = ranges_params_obs

    n_free_first = len(ranges_params[0])
    assert all([len(rp) == n_free_first for rp in ranges_params[1:]])

    jacobians = [None]*n_obs
    residuals = [None]*n_obs
    datasize = np.sum(datasizes) + n_priors
    jacobian = np.zeros((datasize, n_params_jac))
    residual = np.zeros(datasize)
    # jacobian_prior = self.jacobian[datasize:, ].view()

    offset = 0
    for idx_obs in range(n_obs):
        size_obs = datasizes[idx_obs]
        end = offset + size_obs
        shape = (n_rows[idx_obs], n_cols[idx_obs])
        ranges_params_obs = ranges_params[idx_obs]
        jacobians_obs = [None]*(len(ranges_params_obs))
        for idx_param, idx_jac in enumerate(ranges_params_obs):
            jacobians_obs[idx_param] = g2.ImageD(jacobian[offset:end, idx_jac].view().reshape(shape))
        jacobians[idx_obs] = jacobians_obs
        residuals[idx_obs] = g2.ImageD(residual[offset:end].view().reshape(shape))
        offset = end

    model.setup_evaluators(evaluatormode=g2f.Model.EvaluatorMode.loglike)
    likelihood = model.evaluate()

    model_jac.setup_evaluators(
        evaluatormode=g2f.Model.EvaluatorMode.jacobian,
        outputs=jacobians,
        residuals=residuals,
        print=printout,
    )
    model_jac.verify_jacobian()
    likelihood_jac = model_jac.evaluate()

    assert likelihood == likelihood_jac

    result, time = fit_model(model_jac, jacobian[:, 1:], residual)
    for param, value in zip(params_free, result.x):
        param.value_transformed = value
    likelihood = model.evaluate()
    print(f'got like={likelihood} in t={time:.3e}, result: \n{result}')

    model_old.evaluate(do_fit_leastsq_prep=True, do_fit_nonlinear_prep=True, do_jacobian=True)

    n_eval = 10

    n_name_model_max = max(len(x) for x in models.keys())
    format_name = f'{{0: <{n_name_model_max}}}'

    for name, obj in models.items():
        result = np.array(
            timeit.repeat(f'model.evaluate({obj[1]})', repeat=n_eval, number=n_eval,
                          globals={'model': obj[0]})
        )/n_eval
        print(f'{format_name.format(name)}: min={np.min(result, axis=0):.4e}'
              f', med={np.median(result, axis=0):.4e}')

