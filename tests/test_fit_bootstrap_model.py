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

import astropy.table
import gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    CentroidConfig,
    FluxFractionParameterConfig,
    FluxParameterConfig,
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.fit_bootstrap_model import (
    CatalogExposurePsfBootstrap,
    CatalogExposureSourcesBootstrap,
    CatalogPsfBootstrapConfig,
    CatalogSourceBootstrapConfig,
    CatalogSourceFitterBootstrap,
    NoisyObservationConfig,
    NoisyPsfObservationConfig,
)
from lsst.multiprofit.fit_psf import CatalogPsfFitter, CatalogPsfFitterConfig, CatalogPsfFitterConfigData
from lsst.multiprofit.fit_source import CatalogSourceFitterConfig, CatalogSourceFitterConfigData
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.modeller import ModelFitConfig
from lsst.multiprofit.observationconfig import CoordinateSystemConfig
from lsst.multiprofit.plots import ErrorValues, plot_catalog_bootstrap, plot_loglike
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.multiprofit.utils import get_params_uniq
import numpy as np
import pytest

shape_img = (23, 27)
reff_x_src, reff_y_src, rho_src, nser_src = 2.5, 3.6, -0.25, 2.0

# TODO: These can be parameterized; should they be?
compute_errors_no_covar = True
compute_errors_from_jacobian = True
include_point_source = False
n_sources = 3
# Set to True for interactive debugging (but don't commit)
plot = False


@pytest.fixture(scope="module")
def channels():
    return {band: g2f.Channel.get(band) for band in ("R", "G", "B")}


@pytest.fixture(scope="module")
def configfitter_psfs(channels) -> dict[g2f.Channel, CatalogExposurePsfBootstrap]:
    config_datas = {}
    for idx, (band, channel) in enumerate(channels.items()):
        n_rows = 17 + idx*2
        n_cols = 15 + idx*2
        config = CatalogPsfFitterConfig(
            model=SourceConfig(
                component_groups={"": ComponentGroupConfig(
                    centroids={
                        "default": CentroidConfig(
                            x=ParameterConfig(value_initial=n_cols/2.),
                            y=ParameterConfig(value_initial=n_rows/2.),
                        ),
                    },
                    components_gauss={
                        "comp1": GaussianComponentConfig(
                            flux=FluxParameterConfig(value_initial=1.0, fixed=True),
                            fluxfrac=FluxFractionParameterConfig(value_initial=0.5, fixed=False),
                            size_x=ParameterConfig(value_initial=1.5 + 0.1*idx),
                            size_y=ParameterConfig(value_initial=1.7 + 0.13*idx),
                            rho=ParameterConfig(value_initial=-0.035 - 0.007*idx),
                        ),
                        "comp2": GaussianComponentConfig(
                            size_x=ParameterConfig(value_initial=3.1 + 0.24*idx),
                            size_y=ParameterConfig(value_initial=2.7 + 0.16*idx),
                            rho=ParameterConfig(value_initial=0.06 + 0.012*idx),
                            fluxfrac=FluxFractionParameterConfig(value_initial=1.0, fixed=True),
                        ),
                    },
                    is_fractional=True,
                )}
            ),
        )
        config_boot = CatalogPsfBootstrapConfig(
            observation=NoisyPsfObservationConfig(n_rows=n_rows, n_cols=n_cols, gain=1e5),
            n_sources=n_sources,
        )
        config_data = CatalogExposurePsfBootstrap(config=config, config_boot=config_boot)
        config_datas[channel] = config_data

    return config_datas


@pytest.fixture(scope="module")
def configfitter_source(channels) -> CatalogSourceFitterConfigData:
    config = CatalogSourceFitterConfig(
        config_fit=ModelFitConfig(fit_linear_iter=3),
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_gauss={
                                "ps": GaussianComponentConfig(
                                    flux=FluxParameterConfig(value_initial=1000),
                                    rho=ParameterConfig(value_initial=0, fixed=True),
                                    size_x=ParameterConfig(value_initial=0, fixed=True),
                                    size_y=ParameterConfig(value_initial=0, fixed=True),
                                )
                            } if include_point_source else {},
                            components_sersic={
                                "ser": SersicComponentConfig(
                                    prior_size_mean=reff_y_src,
                                    prior_size_stddev=1.0,
                                    prior_axrat_mean=reff_x_src / reff_y_src,
                                    prior_axrat_stddev=0.2,
                                    flux=FluxParameterConfig(value_initial=5000),
                                    rho=ParameterConfig(value_initial=rho_src),
                                    size_x=ParameterConfig(value_initial=reff_x_src),
                                    size_y=ParameterConfig(value_initial=reff_y_src),
                                    sersic_index=SersicIndexParameterConfig(fixed=False, value_initial=1.0),
                                ),
                            }
                        )
                    }
                ),
            },
        ),
        convert_cen_xy_to_radec=False,
        compute_errors_no_covar=compute_errors_no_covar,
        compute_errors_from_jacobian=compute_errors_from_jacobian,
    )
    config_data = CatalogSourceFitterConfigData(
        channels=tuple(channels.values()),
        config=config,
    )
    return config_data


@pytest.fixture(scope="module")
def tables_psf_fits(configfitter_psfs) -> dict[g2f.Channel, astropy.table.Table]:
    fitter = CatalogPsfFitter()
    fits = {
        channel: fitter.fit(
            catexp=configfitter_psf,
            config_data=configfitter_psf,
        )
        for channel, configfitter_psf in configfitter_psfs.items()
    }
    return fits


@pytest.fixture(scope="module")
def config_data_sources(
    configfitter_psfs, tables_psf_fits,
) -> dict[g2f.Channel, CatalogExposureSourcesBootstrap]:
    config_datas = {}
    for idx, (channel, configfitter_psf) in enumerate(configfitter_psfs.items()):
        table_psf_fits = tables_psf_fits[channel]
        n_rows = shape_img[0] + idx*2
        n_cols = shape_img[1] + idx*2
        config_boot = CatalogSourceBootstrapConfig(
            observation=NoisyObservationConfig(
                n_rows=n_rows, n_cols=n_cols, band=channel.name, background=100,
                coordsys=CoordinateSystemConfig(x_min=-2 + 3*idx, y_min=5 - 4*idx),
            ),
            n_sources=n_sources,
        )
        config_data = CatalogExposureSourcesBootstrap(
            config_boot=config_boot,
            table_psf_fits=table_psf_fits,
        )
        config_datas[channel] = config_data

    return config_datas


def test_fit_psf(configfitter_psfs, tables_psf_fits):
    for band, results in tables_psf_fits.items():
        assert len(results) == n_sources
        assert np.sum(results["mpf_psf_unknown_flag"]) == 0
        assert all(np.isfinite(list(results[0].values())))
        config_data_psf = configfitter_psfs[band]
        psfmodel_init = config_data_psf.config.make_psfmodel()
        psfdata = CatalogPsfFitterConfigData(config=config_data_psf.config)
        psfmodel_fit = psfdata.psfmodel
        psfdata.init_psfmodel(results[0])
        assert len(psfmodel_init.components) == len(psfmodel_fit.components)
        params_init = psfmodel_init.parameters()
        params_fit = psfmodel_fit.parameters()
        assert len(params_init) == len(params_fit)
        for p_init, p_meas in zip(params_init, params_fit):
            assert p_meas.fixed == p_init.fixed
            if p_meas.fixed:
                assert p_init.value == p_meas.value
            else:
                # TODO: come up with better (noise-dependent) thresholds here
                if isinstance(p_init, g2f.IntegralParameterD):
                    atol, rtol = 0, 0.02
                elif isinstance(p_init, g2f.ProperFractionParameterD):
                    atol, rtol = 0.1, 0.01
                elif isinstance(p_init, g2f.RhoParameterD):
                    atol, rtol = 0.05, 0.1
                else:
                    atol, rtol = 0.01, 0.1
                assert np.isclose(p_init.value, p_meas.value, atol=atol, rtol=rtol)


def test_fit_source(configfitter_source, config_data_sources):
    fitter = CatalogSourceFitterBootstrap()
    # We don't have or need multiband input catalog, so just pretend the first one is
    catalog_multi = next(iter(config_data_sources.values())).get_catalog()
    catexps = list(config_data_sources.values())
    results = fitter.fit(catalog_multi=catalog_multi, catexps=catexps, config_data=configfitter_source)
    assert len(results) == n_sources

    model = fitter.get_model(
        0, catalog_multi=catalog_multi, catexps=catexps, config_data=configfitter_source, results=results
    )

    model_sources, priors = configfitter_source.config.make_sources(channels=list(config_data_sources.keys()))
    model_true = g2f.Model(data=model.data, psfmodels=model.psfmodels, sources=model_sources)
    fitter.initialize_model(model_true, catalog_multi[0], catexps=catexps)
    params_true = tuple(param.value for param in get_params_uniq(model_true, fixed=False))
    plot_catalog_bootstrap(
        results, histtype="step", paramvals_ref=params_true, plot_total_fluxes=True, plot_colors=True
    )
    if plot:
        import matplotlib.pyplot as plt

        plt.show()

    assert np.sum(results["mpf_unknown_flag"]) == 0
    assert all(np.isfinite(list(results[0].values())))

    variances = []
    for return_negative in (False, True):
        variances.append(
            fitter.modeller.compute_variances(
                model, transformed=False, options=g2f.HessianOptions(return_negative=return_negative),
                use_diag_only=True,
            )
        )
        assert np.all(variances[-1] > 0)
        if return_negative:
            variances = np.array(variances)
            variances[variances <= 0] = 0
            variances = list(variances)

    # Bootstrap errors
    model.setup_evaluators(evaluatormode=model.EvaluatorMode.image)
    model.evaluate()
    img_data_old = []
    for obs, output in zip(model.data, model.outputs):
        img_data_old.append(obs.image.data.copy())
        img = obs.image.data
        img.flat = output.data.flat
    options_hessian = g2f.HessianOptions(return_negative=return_negative)
    variances_bootstrap = fitter.modeller.compute_variances(model, transformed=False, options=options_hessian)
    variances_bootstrap_diag = fitter.modeller.compute_variances(
        model, transformed=False, options=options_hessian, use_diag_only=True
    )
    for obs, img_datum_old in zip(model.data, img_data_old):
        obs.image.data.flat = img_datum_old.flat
    variances_jac = fitter.modeller.compute_variances(model, transformed=False)
    variances_jac_diag = fitter.modeller.compute_variances(model, transformed=False, use_diag_only=True)

    errors_plot = {
        "inv_hess": ErrorValues(values=np.sqrt(variances[0]), kwargs_plot={"linestyle": "-", "color": "r"}),
        "-inv_hess": ErrorValues(values=np.sqrt(variances[1]), kwargs_plot={"linestyle": "--", "color": "r"}),
        "inv_jac": ErrorValues(values=np.sqrt(variances_jac), kwargs_plot={"linestyle": "-.", "color": "r"}),
        "boot_hess": ErrorValues(
            values=np.sqrt(variances_bootstrap), kwargs_plot={"linestyle": "-", "color": "b"}
        ),
        "boot_diag": ErrorValues(
            values=np.sqrt(variances_bootstrap_diag), kwargs_plot={"linestyle": "--", "color": "b"}
        ),
        "boot_jac_diag": ErrorValues(
            values=np.sqrt(variances_jac_diag), kwargs_plot={"linestyle": "-.", "color": "m"}
        ),
    }
    fig, ax = plot_loglike(model, errors=errors_plot, values_reference=params_true)
    if plot:
        plt.tight_layout()
        plt.show()
