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

import gauss2d.fit as g2f
import numpy as np
import pytest
from lsst.multiprofit.componentconfig import (
    GaussianConfig,
    ParameterConfig,
    SersicConfig,
    SersicIndexConfig,
    init_component,
)
from lsst.multiprofit.fit_bootstrap_model import (
    CatalogExposurePsfBootstrap,
    CatalogExposureSourcesBootstrap,
    CatalogSourceFitterBootstrap,
)
from lsst.multiprofit.fit_psf import CatalogPsfFitter, CatalogPsfFitterConfig
from lsst.multiprofit.fit_source import CatalogSourceFitterConfig
from lsst.multiprofit.modeller import ModelFitConfig
from lsst.multiprofit.plots import ErrorValues, plot_catalog_bootstrap, plot_loglike
from lsst.multiprofit.utils import get_params_uniq

channels = (g2f.Channel.get("g"), g2f.Channel.get("r"), g2f.Channel.get("i"))
shape_img = (23, 27)
sigma_psf = 2.1
reff_x_src, reff_y_src, rho_src, nser_src = 2.5, 3.6, -0.25, 2.0

# TODO: These can be parameterized; should they be?
compute_errors_no_covar = True
n_sources = 5
# This is for interactive debugging
plot = False


@pytest.fixture(scope="module")
def config_psf():
    return CatalogPsfFitterConfig(
        gaussians={
            "comp1": GaussianConfig(
                size_x=ParameterConfig(value_initial=sigma_psf),
                size_y=ParameterConfig(value_initial=sigma_psf),
            )
        },
    )


@pytest.fixture(scope="module")
def config_source_fit():
    # TODO: Separately test n_pointsources=0 and sersics={}
    return CatalogSourceFitterConfig(
        config_fit=ModelFitConfig(fit_linear_iter=3),
        n_pointsources=1,
        sersics={
            "comp1": SersicConfig(
                prior_size_mean=reff_y_src,
                prior_size_stddev=1.0,
                prior_axrat_mean=reff_x_src / reff_y_src,
                prior_axrat_stddev=0.2,
                sersicindex=SersicIndexConfig(fixed=False, value_initial=1.0),
            )
        },
        convert_cen_xy_to_radec=False,
        compute_errors_no_covar=compute_errors_no_covar,
        compute_errors_from_jacobian=False,
    )


@pytest.fixture(scope="module")
def table_psf_fits(config_psf):
    fitter = CatalogPsfFitter()
    fits = {
        channel.name: fitter.fit(
            CatalogExposurePsfBootstrap(
                sigma_x=reff_x_src,
                sigma_y=reff_y_src,
                rho=rho_src,
                nser=nser_src,
                n_sources=n_sources,
            ),
            config_psf,
        )
        for channel in channels
    }
    return fits


def test_fit_psf(config_psf, table_psf_fits):
    for results in table_psf_fits.values():
        assert len(results) == n_sources
        assert np.sum(results["mpf_psf_unknown_flag"]) == 0
        assert all(np.isfinite(list(results[0].values())))
        psfmodel = config_psf.rebuild_psfmodel(results[0])
        assert len(psfmodel.components) == len(config_psf.gaussians)


def test_fit_source(config_source_fit, table_psf_fits):
    model_source, *_ = config_source_fit.make_source(channels=channels)
    # Have to do this here so that the model initializes its observation with
    # the extended component having the right size
    init_component(model_source.components[1], sigma_x=sigma_psf, sigma_y=sigma_psf, rho=0)
    CatalogExposureSourcesBootstrap(
        channel_name=channels[0].name,
        config_fit=config_source_fit,
        model_source=model_source,
        table_psf_fits=table_psf_fits[channels[0].name],
        n_sources=n_sources,
    )
    catexps = tuple(
        CatalogExposureSourcesBootstrap(
            channel_name=channel.name,
            config_fit=config_source_fit,
            model_source=model_source,
            table_psf_fits=table_psf_fits[channel.name],
            n_sources=n_sources,
        )
        for channel in channels
    )
    fitter = CatalogSourceFitterBootstrap(reff_x=reff_x_src, reff_y=reff_y_src, rho=rho_src, nser=nser_src)
    catalog_multi = catexps[0].get_catalog()
    results = fitter.fit(catalog_multi=catalog_multi, catexps=catexps, config=config_source_fit)
    assert len(results) == n_sources

    model = fitter.get_model(
        0, catalog_multi=catalog_multi, catexps=catexps, config=config_source_fit, results=results
    )

    model_true = g2f.Model(data=model.data, psfmodels=model.psfmodels, sources=[model_source])
    fitter.initialize_model(model_true, catalog_multi[0])
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
                model, transformed=False, options=g2f.HessianOptions(return_negative=return_negative)
            )
        )
        if return_negative:
            variances = np.array(variances)
            variances[variances <= 0] = 0
            variances = list(variances)
        else:
            assert np.all(variances[-1] > 0)

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
    fig, ax = plot_loglike(model, errors=errors_plot, values_reference=fitter.params_values_init)
    if plot:
        plt.tight_layout()
        plt.show()
        plt.close()
