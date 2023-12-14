import astropy.table
from collections import defaultdict
from dataclasses import dataclass, field
import gauss2d.fit as g2f
from itertools import cycle
import matplotlib as mpl
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import typing

from .utils import get_params_uniq

__all__ = ["ErrorValues", "plot_catalog_bootstrap", "plot_loglike", "plot_sersicmix_interp"]


linestyles_default = ["--", "-.", ":"]
ln10 = np.log(10)


@dataclass(kw_only=True)
class ErrorValues:
    """Configuration for plotting uncertainties."""

    kwargs_plot: dict = field(default_factory=dict)
    values: np.ndarray


def plot_catalog_bootstrap(
    catalog_bootstrap: astropy.table.Table,
    n_bins: int = None,
    paramvals_ref=None,
    plot_total_fluxes: bool = False,
    plot_colors: bool = False,
    **kwargs,
):
    """Plot a bootstrap catalog for a single source model.

    Parameters
    ----------
    catalog_bootstrap
        A bootstrap catalog, as returned by
        `multiprofit.fit_bootstrap_model.CatalogSourceFitterBootstrap`.
    n_bins
        The number of bins for parameter value histograms. Default
        is sqrt(N) with a minimum of 10.
    paramvals_ref
        Reference parameter values to plot, if any.
    plot_total_fluxes
        Whether to plot total fluxes, not just component.
    plot_colors
        Whether to plot colors in addition to fluxes.
    kwargs
        Keyword arguments to pass to matplotlib hist calls.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis handles, as returned by plt.subplots.
    """
    n_sources = len(catalog_bootstrap)
    if n_bins is None:
        n_bins = np.max([int(np.ceil(np.sqrt(n_sources))), 10])

    config = catalog_bootstrap.meta["config"]
    prefix = config["prefix_column"]
    suffix_err = "_err"

    # TODO: There are probably better ways of doing this
    colnames_err = [col for col in catalog_bootstrap.colnames if col.endswith(suffix_err)]
    colnames_meas = [col[:-4] for col in colnames_err]
    n_params_init = len(colnames_meas)
    if paramvals_ref and (len(paramvals_ref) != n_params_init):
        raise ValueError(f"{len(paramvals_ref)=} != {n_params_init=}")

    results_good = catalog_bootstrap[catalog_bootstrap["mpf_n_iter"] > 0]

    if plot_total_fluxes or plot_colors:
        if paramvals_ref:
            paramvals_ref = {
                colname: paramval_ref for colname, paramval_ref in zip(colnames_meas, paramvals_ref)
            }
        results_dict = {}
        for colname_meas, colname_err in zip(colnames_meas, colnames_err):
            results_dict[colname_meas] = results_good[colname_meas]
            results_dict[colname_err] = results_good[colname_err]

        colnames_flux = [colname for colname in colnames_meas if colname.endswith("_flux")]

        colnames_flux_band = defaultdict(list)
        colnames_flux_comp = defaultdict(list)

        for colname in colnames_flux:
            colname_short = colname.partition(prefix)[-1]
            comp, band = colname_short[:-5].split("_")
            colnames_flux_band[band].append(colname)
            colnames_flux_comp[comp].append(colname)

        band_prev = None
        for band, colnames_band in colnames_flux_band.items():
            for suffix, target in (("", colnames_meas), ("_err", colnames_err)):
                is_err = suffix == "_err"
                colname_flux = f"{prefix}{band}_flux{suffix}"
                total = np.sum(
                    [results_good[f"{colname}{suffix}"] ** (1 + is_err) for colname in colnames_band], axis=0
                )
                if is_err:
                    total = np.sqrt(total)
                elif paramvals_ref:
                    paramvals_ref[colname_flux] = sum((paramvals_ref[colname] for colname in colnames_band))
                results_dict[colname_flux] = total
                if plot_total_fluxes:
                    target.append(colname_flux)

            if band_prev:
                flux_prev, flux = (results_dict[f"{prefix}{b}_flux"] for b in (band_prev, band))
                mag_prev, mag = (-2.5 * np.log10(flux_b) for flux_b in (flux_prev, flux))
                mag_err_prev, mag_err = (
                    results_dict[f"{prefix}{b}_flux{suffix_err}"] / (-0.4 * flux_b * ln10)
                    for b, flux_b in ((band_prev, flux_prev), (band, flux))
                )
                colname_color = f"{prefix}{band_prev}-{band}_flux"
                colnames_meas.append(colname_color)
                colnames_err.append(f"{colname_color}{suffix_err}")

                results_dict[colname_color] = mag_prev - mag
                results_dict[f"{colname_color}{suffix_err}"] = 2.5 / ln10 * np.hypot(mag_err, mag_err_prev)
                if paramvals_ref:
                    mag_prev_ref, mag_ref = (
                        -2.5 * np.log10(paramvals_ref[f"{prefix}{b}_flux"]) for b in (band_prev, band)
                    )
                    paramvals_ref[colname_color] = mag_prev_ref - mag_ref

            band_prev = band

        results_good = results_dict
        if paramvals_ref:
            paramvals_ref = tuple(paramvals_ref.values())

    n_colnames = len(colnames_err)
    n_cols = 3
    n_rows = int(np.ceil(n_colnames / n_cols))

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, constrained_layout=True)
    idx_row, idx_col = 0, 0

    for idx_colname in range(n_colnames):
        colname_meas = colnames_meas[idx_colname]
        colname_short = colname_meas.partition(prefix)[-1]
        values = results_good[colname_meas]
        errors = results_good[colnames_err[idx_colname]]
        median = np.median(values)
        std = np.std(values)

        median_err = np.median(errors)

        axis = ax[idx_row][idx_col]
        axis.hist(values, bins=n_bins, color="b", label="fit values", **kwargs)

        label = "median +/- stddev"
        for offset in (-std, 0, std):
            axis.axvline(median + offset, label=label, color="k")
            label = None
        if paramvals_ref is not None:
            value_ref = paramvals_ref[idx_colname]
            label_value = f" {value_ref=:.3e} bias={median - value_ref:.3e}"
            axis.axvline(value_ref, label="reference", color="k", linestyle="--")
        else:
            label_value = f" {median=:.3e}"
        axis.hist(median + errors, bins=n_bins, color="r", label="median + error", **kwargs)
        axis.set_title(f"{colname_short} {std=:.3e} vs {median_err=:.3e}")
        axis.set_xlabel(f"{colname_short} {label_value}")
        axis.legend()

        idx_col += 1

        if idx_col == n_cols:
            idx_row += 1
            idx_col = 0

    return fig, ax


def plot_loglike(
    model: g2f.Model,
    params: list[g2f.ParameterD] = None,
    n_values: int = 15,
    errors: dict[str, ErrorValues] = None,
    values_reference: np.ndarray = None,
):
    """Plot the loglikehood and derivatives vs free parameter values around
       best-fit values.

    Parameters
    ----------
    model
        The model to evaluate.
    params
        Free parameters to plot marginal loglikelihood for.
    n_values
        The number of evaluations to make on either side of each param value.
    errors
        A dict keyed by label of uncertainties to plot. Values must be the same
        length as `params`.
    values_reference
        Reference values to plot (e.g. true parameter values). Must be the same
        length as `params`.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis handles, as returned by plt.subplots.
    """
    if errors is None:
        errors = {}
    loglike_grads = np.array(model.compute_loglike_grad())
    loglike_init = np.array(model.evaluate())

    if params is None:
        params = tuple(get_params_uniq(model, fixed=False))

    n_params = len(params)

    if values_reference is not None and len(values_reference) != n_params:
        raise ValueError(f"{len(values_reference)=} != {n_params=}")

    n_rows = n_params
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(10, 3 * n_rows))
    axes = [ax] if (n_rows == 1) else ax

    n_loglikes = len(loglike_init)
    labels = [channel.name for channel in model.data.channels]
    labels.extend(["prior", "total"])

    for param in params:
        param.fixed = True

    for row, param in enumerate(params):
        value_init = param.value
        param.fixed = False
        values = [value_init]
        loglikes = [loglike_init * 0]
        dlls = [loglike_grads[row]]

        diff_init = 1e-4 * np.sign(loglike_grads[row])
        diff = diff_init

        # TODO: This entire scheme should be improved/replaced
        # It sometimes takes excessively large steps
        # Option: Try to fit a curve once there are a couple of points
        # on each side of the peak
        idx_prev = -1
        for idx in range(2 * n_values):
            try:
                param.value_transformed += diff
                loglikes_new = np.array(model.evaluate()) - loglike_init
                dloglike_actual = np.sum(loglikes_new) - np.sum(loglikes[idx_prev])
                values.append(param.value)
                loglikes.append(loglikes_new)
                dloglike_actual_abs = np.abs(dloglike_actual)
                if dloglike_actual_abs > 1:
                    diff /= dloglike_actual_abs
                elif dloglike_actual_abs < 0.5:
                    diff /= np.clip(dloglike_actual_abs, 0.2, 0.5)
                dlls.append(model.compute_loglike_grad()[0])
                if idx == n_values:
                    diff = -diff_init
                    param.value = value_init
                    idx_prev = 0
                else:
                    idx_prev = -1
            except RuntimeError:
                break
        param.value = value_init
        param.fixed = True

        subplot = axes[row][0]
        sorted = np.argsort(values)
        values = np.array(values)[sorted]
        loglikes = [loglikes[idx] for idx in sorted]
        dlls = np.array(dlls)[sorted]

        for idx in range(n_loglikes):
            subplot.plot(values, [loglike[idx] for loglike in loglikes], label=labels[idx])
        subplot.plot(values, np.sum(loglikes, axis=1), label=labels[-1])
        vline_kwargs = dict(ymin=np.min(loglikes) - 1, ymax=np.max(loglikes) + 1, color="k")
        subplot.vlines(value_init, **vline_kwargs)

        suffix = f" {param.label}" if param.label else ""
        subplot.legend()
        subplot.set_title(f"{param.name}{suffix}")
        subplot.set_ylabel("loglike")
        subplot.set_ylim(vline_kwargs["ymin"], vline_kwargs["ymax"])

        subplot = axes[row][1]
        subplot.plot(values, dlls)
        subplot.axhline(0, color="k")
        subplot.set_ylabel("dloglike/dx")

        vline_kwargs = dict(ymin=np.min(dlls), ymax=np.max(dlls))
        subplot.vlines(value_init, **vline_kwargs, color="k", label="fit")
        if values_reference is not None:
            subplot.vlines(values_reference[row], **vline_kwargs, color="b", label="ref")

        cycler_linestyle = cycle(linestyles_default)
        for name_error, valerr in errors.items():
            linestyle = valerr.kwargs_plot.pop("linestyle", next(cycler_linestyle))
            for idx_ax in range(2):
                axes[row][idx_ax].vlines(
                    [value_init - valerr.values[row], value_init + valerr.values[row]],
                    linestyles=[linestyle, linestyle],
                    label=name_error if (idx_ax == 1) else None,
                    **valerr.kwargs_plot,
                    **vline_kwargs,
                )
        subplot.legend()

    for param in params:
        param.fixed = False

    return fig, ax


Interpolator: typing.TypeAlias = g2f.SersicMixInterpolator | tuple[typing.Type, dict[str, typing.Any]]


def plot_sersicmix_interp(
    interps: dict[str, tuple[Interpolator, str | tuple]], n_ser: np.ndarray, **kwargs
) -> matplotlib.figure.Figure:
    """Plot Gaussian mixture Sersic profile interpolated values.

    Parameters
    ----------
    interps
        Dict of interpolators by name.
    n_ser
        Array of Sersic index values to plot interpolated quantities for.
    kwargs
        Keyword arguments to pass to matplotlib.pyplot.subplots.

    Returns
    -------
        The resulting figure.
    """
    orders = {
        name: interp.order
        for name, (interp, _) in interps.items()
        if isinstance(interp, g2f.SersicMixInterpolator)
    }
    order = set(orders.values())
    if not len(order) == 1:
        raise ValueError(f"len(set({orders})) != 1; all interpolators must have the same order")
    order = tuple(order)[0]

    cmap = mpl.cm.get_cmap("tab20b")
    colors_ord = [None] * order
    for i_ord in range(order):
        colors_ord[i_ord] = cmap(i_ord / (order - 1.0))

    n_ser_min = np.min(n_ser)
    n_ser_max = np.max(n_ser)
    knots = g2f.sersic_mix_knots(order=order)
    n_knots = len(knots)
    integrals_knots = np.empty((n_knots, order))
    sigmas_knots = np.empty((n_knots, order))
    n_ser_knots = np.empty(n_knots)

    i_knot_first = None
    i_knot_last = n_knots
    for i_knot, knot in enumerate(knots):
        if i_knot_first is None:
            if knot.sersicindex > n_ser_min:
                i_knot_first = i_knot
            else:
                continue
        if knot.sersicindex > n_ser_max:
            i_knot_last = i_knot
            break
        n_ser_knots[i_knot] = knot.sersicindex
        for i_ord in range(order):
            values = knot.values[i_ord]
            integrals_knots[i_knot, i_ord] = values.integral
            sigmas_knots[i_knot, i_ord] = values.sigma
    range_knots = range(i_knot_first, i_knot_last)
    integrals_knots = integrals_knots[range_knots, :]
    sigmas_knots = sigmas_knots[range_knots, :]
    n_ser_knots = n_ser_knots[range_knots]

    n_values = len(n_ser)
    integrals, dintegrals, sigmas, dsigmas = (
        {name: np.empty((n_values, order)) for name in interps} for _ in range(4)
    )

    for name, (interp, _) in interps.items():
        if not isinstance(interp, g2f.SersicMixInterpolator):
            kwargs = interp[1] if interp[1] is not None else {}
            interp = interp[0]
            x = [knot.sersicindex for knot in knots]
            for i_ord in range(order):
                integrals_i = np.empty(n_knots, dtype=float)
                sigmas_i = np.empty(n_knots, dtype=float)
                for i_knot, knot in enumerate(knots):
                    integrals_i[i_knot] = knot.values[i_ord].integral
                    sigmas_i[i_knot] = knot.values[i_ord].sigma
                interp_int = interp(x, integrals_i, **kwargs)
                dinterp_int = interp_int.derivative()
                interp_sigma = interp(x, sigmas_i, **kwargs)
                dinterp_sigma = interp_sigma.derivative()
                for i_val, value in enumerate(n_ser):
                    integrals[name][i_val, i_ord] = interp_int(value)
                    sigmas[name][i_val, i_ord] = interp_sigma(value)
                    dintegrals[name][i_val, i_ord] = dinterp_int(value)
                    dsigmas[name][i_val, i_ord] = dinterp_sigma(value)

    for i_val, value in enumerate(n_ser):
        for name, (interp, _) in interps.items():
            if isinstance(interp, g2f.SersicMixInterpolator):
                values = interp.integralsizes(value)
                derivs = interp.integralsizes_derivs(value)
                for i_ord in range(order):
                    integrals[name][i_val, i_ord] = values[i_ord].integral
                    sigmas[name][i_val, i_ord] = values[i_ord].sigma
                    dintegrals[name][i_val, i_ord] = derivs[i_ord].integral
                    dsigmas[name][i_val, i_ord] = derivs[i_ord].sigma

    fig, axes = plt.subplots(2, 2, **kwargs)
    for idx_row, (yv, yd, yk, y_label) in (
        (0, (integrals, dintegrals, integrals_knots, "integral")),
        (1, (sigmas, dsigmas, sigmas_knots, "sigma")),
    ):
        is_label_row = idx_row == 1
        for idx_col, y_i, y_prefix in ((0, yv, ""), (1, yd, "d")):
            is_label_col = idx_col == 0
            make_label = is_label_col and is_label_row
            axis = axes[idx_row, idx_col]
            if is_label_col:
                for i_ord in range(order):
                    axis.plot(
                        n_ser_knots,
                        yk[:, i_ord],
                        "kx",
                        label="knots" if make_label and (i_ord == 0) else None,
                    )
            for name, (_, lstyle) in interps.items():
                for i_ord in range(order):
                    label = f"{name}" if make_label and (i_ord == 0) else None
                    axis.plot(n_ser, y_i[name][:, i_ord], c=colors_ord[i_ord], label=label, linestyle=lstyle)
                axis.set_xlim((n_ser_min, n_ser_max))
                axis.set_ylabel(f"{y_prefix}{y_label}")
            if make_label:
                axis.legend(loc="upper left")
    return fig
