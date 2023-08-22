from dataclasses import dataclass, field
import gauss2d.fit as g2f
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_params_uniq

linestyles_default = ["--", "-.", ":"]


@dataclass(kw_only=True)
class ErrorValues:
    kwargs_plot: dict = field(default_factory=dict)
    values: np.ndarray


def plot_loglike(model: g2f.Model, params: list[g2f.ParameterD] = None, n_values: int = 15,
                 errors: dict[str, ErrorValues] = None, values_reference: np.ndarray = None):
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
    fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=(8, 4*n_rows))
    if n_rows == 1:
        ax = [ax]

    n_loglikes = len(loglike_init)
    labels = [channel.name for channel in model.data.channels]
    labels.extend(['prior', 'total'])

    for param in params:
        param.fixed = True

    for row, param in enumerate(params):
        value_init = param.value
        param.fixed = False
        values = [value_init]
        loglikes = [loglike_init*0]
        dlls = [loglike_grads[row]]

        diff_init = 1e-4*np.sign(loglike_grads[row])
        diff = diff_init

        idx_prev = -1
        for idx in range(2*n_values):
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
            except RuntimeError as e:
                break
        param.value = value_init
        param.fixed = True

        subplot = ax[row][0]
        sorted = np.argsort(values)
        values = np.array(values)[sorted]
        loglikes = [loglikes[idx] for idx in sorted]
        dlls = np.array(dlls)[sorted]

        for idx in range(n_loglikes):
            subplot.plot(values, [loglike[idx] for loglike in loglikes], label=labels[idx])
        subplot.plot(values, np.sum(loglikes, axis=1), label=labels[-1])
        vline_kwargs = dict(ymin=np.min(loglikes) - 1, ymax=np.max(loglikes) + 1, color='k')
        subplot.vlines(value_init, **vline_kwargs)

        suffix = f' {param.label}' if param.label else ''
        subplot.legend(loc='lower center')
        subplot.set_title(f"{param.name}{suffix}")
        subplot.set_ylabel('loglike')
        subplot.set_ylim(vline_kwargs['ymin'], vline_kwargs['ymax'])

        subplot = ax[row][1]
        subplot.plot(values, dlls)
        subplot.axhline(0, color='k')

        vline_kwargs = dict(ymin=np.min(dlls), ymax=np.max(dlls))
        subplot.vlines(value_init, **vline_kwargs, color='k', label='fit')
        if values_reference is not None:
            subplot.vlines(values_reference[row], **vline_kwargs, color='b', label='ref')

        cycler_linestyle = cycle(linestyles_default)
        for name_error, valerr in errors.items():
            for idx_ax in range(2):
                linestyle = valerr.kwargs_plot.pop('linestyle', next(cycler_linestyle))
                ax[row][idx_ax].vlines(
                    [value_init - valerr.values[row], value_init + valerr.values[row]],
                    linestyles=[linestyle, linestyle],
                    label=name_error if (idx_ax == 1) else None,
                    **valerr.kwargs_plot,
                    **vline_kwargs,
                )
        subplot.legend(loc='upper right')

    for param in params:
        param.fixed = False
