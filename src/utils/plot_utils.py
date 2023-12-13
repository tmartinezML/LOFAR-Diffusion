from collections.abc import Iterable
from itertools import compress

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter

from utils.stats_utils import norm, norm_err, cdf, cdf_err, centers


def plot_losses(loss_files, logscale=True, stride=1, smooth=20,
                title=None, fig_ax=None):
    # Create figure and axes
    fig, ax = fig_ax or plt.subplots(
        dpi=100, figsize=(9, 5), tight_layout=True)

    # If loss_files is a list, plot each file in a different color and
    # with labels
    if isinstance(loss_files, Iterable):
        show_legend = True
        colors = mpl.colormaps['turbo'](np.linspace(0, 1, len(loss_files)))

        # Plot each loss file
        for i, lf in enumerate(loss_files):
            sm = smooth[i] if isinstance(smooth, Iterable) else smooth
            add_loss_plot(lf, ax, stride=stride, smooth=sm, color=colors[i],
                          label=lf.stem.replace("losses_", ""))

    # If loss_files is a single file, plot it in black
    else:
        show_legend = False
        add_loss_plot(loss_files, ax, stride=stride, smooth=20)

    # Set plot properties
    if logscale:
        ax.set_yscale('log')
    ax.set_xlabel("Total training step")
    ax.set_ylabel("Training Loss")
    ax.grid(alpha=0.5)
    if show_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_losses_from_folder(path, **kwargs):
    loss_files = sorted(path.glob("losses_*.csv"))
    return plot_losses(loss_files, smooth=[20, 0], **kwargs)


def add_loss_plot(loss_file, ax, stride=1, smooth=20, color="black",
                  hline=True,
                  **kwargs):
    # Read loss file
    df = pd.read_csv(loss_file, delimiter=";", dtype=float)
    step, loss = [df[c] for c in df][:2]  # Can also have 'ema_loss' column

    # If smoothing is desired, plot kwargs are passed to the smoothed plot.
    kwargs_loss = {} if smooth else kwargs

    # Plot loss
    ax.plot(
        step[::stride], loss[::stride],
        color=color, alpha=(0.2 if smooth else 0.6), **kwargs_loss
    )

    # Calculate and plot EMA of loss
    if smooth:
        ema = pd.Series(loss).ewm(span=smooth).mean()
        ax.plot(step, ema, color=color, **kwargs)

    # Plot horizontal line at last loss value
    if hline:
        ax.axhline(
            (ema if smooth else loss).to_numpy()[-1],
            color=color, alpha=0.5, ls="--", lw=1
        )

    # Plot ema loss if availble
    if 'ema_loss' in df.columns:
        ax.plot(step, df['ema_loss'], color=color, ls='--', lw=1)


def plot_image_grid(imgs, suptitle=None, vmin=-1, vmax=1, savefig=None,
                    n_rows=None, n_cols=None, titles=None):
    
    if isinstance(imgs, list):
        imgs = np.array(imgs)

    if n_rows is None and n_cols is None:
        n = int(np.sqrt(imgs.shape[0]))
        n_cols = n
        n_rows = n + imgs.shape[0] % n + int(n == 1) 
    elif n_rows is None or n_cols is None:
        known = n_rows or n_cols
        n = int(imgs.shape[0] // known)
        n_rows = n_rows or n
        n_cols = n_cols or n

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, tight_layout=True,
                            figsize=(3 * n_cols, 3 * n_rows))
    flat_axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

    if titles is not None:
        assert len(titles) == len(imgs), (
            f"Number of titles ({len(titles)}) should match number of images "
            f"({len(imgs)})."
        )
    for i, (ax, img) in enumerate(zip(flat_axs, imgs)):
        ax.axis('off')
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax)

        # Set axis title if titles are passed
        if titles is not None:
            title = titles[i]
            # Convert title to string if necessary
            match title:
                case float() | np.float32() | np.float64():
                    t_str = f"{title:.2e}"
                case int():
                    t_str = str(title)
                case _:
                    t_str = title

            ax.set_title(t_str)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize='xx-large')

    if savefig is not None:
        fig.savefig(savefig)

    return fig, axs

# -----------------------------#
# Pixel Distributions
# -----------------------------#


def add_distribution_plot(counts, edges, ax,
                          label='', color="black", alpha=0.5, fill=False):
    # Extract counts and edges from distribution
    c_norm = norm(counts)
    c_norm_err = norm_err(counts)

    # Add number of images to label
    if label is not None:
        label += f" (n={int(np.sum(counts)):_})"

    # Check if lines or collections are already present in axes
    # (relevant for x limits)
    data_present = len(ax.lines) + len(ax.collections) > 0

    # Plot distribution and error bars
    ax.stairs(c_norm, edges, alpha=alpha, fill=fill, label=label, color=color)
    ax.errorbar(centers(edges), c_norm, yerr=c_norm_err,
                alpha=0.75 * alpha, color=color, ls="none", elinewidth=0.5,
                capsize=2.5, capthick=0.5)

    # Set x limits
    _, x_max = ax.get_xlim()
    new_xmax = max(edges[1:][counts > 0]) * 1.02
    if data_present:
        new_xmax = max(new_xmax, x_max)
    ax.set_xlim(-0.02 * new_xmax, new_xmax)
    return counts, edges


def pixel_metrics_plot(distributions_dict, pixel_dist=None,
                       fig_ax=None, color=None, label=None):
    # Init figure and axes
    if fig_ax is None:
        fig, axs = plt.subplots(
            3, 2, figsize=(10, 10), dpi=150, tight_layout=True,
            sharex=False
        )
    else:
        fig, axs = fig_ax

    # Assert pixel distribution is either in dict or passed
    assert pixel_dist is not None or 'Pixel_Intensity' in distributions_dict, (
        "Pixel distribution should be passed or be present in distributions_dict."
    )
    pixel_dist = pixel_dist or distributions_dict['Pixel_Intensity'][0]

    # Keys to plot
    keys = [
        'Active_Pixels', 'Image_Mean', 'Active_Mean',
        'Image_Sigma', 'Active_Sigma'
    ]
    color = color if color is not None else "black"

    # Assert all keys are present in distributions_dict
    assert all(mask := [k in distributions_dict for k in keys]), (
        f"Not all keys are present in distributions_dict:"
        f" {compress(keys, ~mask)}"
    )

    # Plot pixel distribution first
    ax = axs[0, 0]
    add_distribution_plot(
        pixel_dist, np.linspace(0, 1, len(pixel_dist) + 1), ax,
        label=label, color=color
    )
    ax.set_xlabel('Pixel Value')

    # Plot data & set axes properties
    for ax, key in zip(axs.flatten()[1:], keys):
        counts, edges = distributions_dict[key]
        add_distribution_plot(
            counts, edges, ax,
            label=label, color=color
        )
        ax.set_xlabel(key.replace('_', ' '))

    # Set axes properties
    for ax in axs.flatten():
        ax.grid(alpha=0.2)
        if label is not None:
            ax.legend(fontsize='small')
        ax.set_yscale("log")
    # Put ticks on rhs for plots on the right
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()

    return fig, axs


def shape_metrics_plot(distributions_dict, COM=None,
                       fig_ax=None, color=None, label=None):
    if fig_ax is None:
        fig, axs = plt.subplots(
            3, 2, figsize=(10, 10), dpi=150, tight_layout=True
        )
    else:
        fig, axs = fig_ax

    # Assert COM is either in dict or passed
    assert COM is not None or 'COM' in distributions_dict, (
        "COM should be passed or be present in distributions_dict."
    )
    COM = COM or distributions_dict['COM']

    # Keys to plot
    keys = [
        'WPCA_Angle', 'WPCA_Ratio', 'COM_Radius', 'COM_Angle', 'Scatter'
    ]
    color = color if color is not None else "black"

    for ax, key in zip(axs.flatten()[:-1], keys):
        counts, edges = distributions_dict[key]
        add_distribution_plot(
            counts, edges, ax,
            label=label, color=color
        )
        ax.set_xlabel(key.replace('_', ' '))
        if key == 'WPCA_Ratio':
            ax.set_xlim(left=0.48)

    # COM Distribution
    ax = axs[-1][-1]
    ax.set_aspect('equal')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.invert_yaxis()
    ax.set_xlabel('COM x')
    ax.set_ylabel('COM y')
    ax.set_title("COM Scatter Plot")
    # Plot
    ax.scatter(*COM.T, s=0.2, color=color, alpha=0.2, label=label)

    for ax in axs.flatten()[:-1]:
        if label is not None:
            ax.legend(fontsize='small')
        ax.grid(alpha=0.2)
        ax.set_yscale("log")
    # Put ticks on rhs for plots on the right
    for ax in axs[:, 1]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    return fig, axs


def plot_collection(distr_dict_list, plot_fn, labels=None,
                    colors=None, cmap='cividis', **plot_kwargs):
    # Set colors if not passed
    if colors is None:
        colors = mpl.colormaps[cmap](np.linspace(0, 1, len(distr_dict_list)))
    elif len(colors) < len(distr_dict_list):
        colors = [*colors, *mpl.colormaps[cmap](
            np.linspace(0, 1, len(distr_dict_list) - len(colors))
        )]

    fig_axs = None
    for i, distr_dict in enumerate(distr_dict_list):
        fig_axs = plot_fn(
            distr_dict, fig_ax=fig_axs, color=colors[i],
            label=labels[i] if labels is not None else None,
            **plot_kwargs
        )

    return fig_axs


def add_distribution_delta_plot(delta, ax2, key, color="tab:blue"):
    # Extract delta and edges from distribution
    delta, delta_err, edges = delta[key]

    # Plot delta and error bars
    ax2.stairs(delta, edges, fill=True, alpha=0.5, color=color)
    ax2.errorbar(centers(edges), delta, yerr=delta_err,
                 alpha=0.5, color=color, ls="none", elinewidth=0.5,
                 capsize=2.5, capthick=0.5)


def plot_distributions(distr_gen, distr_lofar, error,
                       title="Real vs. Generated Distributions", labels=None,
                       landscape_mode=False):
    # Set plot size and layout
    if not landscape_mode:  # Default
        rows = 6
        cols = 1
        figsize = (8, 16)
    else:
        rows = 2
        cols = 3
        figsize = (24, 6)

    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, dpi=150, tight_layout=True,
                            figsize=figsize,
                            height_ratios=[5, 1] * int(rows / 2))

    # Loop through axes to plot distributions. Every 2nd axis is for the
    # per-bin delta between both distributions.
    for ax, ax2, key in zip(
        axs.transpose().flat[::2],  # Large plots for distributions
        axs.transpose().flat[1::2],  # Small plots for per-bin delta
        distr_lofar.keys()
    ):

        # Plot generated distributions.
        # distr_gen can be either a single dictionary or a list of
        # dictionaries.
        if isinstance(distr_gen, dict):  # Single dictionary
            c_gen, e_gen = distr_gen[key]
            add_distribution_plot(
                c_gen, e_gen, ax, color="tab:blue", alpha=0.5, fill=True,
                label=f"Generated"
            )
        elif isinstance(distr_gen, list):  # List of dictionaries
            assert all(isinstance(d, dict) for d in distr_gen), (
                "If distr_gen is a list, it should contain dictionaries."
            )
            # Set colors for each distribution
            cmap = 'cividis_r'
            colors = mpl.colormaps[cmap](np.linspace(0, 1, len(distr_gen)))

            # Plot each distribution
            for i, dg in enumerate(distr_gen):
                label = labels[i] if labels is not None else None
                c_gen, e_gen = dg[key]
                add_distribution_plot(
                    c_gen, e_gen, ax, color=colors[i], alpha=0.8, label=label
                )

        # Plot distributions of real lofar data
        c_lofar, e_lofar = distr_lofar[key]
        add_distribution_plot(
            c_lofar, e_lofar, ax, color="tab:orange", alpha=0.7,
            label=f"Real"
        )

        # Set properties for distributions plot
        # (x-axis limits, labels, log-scale, grid, legend)
        xmax = np.max(np.concatenate([
            e_gen[1:][c_gen > 0], e_lofar[1:][c_lofar > 0]
        ])).item()
        ax.set_xlim(left=0, right=xmax)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(key)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Plot per-bin relative delta between both distributions (thin plots)
        if isinstance(error, dict):  # Single dictionary
            add_distribution_delta_plot(error, ax2, key)

        elif isinstance(error, list):  # List of dictionaries
            assert all(isinstance(d, dict) for d in error), (
                "If error is a list, it should contain dictionaries."
            )
            # cmap and colors are defined above, where distributions are plotted
            for i, e in enumerate(error):
                add_distribution_delta_plot(e, ax2, key, color=colors[i])

        # Set properties for delta plot
        # (x-axis limits, labels, grid)
        ax2.set_ylabel(r"$2 \cdot \frac{Real-Gen}{Real+Gen}$")
        ax2.grid(alpha=0.3)
        ax2.set_xlim(left=0, right=xmax)
        ax2.set_ylim(-2.5, 2.5)

    # If necessary, remove y labels from all but the first column
    if landscape_mode:
        for ax in axs.transpose()[1:].flat:
            ax.set_ylabel(None)

    # Add title to figure
    fig.suptitle(f"{title}")

    return fig


def plot_W1_distances(W1_dict_list, x_values=None, x_label=''):
    n_trials = len(W1_dict_list)
    n_metrics = len(W1_dict_list[0].keys())

    # Set x values if not passed
    x_values = x_values or range(n_trials)

    # Set colors for each distribution
    cmap = 'tab20'
    colors = mpl.colormaps[cmap](np.linspace(0, 1, n_metrics))

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150,
                           tight_layout=True)

    # Loop through axes & keys to plot W1 values.
    # Get keys from first dictionary in list
    for i, key in enumerate(W1_dict_list[0].keys()):
        # Plot W1 values with error
        W1, W1_err = np.split(np.array([d[key] for d in W1_dict_list]).T, 2)
        ax.errorbar(
            x_values, W1.squeeze(), yerr=W1_err.squeeze(),
            color=colors[i], marker='.', label=key.replace('_', ' '),
            ls="--", elinewidth=0.5, capsize=2.5, capthick=0.5,
        )
        # Set properties for W1 plot
        ax.grid(alpha=0.3)

    # Set labels depending on layout
    ax.set_ylabel('W1 Distance')
    ax.set_xlabel(x_label)
    ax.legend()
    return fig, ax

# -----------------------------#
# Visualizing PBT results
# -----------------------------#


def get_init_theta():
    return np.array([0.9, 0.9])


def Q_batch(theta):
    """Returns the true function value for a batch of parameters with size (B, 2)"""
    return 1.2 - (3 / 4 * theta[:, 0] ** 2 + theta[:, 1] ** 2)


def get_arrows(theta_history, perturbation_interval):
    theta_history = theta_history[1:, :]
    arrow_start = theta_history[
        np.arange(perturbation_interval - 1,
                  len(theta_history), perturbation_interval)
    ]
    arrow_end = theta_history[
        np.arange(perturbation_interval, len(
            theta_history), perturbation_interval)
    ]
    if len(arrow_end) > len(arrow_start):
        arrow_end = arrow_end[: len(arrow_start)]
    else:
        arrow_start = arrow_start[: len(arrow_end)]
    deltas = arrow_end - arrow_start
    return arrow_start, deltas


def plot_parameter_history(
    results,
    colors,
    labels,
    perturbation_interval=None,
    fig=None,
    ax=None,
    plot_until_iter=None,
    include_colorbar=True,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    theta_0 = get_init_theta()

    x = np.linspace(-0.2, 1.0, 50)
    y = np.linspace(-0.2, 1.0, 50)
    xx, yy = np.meshgrid(x, y)
    xys = np.transpose(np.stack((xx, yy)).reshape(2, -1))
    contour = ax.contourf(xx, yy, Q_batch(xys).reshape(xx.shape), 20)
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_title("Q(theta)")

    scatters = []
    for i in range(len(results)):
        df = results[i].metrics_dataframe

        # Append the initial theta values to the history
        theta0_history = np.concatenate(
            [[theta_0[0]], df["theta0"].to_numpy()])
        theta1_history = np.concatenate(
            [[theta_0[1]], df["theta1"].to_numpy()])
        training_iters = np.concatenate(
            [[0], df["training_iteration"].to_numpy()])

        if plot_until_iter is None:
            plot_until_iter = len(training_iters)

        scatter = ax.scatter(
            theta0_history[:plot_until_iter],
            theta1_history[:plot_until_iter],
            # Size of scatter point decreases as training iteration increases
            s=100 / ((training_iters[:plot_until_iter] + 1) ** 1 / 3),
            alpha=0.5,
            c=colors[i],
            label=labels[i],
        )
        scatters.append(scatter)
        for i, theta0, theta1 in zip(training_iters, theta0_history, theta1_history):
            if i % (perturbation_interval or 1) == 0 and i < plot_until_iter:
                ax.annotate(i, (theta0, theta1))

        if perturbation_interval is not None:
            theta_history = np.hstack(
                (theta0_history.reshape(-1, 1), theta1_history.reshape(-1, 1))
            )[:plot_until_iter, :]
            arrow_starts, deltas = get_arrows(
                theta_history, perturbation_interval)
            for arrow_start, delta in zip(arrow_starts, deltas):
                ax.arrow(
                    arrow_start[0],
                    arrow_start[1],
                    delta[0],
                    delta[1],
                    head_width=0.01,
                    length_includes_head=True,
                    alpha=0.25,
                )
    ax.legend(loc="upper left")
    if include_colorbar:
        fig.colorbar(contour, ax=ax, orientation="vertical")
    return scatters


def plot_Q_history(results, colors, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title("True function (Q) value over training iterations")
    ax.set_xlabel("training_iteration")
    ax.set_ylabel("Q(theta)")
    for i in range(len(results)):
        df = results[i].metrics_dataframe
        ax.plot(df["Q"], label=labels[i], color=colors[i])
    ax.legend()


def make_animation(
    results, colors, labels, perturbation_interval=None, filename="pbt.gif"
):
    fig, ax = plt.subplots(figsize=(8, 8))

    def animate(i):
        ax.clear()
        return plot_parameter_history(
            results,
            colors,
            labels,
            perturbation_interval=perturbation_interval,
            fig=fig,
            ax=ax,
            plot_until_iter=i,
            include_colorbar=False,
        )

    ani = FuncAnimation(
        fig, animate, interval=200, blit=True, repeat=True, frames=range(1, 101)
    )
    ani.save(filename, writer=PillowWriter())
    plt.close()
