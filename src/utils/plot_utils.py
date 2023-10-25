from collections.abc import Iterable

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_losses(loss_files, logscale=True, stride=1, smooth=20,
                title=None, ):
    fig, ax = plt.subplots(dpi=100, figsize=(9, 5), tight_layout=True)
    # Add data to plot
    if isinstance(loss_files, Iterable):
        show_legend = True
        colors = mpl.colormaps['turbo'](np.linspace(0, 1, len(loss_files)))
        for i, lf in enumerate(loss_files):
            sm = smooth[i] if isinstance(smooth, Iterable) else smooth
            add_loss_plot(lf, ax, stride=stride, smooth=sm, color=colors[i],
                          label=lf.stem.replace("losses_", ""))
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


def add_loss_plot(loss_file, ax, stride=1, smooth=20, color="black", **kwargs):
    df = pd.read_csv(loss_file, delimiter=";", dtype=float)
    step, loss = [df[c] for c in df][:2]  # Can also have 'ema_loss' column
    k = kwargs if not smooth else {}
    ax.plot(step[::stride], loss[::stride],
            color=color,
            alpha=(0.2 if smooth else 1),
            **k)
    if smooth:
        ema = pd.Series(loss).ewm(span=smooth).mean()
        ax.plot(step, ema, color=color, **kwargs)


def plot_samples(imgs, title=None, vmin=-1, vmax=1, savefig=None):
    n = int(np.sqrt(imgs.shape[0]))
    fig, axs = plt.subplots(nrows=n, ncols=n, tight_layout=True,
                            figsize=(12, 12))
    for ax, img in zip(axs.flat, imgs):
        ax.axis('off')
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax)

    if title is not None:
        fig.suptitle(title, fontsize='xx-large')

    if savefig is not None:
        fig.savefig(savefig)

    return fig, axs

# -----------------------------#
# Pixel Distributions
# -----------------------------#


def get_centers(edges):
    # Helper function to get bin centers from bin edges
    return (edges[1:] + edges[:-1]) / 2


def add_distribution_plot(distr_gen, ax, key,
                          label='', color="black", alpha=0.5, fill=False):
    c_gen, e_gen = distr_gen[key]
    norm = torch.sum(c_gen)
    err_gen = torch.sqrt(c_gen)
    if label is not None:
        label += f" (n={int(norm):_})"
    ax.stairs(c_gen / norm, e_gen,
              alpha=alpha, fill=fill, label=label,
              color=color)
    ax.errorbar(get_centers(e_gen), c_gen / norm,
                yerr=err_gen / norm,
                alpha=0.75 * alpha, ls="none", color=color, elinewidth=0.5,
                capsize=2.5, capthick=0.5)

    return c_gen, e_gen


def add_distribution_error_plot(error, ax2, key, color="tab:blue"):
    c, dc_sq, e = error[key]
    ax2.stairs(c, e, fill=True, alpha=0.5,
               color=color)
    ax2.errorbar(get_centers(e), c,
                 yerr=torch.sqrt(dc_sq),
                 alpha=0.5, ls="none", color=color, elinewidth=0.5,
                 capsize=2.5, capthick=0.5)


def plot_distributions(distr_gen, distr_lofar, error,
                       title="Real vs. Generated Distributions", labels=None,
                       landscape_mode=False):
    if not landscape_mode:  # Default
        rows = 6
        cols = 1
        figsize = (8, 16)
    else:
        rows = 2
        cols = 3
        figsize = (24, 6)
    fig, axs = plt.subplots(rows, cols, dpi=150, tight_layout=True,
                            figsize=figsize,
                            height_ratios=[5, 1] * int(rows / 2))

    for ax, ax2, key in zip(
            axs.transpose().flat[::2],  # Large plots for distributions
            axs.transpose().flat[1::2],  # Small plots for per-bin error
            distr_lofar.keys()):
        # Plot distributions of generated data
        # Single distribution:
        if isinstance(distr_gen, dict):
            c_gen, e_gen = add_distribution_plot(
                distr_gen, ax, key, color="tab:blue", alpha=0.5, fill=True,
                label=f"Generated"
            )
        # Multiple distributions:
        elif isinstance(distr_gen, list):
            assert all(isinstance(d, dict) for d in distr_gen), (
                "If distr_gen is a list, it should contain dictionaries."
            )
            cmap = 'cividis_r'
            colors = mpl.colormaps[cmap](np.linspace(0, 1, len(distr_gen)))
            for i, dg in enumerate(distr_gen):
                label = labels[i] if labels is not None else None
                c_gen, e_gen = add_distribution_plot(
                    dg, ax, key, color=colors[i], alpha=0.8, label=label
                )

        # Plot distributions of real lofar data
        c_lofar, e_lofar = add_distribution_plot(
            distr_lofar, ax, key, color="tab:orange", alpha=0.7,
            label=f"Real"
        )

        # Set plot properties
        xmax = torch.max(torch.cat([
            e_gen[1:][c_gen > 0], e_lofar[1:][c_lofar > 0]
        ])).item()
        ax.set_xlim(left=0, right=xmax)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(key)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Plot per-bin relative error between both distributions (thin plots)
        if isinstance(error, dict):
            add_distribution_error_plot(error, ax2, key)

        elif isinstance(error, list):
            assert all(isinstance(d, dict) for d in error), (
                "If error is a list, it should contain dictionaries."
            )
            # cmap and colors are defined above, where distributions are plotted
            for i, e in enumerate(error):
                add_distribution_error_plot(e, ax2, key, color=colors[i])

        # Set plot properties
        ax2.set_ylabel(r"$2 \cdot \frac{Real-Gen}{Real+Gen}$")
        ax2.grid(alpha=0.3)
        ax2.set_xlim(left=0, right=xmax)
        ax2.set_ylim(-2.5, 2.5)

    if landscape_mode:
        for ax in axs.transpose()[1:].flat:
            # Remove y labels
            ax.set_ylabel(None)

    fig.suptitle(f"{title}")
    # fig.savefig(f"./analysis/dataset_comparisons/{title.replace('.', '')}.pdf")
    return fig


def plot_W1_scores(scores_list, x_values=None, x_label='Trial',
                   landscape_mode=False):
    if x_values is None:
        x_values = range(len(scores_list))
    keys = scores_list[0].keys()

    cmap = 'cividis_r'
    colors = mpl.colormaps[cmap](np.linspace(0, 1, len(scores_list)))

    if not landscape_mode:  # Default
        rows = 3
        cols = 1
        figsize = (6, 9)
    else:
        rows = 1
        cols = 3
        figsize = (18, 3)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, dpi=150,
                            tight_layout=True)

    for ax, key in zip(axs, keys):
        ax.scatter(x_values, [score[key]
                   for score in scores_list], color=colors)
        ax.set_title(key)
        ax.grid(alpha=0.3)

    [ax.set_ylabel('W1 score') for ax in axs[:(1 if landscape_mode else 3)]]
    [ax.set_xlabel(x_label) for ax in axs[(0 if landscape_mode else 2):]]
    return fig

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
