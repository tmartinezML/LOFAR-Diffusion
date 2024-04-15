from collections.abc import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
