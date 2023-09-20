from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_losses(loss_files, logscale=True, stride=1, smooth=20,
                title=None, ):
    fig, ax = plt.subplots(dpi=80, figsize=(10, 6), tight_layout=True)
    # Add data to plot
    if isinstance(loss_files, list):
        show_legend = True
        colors = mpl.colormaps['turbo'](np.linspace(0, 1, len(loss_files)))
        for lf, c in zip(loss_files, colors):
            add_loss_plot(lf, ax, stride=stride, smooth=smooth, color=c, 
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

def add_loss_plot(loss_file, ax, stride=1, smooth=20, color="black", **kwargs):
    df = pd.read_csv(loss_file, delimiter=";", dtype=float)
    step, loss = [df[c] for c in df]
    k = kwargs if not smooth else {}
    ax.plot(step[::stride], loss[::stride],
            color=color, alpha=0.2, **k)
    if smooth:
        ema = pd.Series(loss).ewm(span=smooth).mean()
        ax.plot(step, ema, color=color, **kwargs)


def plot_samples(imgs, title=None, vmin=-1, vmax=1, savefig=None):
    n = int(np.sqrt(imgs.shape[0]))
    fig, axs = plt.subplots(nrows=n, ncols=n, tight_layout=True,
                            figsize=(12,12))
    for ax, img in zip(axs.flat, imgs):
        ax.axis('off')
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax)

    if title is not None:
        fig.suptitle(title, fontsize='xx-large')

    if savefig is not None:
        fig.savefig(savefig)

    return fig, axs