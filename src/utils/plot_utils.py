from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_losses(loss_files, logscale=True, stride=1, smooth=20,
                title=None, ):
    fig, ax = plt.subplots(dpi=80, figsize=(9, 5), tight_layout=True)
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

def plot_distributions(distr_gen, distr_lofar, error, n_gen, n_lofar,
                       title="Real vs. Generated Distributions"):
    fig, axs = plt.subplots(6, 1, dpi=150, tight_layout=True, figsize=(8, 16),
                        height_ratios=[5,1,5,1,5,1])
    get_centers = lambda edges: (edges[1:] + edges[:-1]) / 2

    for ax, ax2, key in zip(axs[::2], axs[1::2], distr_lofar.keys()):
        # Plot distributions of generated data
        c_gen, e_gen = distr_gen[key]
        norm = torch.sum(c_gen)
        err_gen = torch.sqrt(c_gen)
        ax.stairs(c_gen / norm, e_gen,
                alpha=0.5, fill=True, label=f"Generated (n={n_gen})",
                color="tab:blue")
        ax.errorbar(get_centers(e_gen), c_gen / norm,
                    yerr=err_gen / norm,
                    alpha=0.5, ls="none", color="tab:blue",
                    capsize=2.5, capthick=0.5)
        
        # Plot distributions of real lofar data
        c_lofar, e_lofar = distr_lofar[key]  # Counts, edges
        norm = torch.sum(c_lofar)
        err_lofar = torch.sqrt(c_lofar)
        ax.stairs(c_lofar / c_lofar.sum(), e_lofar,
                alpha=0.7, label=f"Real (n={n_lofar})", color="tab:orange")
        ax.errorbar(get_centers(e_lofar), c_lofar / norm,
                    yerr=err_lofar / norm,
                    alpha=0.5, ls="none", color="tab:orange",
                    capsize=2.5, capthick=0.5)
        
        # Set plot properties
        xmax = torch.max(torch.cat([
            e_gen[1:][c_gen>0], e_lofar[1:][c_lofar>0]
        ])).item()
        ax.set_xlim(left=0, right=xmax)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(key)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

        # Plot per-bin relative error between both distributions (thin plots)
        c, dc_sq, e = error[key]
        ax2.stairs(c, e, fill=True, alpha=0.5,
                color="tab:blue")
        ax2.errorbar(get_centers(e), c,
                    yerr=torch.sqrt(dc_sq),
                    alpha=0.5, ls="none", color="tab:blue",
                    capsize=2.5, capthick=0.5)

        # Set plot properties
        ax2.set_ylabel(r"$2 \cdot \frac{Real-Gen}{Real+Gen}$")
        ax2.grid(alpha=0.3)
        ax2.set_xlim(left=0, right=xmax)
        ax2.set_ylim(-2.5, 2.5)

    fig.suptitle(f"{title}")
    # fig.savefig(f"./analysis/dataset_comparisons/{title.replace('.', '')}.pdf")
    return fig