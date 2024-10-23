from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from scipy.stats import norm as sp_norm
from scipy.optimize import curve_fit
from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import utils.paths as paths
import analysis.model_evaluation as meval
import analysis.bdsf_evaluation as bdsfeval
from analysis.stats_utils import norm, centers
from plotting.image_plots import plot_image_grid, quantile_contour_plot, remove_axes
from plotting.plot_utils import add_distribution_plot, plot_collection


# Plot the following metrics separately and save each of them to pdf:
# Pixel Value distribution, Image Mean, Image Sigma, COM Contours,
# Total flux gaus, nsrc, q0.1_area

mm = 1 / 25.4  # mm in inches
fig_width = 88 * mm
# plt.rcParams.update({"font.size": 5})
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "figure.dpi": 300,
        "font.family": "Nimbus Roman",
        "font.size": 4.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Roman",
        "mathtext.it": "Nimbus Roman:italic",
        "mathtext.bf": "Nimbus Roman:bold",
        "mathtext.cal": "Nimbus Roman:italic",
        "text.usetex": False,
    }
)
out_path = Path(
    "/hs/fs08/data/group-brueggen/tmartinez/diffusion/analysis_results/paper_plots"
)


def pyBDSF_residual_plot(img_arr, model_arr, residuals, fmax):
    n = len(img_arr)
    assert len(model_arr) == n, "Number of images must match"
    assert len(residuals) == n, "Number of residuals must match"

    fig, axs = plt.subplots(
        3, n, figsize=(fig_width, fig_width * 2 / 3), tight_layout=True
    )

    for i, ax in zip(range(len(img_arr)), axs.T):
        img = img_arr[i]
        res = residuals[i]
        ax[0].imshow(img)
        ax[0].set_title(f"{fmax[i]:.2f}")
        ax[1].imshow(model_arr[i])
        ax[2].imshow(res, vmin=0, vmax=residuals.max(), cmap="binary_r")
        print(residuals.max())

        delta = res[res >= 0].sum()
        ax[2].set_xlabel(f"{delta:.2f}")

        if i == 0:
            ax[0].set_ylabel("Image")
            ax[1].set_ylabel("Model")
            ax[2].set_ylabel(r"Res. $\geq$0")

    # Add annotation on bottom left:
    fig.text(
        0.095, 0.927, r"$\hat{f}_\mathrm{scaled}$:", ha="right", va="bottom", fontsize=9
    )
    fig.text(0.095, 0.028, r"$\Delta_+$:", ha="right", va="bottom", fontsize=9)

    for ax in axs.flatten():
        remove_axes(ax)

    return fig, axs


def pyBDSF_examples(imgs, model_imgs, a50=False):
    # Two patches with 6 cols and two rows each, upper row is images,
    # lower row is model images. The separation between the two patches is
    # larger than the separation between the images in each patch.
    # Create a figure
    fig = plt.figure(
        figsize=(fig_width, fig_width * 0.5),
        dpi=150,
        tight_layout=True,
    )

    assert len(imgs) == len(model_imgs), "Number of images must match"

    n_subplots = 1
    n_cols = len(imgs) // n_subplots

    gs_outer = gridspec.GridSpec(
        n_subplots,
        1,
        height_ratios=[
            1,
        ]
        * n_subplots,
        # hspace=0.1,
    )

    axs = []
    for i in range(n_subplots):
        # Create a GridSpec for each subplot
        gs = gridspec.GridSpecFromSubplotSpec(
            2,
            n_cols,
            subplot_spec=gs_outer[i],
            wspace=0.1,  # adjust the space between the plots in the grid
            hspace=0.02,
        )

        # Create the 2xn_cols grid of axes
        for j in range(2):
            for k in range(n_cols):
                ax = fig.add_subplot(gs[j, k])
                axs.append(ax)
                remove_axes(ax)  # Turn off the axis (or customize as needed)

    axs = np.array(axs).reshape(n_subplots, 2, n_cols)

    for i, (img, model_img) in enumerate(zip(imgs, model_imgs)):
        ax_img, ax_model = axs[i // n_cols, :, i % n_cols]
        ax_img.imshow(img.squeeze())
        if a50:
            (q,) = quantile_contour_plot(model_img.squeeze(), p_vals=[0.5], ax=ax_model)
            area = (model_img >= q).sum()
            ax_model.set_xlabel(f"{area}")
        else:
            ax_model.imshow(model_img.squeeze())

    # Add annotation on bottom left:
    fig.text(-0.025, 0.045, r"$A_{50\%}$:", ha="left", va="bottom", fontsize=9)

    return fig, ax


def boxcox_plot(dset):
    dset.transform_max_vals()

    fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_width * 4 / 3), dpi=150)

    add_distribution_plot(
        *np.histogram(dset.max_values, bins=100),
        axs[0],
        label="Dataset",
        color="black",
        normalize=False,
        label_count=False,
        errorbar=False,
        alpha=0.7,
    )
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\hat{f}$")
    axs[0].set_ylabel("Count")
    axs[0].legend()

    add_distribution_plot(
        *(hist := np.histogram(dset.max_values_tr, bins=100)),
        axs[1],
        label="Dataset",
        color="black",
        normalize=False,
        label_count=False,
        errorbar=False,
        alpha=0.7,
    )

    # Plot standard normal gaussian with amplitude fitted to histogram
    # Step 3: Define the standard normal distribution with amplitude as the free parameter
    def gaussian(x, amplitude):
        return amplitude * sp_norm.pdf(x, 0, 1)

    # Step 4: Fit the amplitude
    popt, _ = curve_fit(gaussian, centers(hist[1]), hist[0])

    axs[1].plot(
        hist[1],
        gaussian(hist[1], *popt),
        color="grey",
        label="Gaussian",
        alpha=0.5,
        ls="--",
    )
    axs[1].set_xlabel(r"$\hat{f}_\mathrm{scaled}$")
    axs[1].set_ylabel("Count")
    axs[1].legend(loc="upper right")

    return fig, axs


def selection_dropout_plot(edge_imgs, broken_imgs):
    fig, ax = plt.subplots(
        2,
        len(edge_imgs),
        figsize=(fig_width, fig_width * 0.45),
        dpi=150,
        tight_layout=True,
    )

    for i, (edge_img, broken_img) in enumerate(zip(edge_imgs, broken_imgs)):
        ax[0, i].imshow(edge_img.squeeze())
        ax[0, i].axis("off")
        ax[1, i].imshow(broken_img.squeeze())
        ax[1, i].axis("off")

    return fig, ax


def SNR_example_plot(SNR, edges, images, n_row=5):

    # Figure should have one plot on top for the distribution and a grid of images below
    fig = plt.figure(
        dpi=150,
        figsize=(fig_width, fig_width * 4 / 3 * 0.9),
        tight_layout=True,
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0])

    # Make grid on the bottom
    n_col = len(edges) - 1
    gs_lower = gridspec.GridSpecFromSubplotSpec(
        n_row, n_col, subplot_spec=gs[1], wspace=0.05, hspace=0.05
    )

    # Make 2D grid with axes
    grid_axs = []
    for i in range(n_row):
        for j in range(n_col):
            ax = fig.add_subplot(gs_lower[i, j])
            ax.axis("off")
            grid_axs.append(ax)
    grid_axs = np.array(grid_axs).reshape(n_row, n_col)

    # Plot the distribution
    counts = np.histogram(SNR, bins=edges)[0]
    add_distribution_plot(
        counts,
        edges,
        ax0,
        color="black",
        alpha=0.8,
        normalize=False,
        label="Cutouts",
        label_count=False,
    )
    ax0.axvline(5, color="grey", linestyle="--", label="Sel. cut")
    ax0.set_xlabel(r"$\mathit{S/N}_\sigma$")
    ax0.set_ylabel("Count")
    # Set ticks format to abbreviate thousands
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    # Plot inset with entire distribution
    ax_inset = inset_axes(ax0, width="40%", height="40%", loc="upper right")
    ax_inset.hist(SNR, bins=25, color="black", alpha=0.8)
    ax_inset.set_yscale("log")
    # ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    ax0.legend(loc=(0.72, 0.2))

    # Pick n_row examples from each bin
    binned_idxs = np.digitize(SNR, edges)  # Bin index for every SNR value
    bin_idxs = np.arange(0, n_col) + 1
    img_idxs = [np.argwhere(binned_idxs == i_bin).flatten() for i_bin in bin_idxs]
    img_idxs = [
        np.random.choice(idxs, n_row, replace=False) if len(idxs) > 0 else []
        for idxs in img_idxs
    ]
    print(len(img_idxs))

    for i, idxs in enumerate(img_idxs):
        for j, img_idx in enumerate(idxs):
            ax = grid_axs[j, i]
            ax.imshow(images[img_idx].squeeze())
            ax.axis("off")

    return fig


def COM_contour_plot(COM, fig_ax=None, color="black", label=None):

    if fig_ax is None:
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(88 * mm,) * 2)
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.invert_yaxis()
        ax.set_xlabel("COM x")
        ax.set_ylabel("COM y")

    else:
        fig, ax = fig_ax

    X = np.arange(-40, 40) + 0.5  # Should be centered on pixels
    counts_norm = norm(COM)
    cs = ax.contour(
        X,
        X,
        counts_norm,
        levels=[1e-4, 1e-3, 1e-2],
        colors=color,
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.e")

    ax._children.append(mlines.Line2D([0], [0], color=color))
    ax._children[-1].set_label(label)
    ax.legend()

    return fig, ax


def single_metric_plot(
    distr_dict,
    key,
    fig_ax=None,
    label=None,
    xlabel=None,
    color="black",
    alpha=0.8,
):

    # Get data
    counts, edges = distr_dict[key]

    if "area" in key:
        counts, edges = counts[:-1], edges[:-1]

    if key == "COM":
        return COM_contour_plot(counts, fig_ax=fig_ax, color=color, label=label)

    if fig_ax is None:
        # Initialize figure
        fig, ax = plt.subplots(
            1, 1, dpi=150, tight_layout=True, figsize=(88 * mm, 88 * 2 / 3 * mm)
        )
    else:
        fig, ax = fig_ax

    add_distribution_plot(
        counts,
        edges,
        ax,
        label=label,
        color=color,
        alpha=alpha,
        label_count=False,
    )

    # Axis and plot properties
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel(xlabel if xlabel is not None else key.replace("_", " "))

    if label is not None:
        ax.legend()
    ax.set_yscale("log")

    return fig, ax


def paper_plots_bdsf(
    img_dir,
    labels=None,
    cmap="cividis",
    colors=None,
    plot_train=True,
    force_train=False,
    train_path=paths.LOFAR_SUBSETS["0-clip"],
    train_label="LOFAR Dataset",
    bdsf_dir=None,
    h5_kwargs={},
    **distribution_kwargs,
):

    match img_dir:
        case Path():
            # Set output path
            fig_name = img_dir.stem
            img_dir = [img_dir]
            bdsf_dir = [bdsf_dir]
            if labels is None:
                labels = ["Generated"]

        case list():
            if labels is None:
                labels = [d.stem for d in img_dir]

    metrics = ["total_flux_gaus", "nsrc", "q0.5_area"]
    xlabels = ["Model Flux (a.u.)", "Identified Sources", r"$A_\mathrm{50\%}$ (px)"]
    gen_distr_dict_list = [
        bdsfeval.get_bdsf_distributions(d, out_dir=p) for d, p in zip(img_dir, bdsf_dir)
    ]
    # Get metrics dict
    if train_path is not None:
        lofar_distr_dict = bdsfeval.get_bdsf_distributions(
            train_path, force=force_train
        )

    # Set colors for paper plots
    colors = colors or [cm["viridis"](i) for i in [0.2, 0.85]]

    out = {}
    for metric, xlabel in zip(metrics, xlabels):
        collection = (
            [lofar_distr_dict, *gen_distr_dict_list]
            if plot_train
            else gen_distr_dict_list
        )
        fig, axs = plot_collection(
            collection,
            single_metric_plot,
            colors=colors or (["darkviolet"] if plot_train else None),
            labels=[train_label, *labels] if plot_train else labels,
            cmap=cmap,
            key=metric,
            xlabel=xlabel,
        )
        fig.savefig(paths.PAPER_PLOT_DIR / f"{metric}.pdf", bbox_inches="tight")

        out[metric] = (fig, axs)

        # Plot rmae between distributions
        c_lofar, e = lofar_distr_dict[metric]
        c_gen, _ = gen_distr_dict_list[0][metric]
        if "area" in metric:
            c_lofar, c_gen, e = c_lofar[:-1], c_gen[:-1], e[:-1]
        c_lofar, c_gen = norm(c_lofar), norm(c_gen)
        rmae = (c_lofar - c_gen) / c_lofar
        rmae[np.isnan(rmae)] = 0

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 0.2))
        add_distribution_plot(
            rmae, e, ax, normalize=False, errorbar=False, label_count=False
        )
        fig.show()

    return out


def paper_plots_pxstats(
    img_dir,
    labels=None,
    cmap="cividis",
    colors=None,
    plot_train=True,
    force_train=False,
    train_path=paths.LOFAR_SUBSETS["0-clip"],
    train_label="LOFAR Dataset",
    h5_kwargs={},
    **distribution_kwargs,
):

    match img_dir:
        case Path():
            # Set output path
            fig_name = img_dir.stem
            img_dir = [img_dir]
            if labels is None:
                labels = ["Generated"]

        case list():
            if labels is None:
                labels = [d.stem for d in img_dir]

    metrics = ["Pixel_Intensity", "Image_Mean", "Image_Sigma"]
    xlabels = ["Pixel Values", "Image Mean", "Image Std. Dev."]

    gen_distr_dict_list = [meval.get_distributions(d) for d in img_dir]
    # Get metrics dict
    if train_path is not None:
        lofar_distr_dict = meval.get_distributions(train_path, force=force_train)

    # Set colors for paper plots
    colors = colors or [cm["viridis"](i) for i in [0.2, 0.85]]

    out = {}
    for metric, xlabel in zip(metrics, xlabels):
        collection = (
            [lofar_distr_dict, *gen_distr_dict_list]
            if plot_train
            else gen_distr_dict_list
        )
        fig, axs = plot_collection(
            collection,
            single_metric_plot,
            colors=colors or (["darkviolet"] if plot_train else None),
            labels=[train_label, *labels] if plot_train else labels,
            cmap=cmap,
            key=metric,
            xlabel=xlabel,
        )
        fig.savefig(paths.PAPER_PLOT_DIR / f"{metric}.pdf", bbox_inches="tight")

        out[metric] = (fig, axs)

        # Plot rmae between distributions
        c_lofar, e = lofar_distr_dict[metric]
        c_gen, _ = gen_distr_dict_list[0][metric]
        c_lofar, c_gen = norm(c_lofar), norm(c_gen)
        rmae = c_lofar - c_gen  # / (c_lofar + c_gen)
        rmae[np.isnan(rmae)] = 0

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 0.2))
        add_distribution_plot(
            rmae, e, ax, normalize=False, errorbar=False, label_count=False
        )
        fig.show()

    return out
