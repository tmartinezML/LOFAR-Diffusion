import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import torch

import analysis.bdsf_evaluation as bdsfeval


def random_image_grid(dset, n_img=25, idx_titles=False, **kwargs):
    def has_context(img):
        return isinstance(img, list) or isinstance(img, tuple)

    idxs = np.random.choice(len(dset), n_img, replace=False)
    imgs = [img if not has_context(img := dset[i]) else img[0] for i in idxs]

    if idx_titles:
        kwargs["titles"] = idxs
    return plot_image_grid(imgs, **kwargs)


def plot_image_grid(
    imgs,
    masks=None,
    suptitle=None,
    vmin=0,
    vmax=1,
    savefig=None,
    fig_axs=None,
    n_rows=None,
    n_cols=None,
    titles=None,
    **imshow_kwargs,
):
    if isinstance(imgs, list):
        imgs = np.array(imgs)

    if n_rows is None and n_cols is None:
        n = int(np.sqrt(imgs.shape[0]))
        n_cols = n
        n_rows = n + np.ceil((imgs.shape[0] - n**2) / n).astype(int)
    elif n_rows is None or n_cols is None:
        known = n_rows or n_cols
        n = int(imgs.shape[0] // known) + int(bool(imgs.shape[0] % known))
        n_rows = n_rows or n
        n_cols = n_cols or n

    fig, axs = fig_axs or plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        constrained_layout=True,
        figsize=(3 * n_cols, 3 * n_rows),
    )
    flat_axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

    if titles is not None:
        assert len(titles) == len(imgs), (
            f"Number of titles ({len(titles)}) should match number of images "
            f"({len(imgs)})."
        )
    for i, (ax, img) in enumerate(zip(flat_axs, imgs)):
        ax.axis("off")
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax, **imshow_kwargs)

        # Plot mask contours if masks are passed
        if masks is not None:
            mask = masks[i]
            ax.contour(mask, colors="cyan", levels=[0.5], alpha=0.5, linewidths=0.3)

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
        fig.suptitle(suptitle, fontsize="xx-large")

    if savefig is not None:
        fig.savefig(savefig)

    return fig, axs


def plot_image_grid_from_file(path, save=False, **kwargs):
    imgs = torch.load(path, map_location="cpu").numpy()[:, -1, :, :]
    savefig = path.parent / f"{path.stem}_grid.png" if save else None
    return random_image_grid(imgs, savefig=savefig, **kwargs)


def metric_peek(metric, edges, images, names=None, n_examples=10, metric_name="Metric"):
    # Find indices of the bins to which each value in input array belongs.
    print("Digitizing...")
    binned_idxs = np.digitize(metric, edges)

    # Go through each bin
    for i_bin in np.unique(binned_idxs):

        # Print the bin range
        if i_bin == 0:
            suptitle = f"Bin {i_bin} - {metric_name} <= {edges[i_bin]:.2g}"

        elif i_bin == len(edges):
            suptitle = f"Bin {i_bin} - {edges[i_bin-1]:.2g} <= {metric_name}"

        else:
            suptitle = (
                f"Bin {i_bin} - "
                f"{metric_name} in [{edges[i_bin-1]:.2g}, {edges[i_bin]:.2g})"
            )

        # Get the indices of the images in the bin
        img_idxs = np.argwhere(binned_idxs == i_bin).flatten()

        # Randomly select n_examples from the bin
        if len(img_idxs) > n_examples:
            img_idxs = np.random.choice(img_idxs, n_examples, replace=False)

        # If there are no images in the bin, continue to the next bin
        if not len(img_idxs):
            continue

        # Plot the images
        n_col = min(len(img_idxs), 5)
        n_row = int(np.ceil(len(img_idxs) / n_col))

        fig, axes = plt.subplots(
            n_row,
            n_col,
            dpi=150,
            figsize=(3 * n_col, 3 * n_row + 0.75),
            constrained_layout=True,
        )
        if len(img_idxs) == 1:
            axes = np.array([axes])

        for i, img in enumerate(images[img_idxs]):
            ax = axes.flatten()[i]
            ax.imshow(img)
            value = metric[img_idxs[i]]
            ax.set_title(
                f"{names[img_idxs[i]] if names is not None else img_idxs[i]}\n"
                f"{metric_name} = {value:.2g}"
            )
            ax.axis("off")
        fig.suptitle(suptitle, fontsize="xx-large")
        fig.show()


def quantile_contour_plot(img_arr, p_vals=[0.9, 0.5, 0.1], ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    q_vals = bdsfeval.quantile_values(img_arr, p_vals)

    ax.imshow(img_arr)

    cnt = ax.contour(
        img_arr, levels=np.unique(q_vals), colors=cm["PiYG"](p_vals), linewidths=0.5
    )
    if label:
        ax.clabel(
            cnt, inline=True, fontsize=8, fmt={q: p for q, p in zip(q_vals, p_vals)}
        )

    if ax is None:
        plt.show()
    return q_vals


def remove_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
