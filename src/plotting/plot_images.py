import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_grid(imgs, suptitle=None, vmin=-1, vmax=1, savefig=None,
                    n_rows=None, n_cols=None, titles=None, **imshow_kwargs):

    if isinstance(imgs, list):
        imgs = np.array(imgs)

    if n_rows is None and n_cols is None:
        n = int(np.sqrt(imgs.shape[0]))
        n_cols = n
        n_rows = n + np.ceil((imgs.shape[0] - n**2) / n).astype(int)
    elif n_rows is None or n_cols is None:
        known = n_rows or n_cols
        n = int(imgs.shape[0] // known)
        n_rows = n_rows or n
        n_cols = n_cols or n

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, constrained_layout=True,
                            figsize=(3 * n_cols, 3 * n_rows))
    flat_axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

    if titles is not None:
        assert len(titles) == len(imgs), (
            f"Number of titles ({len(titles)}) should match number of images "
            f"({len(imgs)})."
        )
    for i, (ax, img) in enumerate(zip(flat_axs, imgs)):
        ax.axis('off')
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax, **imshow_kwargs)

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


def plot_image_grid_from_file(path, n_rows=5, n_cols=None, save=False,
                              idx_titles=False, **kwargs):
    imgs = torch.load(path, map_location='cpu')

    if n_cols is None:
        n_cols = n_rows

    idxs = np.random.choice(len(imgs), n_rows * n_cols, replace=False)
    imgs = imgs[idxs].numpy()[:, -1, :, :]
    savefig = path.parent / f"{path.stem}_grid.png" if save else None
    titles = idxs if idx_titles else None
    return plot_image_grid(
        imgs, suptitle=path.stem, savefig=savefig, titles=titles, **kwargs
    )


def metric_peek(metric, edges, images,
                names=None, n_examples=10, metric_name='Metric'
                ):
    # Find indices of the bins to which each value in input array belongs.
    binned_idxs = np.digitize(metric, edges)

    # Go through each bin
    for i_bin in np.unique(binned_idxs):

        # Print the bin range
        if i_bin == len(edges):
            suptitle = f'Bin {i_bin} - {edges[i_bin-1]:.2f} <= {metric_name}'
        else:
            suptitle = (
                f'Bin {i_bin} - '
                f'{edges[i_bin-1]:.2f} <= {metric_name} < {edges[i_bin]:.2f}'
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
            n_row, n_col, dpi=150, figsize=(3 * n_col, 3 * n_row + 0.75)
        )
        if len(img_idxs) == 1:
            axes = np.array([axes])
        for i, img in enumerate(images[img_idxs]):
            ax = axes.flatten()[i]
            ax.imshow(img)
            value = (
                metric.values[img_idxs[i]] if hasattr(metric, "values")
                else metric[img_idxs[i]]
            )
            ax.set_title(
                f'{names[img_idxs[i]] if names is not None else img_idxs[i]}\n'
                f'{metric_name} = {value:.2f}'
            )
            ax.axis('off')
        fig.suptitle(suptitle)
        fig.show()
