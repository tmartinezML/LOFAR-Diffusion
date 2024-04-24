from functools import partial

from wpca import WPCA
import numpy as np
from tqdm import tqdm


def metrics_dict_from_iterable(images, n_bins=4):
    metric_funcs = {
        "Image_Mean": image_mean,
        "Image_Sigma": image_sigma,
        "Active_Pixels": active_pixels,
        "Bin_Pixels": function_in_value_bins(pixels_in_range, n_bins=n_bins),
        "Bin_Mean": function_in_value_bins(mean_in_range, n_bins=n_bins),
        "Bin_Sigma": function_in_value_bins(sigma_in_range, n_bins=n_bins),
        "WPCA": wpca_angle_and_elongation,
        "COM": center_of_mass,
        "Scatter": signal_scatter,
    }
    metrics = list(metric_funcs.keys()) + [
        "WPCA_Angle",
        "WPCA_Elongation",
        "COM_Radius",
        "COM_Angle",
    ]
    metrics.remove("WPCA")

    pixel_dist = np.zeros(256, dtype=int)
    stats_dict = {key: [] for key in metrics}

    print("Calculating metrics...")
    for batch in tqdm(images, total=len(images)):

        # Remove label if present
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        img = batch.squeeze()
        for key, fnc in metric_funcs.items():
            val = fnc(img)
            match key:
                case "WPCA":
                    stats_dict[key + "_Angle"].append(val[0])
                    stats_dict[key + "_Elongation"].append(val[1])
                case "COM":
                    stats_dict[key].append(val[0])
                    stats_dict["COM_Radius"].append(val[1])
                    stats_dict["COM_Angle"].append(val[2])
                case _:
                    stats_dict[key].append(val)

        # Remove pixels that are 1 and 0 for calculating pixel distribution
        # minmax_mask = (img.squeeze().numpy() > 0) & (img.squeeze().numpy() < 1)
        minmax_mask = np.ones(img.squeeze.shape, dtype=bool)
        pixel_dist += np.histogram(
            img.squeeze().numpy()[minmax_mask], bins=np.linspace(0, 1, 257)
        )[0].astype(int)

    # Convert entries to right format
    for key in stats_dict.keys():
        if key.startswith("Bin_") and isinstance(stats_dict[key], list):
            # Here, stats_dict[key] is list of dicts with single key-value
            # pair. Convert to dict of np arrays with unique key-value pairs.
            unique_keys = set([k for d in stats_dict[key] for k in d.keys()])
            stats_dict[key] = {
                k: np.array([d[k] for d in stats_dict[key] if k in d.keys()])
                for k in sorted(list(unique_keys))
            }
        else:
            stats_dict[key] = np.array(stats_dict[key])

    # Add pixel distribution
    stats_dict["Pixel_Intensity"] = (pixel_dist, np.linspace(0, 1, 257))

    return stats_dict


def image_mean(img):
    return img.squeeze().numpy().mean()


def image_sigma(img):
    return img.squeeze().numpy().std()


def active_pixels(img):
    return np.sum(img.squeeze().numpy() > 0)


def is_between(img, thr=0, lim=1, closed=False):
    return (img >= thr) == ((img <= lim) if closed else (img < lim))


def pixels_in_range(img, **bin_kwargs):
    return np.sum(is_between(img.squeeze().numpy(), **bin_kwargs))


def func_in_range(img, fnc, **bin_kwargs):
    img = img.squeeze().numpy()
    mask = is_between(img, **bin_kwargs)
    if np.sum(mask) == 0:
        return np.nan
    return fnc(img[is_between(img, **bin_kwargs)])


def mean_in_range(img, **bin_kwargs):
    return func_in_range(img, np.mean, **bin_kwargs)


def sigma_in_range(img, **bin_kwargs):
    return func_in_range(img, np.std, **bin_kwargs)


def stats_in_value_bins(img, fnc, n_bins=4):
    if n_bins == 1:
        return fnc(img, thr=0, lim=1, closed=True)

    limits = np.linspace(0, 1, n_bins + 1)
    out = {
        (limits[i], limits[i + 1]): fnc(
            img, thr=limits[i], lim=limits[i + 1], closed=(i == n_bins - 1)
        )
        for i in range(n_bins)
    }
    return out


def function_in_value_bins(fnc, n_bins=4, **bin_kwargs):
    return partial(stats_in_value_bins, fnc=fnc, n_bins=n_bins, **bin_kwargs)


def wpca_angle_and_elongation(img):
    img_np = img.squeeze().numpy()
    active_coords = np.array(img_np.nonzero())
    weights = img_np[active_coords[0], active_coords[1]]

    wpca = WPCA(n_components=2)
    wpca.fit(active_coords.T, weights=np.tile(weights, (2, 1)).T)

    y, x = wpca.components_[0]  # rows, columns --> y, x
    v_max = np.max(wpca.explained_variance_ratio_)
    return (np.arctan2(y, x) / np.pi * 180) % 180, (v_max - 1 / 2) * 2


def center_of_mass(img):
    img = img.squeeze().numpy()
    coordinates = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].reshape(2, -1).T
    weights = img.reshape(-1)
    COM = np.average(coordinates, weights=weights, axis=0)
    rho = np.sqrt(((COM - 40) ** 2).sum())
    theta = np.angle(COM[0] - 40 + 1j * (COM[1] - 40), deg=True) % 360
    return COM, rho, theta


def signal_scatter(img):
    img = img.squeeze().numpy()
    coordinates = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].reshape(2, -1).T
    weights = img.reshape(-1)
    COM = np.average(coordinates, weights=weights, axis=0)
    sigma = np.average(
        np.sqrt(((coordinates - COM) ** 2).sum(axis=1)), weights=weights, axis=0
    )
    return sigma


def batch_metric(batch, metric):
    return np.array([metric(img) for img in batch])
