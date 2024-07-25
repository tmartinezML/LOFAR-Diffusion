import pickle
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm
from scipy.optimize import fminbound

import utils.paths as paths
from plotting.plot_utils import plot_collection
from plotting.metric_plots import bdsf_metrics_plot


def get_bdsf_metrics_plots(
    img_dir,
    save=True,
    labels=None,
    cmap="cividis",
    plot_train=True,
    force_train=False,
    train_path=None,
    train_label="Train Data",
    parent=paths.ANALYSIS_PARENT,
    bdsf_dir=None,
    h5_kwargs={},
    **distribution_kwargs,
):

    match img_dir:
        case Path():
            # Set output path
            if img_dir.is_file():
                out_dir = img_dir.parent
                fig_name = img_dir.stem
            else:
                out_dir = parent / img_dir.name
                fig_name = img_dir.name
            img_dir = [img_dir]
            bdsf_dir = [bdsf_dir]
            save_fig = save

            if labels is None:
                labels = [fig_name]

        case list():
            warnings.warn(
                "Multiple image directories passed to metrics plot, "
                "therefore save_fig is set to False."
            )
            save_fig = False
            if labels is None:
                labels = [d.name if d.is_dir() else d.stem for d in img_dir]

    # Get metrics dict
    gen_distr_dict_list = [
        get_bdsf_distributions(d, out_dir=p) for d, p in zip(img_dir, bdsf_dir)
    ]
    if train_path is not None:
        lofar_distr_dict = get_bdsf_distributions(train_path, force=force_train)

    collection = (
        [lofar_distr_dict, *gen_distr_dict_list] if plot_train else gen_distr_dict_list
    )
    fig, axs = plot_collection(
        collection,
        bdsf_metrics_plot,
        colors=["darkviolet"] if plot_train else None,
        labels=[train_label, *labels] if plot_train else labels,
        cmap=cmap,
    )
    if save_fig:
        fig.savefig(out_dir / f"{fig_name}_bdsf_metrics_plot.pdf")

    return fig, axs


def get_bdsf_distributions(img_path, **metric_kwargs):
    bdsf_metric_dict = get_metric_dict(img_path, **metric_kwargs)
    return bdsf_distributions_from_metrics(bdsf_metric_dict)


def get_metric_dict(
    img_path, parent=paths.ANALYSIS_PARENT, force=False, save=True, out_dir=None
):

    out_dir = out_dir or "bdsf"
    match (img_path.is_file(), img_path.suffix):
        case (True, ".pt"):
            out_dir = img_path.parent / out_dir

        case (True, ".h5") | (True, ".hdf5"):
            out_dir = parent / img_path.stem / out_dir

        case (False, _):
            out_dir = parent / img_path.name / out_dir

    assert (
        out_dir.exists()
    ), f"Directory {out_dir} required for bdsf results does not exist."

    dicts_path = out_dir / "dicts"
    assert any(dicts_path.glob("*.pkl")), f"No pickle files found in {dicts_path}"

    out_file = out_dir / "bdsf_metrics.npy"

    if not force and out_file.exists():
        print(f"Found existing distribution file for {img_path.name}.")
        return np.load(out_file, allow_pickle=True).item()

    bdsf_metric_dict = load_bdsf_metric_dict(dicts_path)
    bdsf_metric_dict = add_quantile_areas(bdsf_metric_dict)

    # Optionally save
    if save:
        np.save(out_file, bdsf_metric_dict)

    return bdsf_metric_dict


def bdsf_distributions_from_metrics(metrics_dict, bins: dict = {}):
    # Define bins for distributions
    bins_dict = {
        "total_flux_gaus": np.linspace(0, 75, 76),
        "ngaus": np.linspace(0, 100, 101),
        "nsrc": np.linspace(0, 100, 101),
        "q0.9_area": np.linspace(0, 6400, 257),
        "q0.5_area": np.linspace(0, 6400, 257),
        "q0.1_area": np.linspace(0, 6400, 257),
    }

    # Copy metrics dict and remove unnecessary keys
    metrics_dict = metrics_dict.copy()

    # Update bins dict with user-specified bins
    if len(bins):
        # Assert keys are contained in bins dict
        assert all([key in bins_dict.keys() for key in bins.keys()]), (
            f"Unknown key in bins dict:" f" {set(bins.keys()) - set(bins_dict.keys())}"
        )
        bins_dict.update(bins)

    # Assert keys of bins_dict are in metrics_dict
    assert set(bins_dict.keys()) <= set(metrics_dict.keys()), (
        f"Keys of bins dict and metrics dict do not match:"
        f" {set(bins_dict.keys()) - set(metrics_dict.keys())}"
    )

    # Calculate distributions
    distributions_dict = {}
    for key in tqdm(
        bins_dict.keys(),
        total=len(bins_dict.keys()),
        desc="Calculating metric distributions",
    ):

        match key:

            case _:
                counts, edges = np.histogram(metrics_dict[key], bins=bins_dict[key])
                distributions_dict[key] = counts, edges

    return distributions_dict


def load_bdsf_metric_dict(bdsf_pkl_dir):

    # Function for loading single pickle file
    def load_single_pickle(f):
        with open(f, "rb") as f:
            return pickle.load(f)

    # helper function used for properly sorting files
    def try_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    # Load the dicts, which are stored as single pickle files each
    bdsf_dicts = list(
        map(
            load_single_pickle,
            tqdm(
                sorted(bdsf_pkl_dir.glob("*.pkl"), key=lambda x: try_int(x.stem)),
                desc="Loading .pkl files...",
            ),
        )
    )

    # Merge them to one single dict with arrays as values
    out_dict = {
        k: elements_from_dict(bdsf_dicts, k)
        for k in tqdm(bdsf_dicts[0].keys(), desc="Merging dicts...")
    }
    return out_dict


def elements_from_dict(dcts, key):
    # Return the values of a key from a list of dicts as np array
    return np.array([d[key] for d in dcts])


def add_quantile_areas(bdsf_metric_dict):
    pvals = [0.9, 0.5, 0.1]
    avals = quantile_area_image_list(bdsf_metric_dict["model_gaus_arr"], pvals)
    for p, q in zip(pvals, avals.T):
        bdsf_metric_dict[f"q{p}_area"] = q
    return bdsf_metric_dict


def quantile_area_image_list(img_list, p_vals, q_vals_list=None):
    if q_vals_list is None:
        q_vals_list = quantile_value_img_list(img_list, p_vals)
    return np.stack(
        [
            quantile_area(img, p_vals, q_vals=q_vals)
            for img, q_vals in tqdm(zip(img_list, q_vals_list), total=len(img_list))
        ]
    )


def quantile_value_img_list(img_list, p_vals):
    out = []
    for img in tqdm(img_list, desc="Calculating quantiles"):
        out.append(quantile_values(img, p_vals))
    return np.stack(out)


def quantile_area(img_arr, p_vals, q_vals=None):
    if q_vals is None:
        q_vals = quantile_values(img_arr, p_vals)
    return [np.sum(img_arr >= q) for q in q_vals]


def quantile_values(img_arr, p_vals):
    return np.array(
        [
            fminbound(errfn, 0, img_arr.max(), args=(img_arr, p_target), disp=0)
            for p_target in p_vals
        ]
    )


def errfn(q, img_arr, p_target):
    p = img_arr[img_arr >= q].sum() / img_arr.sum()
    return (p - p_target) ** 2
