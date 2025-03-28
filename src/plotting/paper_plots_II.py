from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import data.datasets as datasets
from tqdm import tqdm
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from matplotlib.colors import Normalize, SymLogNorm, LogNorm
from skimage.measure import regionprops


import utils.paths as paths
import data.segment as seg
from data.datasets import ImagePathDataset, LOFARPrototypesDataset, SamplesDataset
from model.model_utils import load_data_transforms
from maps.map_utils import get_image, beam_solid_angle
from data.transforms import max_scale_batch, minmax_scale, minmax_scale_batch
from analysis.stats_utils import err_poisson
from data.segment import get_circle, get_sample_mask, circular_mask
from plotting.plot_utils import add_distribution_plot, auto_log_bins
from plotting.image_plots import plot_image_grid
from plotting.map_plots import double_map_plot

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
out_path = paths.ANALYSIS_PARENT / "paper_plots_II"


def image_example_grids():

    # All Cutouts
    print("Plotting all cutouts...")
    seed = 42
    cutouts_file = Path(
        "/hs/fs08/data/group-brueggen/tmartinez/image_data/LOFAR/cutouts/cutouts_200p_optC.hdf5"
    )
    with h5py.File(cutouts_file, "r") as f:
        imgs = f["cutouts"]
        idxs = np.random.RandomState(seed=seed).choice(len(imgs), size=18)
        print("Reading...")
        imgs = np.array(imgs[np.sort(idxs)])

    fig, _ = plot_image_grid(minmax_scale_batch(imgs), vmin=0, vmax=1, n_cols=6)
    fig.savefig(out_path / "Image_Example_Grids_All-Cutouts.pdf")
    fig.show()

    # Cutouts init. Selection
    print("Plotting cutouts...")
    seed = 42
    file = paths.LOFAR_SUBSETS["200p"]
    with h5py.File(file, "r") as f:
        imgs = f["images"]
        idxs = np.random.RandomState(seed=seed).choice(len(imgs), size=18)
        print("Reading...")
        imgs = np.array(imgs[np.sort(idxs)])

    fig, _ = plot_image_grid(minmax_scale_batch(imgs), vmin=0, vmax=1, n_cols=6)
    fig.savefig(out_path / "Image_Example_Grids_Cutout-Selection.pdf")
    fig.show()

    # Prototypes
    print("Plotting prototypes...")
    seed = 42
    proto_dset = datasets.LOFARPrototypesDataset(
        "prototypes",
        train_mode=False,
        img_size=80,
    )
    idxs = np.random.RandomState(seed=seed).choice(len(proto_dset), size=18)
    fig, _ = proto_dset.plot_image_grid(idxs, n_cols=6, show_titles=False)
    fig.savefig(out_path / "Image_Example_Grids_Prototypes.pdf")
    fig.show()

    # Samples
    print("Plotting samples...")
    seed = 42
    samples_dset = datasets.SamplesDataset("Prototypes_Model")
    idxs = np.random.RandomState(seed=seed).choice(len(samples_dset), size=18)
    imgs = max_scale_batch(samples_dset.data.numpy().squeeze()[idxs])
    masks = np.array(
        [
            get_sample_mask(img, dilate=0)
            for img in tqdm(imgs, desc=f"{'Sample source masks':<30}")
        ]
    )
    fig, _ = plot_image_grid(imgs, masks=masks, n_cols=6, vmin=0, vmax=1)
    fig.savefig(out_path / "Image_Example_Grids_DM-Samples.pdf")
    fig.show()


def training_data_selection_barchart():
    selection_counts = {
        "Step": [
            "Initial Selection",
            "Edge Max.",
            "# Islands",
            "Ghosts",
            "Multiples",
            "Duplicates",
            "On Cutout Edge",
            "Final Selection",
        ],
        "Selected": np.array([127559, 42021, 39014, 29382, 28162, 23856, 23817, 23817]),
    }
    selection_counts["Removed"] = np.zeros_like(selection_counts["Selected"])
    selection_counts["Removed"][1:] = np.diff(selection_counts["Selected"]) * -1
    selection_counts["Fraction Removed"] = np.zeros_like(
        selection_counts["Selected"], dtype=float
    )
    selection_counts["Fraction Removed"][1:] = (
        selection_counts["Removed"][1:] / selection_counts["Selected"][:-1]
    )

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 2 / 3), dpi=200)

    clr0, clr1 = [plt.get_cmap("viridis")(i) for i in [0.05, 0.75]]
    annt_kw = dict(
        fontsize=7,
    )

    # Plot 'Selected' bars
    n_steps = len(selection_counts["Step"])
    b = ax.barh(
        np.arange(n_steps),
        sel_rev := np.flip(selection_counts["Selected"]),
        color=clr0,
        label="Selected",
        alpha=0.5,
    )
    labels = sel_rev / sel_rev[-1]
    ax.bar_label(
        b,
        labels=[f"{l:.1%}" for l in labels],
        label_type="center",
        color=clr0,
        **annt_kw,
    )

    # Plot 'Removed' bars
    b = ax.barh(
        np.arange(n_steps - 1)[1:],
        rem_rev := np.flip(selection_counts["Removed"])[1:-1],
        color=clr1,
        label="Removed",
        left=sel_rev[1:-1],
        alpha=0.5,
    )
    labels = rem_rev / sel_rev[-1]
    ax.bar_label(
        b,
        labels=[f"{l:.2%}" for l in labels],
        label_type="edge",
        color=clr1,
        padding=2,
        **annt_kw,
    )

    ax.axhline(0.5, color=clr0, lw=0.5, ls="--")
    ax.axhline(0.5 + n_steps - 2, color=clr0, lw=0.5, ls="--")

    # Set labels
    ax.set_yticks(np.arange(n_steps))
    ax.set_yticklabels(np.flip(selection_counts["Step"]))
    ax.grid(alpha=0.1, ls="--")
    ax.set_xlabel("Number of Cutouts")
    ax.legend()
    ax.set_xlim(right=selection_counts["Selected"][0] * 1.15)

    fig.savefig(out_path / "Training_Data_Selection_Barchart.pdf")

    # Print Latex Table
    df = pd.DataFrame(selection_counts)
    df.drop(["Fraction Removed"], axis=1, inplace=True)
    df = df.reindex(columns=["Step", "Removed", "Selected"])
    fmt = lambda x: "\\num{" + str(x) + "}"
    print(
        df.to_latex(
            index=False,
            formatters=[str, fmt, fmt],
        )
        .replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline")
    )

    return


def mask_edge_threshold_dropouts():
    idxs = [
        88809,
        120895,
        122825,
    ]

    dset_file = paths.LOFAR_SUBSETS["200p"]

    with h5py.File(dset_file, "r") as f:
        imgs = np.array(f["images"][idxs]).squeeze()
        masks = np.array(f["masks_refined"][idxs])

    fig, axss = plt.subplots(
        len(idxs),
        3,
        figsize=(fig_width, fig_width * 1 / 3 * len(idxs)),
        tight_layout=True,
        dpi=100,
    )

    for i, (img, mask) in enumerate(zip(imgs, masks)):

        axs = axss[i]

        for ax in axs.flatten():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.xaxis.set_label_position("top")

        axs[0].imshow(img)
        axs[1].imshow(img * mask)
        axs[2].imshow(img * (1 - mask))

        if i == 0:
            axs[0].set_xlabel("Original Cutout")
            axs[1].set_xlabel("Mask Applied")
            axs[2].set_xlabel("Inverse Mask Applied")

    fig.savefig(out_path / f"Mask_Edge_Threshold_Dropouts.pdf")


def mask_post_processing():
    cutout_index = 109451

    dset_file = paths.LOFAR_SUBSETS["200p"]

    with h5py.File(dset_file, "r") as f:
        img = np.array(f["images"][cutout_index]).squeeze()
        bdsf_mask = np.array(f["island_labels"][cutout_index])

    img = minmax_scale(img)
    smoothed_mask = seg.smooth_mask(bdsf_mask)
    refined_mask = seg.refine_mask(img, smoothed_mask)

    fig, axs = plt.subplots(3, 3, figsize=(fig_width,) * 2, tight_layout=True)

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for aa in axs.T:
        aa[0].imshow(img)

    for j, mask in enumerate([bdsf_mask.astype(bool), smoothed_mask, refined_mask]):
        axs[0][j].contour(
            mask, levels=[0.5], colors="orange", linewidths=0.5, alpha=0.9
        )
        axs[1][j].imshow(img * mask)
        axs[2][j].imshow(img * (1 - mask))

    # Set titles
    axs[0][0].set_xlabel(f"PyBDSF Islands")
    axs[0][1].set_xlabel("Smoothed Mask")
    axs[0][2].set_xlabel("Refined Mask")
    for ax in axs[0]:
        ax.xaxis.set_label_position("top")

    # Set ylabels
    axs[0][0].set_ylabel("Original Cutout")
    axs[1][0].set_ylabel("Mask Applied")
    axs[2][0].set_ylabel("Inv. Mask Applied")

    fig.savefig(out_path / "Mask_Post_Processing.pdf")

    return fig


def map_images(save=True):
    map_name = "map_5deg_v4.1"

    # Load map and model images
    map_img, map_wcs = get_image(
        paths.SKY_MAP_PARENT / map_name / f"ddf/{map_name}.int.restored.fits"
    )
    model_img, _ = get_image(paths.SKY_MAP_PARENT / map_name / f"{map_name}.fits")

    # Scale model image from Jy/pixel to mJy/Beam
    model_img *= beam_solid_angle(6) / 1.5**2 * 1e3

    # Scale map image from Jy/Beam to mJy/Beam
    map_img *= 1e3

    names = ["Sky Model", "Sky Map"]
    maps = [model_img, map_img]
    cutout_sizes = [
        5,
    ]  # 1, 0.5]
    cutout_center = SkyCoord("00h00m55s", "+22d17m00s")

    # Plot images
    for cutout_size in cutout_sizes:
        cutouts = []
        for i in range(len(names)):
            img = maps[i].squeeze()
            if cutout_size < 5:
                cutout = Cutout2D(
                    img,
                    cutout_center,
                    u.Quantity(
                        [
                            cutout_size,
                        ]
                        * 2,
                        u.deg,
                    ),
                    wcs=map_wcs,
                )
                img = cutout.data
                wcs = cutout.wcs

            else:
                wcs = map_wcs

            cutouts.append((img, wcs))

        # Set values for color norm
        imgs_flat = np.concatenate([img.flatten() for img, _ in cutouts])
        clip_quantile = 0.999
        vmin = np.quantile(imgs_flat[imgs_flat < 0], clip_quantile)
        vmax = np.quantile(imgs_flat[imgs_flat > 0], clip_quantile)
        # Base linthr on max. tile RMS value of 0.153 mJy/beam
        rms = 0.153
        linthresh = 1.0 * rms
        norm_kw = dict(linthresh=linthresh, vmin=vmin, vmax=vmax, clip=True)
        norm_map = SymLogNorm(**norm_kw)

        fig, axs = double_map_plot(
            cutouts[0][0],
            cutouts[1][0],
            wcs=(cutouts[0][1], cutouts[1][1]),
            size=(2 * fig_width, fig_width),
            norm=norm_map,
            cbar_label="Flux Density (mJy/beam)",
        )

        # Add zoomed inset
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        def center_cutout_arrlim(n_deg, arr):
            n_px = int(n_deg * 3600 / 1.5)
            N = arr.shape[0]
            n1 = (N - n_px) // 2
            n2 = n1 + n_px
            return n1, n2

        aa = cutouts[0][0], cutouts[1][0]
        ww = cutouts[0][1], cutouts[1][1]

        n_deg_inset = 1 / 6  # 10 arcmin
        for i, (ax, arr, wcs) in enumerate(zip(axs, aa, ww)):
            n1, n2 = center_cutout_arrlim(n_deg_inset, arr)
            ax_inset = ax.inset_axes([0.01, 0.01, 0.4, 0.4], projection=wcs)

            ax_inset.imshow(arr, norm=norm_map)
            ax_inset.set_xlim(n1, n2)
            ax_inset.set_ylim(n1, n2)

            for c in ax_inset.coords:
                c.set_ticks_visible(False)
                c.set_ticklabel_visible(False)
                c.set_axislabel("")

            mark_inset(
                ax,
                ax_inset,
                loc1=2,
                loc2=4,
                fc="none",
                ec="white",
                lw=0.5,
                alpha=0.3,
                ls="--",
            )

            # Add scalebar
            import astropy.units as u
            from astropy.visualization.wcsaxes import add_scalebar

            add_scalebar(
                ax,
                10 * u.arcmin,
                color="white",
                label="10 arcmin",
                fontproperties=dict(size=plt.rcParams["font.size"]),
            )
            add_scalebar(
                ax_inset,
                1 * u.arcmin,
                color="white",
                label="1 arcmin",
                fontproperties=dict(size=plt.rcParams["font.size"]),
            )

        if save:
            fig.savefig(out_path / f"Map_Images_{cutout_size}deg.pdf")


def map_vs_lotss_image(save=True):
    map_name = "map_5deg_v4.1"
    pointing_name = "P181+40"
    n_deg = 2.5

    # Load map and model images
    map_img, map_wcs = get_image(
        paths.SKY_MAP_PARENT / map_name / f"ddf/{map_name}.int.restored.fits"
    )
    map_img *= 1e3

    pointing_dir = paths.LOFAR_DATA_PARENT / f"pointings/{pointing_name}"
    lotss_file = pointing_dir / "mosaic-blanked.fits"
    lotss_arr, lotss_wcs = get_image(lotss_file)
    lotss_arr *= 1e3

    def center_cutout(n_deg, img, wcs):
        img = img.squeeze()
        n_px = int(n_deg * 3600 / 1.5)
        N = img.shape[0]
        n1 = (N - n_px) // 2
        n2 = n1 + n_px
        img = img[n1:n2, n1:n2]
        wcs = wcs[n1:n2, n1:n2]
        return img, wcs

    lotss_arr_cut, lotss_wcs_cut = center_cutout(n_deg, lotss_arr.squeeze(), lotss_wcs)
    map_arr_cut, map_wcs_cut = center_cutout(n_deg, map_img.squeeze(), map_wcs)

    imgs_flat = np.concatenate([lotss_arr_cut.flatten(), map_arr_cut.flatten()])

    # Color normalization
    # Base linthr on max. tile RMS value of 0.153 mJy/beam
    rms = 0.153
    linthresh = rms
    clip_quantile = 0.999
    vmin = np.quantile(imgs_flat[imgs_flat < 0], clip_quantile)
    vmax = np.quantile(imgs_flat[imgs_flat > 0], clip_quantile)
    norm_map = SymLogNorm(
        # linthresh=norm_quantile,
        linthresh=linthresh,
        vmin=vmin,
        vmax=vmax,
        clip=True,
    )

    mm = 1 / 25.4  # mm in inches
    fig_width = 88 * mm

    fig, axs = double_map_plot(
        lotss_arr_cut,
        map_arr_cut,
        wcs=(lotss_wcs_cut, map_wcs_cut),
        size=(2 * fig_width, fig_width),
        norm=norm_map,
        cbar_label="Flux Density (mJy/beam)",
    )
    title_kw = dict(style="italic", weight="bold")
    axs[0].set_title("LoTSS-DR2:", **title_kw)
    axs[1].set_title("Simulated:", **title_kw)

    # Add zoomed inset
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    def center_cutout_arrlim(n_deg, arr):
        n_px = int(n_deg * 3600 / 1.5)
        N = arr.shape[0]
        n1 = (N - n_px) // 2
        n2 = n1 + n_px
        return n1, n2

    n_deg_inset = 1 / 6  # 10 arcmin
    for i, (ax, arr, wcs) in enumerate(
        zip(axs, [lotss_arr_cut, map_arr_cut], [lotss_wcs_cut, map_wcs_cut])
    ):
        n1, n2 = center_cutout_arrlim(n_deg_inset, arr)
        ax_inset = ax.inset_axes([0.01, 0.01, 0.4, 0.4], projection=wcs)

        ax_inset.imshow(arr, norm=norm_map)
        ax_inset.set_xlim(n1, n2)
        ax_inset.set_ylim(n1, n2)

        for c in ax_inset.coords:
            c.set_ticks_visible(False)
            c.set_ticklabel_visible(False)
            c.set_axislabel("")

        mark_inset(
            ax,
            ax_inset,
            loc1=2,
            loc2=4,
            fc="none",
            ec="white",
            lw=0.5,
            alpha=0.3,
            ls="--",
        )

        # Add scalebar
        import astropy.units as u
        from astropy.visualization.wcsaxes import add_scalebar

        add_scalebar(
            ax,
            10 * u.arcmin,
            color="white",
            label="10 arcmin",
            fontproperties=dict(size=plt.rcParams["font.size"]),
        )
        add_scalebar(
            ax_inset,
            1 * u.arcmin,
            color="white",
            label="1 arcmin",
            fontproperties=dict(size=plt.rcParams["font.size"]),
        )

    if save:
        fig.savefig(out_path / f"Map_Images_LoTSS_comparison.png")

    return fig, axs


def residual_RMS():

    # Load residual map
    map_name = "map_5deg_v4.1"
    ddf_resid_map, wcs = get_image(
        paths.SKY_MAP_PARENT / map_name / f"ddf/{map_name}.int.residual.fits",
    )

    # RMS of DDF residual
    rms_ddf_resid = np.sqrt(np.nanmean(ddf_resid_map**2))
    print(f"RMS of DDF residual: {rms_ddf_resid:.2e} Jy/beam")

    # Function for calculating RMS in quadrants
    def quadrants_rms(image, n=10):
        # n is the number of quadrants in each dimension
        image = image.squeeze()
        quad_shape = (image.shape[0] // n, image.shape[1] // n)

        rms = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                quad = image[
                    i * quad_shape[0] : (i + 1) * quad_shape[0],
                    j * quad_shape[1] : (j + 1) * quad_shape[1],
                ]
                rms[i, j] = np.sqrt(np.nanmean(quad**2))
        return rms

    # Calculate RMS in quadrants (1e3 for mJy)
    quad_rms = quadrants_rms(ddf_resid_map * 1e6, n=10)

    # Slice wcs to match quadrant shape
    # Slice wcs to match quadrant shape
    sub_wcs = wcs.deepcopy()
    sub_wcs.wcs.cdelt *= wcs.array_shape[-1] / 10
    sub_wcs.wcs.crpix = [5.5, 5.5]
    sub_wcs.array_shape = (10, 10)

    print(
        f"RMS in quadrants (micro-Jy/beam): Min = {quad_rms.min():.2e}, Max = {quad_rms.max():.2e}"
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_width),
        subplot_kw={"projection": sub_wcs},
        constrained_layout=True,
        dpi=120,
    )

    im = ax.imshow(
        quad_rms,
        origin="lower",
        cmap="viridis",
    )
    fig.colorbar(
        im, ax=ax, fraction=0.05, pad=0.05, label="RMS ($\mathrm{\mu}$Jy/beam)"
    )
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    # Rotate y labels
    ax.coords[1].set_ticklabel(rotation="vertical")
    ax.grid(alpha=0.1, ls="--")

    fig.savefig(out_path / "Residual_RMS.pdf")


def map_catalog_histograms():
    # Specify map name and ddf directory
    map_name = "map_5deg_v4.1"
    ddf_parent = "ddf"

    print("Loading catalogs...")
    # Load Map catalog to pandas
    map_cat_file = (
        paths.SKY_MAP_PARENT
        / map_name
        / ddf_parent
        / f"{map_name}.int.restored_pybdsf"
        / f"catalogues/{map_name}.int.restored.pybdsf.srl.FITS"
    )
    map_cat = Table.read(map_cat_file).to_pandas()

    # Load model catalog
    model_cat_file = (
        paths.SKY_MAP_PARENT
        / map_name
        / f"{map_name}.bdsfNoise_pybdsf"
        / f"catalogues/{map_name}.bdsfNoise.pybdsf.srl.FITS"
    )
    model_cat = Table.read(model_cat_file).to_pandas()

    # Load TRECS catalog
    trecs_file = (
        paths.SKY_MAP_PARENT / map_name / f"trecs/catalogue_continuum_wrapped.fits"
    )
    trecs_cat = Table.read(trecs_file, hdu=1, unit_parse_strict="silent").to_pandas()

    # Load lofar catalog
    lofar_cat = Table.read(paths.LOTSS_DR2_CAT).to_pandas()

    # Plot those quantities:
    metrics = ["Integrated Flux", "Peak Flux", "Major Axis"]
    # Define getter functions
    get_raw = lambda df, kw: df[kw].values
    get_f_mJy = lambda df, kw: df[kw].values * 1e-3
    get_maj_deg = lambda df, kw: df[kw].values * 3600

    def get_maj_trecs(df, kw):
        size = df["size"].values
        sfg_flag = df["RadioClass"].values < 4
        size[sfg_flag] = df["bmaj"].values[sfg_flag]
        return size / 2

    plot_dict = {
        "Catalog Names": ["Sky Model", "Sim. Sky Map", "TRECS", "LoTSS-DR2"],
        "Plot_flags": [False, True, False, True],
        "Catalogs": [model_cat, map_cat, trecs_cat, lofar_cat],
        "Integrated Flux": [
            ("Total_flux", get_raw),
            ("Total_flux", get_raw),
            ("I144", get_f_mJy),
            ("Total_flux", get_f_mJy),
        ],
        "Peak Flux": [
            ("Peak_flux", get_raw),
            ("Peak_flux", get_raw),
            (None, None),
            ("Peak_flux", get_f_mJy),
        ],
        "Major Axis": [
            ("Maj", get_maj_deg),
            ("Maj", get_maj_deg),
            (None, get_maj_trecs),
            ("Maj", get_raw),
        ],
        "Colors": [
            plt.get_cmap("viridis")(0.8),
            plt.get_cmap("viridis")(0.15),
            plt.get_cmap("plasma")(0.35),
            plt.get_cmap("plasma")(0.7),
        ],
    }
    areas_degsq = [25, 5634]

    figs = {}
    print("Making plots...")
    for i_metric, metric in enumerate(metrics):
        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 2 / 3), dpi=100)

        metric_values = []
        for i_cat, cat_name in enumerate(plot_dict["Catalog Names"]):
            if not plot_dict["Plot_flags"][i_cat]:
                metric_values.append(None)
                continue
            # Get catalog and metric getter
            cat = plot_dict["Catalogs"][i_cat]
            kw, get_metric = plot_dict[metric][i_cat]
            if get_metric is None:
                metric_values.append(None)
                continue

            # Get metric values
            metric_values.append(get_metric(cat, kw))

        # Calculate histograms
        bins = auto_log_bins([m for m in metric_values if m is not None])

        # This will be used for setting the x limits
        all_counts = []
        for i_cat, values in enumerate(metric_values):

            if values is None:
                continue

            c, _ = np.histogram(values, bins=bins, density=False)
            err_lo, err_hi = err_poisson(c)
            area_sr = (
                areas_degsq[
                    (1 if plot_dict["Catalog Names"][i_cat] == "LoTSS-DR2" else 0)
                ]
                * (np.pi / 180) ** 2
            )
            c = c / area_sr
            err_lo = err_lo / area_sr
            err_hi = err_hi / area_sr
            p = ax.stairs(
                c,
                bins,
                label=plot_dict["Catalog Names"][i_cat],
                color=plot_dict["Colors"][i_cat],
                lw=1.5,
                alpha=0.5,
            )
            ax.errorbar(
                (bins[1:] + bins[:-1]) / 2,
                c,
                yerr=(err_lo, err_hi),
                ls="none",
                color=p.get_edgecolor(),
                alpha=0.5 * 0.75,
                elinewidth=0.5,
                capsize=2.5,
                capthick=0.5,
            )

            all_counts.append(c)

        # Set x range
        nonzero_idx = np.logical_or.reduce([c > 1e1 for c in all_counts])
        xmin, xmax = bins[:-1][nonzero_idx].min(), bins[1:][nonzero_idx].max()
        if metric == "Major Axis":
            xmin, xmax = 3.5e0, 2.5e2
        side_gap = 0.05
        log_factor = xmax ** (side_gap) * xmin ** (-side_gap)
        ax.set_xlim(xmin / log_factor, xmax * log_factor)

        # Axis settings
        ax.set_xlabel(metric + (" (arcsec)" if metric == "Major Axis" else " (Jy)"))
        ax.set_ylabel("Count Density (sr$^{-1}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.set_ylim(bottom=1e1)

        # Add figure to dict
        figs[metric] = fig

        # Save figure
        fig.savefig(out_path / f"Map_Catalog_{metric.replace(' ', '_')}.pdf")

    return figs


def DM_histograms(save=True):

    # Load unconditioned sampled images
    samples_dset = datasets.SamplesDataset(
        "Prototypes_Model_SizeCond", key="samples_Size-Conditioned"
    )
    images = max_scale_batch(samples_dset.data.numpy().squeeze())

    # Generate source masks for the samples
    masks = np.array(
        [
            get_sample_mask(
                img,
                dilate=0,
            )
            for img in tqdm(images, desc=f"{'Sample source masks':<30}")
        ]
    ).astype(bool)

    # Get sample sizes
    sample_sizes = [
        regionprops(m)[0].feret_diameter_max
        for m in tqdm(masks.astype(int), desc="Sample sizes")
    ]

    # Get outer circles for the sample source masks
    # This gives center and radius:
    sample_circles = np.array(
        [get_circle(mask) for mask in tqdm(masks, desc=f"{'Sample outer circles':<30}")]
    )
    sample_centers, sample_radii = sample_circles[:, :2], sample_circles[:, 2]
    # This gives the actual masks:
    sample_circle_masks = np.array(
        [
            circular_mask(img.shape, center=c, radius=r)
            for img, c, r in zip(images, sample_centers, sample_radii)
        ]
    )

    # Calculate compactness and mask area.
    # Compactness is the ratio of the area of the mask to the area of the circle.
    sample_mask_areas = masks.sum(axis=(1, 2))
    sample_compactness = sample_mask_areas / sample_circle_masks.sum(axis=(1, 2))

    # Get mean and std of the source pixels
    sample_source_pixels = images[masks]
    sample_source_means = [
        np.mean(img[mask])
        for img, mask in tqdm(
            zip(images, masks), total=len(images), desc=f"{'Sample source means':<30}"
        )
    ]
    sample_source_stds = [
        np.std(img[mask])
        for img, mask in tqdm(
            zip(images, masks),
            total=len(images),
            desc=f"{'Sample source std. devs':<30}",
        )
    ]

    # Load prototype images
    proto_dset = datasets.LOFARPrototypesDataset(
        "prototypes",
        train_mode=False,
        img_size=80,
    )
    proto_radii = proto_dset.mask_metadata["Model_Radius"].values
    proto_sizes = proto_dset.mask_metadata["feret_diameter_max"].values

    # Generate outer circle masks for the prototypes
    proto_circle_masks = np.array(
        [
            circular_mask((80,) * 2, radius=r)
            for r in tqdm(proto_radii, desc=f"{'Prototype outer circles':<30}")
        ]
    )

    # Calculate compactness and mask area.
    proto_mask_areas = proto_dset.masks.sum(axis=(1, 2))
    proto_compactness = proto_mask_areas / proto_circle_masks.sum(axis=(1, 2))

    # Get mean and std of the prototypes source pixels
    proto_source_pixels = proto_dset.data.numpy()[proto_dset.masks.numpy().astype(bool)]
    proto_source_means = [
        np.mean(img[mask])
        for img, mask in tqdm(
            zip(proto_dset.data.numpy(), proto_dset.masks.numpy().astype(bool)),
            total=len(proto_dset),
            desc=f"{'Prototype source means':<30}",
        )
    ]
    proto_source_stds = [
        np.std(img[mask])
        for img, mask in tqdm(
            zip(proto_dset.data.numpy(), proto_dset.masks.numpy().astype(bool)),
            total=len(proto_dset),
            desc=f"{'Prototype source std. devs':<30}",
        )
    ]

    # Start making plots. Output will be saved in this dict.
    figs = {}
    metrics = [
        "Pixel Values",
        "Pixel Means",
        "Pixel Std. Devs",
        "Mask Sizes",
        "Compactness",
        "Mask Areas",
    ]
    metric_values = [
        (proto_source_pixels, sample_source_pixels),
        (proto_source_means, sample_source_means),
        (proto_source_stds, sample_source_stds),
        (proto_sizes, sample_sizes),
        (proto_compactness, sample_compactness),
        (proto_mask_areas, sample_mask_areas),
    ]
    xlabels = [
        "Source Pixel Value",
        "Source Pixel Mean",
        "Source Pixel St. Dev.",
        "Mask Size (px)",
        "Compactness",
        "Mask Area (px$^2$)",
    ]
    bin_ranges = [
        (0, 1, 100),
        (0, 0.5, 100),
        (0, 0.5, 100),
        (0, 80, 81),
        (0, 1, 50),
        (0, 2500, 100),
    ]

    # Set colors for train data and samples
    prot_clr, smpl_clr = [plt.get_cmap("viridis")(i) for i in [0.05, 0.8]]

    for i_metric in range(len(metrics)):

        # Initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 2 / 3), dpi=100)

        # Calculate histograms
        bins = np.linspace(*bin_ranges[i_metric])
        c_prot, _ = np.histogram(metric_values[i_metric][0], bins=bins)
        c_smpl, _ = np.histogram(metric_values[i_metric][1], bins=bins)

        # Plot histograms
        plot_kw = dict(label_count=False, alpha=0.8, auto_xlim=False)
        add_distribution_plot(
            c_prot, bins, ax, label="Train Data", color=prot_clr, **plot_kw
        )
        add_distribution_plot(
            c_smpl, bins, ax, label="Samples", color=smpl_clr, **plot_kw
        )

        # Set xlimits symmetrically
        # Symmetric difference between axis limits and nonzero bins on both sides.
        nonzero_idx = (c_prot > 0) | (c_smpl > 0)
        xmin, xmax = bins[:-1][nonzero_idx].min(), bins[1:][nonzero_idx].max()
        xrange = xmax - xmin
        ax.set_xlim(xmin - 0.05 * xrange, xmax + 0.05 * xrange)

        # Set labels and legend
        ax.set_xlabel(xlabels[i_metric])
        ax.set_ylabel("Normalized Counts")
        # If not pixel mean or std, set y scale to log
        # if i_metric not in [1, 2]:
        ax.set_yscale("log")
        ax.legend()

        # Save figure to dict
        figs[metrics[i_metric]] = fig

        if save:
            fig.savefig(
                out_path / f"Diffusion_Model_{metrics[i_metric].replace(' ', '_')}.png"
            )

    # Plot size offsets
    sizes_out = sample_sizes
    ctx_transform = load_data_transforms("Prototypes_Model_SizeCond")["mask_sizes"]
    ctx_in = samples_dset.samples_SizeConditioned_context.reshape(-1, 1)
    sizes_in = ctx_transform.inverse_transform(ctx_in).squeeze()

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width * 2 / 3), dpi=100)

    bins = np.linspace(-10, 10, 75)
    counts, bins = np.histogram((sizes_out - sizes_in), bins=bins)

    add_distribution_plot(
        counts,
        bins,
        ax,
        label="Samples",
        color=smpl_clr,
        **plot_kw,
    )

    # Set axis properties
    ax.set_xlim(-4, 5)
    ax.set_xlabel("Size Offset (px)")
    ax.set_ylabel("Normalized Counts")
    ax.legend()

    figs["Size Offsets"] = fig

    if save:
        fig.savefig(out_path / "Diffusion_Model_Size_Offsets.png")

    return figs
