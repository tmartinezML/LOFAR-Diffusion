import os
import sys
import json
import shutil
import logging
import tempfile
import itertools
import subprocess
from io import StringIO
from pathlib import Path
from ast import literal_eval
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import bdsf
import utils.paths as paths
import utils.logging as logging
from maps.telsim_utils import *
from maps.map_utils import lofar_num2nu


# To do:
# - Maybe copy ddf-pipeline extract/image file from output jungle
# - Document options for config file


class TelescopeSimulator:

    # Some global paths that are used within the class
    shell_script_dir = paths.BASE_PARENT / "src/maps/shell_scripts"
    defualt_file_dir = paths.BASE_PARENT / "src/maps/default_files"

    # Class function for parsing config name, which can be Path or str
    @classmethod
    def parse_config_name(cls, config_name):
        match config_name:
            case str():
                # If the string contains a '/', it is a path
                if "/" in config_name:
                    path = Path(config_name)

                    # If not config file itself, then look for config file in the directory
                    if not path.is_file():
                        path = path / "TelSim_config.conf"

                    # If the file does not exist, raise an error
                    if not path.exists():
                        raise FileNotFoundError(f"Config file {path} not found.")

                # If the string does not contain a '/', it is a map parent name
                else:
                    path = paths.SKY_MAP_PARENT / f"{config_name}/TelSim_config.conf"

            case Path():
                path = config_name

            case _:
                raise TypeError(f"Invalid type for config_name: {type(config_name)}")

        # Raise error if file does not exist
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found:\n\t{path}\nInferred from input:\n\t{config_name}"
            )

        return path

    def __init__(self, config_name):

        # Logger
        self.logger = logging.get_logger("TelSim")

        # Read config
        self.config_file = TelescopeSimulator.parse_config_name(config_name)
        self.config = ConfigParser()
        self.config.read(self.config_file)

        # Set paths and data attributes
        # TODO: At the moment, local paths are hard-coded into the generation
        # of shell scripts for sub-processes, see functions like prepare_*().
        # This should ideally be more flexible.
        self.sky_model_file = paths.Path(self.config["data"]["sky_model"])
        self.mask_file = paths.Path(self.config["data"]["fits_mask"])
        self.override = self.config.getboolean("data", "override")
        self.parent = self.config_file.parent
        # In singularity, the storage parent folder is mounted as root directory
        # (See singularity command in shell scripts)
        self.mount_parent = Path(
            f"/{paths.STORAGE_PARENT.name}"
        ) / self.parent.relative_to(paths.STORAGE_PARENT)

        # Define directories & files for each step
        if (
            self.config.has_option("data", "ms_dir")
            and (dir_name := self.config["data"]["ms_dir"]) is not None
        ):
            self.ms_dir = self.parent / dir_name
        else:
            self.ms_dir = self.parent / "ms"
        self.DP3_dir = self.parent / "DP3"
        self.losito_dir = self.parent / "losito"
        self.predict_dir = self.parent / "predict"
        self.ddf_dir = self.parent / "ddf"
        self.ddfpipeline_dir = self.parent / "ddf-pipeline"
        self.ClusterCat_dir = self.parent / "ClusterCat"

        # Welcome message
        self._welcome_message()

        # Set control flags, indicating which steps to run
        self.do_DP3 = self._get_bool_default("control", "DP3", default=False)
        self.do_synthms = self._get_bool_default("control", "synthms", default=False)
        self.do_losito = self._get_bool_default("control", "losito", default=False)
        self.do_predict = self._get_bool_default("control", "predict", default=False)
        self.do_ClusterCat = self._get_bool_default(
            "control", "ClusterCat", default=False
        )
        self.do_ddf = self._get_bool_default("control", "ddf", default=False)
        self.do_ddfpipeline = self._get_bool_default(
            "control", "ddf-pipeline", default=False
        )

        # Some sensible checks on the control flags:

        # Predict and losito predict should not both be run.
        if (
            self.do_losito
            and self._get_bool_default("losito", "do_predict")
            and self.do_predict
        ):
            self.logger.error(
                'I detected that both "losito" and "predict" are set to True, '
                'with "do_predict" also set to True in the losito section. '
                "This would result in redundance, since one would override the "
                "other. Please set only one of them to True. Exiting application."
            )
            sys.exit()

        # For predict, we need a fits file with 3 axes for channel dimension.
        if self.do_predict:
            self.sky_model_file_chandim = f"{self.sky_model_file.stem}_chanDim.fits"
            if not (self.parent / self.sky_model_file_chandim).exists():
                self.logger.error(
                    f"Predict step requires a fits file with 3 axes for channel dimension. "
                    f"The file {self.sky_model_file_chandim} was not found. Exiting application."
                )
                sys.exit()
            elif not (ndim := get_ndim(self.parent / self.sky_model_file_chandim)) == 3:
                self.logger.error(
                    f"Predict step requires a fits file with 3 axes for channel dimension. "
                    f"The file {self.sky_model_file_chandim} has {ndim} axes. Exiting application."
                )
                sys.exit()

        # Deprecated flag
        if (flg := self._get_bool_default("control", "DP3")) is not None:
            self.logger.warning(
                f"DP3 control flag ({flg}) is deprecated since it was"
                f" integrated in the synthms step. Will ignore."
            )

        # Set map properties
        self.fits_header = fits.getheader(self.parent / self.sky_model_file)
        self.center_radec = SkyCoord(
            self.fits_header["CRVAL1"], self.fits_header["CRVAL2"], unit="deg"
        )
        self.map_size_px = self.fits_header["NAXIS1"]
        self.arcsec_per_px = abs(self.fits_header["CDELT1"]) * 3600

    def _get_default(self, *args, default=None):
        if self.config.has_option(*args):
            return self.config.get(*args)
        else:
            return default

    def _get_bool_default(self, *args, default=False):
        if self.config.has_option(*args):
            return self.config.getboolean(*args)
        else:
            return default

    def _prepare_dir(self, dir):

        if dir.exists():
            if not self.override:
                raise FileExistsError(
                    f"Directory {dir} already exists. Set override to True to replace."
                )
            else:
                shutil.rmtree(dir, ignore_errors=False)

        dir.mkdir()

    def _run_command_with_logging(self, cmd, log_file):
        self.logger.info(f"Running command: {' '.join(cmd)}")
        with (
            subprocess.Popen(
                cmd,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ) as p,
            open(log_file, "w") as log,
        ):

            while p.poll() is None:
                line = p.stdout.readline()
                log.write(line)
                print(line, end="")  # Not sure if this will lead to double printing

            p.wait()

        return p

    def _prepare_DP3(self):
        # Prepare directories
        self._prepare_dir(self.DP3_dir)

        # Input MS files
        ms_list = sorted(list(self.ms_dir.glob("*.MS")))

        # Divide into chunks of 10 MS files
        # i + 1 because we're skipping the first MS file,
        # and range(0, 24, 10) because we have 24 chunks, i.e. the last bunch
        # of MS files will also be excluded (this is very hard-coded here).
        ms_list_chunks = [
            [
                f'"../{self.ms_dir.name}/{ms.name}"'
                for ms in ms_list[(i + 1) : (i + 1) + 10]
            ]
            for i in range(0, 24, 10)
        ]

        # Write parset files
        for i, ms in enumerate(ms_list_chunks):
            msin_str = ", ".join(ms)
            nu = int(np.round(np.mean([get_mean_freq(ms) for ms in ms])))
            msout_str = (
                f"../{self.ms_dir.name}/synth_DP3avg_SB{1 + 10*i:03d}_{nu}MHz.MS"
            )
            with open(self.DP3_dir / f"DP3.parset_{i}", "w") as f:
                f.write(
                    f"msin = [{msin_str}]\n"
                    f"msout = {msout_str}\n"
                    f"steps = [avg]\n"
                    f"avg.type = average\n"
                )

        # Write shell script
        with open(self.DP3_dir / "DP3_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source ./envs/losito_venv/bin/activate\n"
                f"cd {self.mount_parent / self.DP3_dir.name}\n"
            )
            for i, ms in enumerate(ms_list_chunks):
                f.write(f"DP3 DP3.parset_{i}\n")

    def _run_DP3(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.mount_parent / self.DP3_dir.name / "DP3_run.sh"),
        ]
        log_file = self.DP3_dir / "TelSim_DP3.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def _welcome_message(self):
        # Read ASCII art logo from file and display
        with open(self.defualt_file_dir / "TelescopeSimulator_logo.txt") as f:
            self.logger.info(
                "Welcome to the"
                + (lspace := "\n\n")
                + (indent := "\t\t")
                + indent.join([l for l in f.readlines()])
                + lspace
            )
        self.logger.info(f"Running with config file:\n\t{self.config_file}\n")

        # Iterate through config control options and print which steps will be run
        steps = [
            opt
            for opt in self.config["control"]
            if self.config.getboolean("control", opt)
        ]

        self.logger.info(
            "The following steps will be run:\n\n\t" + "\n\t".join(steps) + "\n"
        )

        # For testing: Exit applicaiton here
        if False:
            self.logger.info("Exiting application.")
            sys.exit()

    def run(
        self,
    ):

        # If desired, synthesize measurement set data.
        if self.do_synthms:
            self.prepare_synthms()
            p = self.run_synthms()
            p.wait()

        # Otherwise, check if MS files are available.
        # If not, import from default files for steps that require them.
        elif any(
            [  # Steps that need MS files
                self.do_losito,
                self.do_predict,
                self.do_ddf,
                self.do_ddfpipeline,
            ]
        ):
            if not self.ms_dir.exists() or not list(self.ms_dir.glob("*.MS")):
                self.import_synthms_files()

        # Predict: Simulate visibilities into MS with DDF
        if self.do_predict:
            self.prepare_predict()
            p = self.run_predict()
            p.wait()

        # LoSiTo: Simulate noise (or visibilities + noise) into MS
        if self.do_losito:
            self.prepare_losito()
            p = self.run_losito()
            p.wait()

        # ClusterCat: Run clustering algorithm on the input sky model
        if self.do_ClusterCat:
            self.prepare_ClusterCat()
            p = self.run_ClusterCat()
            p.wait()

        # DDF: Run DDF on the input MS files
        if self.do_ddf:
            self.prepare_ddf()
            p = self.run_ddf()
            p.wait()

        # DDF-pipeline: Run the full LoTSS DDF pipeline
        if self.do_ddfpipeline:
            self.prepare_ddfpipeline()
            p = self.run_ddfpipeline()
            p.wait()

    def import_synthms_files(self):
        # TODO: It should be possible to accelerate this through parallelization
        # Prepare directories
        self._prepare_dir(self.ms_dir)

        ms_files = sorted(list((self.defualt_file_dir / "synthms").glob("*.MS")))
        self.logger.info(f"Copying {len(ms_files)} synthms files from default files...")
        for ms in tqdm(ms_files):
            shutil.copytree(ms, self.ms_dir / f"{self.parent.name}_{ms.name}")

    def prepare_synthms(self):
        #
        # Prepare directories
        self._prepare_dir(self.ms_dir)

        # Set settings for synthms
        tstart = Time(self.config["synthms"]["tstart"]).mjd
        tstart *= 3600 * 24  # Because of bug in synthms
        ra, dec = self.center_radec.ra.rad, self.center_radec.dec.rad
        # Set freq range so we get 2 .MS files. This is required because
        # the ddf-pipeline crashes when only 1 MS is provided.
        # Set -1 for full range
        minfreq = -1  # 143652344
        maxfreq = -1  # 143847656
        chanpersb = 2

        # Write shell script
        with open(self.ms_dir / "synthms_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source ./envs/losito_venv/bin/activate\n"
                f"cd {self.mount_parent / self.ms_dir.name}\n"
                f"synthms  --name {self.parent.name} --start {tstart}"
                f" --tobs 8 --tres 8 --ra {ra} --dec {dec} --station HBA"
                f" --minfreq {minfreq} --maxfreq {maxfreq}"
                f" --chanpersb {chanpersb}"
            )

        self._prepare_DP3()

    def run_synthms(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.mount_parent / self.ms_dir.name / "synthms_run.sh"),
        ]
        log_file = self.ms_dir / "TelSim_synthms.log"
        p1 = self._run_command_with_logging(cmd, log_file)
        p2 = self._run_DP3()
        return [p1, p2]

    def prepare_predict(self):
        # Prepare directories
        self._prepare_dir(self.predict_dir)

        # Read default config
        pred_config = ConfigParser()
        pred_config.optionxform = str  # Preserve case
        pred_config.read(self.defualt_file_dir / f"ddf_config-Predict.cfg")

        # Update config
        # Check if self.config has MS options
        if self.config.has_option("data", "MS"):
            pred_config["Data"]["MS"] = self.config["data"]["MS"]
        else:
            pred_config["Data"]["MS"] = f"../{self.ms_dir.name}/*.MS"
        pred_config["Predict"]["FromImage"] = f"../{self.sky_model_file_chandim}"
        pred_config["Output"]["Name"] = str(self.predict_dir / self.parent.name)
        pred_config["Image"]["NPix"] = str(self.map_size_px)

        # Write config
        with open(self.predict_dir / "predict_config.cfg", "w") as f:
            pred_config.write(f)

        # Prepare shell script
        with open(self.predict_dir / "predict_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"source /hsopt/anaconda3/base.env\n"
                f"conda activate cenv_ddf\n"
                f"cd {self.predict_dir}\n"
                f"DDF.py predict_config.cfg"
            )

    def run_predict(self):
        cmd = [
            "bash",
            str(self.predict_dir / "predict_run.sh"),
        ]
        log_file = self.predict_dir / "TelSim_predict.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def prepare_losito(self):
        # Prepare directories
        self._prepare_dir(self.losito_dir)

        # Set parset file name based on whether we are doing the predict
        # or noise-only run
        parset_name = (
            "losito_beam-predict-noise.parset"
            if self._get_bool_default("losito", "do_predict", default=False)
            else "losito_beam-noise.parset"
        )

        # Read settings from default losito parset file
        parser = ConfigParser()
        config_in = StringIO()
        # Add _global section to the beginning of the file
        # (Because of the way LoSiTo reads the parset file)
        with open(self.defualt_file_dir / parset_name) as f:
            config_in.write("[_global]\n" + f.read())
        config_in.seek(0, os.SEEK_SET)
        parser.read_file(config_in)

        # Update general settings
        parser["_global"]["skymodel"] = f"../{self.sky_model_file}"
        parser["_global"]["regions"] = "single_region.ds9"

        # Check if independent run for every MS is required
        run_indep = self.config.getboolean("losito", "run_independent")
        # If independent run is required, we will create one parset for every MS.
        # Otherwise, we will have a list with only one element, representing
        # all MS with a wildcard.
        if run_indep:
            MSList = [f.name for f in self.ms_dir.glob("*.MS")]
        else:
            MSList = ["*.MS"]

        for i, MS in enumerate(MSList):
            parser["_global"]["msin"] = f"../{self.ms_dir.name}/{MS}"
            # Write settings to losito.parset with first line removed
            config_out = StringIO()
            parser.write(config_out)
            config_out.seek(0, os.SEEK_SET)
            config_out = "".join(config_out.readlines()[1:])
            with open(self.losito_dir / f"losito.parset{i}", "w") as f:
                f.write(config_out)

        # Write region file based on sky model file header
        if (s := literal_eval(self.config["losito"]["region_size"])) is not None:
            self.logger.info(f"LoSiTo: Using region size {s} deg from config file.")
            map_size_deg = float(s)
        else:
            map_size_deg = self.fits_header["CDELT1"] * self.map_size_px
        plus = lambda x: x + map_size_deg / 2
        minus = lambda x: x - map_size_deg / 2
        ra, dec = self.center_radec.ra.deg, self.center_radec.dec.deg
        corner = lambda ff: (ff[0](ra), ff[1](dec))
        corners = list(map(corner, itertools.product([plus, minus], repeat=2)))
        corners = tuple(
            itertools.chain.from_iterable([corners[i] for i in [0, 1, 3, 2]])
        )
        out_str = f"fk5\npolygon{corners}\npoint({ra},{dec})\n"
        with open(self.losito_dir / "single_region.ds9", "w") as f:
            f.write(out_str)

        # The losito run command depends on whether we need independent runs
        losito_cmd = "\n".join([f"losito losito.parset{i}" for i in range(len(MSList))])

        # Prepare shell script
        with open(self.losito_dir / "losito_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source envs/losito_venv/bin/activate\n"
                f"cd {self.mount_parent / self.losito_dir.name}\n"
                f"{losito_cmd}"
            )

    def run_losito(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.mount_parent / self.losito_dir.name / "losito_run.sh"),
        ]
        log_file = self.losito_dir / "TelSim_losito.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def prepare_ddf(self):
        # Prepare directories
        self._prepare_dir(self.ddf_dir)

        # Read default config
        ddf_config = ConfigParser()
        ddf_config.optionxform = str  # Preserve case
        ddf_preset = str(self.config.get("ddf", "preset", fallback="SSD"))
        ddf_config.read(self.defualt_file_dir / f"ddf_config-{ddf_preset}.cfg")

        # Update config
        # Check if self.config has MS options
        if self.config.has_option("data", "MS"):
            ddf_config["Data"]["MS"] = self.config["ddf"]["MS"]
        else:
            ddf_config["Data"]["MS"] = f"../{self.ms_dir.name}/*.MS"
        ddf_config["Output"]["Name"] = str(self.ddf_dir / self.parent.name)
        ddf_config["Image"]["NPix"] = str(self.map_size_px)
        ddf_config["Mask"]["External"] = f"../{self.mask_file.name}"

        # For DDF we need cell size in arcsec
        ddf_config["Image"]["Cell"] = str(self.arcsec_per_px)

        # Write config
        with open(self.ddf_dir / "ddf_config.cfg", "w") as f:
            ddf_config.write(f)

        # Prepare shell script
        with open(self.ddf_dir / "ddf_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"source /hsopt/anaconda3/base.env\n"
                f"conda activate cenv_ddf\n"
                f"cd {self.ddf_dir}\n"
                f"DDF.py ddf_config.cfg"
            )

    def run_ddf(self):
        cmd = [
            "bash",
            str(self.ddf_dir / "ddf_run.sh"),
        ]
        log_file = self.ddf_dir / "TelSim_ddf.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def prepare_ClusterCat(self):
        self._prepare_dir(self.ClusterCat_dir)
        # This file will be created in the run function
        bdsf_cat = self.sky_model_file.with_suffix(".pybdsf.srl.fits").name

        ndir = self.config.get("ddf-pipeline", "ndir", fallback=45)
        min_flux = self.config.get("ClusterCat", "FluxMin", fallback=0.03)

        cmd = f"ClusterCat.py --SourceCat {bdsf_cat} --FluxMin {min_flux} --DoPlot=0 --NGen 100 --NCPU 96 --NCluster {ndir}"

        # Prepare shell script
        with open(self.ClusterCat_dir / "ClusterCat_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"source /hsopt/anaconda3/base.env\n"
                f"conda activate cenv_ddf\n"
                f"cd {self.ClusterCat_dir}\n"
                f"{cmd}"
            )

    def run_ClusterCat(self):
        # Create pybdsf catalog
        img = bdsf_on_model(
            self.parent / self.sky_model_file,
            out_dir=self.ClusterCat_dir,
        )
        img.write_catalog(
            # TO DO: This is not DRY, since the outfile name is also defined in
            # prepare_ClusterCat. This should be fixed.
            outfile=str(
                self.ClusterCat_dir
                / self.sky_model_file.with_suffix(".pybdsf.srl.fits").name
            ),
            catalog_type="srl",
            clobber=True,
            format="fits",
        )

        # Run ClusterCat
        cmd = [
            "bash",
            str(self.ClusterCat_dir / "ClusterCat_run.sh"),
        ]
        log_file = self.ClusterCat_dir / "TelSim_ClusterCat.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def prepare_ddfpipeline(self):
        # Prepare directories
        if not (
            do_restart := self._get_bool_default("ddf-pipeline", "restart")
        ) or not (self.ddfpipeline_dir.exists()):
            self._prepare_dir(self.ddfpipeline_dir)
        else:
            self.logger.info(
                "Restart flag set to True, will not overwrite existing ddf-pipeline directory."
            )

        # Read available ms files
        ms_list = sorted(list(self.ms_dir.glob("*.MS")))
        if not len(ms_list):
            raise FileNotFoundError(f"No MS files found in {self.ms_dir}.")

        # Write mslist.txt with the files in ms_dir
        with open(self.ddfpipeline_dir / "mslist.txt", "w") as f:
            for file in ms_list:
                f.write(f"../{self.ms_dir.name}/{file.name}\n")

        # If desired, create evenly spaced sub-list of MS files
        if self.config.has_option("ddf-pipeline", "make_ms_sublist") and (
            make_sublist := self.config.getboolean("ddf-pipeline", "make_ms_sublist")
        ):
            self.logger.info("Creating ms sub-list for ddf-pipeline...")

            if (n := len(ms_list)) < 24:
                self.logger.warning(f"Warning -- only {n} ms files found!")

            # Create evenly spaced sub-list
            ms_sublist = ms_list[2::4]
            with open(self.ddfpipeline_dir / "ms_sublist.txt", "w") as f:
                for file in ms_sublist:
                    f.write(f"../{self.ms_dir.name}/{file.name}\n")

        # Read default config
        ddfpipeline_config = ConfigParser()
        ddfpipeline_config.read(self.defualt_file_dir / "ddf-pipeline_config.cfg")

        # Update config:

        # Restart flag
        if do_restart:
            ddfpipeline_config["control"]["restart"] = "True"
            ddfpipeline_config["control"]["remove_columns"] = "False"

        # MS files
        ddfpipeline_config["data"]["mslist"] = (
            "ms_sublist.txt" if make_sublist else "mslist.txt"
        )
        ddfpipeline_config["data"]["full_mslist"] = "mslist.txt"

        # Image size & resolution
        ddfpipeline_config["image"]["imsize"] = str(self.map_size_px)
        if (npix := literal_eval(self.config["ddf-pipeline"]["Npix"])) is not None:
            ddfpipeline_config["Image"]["Npix"] = npix
        if self.config.has_option("ddf-pipeline", "ndir"):
            ddfpipeline_config["solutions"]["ndir"] = self.config["ddf-pipeline"][
                "ndir"
            ]

        # Manual ClusterCat file
        if (
            ClusterCat_file := self.ClusterCat_dir
            / self.sky_model_file.with_suffix(".pybdsf.srl.fits.ClusterCat.npy").name
        ).exists():
            self.logger.info(
                f"ClusterCat file found, using it for ddf-pipeline:\n\t{ClusterCat_file}"
            )
            ddfpipeline_config["image"][
                "clusterfile"
            ] = f"../{self.ClusterCat_dir.name}/{ClusterCat_file.name}"
        elif self.config.has_option("ddf-pipeline", "clusterfile"):
            ddfpipeline_config["image"][
                "clusterfile"
            ] = f'../{self.config["ddf-pipeline"]["clusterfile"]}'

        # External mask
        ddfpipeline_config["masking"][
            "external_fits_mask"
        ] = f"../{self.mask_file.name}"

        # Write config
        with open(self.ddfpipeline_dir / "ddf-pipeline_config.cfg", "w") as f:
            ddfpipeline_config.write(f)

        # Prepare shell script
        with open(self.ddfpipeline_dir / "ddf-pipeline_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"cd {self.mount_parent / self.ddfpipeline_dir.name}\n"
                f"pipeline.py ddf-pipeline_config.cfg"
            )

    def run_ddfpipeline(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "ddf-pipeline_container.sh"),
            str(self.mount_parent / self.ddfpipeline_dir.name / "ddf-pipeline_run.sh"),
        ]
        log_file = self.ddfpipeline_dir / "TelSim_ddfpipeline.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p


if __name__ == "__main__":
    # Config file as first arg
    config_file = sys.argv[1]
    ts = TelescopeSimulator(config_file)
    ts.run()
