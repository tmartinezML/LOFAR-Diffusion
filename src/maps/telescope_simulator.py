import os
import sys
import json
import shutil
import logging
import itertools
import subprocess
from io import StringIO
from pathlib import Path
from ast import literal_eval
from configparser import ConfigParser

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import utils.paths as paths
import utils.logging as logging


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
        config_file = TelescopeSimulator.parse_config_name(config_name)
        self.config = ConfigParser()
        self.config.read(config_file)

        # Set paths and data attributes
        # TODO: At the moment, local paths are hard-coded into the generation
        # of shell scripts for sub-processes, see functions like prepare_*().
        # This should ideally be more flexible.
        self.sky_model_file = paths.Path(self.config["data"]["sky_model"])
        self.mask_file = paths.Path(self.config["data"]["fits_mask"])
        self.override = self.config.getboolean("data", "override")
        self.parent = config_file.parent
        # In singularity, the storage parent folder is mounted as root directory
        # (See singularity command in shell scripts)
        self.mount_parent = Path(
            f"/{paths.STORAGE_PARENT.name}"
        ) / self.parent.relative_to(paths.STORAGE_PARENT)

        # Define directories for each step
        self.synthms_dir = self.parent / "synthms"
        self.losito_dir = self.parent / "losito"
        self.ddf_dir = self.parent / "ddf"
        self.ddfpipeline_dir = self.parent / "ddf-pipeline"

        # Set control flags, indicating which steps to run
        self.do_synthms = self.config.getboolean("control", "synthms")
        self.do_losito = self.config.getboolean("control", "losito")
        self.do_ddf = self.config.getboolean("control", "ddf")
        self.do_ddfpipeline = self.config.getboolean("control", "ddf-pipeline")

        # Set map properties
        self.fits_header = fits.getheader(self.parent / self.sky_model_file)
        self.center_radec = SkyCoord(
            self.fits_header["CRVAL1"], self.fits_header["CRVAL2"], unit="deg"
        )
        self.map_size_px = self.fits_header["NAXIS1"]

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
                # check=True,
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

    def prepare_losito(self):
        # Prepare directories
        self._prepare_dir(self.losito_dir)

        # Read settings from default losito.parset
        parser = ConfigParser()
        config_in = StringIO()
        # Add _global section to the beginning of the file
        with open(self.defualt_file_dir / "losito.parset") as f:
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
            MSList = [f.name for f in self.synthms_dir.glob("*.MS")]
        else:
            MSList = "*.MS"

        for i, MS in enumerate(MSList):
            parser["_global"]["msin"] = f"../{self.synthms_dir.name}/{MS}"
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

    def prepare_synthms(self):
        # Prepare directories
        self._prepare_dir(self.synthms_dir)

        # Set settings for synthms
        tstart = Time(self.config["synthms"]["tstart"]).mjd
        tstart *= 3600 * 24  # Because of bug in synthms
        ra, dec = self.center_radec.ra.rad, self.center_radec.dec.rad
        # Set freq range so we get 2 .MS files. This is required because
        # the ddf-pipeline crashes when only 1 MS is provided.
        minfreq = 143652344
        maxfreq = 143847656

        # Write shell script
        with open(self.synthms_dir / "synthms_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source ./envs/losito_venv/bin/activate\n"
                f"cd {self.mount_parent / self.synthms_dir.name}\n"
                f"synthms  --name {self.parent.name} --start {tstart}"
                f" --tobs 8 --ra {ra} --dec {dec} --station HBA --minfreq {minfreq}"
                f" --maxfreq {maxfreq} --chanpersb 2"
            )

    def prepare_ddf(self):
        # Prepare directories
        self._prepare_dir(self.ddf_dir)

        # Read default config
        ddf_config = ConfigParser()
        # Preserve case
        ddf_config.optionxform = str
        ddf_config.read(self.defualt_file_dir / "ddf_config.cfg")

        # Update config
        ddf_config["Data"]["MS"] = str(
            [
                f"../{self.synthms_dir.name}/{f.name}"
                for f in self.synthms_dir.glob("*.MS")
            ]
        )
        ddf_config["Output"]["Name"] = str(self.ddf_dir / self.parent.name)
        ddf_config["Image"]["Npix"] = str(self.map_size_px)
        
        # For DDF we need cell size in arcsec
        ddf_config["Image"]["Cell"] = str(abs(self.fits_header["CDELT1"]) * 3600)

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

    def prepare_ddfpipeline(self):
        # Prepare directories
        self._prepare_dir(self.ddfpipeline_dir)

        # Write mslist.txt with the files in synthms_dir
        with open(self.ddfpipeline_dir / "mslist.txt", "w") as f:
            for file in self.synthms_dir.glob("*.MS"):
                # TODO: Not sure if relative path will work
                f.write(f"../{self.synthms_dir.name}/{file.name}\n")

        # Read default config
        ddfpipeline_config = ConfigParser()
        ddfpipeline_config.read(self.defualt_file_dir / "ddf-pipeline_config.cfg")

        # Update config
        ddfpipeline_config["data"]["mslist"] = "mslist.txt"
        ddfpipeline_config["data"]["full_mslist"] = "mslist.txt"
        ddfpipeline_config["image"]["imsize"] = str(self.map_size_px)
        if (npix := literal_eval(self.config["ddf-pipeline"]["Npix"])) is not None:
            ddfpipeline_config["Image"]["Npix"] = npix
        # TODO: Possibly set [solutions][ndir] depending on imsize

        # TODO: Not sure if relative path will work, but should be fine
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

    def run_synthms(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.mount_parent / self.synthms_dir.name / "synthms_run.sh"),
        ]
        log_file = self.synthms_dir / "TelSim_synthms.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def run_losito(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.mount_parent / self.losito_dir.name / "losito_run.sh"),
        ]
        log_file = self.losito_dir / "TelSim_losito.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def run_ddf(self):
        cmd = [
            "bash",
            str(self.ddf_dir / "ddf_run.sh"),
        ]
        log_file = self.ddf_dir / "TelSim_ddf.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def run_ddfpipeline(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "ddf-pipeline_container.sh"),
            str(self.mount_parent / self.ddfpipeline_dir.name / "ddf-pipeline_run.sh"),
        ]
        log_file = self.ddfpipeline_dir / "TelSim_ddfpipeline.log"
        p = self._run_command_with_logging(cmd, log_file)
        return p

    def run(
        self,
    ):
        if self.do_synthms:
            self.prepare_synthms()
            p = self.run_synthms()
            p.wait()
        if self.do_losito:
            self.prepare_losito()
            p = self.run_losito()
            p.wait()
        if self.do_ddf:
            self.prepare_ddf()
            p = self.run_ddf()
            p.wait()
        if self.do_ddfpipeline:
            self.prepare_ddfpipeline()
            p = self.run_ddfpipeline()
            p.wait()


if __name__ == "__main__":
    # Config file as first arg
    config_file = sys.argv[1]
    ts = TelescopeSimulator(config_file)
    ts.run()
