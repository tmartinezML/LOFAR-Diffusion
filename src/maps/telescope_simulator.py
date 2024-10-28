import os
import sys
import json
import shutil
import logging
import itertools
import subprocess
from io import StringIO
from pathlib import Path
from configparser import ConfigParser

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import utils.paths as paths

# To be tested:
# - See if output of ddf-pipeline will correctly be redirected to log file
# - See if relative paths in config files will work

# To do:
# - Add logging
# - Maybe copy ddf-pipeline extract/image file from output jungle
# - Document options for config file


class TelescopeSimulator:

    # Some global paths that are used within the class
    shell_script_dir = paths.BASE_PARENT / "src/maps/shell_scripts"
    defualt_file_dir = paths.BASE_PARENT / "src/maps/default_files"

    def __init__(self, config_file):

        # Read config
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

        self.synthms_dir = self.parent / "synthms"
        self.losito_dir = self.parent / "losito"
        self.ddf_dir = self.parent / "ddf"
        self.ddfpipeline_dir = self.parent / "ddf-pipeline"

        # Set control flags
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
                shutil.rmtree(dir)

        dir.mkdir()

    def _run_command_with_logging(self, cmd, log_file):
        print(f"Running command: {cmd}")
        res = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
        )
        return res
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=1,
            text=True,
            shell=True,
        ) as p, open(log_file, "w") as log:
            if p.stdout:
                for line in p.stdout:
                    log.write(line)
                    print(line, end="")
            # output = buf.getvalue()

    def prepare_losito(self):
        # Prepare directories
        self._prepare_dir(self.losito_dir)

        # Read settings from default losito.parset
        parser = ConfigParser()
        config_in = StringIO()
        # Add _global section to the beginning of the file
        with open(self.defualt_file_dir / "losito.parset") as f:
            config_in.write(["_global"] + f.read())
        config_in.seek(0, os.SEEK_SET)
        parser.read_file(config_in)

        # Update settings
        parser["_global"]["skymodel"] = f"../{self.sky_model_file}"
        parser["_global"]["msin"] = f"../{self.synthms_dir.name}/*.MS"
        parser["_global"]["regions"] = "single_region.ds9"

        # Write settings to losito.parset with first line removed
        config_out = StringIO()
        parser.write(config_out)
        config_out.seek(0, os.SEEK_SET)
        config_out = "\n".join(config_out.readlines()[1:])
        with open(self.losito_dir / "losito.parset", "w") as f:
            f.write(config_out)

        # Write region file based on sky model file header
        plus = lambda x: x + 0.5
        minus = lambda x: x - 0.5
        ra, dec = self.center_radec.ra.deg, self.center_radec.dec.deg
        corner = lambda ff: (ff[0](ra), ff[1](dec))
        corners = list(map(corner, itertools.product([plus, minus], repeat=2)))
        corners = tuple(
            itertools.chain.from_iterable([corners[i] for i in [0, 1, 3, 2]])
        )
        out_str = f"fk5\npolygon{corners}\npoint{self.center_radec}\n"
        with open(self.losito_dir / "single_region.ds9", "w") as f:
            f.write(out_str)

        # Prepare shell script
        with open(self.losito_dir / "losito_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source envs/losito_venv/bin/activate\n"
                f"cd /tmartinez/sky_maps/{self.parent.name}/{self.losito_dir.name}\n"
                f"losito losito.parset"
            )

    def prepare_synthms(self):
        # Prepare directories
        self._prepare_dir(self.synthms_dir)

        tstart = Time(self.config["synthms"]["tstart"]).mjd
        tstart *= 3600 * 24  # Because of bug in synthms
        ra, dec = self.center_radec.ra.rad, self.center_radec.dec.rad
        minfreq = 143652344
        maxfreq = 143847656

        with open(self.synthms_dir / "synthms_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source envs/synthms_venv/bin/activate\n"
                f"cd /tmartinez/sky_maps/{self.parent.name}/{self.synthms_dir.name}\n"
                f"synthms  --name {self.parent.name} --start {tstart}"
                f" --tobs 8 --ra {ra} --dec {dec} --station HBA --minfreq {minfreq}"
                f" --maxfreq {maxfreq} chanpersb 2"
            )

    def prepare_ddf(self):
        # Prepare directories
        self._prepare_dir(self.ddf_dir)

        # Read default config
        ddf_config = ConfigParser()
        ddf_config.read(self.defualt_file_dir / "ddf_config.cfg")

        # Update config
        ddf_config["Data"]["MS"] = str(
            [
                f"../{self.synthms_dir.name}/{f.name}"
                for f in self.synthms_dir.glob("*.MS")
            ]
        )
        ddf_config["Output"]["Name"] = str(self.ddf_dir / self.parent.name)

        # Write config
        with open(self.ddf_dir / "ddf_config.cfg", "w") as f:
            ddf_config.write(f)

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
        ddfpipeline_config["image"]["imsize"] = self.map_size_px
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
                f"cd /tmartinez/sky_maps/{self.parent.name}/{self.ddfpipeline_dir.name}\n"
                f"pipeline.py ddf-pipeline_config.cfg"
            )

    def run_synthms(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.synthms_dir / "synthms_run.sh"),
        ]
        log_file = self.synthms_dir / "TelSim_synthms.log"
        self._run_command_with_logging(cmd, log_file)

    def run_losito(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.losito_dir / "losito_run.sh"),
        ]
        log_file = self.losito_dir / "TelSim_losito.log"
        self._run_command_with_logging(cmd, log_file)

    def run_ddf(self):
        cmd = f"""
            source /hsopt/anaconda3/base.env
            conda activate cenv_ddf
            cd /tmartinez/sky_maps/{self.parent.name}/{self.ddf_dir.name}
            DDF.py ddf_config.cfg
        """
        log_file = self.ddf_dir / "TelSim_ddf.log"
        self._run_command_with_logging(cmd, log_file)

    def run_ddfpipeline(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.ddfpipeline_dir / "ddf-pipeline_run.sh"),
        ]
        log_file = self.ddfpipeline_dir / "TelSim_ddfpipeline.log"
        self._run_command_with_logging(cmd, log_file)

    def run(
        self,
    ):
        if self.do_synthms:
            self.prepare_synthms()
            self.run_synthms()
        if self.do_losito:
            self.prepare_losito()
            self.run_losito()
        if self.do_ddf:
            self.prepare_ddf()
            self.run_ddf()
        if self.do_ddfpipeline:
            self.prepare_ddfpipeline()
            self.run_ddfpipeline()


if __name__ == "__main__":
    # Config file as first arg
    config_file = Path(sys.argv[1])
    ts = TelescopeSimulator(config_file)
    ts.run()
