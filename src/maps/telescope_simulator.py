import os
import sys
import json
import shutil
import itertools
import subprocess
from io import StringIO
from pathlib import Path
from configparser import ConfigParser

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import utils.paths as paths


class TelescopeSimulator:

    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        self.sky_model_file = paths.Path(self.config["general"]["sky_model"])
        self.fits_header = fits.getheader(self.sky_model_file)
        self.center_radec = SkyCoord(
            self.fits_header["CRVAL1"], self.fits_header["CRVAL2"], unit="deg"
        )
        self.override = self.config["override"]
        self.prepare_folders(self.sky_model_file)

    def prepare_folders(self, sky_model_file):
        self.parent = sky_model_file.parent
        self.shell_script_dir = paths.BASE_PARENT / "src/maps/shell_scripts"
        self.defualt_file_dir = paths.BASE_PARENT / "src/maps/default_files"

        self.synthms_dir = self.parent / "synthms"
        self.losito_dir = self.parent / "losito"
        self.ddf_dir = self.parent / "ddf"

        for dir in [self.synthms_dir, self.losito_dir, self.ddf_dir]:

            if dir.exists():
                if not self.override:
                    raise FileExistsError(
                        f"Directory {dir} already exists. Set override to True to overwrite."
                    )
                else:
                    shutil.rmtree(dir)

            dir.mkdir()

    def prepare_losito(self):

        # Read settings from default losito.parset
        parser = ConfigParser()
        config_in = StringIO()
        # Add _global section to the beginning of the file
        with open(self.defualt_file_dir / "losito.parset") as f:
            config_in.write(["_global"] + f.read())
        config_in.seek(0, os.SEEK_SET)
        parser.read_file(config_in)

        # Update settings
        parser["_global"]["skymodel"] = self.sky_model_file
        parser["_global"]["msin"] = self.synthms_dir.glob("*.MS")[0]
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
        tstart = Time(self.config["synthms"]["tstart"]).mjd
        tstart *= 3600 * 24  # Because of bug in synthms
        ra, dec = self.center_radec.ra.rad, self.center_radec.dec.rad
        freq = 144000000

        with open(self.synthms_dir / "synthms_run.sh", "w") as f:
            f.write(
                f"#!/bin/bash\n"
                f"cd /tmartinez\n"
                f"source envs/synthms_venv/bin/activate\n"
                f"cd /tmartinez/sky_maps/{self.parent.name}/{self.synthms_dir.name}\n"
                f"synthms  --name {self.parent.name} --start {tstart}"
                f" --tobs 8 --ra {ra} --dec {dec} --station HBA --minfreq {freq}"
                f" --maxfreq {freq} chanpersb 2"
            )

    def prepare_ddf(self):
        # Read default config
        ddf_config = ConfigParser()
        ddf_config.read(self.defualt_file_dir / "ddf_config.cfg")

        # Update config
        ddf_config['Data']['MS'] = str(self.synthms_dir.glob("*.MS")[0])
        ddf_config['Output']['Name'] = str(self.ddf_dir / self.parent.name)

        # Write config
        with open(self.ddf_dir / "ddf_config.cfg", "w") as f:
            ddf_config.write(f)

    def run_synthms(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.synthms_dir / "synthms_run.sh"),
        ]
        subprocess.call(cmd)

    def run_losito(self):
        cmd = [
            "sh",
            str(self.shell_script_dir / "container_exec.sh"),
            str(self.losito_dir / "losito_run.sh"),
        ]
        subprocess.call(cmd)

    def run_ddf(self):

