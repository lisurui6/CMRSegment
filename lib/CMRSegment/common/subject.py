import os
import shutil
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Phase(Enum):
    ED = "ED"
    ES = "ES"

    def __str__(self):
        return self.value


@dataclass
class Mesh:
    phase: Union[Phase, str, int]
    dir: Path

    @property
    def rv(self):
        return self.dir.joinpath(f"RV_{self.phase}.vtk")

    @property
    def rv_epi(self):
        return self.dir.joinpath(f"RVepi_{self.phase}.vtk")

    @property
    def lv_endo(self):
        return self.dir.joinpath(f"LVendo_{self.phase}.vtk")

    @property
    def lv_epi(self):
        return self.dir.joinpath(f"LVepi_{self.phase}.vtk")

    @property
    def lv_myo(self):
        return self.dir.joinpath(f"LVmyo_{self.phase}.vtk")

    def exists(self):
        return self.rv.exists() and self.rv_epi.exists() and self.lv_endo.exists() and self.lv_epi.exists()\
               and self.lv_myo.exists()

    def check_valid(self):
        if self.exists():
            return True
        if not self.rv.exists():
            raise FileNotFoundError(f"RV mesh does not exists at {self.rv}.")
        if not self.rv_epi.exists():
            raise FileNotFoundError(f"RV epi mesh does not exists at {self.rv_epi}.")
        if not self.lv_endo.exists():
            raise FileNotFoundError(f"LV endo mesh does not exists at {self.lv_endo}.")
        if not self.lv_epi.exists():
            raise FileNotFoundError(f"LV epi mesh does not exists at {self.lv_epi}.")
        if not self.lv_myo.exists():
            raise FileNotFoundError(f"LV myo mesh does not exists at {self.lv_myo}.")


@dataclass
class Segmentation:
    phase: Union[Phase, str, int]
    path: Path
    image: np.ndarray = None
    predicted: np.ndarray = None


@dataclass
class Image:
    phase: Union[Phase, str, int]
    path: Path
    output_dir: Path = None
    resampled: Path = None
    enlarged: Path = None
    segmented: Path = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.resampled is None:
            self.resampled = self.output_dir.joinpath("resampled", f"lvsa_{self.phase}.nii.gz")
        self.resampled.parent.mkdir(parents=True, exist_ok=True)
        if self.enlarged is None:
            self.enlarged = self.output_dir.joinpath("enlarged", f"lvsa_{self.phase}.nii.gz")
        self.enlarged.parent.mkdir(parents=True, exist_ok=True)
        if self.segmented is None:
            self.segmented = self.output_dir.joinpath("segmented", f"lvsa_{self.phase}.nii.gz")
        self.segmented.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Cine:
    dir: Path

    @property
    def paths(self):
        paths = []
        for phase_name in os.listdir(str(self.dir)):
            phase_path = self.dir.joinpath(phase_name)
            paths.append(phase_path)
        return paths

    def __iter__(self):
        return iter(self.paths)


class Subject:
    def __init__(self, dir: Path, nii_name: str = "LVSA.nii.gz", output_dir: Path = None):
        self.dir = dir
        self.output_dir = self.dir.joinpath("output") if output_dir is None else output_dir
        self.nii_path = self.dir.joinpath(nii_name)
        assert self.nii_path.exists()
        self.es_path = self.output_dir.joinpath("lvsa_ES.nii.gz")
        self.ed_path = self.output_dir.joinpath("lvsa_ED.nii.gz")
        self.mkdir()

    def mkdir(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.gray_phases_dir().mkdir(exist_ok=True)

    @property
    def name(self):
        return self.dir.name

    @property
    def landmark_path(self):
        return self.output_dir.joinpath("landmark.vtk")

    @property
    def contrasted_nii_path(self):
        return self.output_dir.joinpath("contrasted_{}".format(self.nii_path.name))

    @property
    def gray_phases(self):
        paths = []
        for phase_name in os.listdir(str(self.gray_phases_dir())):
            phase_path = self.gray_phases_dir().joinpath(phase_name)
            paths.append(phase_path)
        return paths

    def gray_phases_dir(self):
        return self.output_dir.joinpath("gray_phases")

    def clean(self):
        shutil.rmtree(str(self.output_dir), ignore_errors=True)
        self.mkdir()


@dataclass
class Template:
    dir: Path

    @property
    def landmark(self):
        return self.dir.joinpath("landmarks2.vtk")

    def rv(self, phase: Phase):
        return self.dir.joinpath(f"RV_{phase}.vtk")

    def lv_endo(self, phase: Phase):
        return self.dir.joinpath(f"LVendo_{phase}.vtk")

    def lv_epi(self, phase: Phase):
        return self.dir.joinpath(f"LVepi_{phase}.vtk")

    def lv_myo(self, phase: Phase):
        return self.dir.joinpath(f"LVmyo_{phase}.vtk")

    def vtk_rv(self, phase: Phase):
        return self.dir.joinpath(f"vtk_RV_{phase}.nii.gz")

    def vtk_lv(self, phase: Phase):
        return self.dir.joinpath(f"vtk_LV_{phase}.nii.gz")

    def check_valid(self):
        for phase in [Phase.ED, Phase.ES]:
            if not self.rv(phase).exists():
                raise FileNotFoundError(f"RV {phase} template does not exists at {self.rv(phase)}.")
            if not self.lv_endo(phase).exists():
                raise FileNotFoundError(f"LV endo {phase} template does not exists at {self.lv_endo(phase)}.")
            if not self.lv_epi(phase).exists():
                raise FileNotFoundError(f"LV epi {phase} template does not exists at {self.lv_epi(phase)}.")
            if not self.lv_myo(phase).exists():
                raise FileNotFoundError(f"LV myo {phase} template does not exists at {self.lv_myo(phase)}.")
