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


class Artifact:
    def __init__(self):
        self.enlarge_phases = []
        self.ed_segmentation = None
        self.es_segmentation = None
        self.landmark = None
        self.ed_mesh = None
        self.es_mesh = None

    def set_segmentation(self, segmentation: 'Segmentation'):
        if segmentation.phase == Phase.ED:
            self.ed_segmentation = segmentation
        else:
            self.es_segmentation = segmentation

    def set_mesh(self, mesh: 'Mesh'):
        if mesh.phase == Phase.ED:
            self.ed_mesh = mesh
        else:
            self.es_mesh = mesh


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

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.path.parent

    @property
    def resampled(self):
        return self.output_dir.joinpath(f"resampled_{self.phase}.nii.gz")

    @property
    def enlarged(self):
        return self.output_dir.joinpath(f"enlarged_{self.phase}.nii.gz")

    @property
    def segmented(self):
        return self.output_dir.joinpath(f"segmented_{self.phase}.nii.gz")


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
        self.output_dir.mkdir(exist_ok=True)
        self.rview_dir().mkdir(exist_ok=True)
        self.enlarge_phases_dir().mkdir(exist_ok=True)
        self.gray_phases_dir().mkdir(exist_ok=True)
        self.resample_phases_dir().mkdir(exist_ok=True)
        self.motions_dir().mkdir(exist_ok=True)
        self.tmps_dir().mkdir(exist_ok=True)
        self.vtks_dir().mkdir(exist_ok=True)
        self.dofs_dir().mkdir(exist_ok=True)

    @property
    def name(self):
        return self.dir.name

    @property
    def landmark_path(self):
        return self.output_dir.joinpath("landmark.vtk")

    @property
    def enlarged_nii_path(self):
        return self.output_dir.joinpath("enlarged_{}".format(self.nii_path.name))

    @property
    def contrasted_nii_path(self):
        return self.output_dir.joinpath("contrasted_{}".format(self.nii_path.name))

    @property
    def enlarged_es_path(self):
        return self.output_dir.joinpath("enlarged_{}".format(self.es_path.name))

    @property
    def enlarged_ed_path(self):
        return self.output_dir.joinpath("enlarged_{}".format(self.ed_path.name))

    @property
    def segmented_es_path(self):
        return self.output_dir.joinpath("segmented_ES.nii.gz")

    @property
    def segmented_ed_path(self):
        return self.output_dir.joinpath("segmented_ED.nii.gz")

    @property
    def segmented_ed_es(self):
        return self.segmented_ed_path, self.segmented_es_path

    @property
    def segmented_LR_es_path(self):
        return self.output_dir.joinpath("segmented_LR_{}".format(self.es_path.name))

    @property
    def segmented_LR_ed_path(self):
        return self.output_dir.joinpath("segmented_LR_{}".format(self.ed_path.name))

    @property
    def segmented_LR_ed_es(self):
        return self.segmented_LR_ed_path, self.segmented_LR_es_path

    @property
    def resampled_ed_path(self):
        return self.output_dir.joinpath("resampled_{}".format(self.ed_path.name))

    @property
    def resampled_es_path(self):
        return self.output_dir.joinpath("resampled_{}".format(self.es_path.name))

    @property
    def enlarge_phases(self):
        paths = []
        for phase_name in os.listdir(str(self.enlarge_phases_dir())):
            phase_path = self.enlarge_phases_dir().joinpath(phase_name)
            paths.append(phase_path)
        return paths

    @property
    def resample_phases(self):
        paths = []
        for phase_name in os.listdir(str(self.resample_phases_dir())):
            phase_path = self.resample_phases_dir().joinpath(phase_name)
            paths.append(phase_path)
        return paths

    @property
    def gray_phases(self):
        paths = []
        for phase_name in os.listdir(str(self.gray_phases_dir())):
            phase_path = self.gray_phases_dir().joinpath(phase_name)
            paths.append(phase_path)
        return paths

    def dofs_dir(self):
        return self.output_dir.joinpath("dofs")

    def rview_dir(self):
        return self.output_dir.joinpath("4D_rview")

    def enlarge_phases_dir(self):
        return self.output_dir.joinpath("enlarge_phases")

    def gray_phases_dir(self):
        return self.output_dir.joinpath("gray_phases")

    def resample_phases_dir(self):
        return self.output_dir.joinpath("resample_phases")

    def segs_dir(self):
        return self.output_dir.joinpath("segs")

    def sizes_dir(self):
        return self.output_dir.joinpath("sizes")

    def vtks_dir(self):
        return self.output_dir.joinpath("vtks")

    def tmps_dir(self):
        return self.output_dir.joinpath("tmps")

    def motions_dir(self):
        return self.output_dir.joinpath("motions")

    def output_dirs(self) -> List[Path]:
        for subdir in ["dofs", "segs", "tmps", "sizes", "motions", "vtks"]:
            yield self.output_dir.joinpath(subdir)

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