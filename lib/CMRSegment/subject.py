import os
import shutil
from pathlib import Path
from typing import List


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
        return self.output_dir.joinpath("segmented_{}".format(self.es_path.name))

    @property
    def segmented_ed_path(self):
        return self.output_dir.joinpath("segmented_{}".format(self.ed_path.name))

    @property
    def segmented_LR_es_path(self):
        return self.output_dir.joinpath("segmented_LR_{}".format(self.es_path.name))

    @property
    def segmented_LR_ed_path(self):
        return self.output_dir.joinpath("segmented_LR_{}".format(self.ed_path.name))

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
