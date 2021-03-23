import os
import shutil
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import nibabel as nib


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


class ImageResource:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def from_dir(cls, dir: Path, filename: str):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        nii_path = dir.joinpath(filename)
        return cls(nii_path)

    def __str__(self):
        return str(self.path)

    def __getattr__(self, name: str) -> object:
        # pathlib.Path attributes
        for key in ("stem", "name", "suffix", "suffixes"):
            if name == key:
                return getattr(self.path, key)
        raise AttributeError("{} has no attribute named '{}'".format(type(self).__name__, name))

    def get_data(self) -> np.ndarray:
        nim = nib.load(str(self.path))
        seg = nim.get_data()
        return seg

    def exists(self):
        return self.path.exists()


class Segmentation(ImageResource):
    def __init__(self, path: Path, phase: Union[Phase, str, int]):
        self.path = path
        self.phase = phase
        super().__init__(path)


class NiiData(ImageResource):
    @classmethod
    def from_dir(cls, dir: Path, filename: str = "LVSA.nii.gz"):
        return super().from_dir(dir, filename)


class PhaseImage(ImageResource):
    def __init__(self, path: Path, phase: Union[Phase, str, int]):
        self.path = path
        self.phase = phase
        super().__init__(path)


class EDImage(PhaseImage):
    @classmethod
    def from_dir(cls, dir: Path, filename: str = "lvsa_ED.nii.gz"):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        nii_path = dir.joinpath(filename)
        return cls(nii_path, Phase.ED)


class ESImage(PhaseImage):
    @classmethod
    def from_dir(cls, dir: Path, filename: str = "lvsa_ES.nii.gz"):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        nii_path = dir.joinpath(filename)
        return cls(nii_path, Phase.ED)


class CineImages:
    def __init__(self, images: List[PhaseImage]):
        self.images = images

    @classmethod
    def from_dir(cls, dir: Path):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        images = []
        for idx, phase_name in enumerate(os.listdir(str(dir))):
            phase_path = dir.joinpath(phase_name)
            image = PhaseImage(path=phase_path, phase=idx)
            images.append(image)
        return cls(images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)


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
