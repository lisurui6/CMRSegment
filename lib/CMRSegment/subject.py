import shutil
from pathlib import Path
from typing import List


class Subject:
    def __init__(self, dir: Path, nii_name: str):
        self.dir = dir
        self.nii_path = self.dir.joinpath(nii_name)
        self.es_path = self.dir.joinpath("lvsa_ES.nii.gz")
        self.ed_path = self.dir.joinpath("lvsa_ED.nii.gz")
        self.tmp_nii_path = self.dir.joinpath("lvsa_.nii.gz")
        self.rview_dir().mkdir(exist_ok=True)
        self.enlarge_phases_dir().mkdir(exist_ok=True)
        self.gray_phases_dir().mkdir(exist_ok=True)
        self.resample_phases_dir().mkdir(exist_ok=True)

    @property
    def name(self):
        return self.dir.name

    def dofs_dir(self):
        return self.dir.joinpath("dofs")

    def rview_dir(self):
        return self.dir.joinpath("4D_rview")

    def enlarge_phases_dir(self):
        return self.dir.joinpath("enlarge_phases")

    def gray_phases_dir(self):
        return self.dir.joinpath("gray_phases")

    def resample_phases_dir(self):
        return self.dir.joinpath("resample_phases")

    def segs_dir(self):
        return self.dir.joinpath("segs")

    def sizes_dir(self):
        return self.dir.joinpath("sizes")

    def vtks_dir(self):
        return self.dir.joinpath("vtks")

    def tmps_dir(self):
        return self.dir.joinpath("tmps")

    def motions_dir(self):
        return self.dir.joinpath("motions")

    def subdirs(self) -> List[Path]:
        for subdir in ["dofs", "segs", "tmps", "sizes", "motions", "vtks"]:
            yield self.dir.joinpath(subdir)

    def clean(self):
        for subdir in self.subdirs():
            if subdir.exists():
                shutil.rmtree(str(subdir), ignore_errors=True)
            subdir.mkdir(parents=True, exist_ok=False)
        if self.dir.joinpath("PHsegmentation_ED.gipl").exists():
            for p in self.dir.glob("*.gipl"):
                p.unlink()
        if self.tmp_nii_path.exists():
            for p in self.dir.glob("lvsa_*.nii.gz"):
                p.unlink()
            for p in self.dir.glob("seg_*.nii.gz"):
                p.unlink()
