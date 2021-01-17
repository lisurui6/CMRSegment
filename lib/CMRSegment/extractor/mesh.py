import mirtk
from CMRSegment.common.subject import Mesh, Segmentation
from pathlib import Path
from CMRSegment.common.utils import extract_lv_label, extract_rv_label
import shutil


class MeshExtractor:
    def __init__(self, iso_value: int = 120, blur: int = 2, overwrite: bool = False):
        self.iso_value = iso_value
        self.blur = blur
        self.overwrite = overwrite

    def run(self, segmentation: Segmentation, output_dir: Path) -> Mesh:
        temp_dir = output_dir.joinpath("temp")
        if self.overwrite:
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(str(output_dir), ignore_errors=True)
        mesh = Mesh(dir=output_dir, phase=segmentation.phase)
        if not mesh.exists() or self.overwrite:
            self.rv(segmentation, mesh.rv)
            self.rv_epi(segmentation, mesh.rv_epi)
            self.lv_endo(segmentation, mesh.lv_endo)
            self.lv_epi(segmentation, mesh.lv_epi)
            self.lv_myo(segmentation, mesh.lv_myo)
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        return mesh

    def rv(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            temp_dir = output_path.parent.joinpath("temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            extract_rv_label(
                segmentation_path=segmentation.path,
                output_path=temp_dir.joinpath("vtk_RV_{}.nii.gz".format(segmentation.phase))
            )
            mirtk.extract_surface(
                str(temp_dir.joinpath("vtk_RV_{}.nii.gz".format(segmentation.phase))),
                str(output_path),
                isovalue=self.iso_value, blur=self.blur,
            )

    def rv_epi(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            temp_dir = output_path.parent.joinpath("temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            mirtk.calculate_element_wise(
                str(segmentation.path),
                "-label", 3, 4,
                set=255, pad=0,
                output=str(temp_dir.joinpath("vtk_RVepi_{}.nii.gz".format(segmentation.phase))),
            )
            mirtk.extract_surface(
                str(temp_dir.joinpath("vtk_RVepi_{}.nii.gz".format(segmentation.phase))),
                str(output_path),
                isovalue=self.iso_value, blur=self.blur,
            )

    def lv_endo(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:

            temp_dir = output_path.parent.joinpath("temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            extract_lv_label(
                segmentation_path=segmentation.path,
                output_path=temp_dir.joinpath("vtk_LV_{}.nii.gz".format(segmentation.phase))
            )
            mirtk.calculate_element_wise(
                str(segmentation.path),
                "-label", 1, set=255, pad=0,
                output=str(temp_dir.joinpath("vtk_LVendo_{}.nii.gz".format(segmentation.phase))),
            )
            mirtk.extract_surface(
                str(temp_dir.joinpath("vtk_LVendo_{}.nii.gz".format(segmentation.phase))),
                str(output_path),
                isovalue=self.iso_value, blur=self.blur,
            )

    def lv_epi(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:

            temp_dir = output_path.parent.joinpath("temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            mirtk.calculate_element_wise(
                str(segmentation.path),
                "-label", 1, 2, set=255, pad=0,
                output=str(temp_dir.joinpath("vtk_LVepi_{}.nii.gz".format(segmentation.phase))),
            )
            mirtk.extract_surface(
                str(temp_dir.joinpath("vtk_LVepi_{}.nii.gz".format(segmentation.phase))),
                str(output_path),
                isovalue=self.iso_value, blur=self.blur,
            )

    def lv_myo(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            temp_dir = output_path.parent.joinpath("temp")
            temp_dir.mkdir(exist_ok=True, parents=True)
            mirtk.calculate_element_wise(
                str(segmentation.path),
                "-label", 2, set=255, pad=0,
                output=str(temp_dir.joinpath("vtk_LVmyo_{}.nii.gz".format(segmentation.phase))),
            )
            mirtk.extract_surface(
                str(temp_dir.joinpath("vtk_LVmyo_{}.nii.gz".format(segmentation.phase))),
                str(output_path),
                isovalue=self.iso_value, blur=self.blur,
            )
