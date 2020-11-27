import mirtk
from CMRSegment.subject import Mesh, Segmentation
from pathlib import Path
from CMRSegment.utils import extract_lv_label, extract_rv_label


class MeshExtractor:
    def __init__(self, iso_value: int = 120, blur: int = 2):
        self.iso_value = iso_value
        self.blur = blur

    def run(self, segmentation: Segmentation, output_dir: Path) -> Mesh:
        mesh = Mesh(dir=output_dir, phase=segmentation.phase)
        self.rv(segmentation, mesh.rv)
        self.rv_epi(segmentation, mesh.rv_epi)
        self.lv_endo(segmentation, mesh.lv_endo)
        self.lv_epi(segmentation, mesh.lv_epi)
        self.lv_myo(segmentation, mesh.lv_myo)
        return mesh

    def rv(self, segmentation: Segmentation, output_path: Path):
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
