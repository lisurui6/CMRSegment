import mirtk
import shutil
from pathlib import Path
from CMRSegment.subject import Subject


class Coregister:
    def __init__(self):
        pass

    def run(self, subject: Subject, template_dir: Path):
        print("\n ... Mesh Generation - step [1] -")
        for phase, segmented_path, segmented_LR_path in zip(
                ["ED", "ES"],
                [subject.segmented_ed_path, subject.segmented_es_path],
                [subject.segmented_LR_ed_path, subject.segmented_LR_es_path]
        ):
            self.extract_meshes(segmented_path, phase, subject)
            self.initialize_registration(subject, template_dir)
            print("\n ... Mesh Generation - step [2] -")
            for fr in ['ED', 'ES']:
                self.register(subject, template_dir, fr)
                print("\n ... Mesh Generation - step [3] -")
                self.compute(subject, template_dir, fr)

    @staticmethod
    def extract_meshes(segmented_path: Path, phase: str, subject: Subject):
        """Extract meshes of lvendo, lvepi, lvmyo, rv and rveip at ED and ES, respectively"""
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 3, 4,
            set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("RV_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 3, 4,
            set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_RVepi_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_RVepi_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("RVepi_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-map", 3, 0, 4, 0,
            output=str(subject.tmps_dir().joinpath("vtk_LV_{}.nii.gz".format(phase))),
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 1, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVendo_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVendo_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 1, 2, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVepi_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVepi_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 2, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVmyo_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVmyo_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVmyo_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )

    @staticmethod
    def initialize_registration(subject: Subject, template_dir: Path):
        """Use landmark to initialise the registration"""
        mirtk.register(
            str(subject.landmark_path),
            str(template_dir.joinpath("landmarks2.vtk")),
            model="Rigid",
            dofout=str(subject.dofs_dir().joinpath("landmarks.dof.gz")),
        )

    @staticmethod
    def register(subject, template_dir, fr):
        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.dofs_dir().joinpath("landmarks.dof.gz")),
            dofout=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr)))
        )

        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("RVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVmyo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_image(
            str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            "-invert",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.transform_image(
            str(subject.tmps_dir().joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            "-invert",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        # affine

        mirtk.smooth_image(
            str(template_dir.joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
            1,
            "-float"
        )
        mirtk.register(
            str(subject.tmps_dir().joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            model="Affine",
            dofout=str(subject.tmps_dir().joinpath("rv_{}_areg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segareg.txt")),
        )

        mirtk.register(
            str(template_dir.joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            model="Affine",
            dofout=str(subject.tmps_dir().joinpath("lv_{}_areg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segareg.txt")),
        )
        # non-rigid
        mirtk.register(
            str(template_dir.joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("rv_{}_areg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv_{}_nreg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segreg.txt")),
        )

        mirtk.register(
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            # "-symmetric",
            "-par", "Point set distance correspondence", "CP",
            ds=8,
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("rv_{}_nreg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv{}ds8.dof.gz".format(fr))),
        )

        mirtk.register(
            str(template_dir.joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_areg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv_{}_nreg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segreg.txt")),
        )

        mirtk.register(
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            # "-symmetric",
            "-par", "Energy function", "PCD(T o P(1:2:end), P(2:2:end))",
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_nreg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            ds=4,
        )
        # same number of points
        mirtk.match_points(
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
        )

        mirtk.match_points(
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
        )

        mirtk.transform_points(
            str(template_dir.joinpath("LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
        )

        mirtk.match_points(
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv{}ds8.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
        )

        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVendo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVepi_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVmyo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))
        )

        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("W_LVmyo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))
        )

    @staticmethod
    def compute(subject, template_dir, fr):
        """Compute the quantities of the heart with respect to template"""

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_LVendo_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            name="WallThickness",
        )
        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_LVepi_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )

        mirtk.calculate_surface_attributes(
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            smooth_iterations=64,
        )

        mirtk.calculate_surface_attributes(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            smooth_iterations=64,
        )

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))),
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-normal",
            name="WallThickness",
        )

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )

        if fr == 'ED':
            fr_ = 'ed'
        if fr == 'ES':
            fr_ = 'es'

        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_curvature.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_wallthickness.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_signeddistances.txt".format(fr_))),
        )

        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("W_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_wallthickness.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_curvature.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("S_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_signeddistances.txt".format(fr_))),
        )
