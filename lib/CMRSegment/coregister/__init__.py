import mirtk
import shutil
import logging
from pathlib import Path

from CMRSegment.common.resource import PhaseMesh, Segmentation, Template, Phase
from CMRSegment.common.utils import extract_lv_label, extract_rv_label

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("CMRSegment.coregister")


# TODO: multi process
class Coregister:
    def __init__(self, template_dir: Path, param_dir: Path, overwrite: bool = False):
        self.template = Template(dir=template_dir)
        self.template.check_valid()
        segareg_path = param_dir.joinpath("segareg.txt")
        segreg_path = param_dir.joinpath("segreg.txt")
        spnreg_path = param_dir.joinpath("spnreg.txt")

        if not segreg_path.exists():
            raise FileNotFoundError(f"segreg.txt does not exist at {segreg_path}")
        if not spnreg_path.exists():
            raise FileNotFoundError(f"spnreg_path.txt does not exist at {spnreg_path}")
        if not segareg_path.exists():
            raise FileNotFoundError(f"segareg_path.txt does not exist at {segareg_path}")
        self.segareg_path = segareg_path
        self.segreg_path = segreg_path
        self.spnreg_path = spnreg_path
        self.logger = LOGGER
        self.overwrite = overwrite

    def run(self, mesh: PhaseMesh, segmentation: Segmentation, landmark_path: Path, output_dir: Path):
        if not landmark_path.exists():
            raise FileNotFoundError(
                f"Landmark file does not exist at {landmark_path}. "
                f"To generate landmark, please run landmark extractor first."
            )
        try:
            mesh.check_valid()
        except FileNotFoundError as e:
            self.logger.error(f"Mesh does not exist. To generate mesh, please run mesh extractor first.")
            raise e
        if not segmentation.path.exists():
            self.logger.error(f"Segmentation does not exist at {segmentation.path}. To generate segmenation, "
                              f"please run segmentor first.")
        temp_dir = output_dir.joinpath("temp")
        if self.overwrite:
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(str(output_dir), ignore_errors=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info("\n ... Mesh Generation - step [1] -")
        landmark_dofs = self.initialize_registration(landmark_path, output_dir)
        rv_label = extract_rv_label(
            segmentation_path=segmentation.path,
            output_path=temp_dir.joinpath(f"vtk_RV_{segmentation.phase}.nii.gz")
        )

        lv_label = extract_lv_label(
            segmentation_path=segmentation.path,
            output_path=temp_dir.joinpath(f"vtk_LV_{segmentation.phase}.nii.gz"),
        )
        self.logger.info("\n ... Mesh Generation - step [2] -")
        nonrigid_transformed_mesh = self.register(
            mesh=mesh,
            landmark_dofs=landmark_dofs,
            rv_label=rv_label,
            lv_label=lv_label,
            output_dir=output_dir,
        )
        # self.logger.info("\n ... Mesh Generation - step [3] -")
        # self.compute_wall_thickness(nonrigid_transformed_mesh, output_dir)
        # self.compute_curvature(nonrigid_transformed_mesh, output_dir)
        return nonrigid_transformed_mesh

    def initialize_registration(self, landmark_path: Path, output_dir: Path):
        """Use landmark to initialise the registration"""
        if not output_dir.joinpath("landmarks.dof.gz").exists() or self.overwrite:
            mirtk.register(
                str(landmark_path),
                str(self.template.landmark),
                model="Rigid",
                dofout=str(output_dir.joinpath("landmarks.dof.gz")),
            )
        return output_dir.joinpath("landmarks.dof.gz")

    def rigid_registration(self, mesh: PhaseMesh, landmark_dofs: Path, output_dir: Path):
        """Rigid registration of mesh against template using dof estimated by comparing landmarks."""
        self.logger.info("Perform rigid registration ")
        fr = mesh.phase
        temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        if not temp_dir.joinpath("{}.dof.gz".format(fr)).exists() or self.overwrite:
            mirtk.register_points(
                "-t", str(mesh.rv.rv),
                "-s", str(self.template.rv(fr)),
                "-t", str(mesh.lv.endocardium),
                "-s", str(self.template.lv_endo(fr)),
                "-t", str(mesh.lv.epicardium),
                "-s", str(self.template.lv_epi(fr)),
                "-symmetric",
                dofin=str(landmark_dofs),
                dofout=str(temp_dir.joinpath("{}.dof.gz".format(fr)))
            )

        if not temp_dir.joinpath("lv_{}_rreg.dof.gz".format(fr)).exists() or self.overwrite:
            mirtk.register_points(
                "-t", str(mesh.lv.endocardium),
                "-s", str(self.template.lv_endo(fr)),
                "-t", str(mesh.lv.epicardium),
                "-s", str(self.template.lv_epi(fr)),
                "-symmetric",
                dofin=str(temp_dir.joinpath("{}.dof.gz".format(fr))),
                dofout=str(temp_dir.joinpath("lv_{}_rreg.dof.gz".format(fr))),
            )
        if not temp_dir.joinpath("rv_{}_rreg.dof.gz".format(fr)).exists() or self.overwrite:
            mirtk.register_points(
                "-t", str(mesh.rv.rv),
                "-s", str(self.template.rv(fr)),
                "-symmetric",
                dofin=str(temp_dir.joinpath("{}.dof.gz".format(fr))),
                dofout=str(temp_dir.joinpath("rv_{}_rreg.dof.gz".format(fr))),
            )
        return temp_dir.joinpath("lv_{}_rreg.dof.gz".format(fr)), temp_dir.joinpath("rv_{}_rreg.dof.gz".format(fr))

    def rigid_transform(self, mesh: PhaseMesh, lv_rigid_transform: Path, rv_rigid_transform: Path, output_dir: Path):
        """Transform mesh according to estimated rigid transformation."""
        transformed_mesh = PhaseMesh.from_dir(
            phase=mesh.phase,
            dir=output_dir.joinpath("rigid")
        )
        if not transformed_mesh.exists() or self.overwrite:
            output_dir.joinpath("rigid").mkdir(exist_ok=True, parents=True)
            mirtk.transform_points(
                str(mesh.rv.rv),
                str(transformed_mesh.rv.rv),
                dofin=str(rv_rigid_transform),
            )

            mirtk.transform_points(
                str(mesh.rv.epicardium),
                str(transformed_mesh.rv.epicardium),
                dofin=str(rv_rigid_transform),
            )

            mirtk.transform_points(
                str(mesh.lv.endocardium),
                str(transformed_mesh.lv.endocardium),
                dofin=str(lv_rigid_transform),
            )

            mirtk.transform_points(
                str(mesh.lv.epicardium),
                str(transformed_mesh.lv.epicardium),
                dofin=str(lv_rigid_transform),
            )
            mirtk.transform_points(
                str(mesh.lv.myocardium),
                str(transformed_mesh.lv.myocardium),
                dofin=str(lv_rigid_transform),
            )
        return transformed_mesh

    def label_rigid_transform(self, lv_label: Path, rv_label: Path, lv_rigid_transform: Path, phase: Phase,
                              output_dir: Path):
        """Transform segmentation label according to estimate rigid transformation."""
        temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        lv_label_transformed = temp_dir.joinpath("N_vtk_LV_{}.nii.gz".format(phase))
        rv_label_transformed = temp_dir.joinpath("N_vtk_RV_{}.nii.gz".format(phase))
        if not lv_label_transformed.exists() or self.overwrite:
            mirtk.transform_image(
                str(lv_label),
                str(lv_label_transformed),
                dofin=str(lv_rigid_transform),
            )
        if not rv_label_transformed.exists() or self.overwrite:
            mirtk.transform_image(
                str(rv_label),
                str(rv_label_transformed),
                dofin=str(lv_rigid_transform),
            )
        return lv_label_transformed, rv_label_transformed

    def affine_registration(self, lv_label_transformed: Path, rv_label_transformed: Path, fr: Phase, output_dir: Path, mesh):
        temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        if not temp_dir.joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr)).exists() or self.overwrite:
            mirtk.smooth_image(
                str(self.template.vtk_rv(fr)),
                str(temp_dir.joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
                1,
                "-float"
            )
        if not temp_dir.joinpath("rv_{}_areg.dof.gz".format(fr)).exists() or self.overwrite:
            mirtk.register(
                str(self.template.rv(fr)),
                str(mesh.rv.rv),
                "-par", "Point set distance correspondence", "CP",
                model="Affine",
                dofout=str(temp_dir.joinpath("rv_{}_areg.dof.gz".format(fr))),
                parin=str(self.segareg_path),
            )
            # mirtk.register(
            #     str(temp_dir.joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
            #     str(rv_label_transformed),
            #     model="Affine",
            #     dofout=str(temp_dir.joinpath("rv_{}_areg.dof.gz".format(fr))),
            #     parin=str(self.segareg_path),
            # )
        if not temp_dir.joinpath("lv_{}_areg.dof.gz".format(fr)).exists() or self.overwrite:
            mirtk.register(
                str(self.template.lv_epi(fr)),
                str(mesh.lv.epicardium),
                "-par", "Point set distance correspondence", "CP",
                model="Affine",
                dofout=str(temp_dir.joinpath("lv_{}_areg.dof.gz".format(fr))),
                parin=str(self.segareg_path),
            )
            # mirtk.register(
            #     str(self.template.vtk_lv(fr)),
            #     str(lv_label_transformed),
            #     model="Affine",
            #     dofout=str(temp_dir.joinpath("lv_{}_areg.dof.gz".format(fr))),
            #     parin=str(self.segareg_path),
            # )
        return temp_dir.joinpath("lv_{}_areg.dof.gz".format(fr)), temp_dir.joinpath("rv_{}_areg.dof.gz".format(fr))

    def nonrigid_registration(self, mesh: PhaseMesh, lv_label_transformed: Path, rv_label_transformed: Path,
                              lv_affine_transform: Path, rv_affine_transform: Path, fr: Phase, output_dir: Path):
        temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        rv_transform = temp_dir.joinpath("rv_{}_nreg.dof.gz".format(fr))
        lv_transform = temp_dir.joinpath("lv_{}_nreg.dof.gz".format(fr))
        # if not rv_transform.exists() or self.overwrite:
        #     mirtk.register(
        #         str(self.template.vtk_rv(fr)),
        #         str(rv_label_transformed),
        #         model="FFD",
        #         dofin=str(rv_affine_transform),
        #         dofout=str(rv_transform),
        #         parin=str(self.segreg_path),
        #     )
        if not rv_transform.exists() or self.overwrite:
            mirtk.register(
                str(self.template.rv(fr)),
                str(mesh.rv.rv),
                # "-symmetric",
                "-par", "Point set distance correspondence", "CP",
                ds=8,
                model="FFD",
                dofin=str(rv_affine_transform),
                dofout=str(rv_transform),
            )
        # if not lv_transform.exists() or self.overwrite:
        #     mirtk.register(
        #         str(self.template.vtk_lv(fr)),
        #         str(lv_label_transformed),
        #         model="FFD",
        #         dofin=str(lv_affine_transform),
        #         dofout=str(lv_transform),
        #         parin=str(self.segreg_path),
        #     )
        if not lv_transform.exists() or self.overwrite:
            mirtk.register(
                str(self.template.lv_endo(fr)),
                str(mesh.lv.endocardium),
                str(self.template.lv_epi(fr)),
                str(mesh.lv.epicardium),
                # "-symmetric",
                "-par", "Energy function", "PCD(T o P(1:2:end), P(2:2:end))",
                model="FFD",
                dofin=str(lv_affine_transform),
                dofout=str(lv_transform),
                ds=4,
            )
        return lv_transform, rv_transform

    def nonrigid_transform(self, mesh: PhaseMesh, lv_nonrigid_transform: Path, rv_nonrigid_transform: Path, fr: Phase,
                           output_dir: Path):
        transformed_mesh = PhaseMesh.from_dir(
            phase=mesh.phase,
            dir=output_dir.joinpath("nonrigid")
        )
        if not transformed_mesh.exists() or self.overwrite:
            output_dir.joinpath("nonrigid").mkdir(parents=True, exist_ok=True)
            # mirtk.match_points(
            #     str(self.template.lv_endo(fr)),
            #     str(mesh.lv.endocardium),
            #     dofin=str(lv_nonrigid_transform),
            #     output=str(transformed_mesh.lv.endocardium),
            # )

            mirtk.transform_points(
                str(self.template.lv_endo(fr)),
                str(transformed_mesh.lv.endocardium),
                dofin=str(lv_nonrigid_transform),
            )

            # mirtk.match_points(
            #     str(self.template.lv_epi(fr)),
            #     str(mesh.lv.epicardium),
            #     dofin=str(lv_nonrigid_transform),
            #     output=str(transformed_mesh.lv.epicardium),
            # )

            mirtk.transform_points(
                str(self.template.lv_epi(fr)),
                str(transformed_mesh.lv.epicardium),
                dofin=str(lv_nonrigid_transform),
            )

            mirtk.transform_points(
                str(self.template.lv_myo(fr)),
                str(transformed_mesh.lv.myocardium),
                dofin=str(lv_nonrigid_transform),
            )

            # mirtk.match_points(
            #     str(self.template.rv(fr)),
            #     str(mesh.rv.rv),
            #     dofin=str(rv_nonrigid_transform),
            #     output=str(transformed_mesh.rv.rv),
            # )

            mirtk.transform_points(
                str(self.template.rv(fr)),
                str(transformed_mesh.rv.rv),
                dofin=str(rv_nonrigid_transform),
            )
            # shutil.copy(str(mesh.rv.epicardium), str(transformed_mesh.rv.epicardium))
        return transformed_mesh

    def register(self, mesh: PhaseMesh, landmark_dofs: Path, rv_label: Path, lv_label: Path, output_dir: Path):
        fr = mesh.phase
        lv_rigid_transform, rv_rigid_transform = self.rigid_registration(mesh, landmark_dofs, output_dir)
        transformed_mesh = PhaseMesh.from_dir(
            phase=mesh.phase,
            dir=output_dir.joinpath("debug", "rigid")
        )
        if not transformed_mesh.exists() or self.overwrite:
            output_dir.joinpath("debug", "rigid").mkdir(exist_ok=True, parents=True)
            mirtk.transform_points(
                str(mesh.lv.endocardium),
                str(transformed_mesh.lv.endocardium),
                dofin=str(lv_rigid_transform),
            )
            mirtk.transform_points(
                str(mesh.lv.epicardium),
                str(transformed_mesh.lv.epicardium),
                dofin=str(lv_rigid_transform),
            )
            mirtk.transform_points(
                str(mesh.lv.myocardium),
                str(transformed_mesh.lv.myocardium),
                dofin=str(lv_rigid_transform),
            )
            mirtk.transform_points(
                str(mesh.rv.rv),
                str(transformed_mesh.rv.rv),
                dofin=str(rv_rigid_transform),
            )
        rigid_transformed_mesh = self.rigid_transform(mesh, lv_rigid_transform, rv_rigid_transform, output_dir)
        lv_label_transformed, rv_label_transformed = self.label_rigid_transform(
            lv_label=lv_label,
            rv_label=rv_label,
            lv_rigid_transform=lv_rigid_transform,
            phase=fr,
            output_dir=output_dir
        )

        # affine
        lv_affine_transform, rv_affine_transform = self.affine_registration(
            lv_label_transformed, rv_label_transformed, fr, output_dir, rigid_transformed_mesh
        )
        transformed_mesh = PhaseMesh.from_dir(
            phase=mesh.phase,
            dir=output_dir.joinpath("debug", "affine")
        )
        if not transformed_mesh.exists() or self.overwrite:

            output_dir.joinpath("debug", "affine").mkdir(exist_ok=True, parents=True)
            # mirtk.transform_points(
            #     str(rigid_transformed_mesh.lv.endocardium),
            #     str(transformed_mesh.lv.endocardium),
            #     dofin=str(lv_affine_transform),
            # )
            # mirtk.transform_points(
            #     str(rigid_transformed_mesh.lv.epicardium),
            #     str(transformed_mesh.lv.epicardium),
            #     dofin=str(lv_affine_transform),
            # )
            # mirtk.transform_points(
            #     str(rigid_transformed_mesh.lv.myocardium),
            #     str(transformed_mesh.lv.myocardium),
            #     dofin=str(lv_affine_transform),
            # )
            # mirtk.transform_points(
            #     str(rigid_transformed_mesh.rv.rv),
            #     str(transformed_mesh.rv.rv),
            #     dofin=str(rv_affine_transform),
            # )

            mirtk.transform_points(
                str(self.template.lv_endo(fr)),
                str(transformed_mesh.lv.endocardium),
                dofin=str(lv_affine_transform),
            )
            mirtk.transform_points(
                str(self.template.lv_epi(fr)),
                str(transformed_mesh.lv.epicardium),
                dofin=str(lv_affine_transform),
            )
            mirtk.transform_points(
                str(self.template.lv_myo(fr)),
                str(transformed_mesh.lv.myocardium),
                dofin=str(lv_affine_transform),
            )
            mirtk.transform_points(
                str(self.template.rv(fr)),
                str(transformed_mesh.rv.rv),
                dofin=str(rv_affine_transform),
            )
        # non-rigid
        lv_nonrigid_transform, rv_nonrigid_transform = self.nonrigid_registration(
            transformed_mesh, lv_label_transformed, rv_label_transformed,
            lv_affine_transform, rv_affine_transform, fr, output_dir
        )
        # same number of points
        nonrigid_transformed_mesh = self.nonrigid_transform(
            transformed_mesh, lv_nonrigid_transform, rv_nonrigid_transform, fr, output_dir
        )
        return nonrigid_transformed_mesh

    # def compute_wall_thickness(self, mesh: PhaseMesh, output_dir: Path):
    #     fr = mesh.phase
    #     output_lv_thickness = output_dir.joinpath("wt", f"LVmyo_{fr}.vtk")
    #     output_rv_thickness = output_dir.joinpath("wt", f"RV_{fr}.vtk")
    #     output_lv_thickness.parent.mkdir(parents=True, exist_ok=True)
    #
    #     if not output_lv_thickness.exists() or self.overwrite:
    #         mirtk.evaluate_distance(
    #             str(mesh.lv.endocardium),
    #             str(mesh.lv.epicardium),
    #             str(output_lv_thickness),
    #             name="WallThickness",
    #         )
    #     if not output_rv_thickness.exists() or self.overwrite:
    #         mirtk.evaluate_distance(
    #             str(mesh.rv.rv),
    #             str(mesh.rv.epicardium),
    #             str(output_rv_thickness),
    #             name="WallThickness",
    #         )
    #     if not output_dir.joinpath("rv_{}_wallthickness.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_rv_thickness),
    #             str(output_dir.joinpath("rv_{}_wallthickness.txt".format(fr))),
    #         )
    #     if not output_dir.joinpath("lv_myo{}_wallthickness.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_lv_thickness),
    #             str(output_dir.joinpath("lv_myo{}_wallthickness.txt".format(fr))),
    #         )

    # def compute_curvature(self, mesh: PhaseMesh, output_dir: Path):
    #     fr = mesh.phase
    #     output_lv_curv = output_dir.joinpath("curv", f"LVmyo_{fr}.vtk")
    #     output_rv_curv = output_dir.joinpath("curv", f"RV_{fr}.vtk")
    #     output_rv_curv.parent.mkdir(parents=True, exist_ok=True)
    #
    #     if not output_lv_curv.exists() or self.overwrite:
    #         mirtk.calculate_surface_attributes(
    #             str(mesh.lv.myocardium),
    #             str(output_lv_curv),
    #             smooth_iterations=64,
    #         )
    #
    #     if not output_rv_curv.exists() or self.overwrite:
    #         mirtk.calculate_surface_attributes(
    #             str(mesh.rv.rv),
    #             str(output_rv_curv),
    #             smooth_iterations=64,
    #         )
    #     if not output_dir.joinpath("rv_{}_curvature.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_rv_curv),
    #             str(output_dir.joinpath("rv_{}_curvature.txt".format(fr))),
    #         )
    #     if not output_dir.joinpath("lv_myo{}_curvature.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_lv_curv),
    #             str(output_dir.joinpath("lv_myo{}_curvature.txt".format(fr))),
    #         )
