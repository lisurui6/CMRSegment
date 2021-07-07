from pathlib import Path
from CMRSegment.common.resource import CineImages, Template, Phase, PhaseMesh, MeshResource, Segmentation
from tqdm import tqdm
import mirtk
from typing import List


class MotionTracker:
    def __init__(self, param_dir: Path, template_dir: Path):
        self.param_dir = param_dir
        self.ffd_motion_cfg = self.param_dir.joinpath("ffd_motion_2.cfg")
        self.ffd_refine_cfg = self.param_dir.joinpath("ffd_refine.cfg")
        self.template = Template(dir=template_dir)

    def run(self, cine: CineImages, ed_segmentation: Segmentation, landmark_path: Path, ED_mesh: PhaseMesh,
            output_dir: Path, overwrite: bool = False):
        # Forward image registration
        forward_dofs = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        dof_dir = output_dir.joinpath("dof")
        dof_dir.mkdir(parents=True, exist_ok=True)
        for fr in tqdm(range(1, len(cine))):

            target = cine[fr - 1]
            source = cine[fr]
            par = self.ffd_motion_cfg
            dof_output = dof_dir.joinpath("ffd_{:02d}_to_{:02d}.dof.gz".format(fr-1, fr))
            if not dof_output.exists():
                mirtk.register(
                    str(target),
                    str(source),
                    parin=str(par),
                    dofout=str(dof_output)
                )
            forward_dofs[fr] = dof_output

        # Compose inter-frame transformation fields #
        print("\n ...  Compose inter-frame transformation fields")
        compose_dofs = {}
        for fr in tqdm(range(2, len(cine))):
            dof_out = dof_dir.joinpath("ffd_comp_00_to_{:02d}.dof.gz".format(fr))
            dofs = [str(forward_dofs[k]) for k in range(1, fr+1)]
            if not dof_out.exists():
                mirtk.compose_dofs(
                    *dofs,
                    str(dof_out)
                )
            compose_dofs[fr] = dof_out

        # Refine motion fields
        # Composition of inter-frame motion fields can lead to accumulative errors.
        # At this step, we refine the motion fields by re-registering the n-th frame with the ED frame.
        refine_dofs = {}
        print("\n ...  Refine motion fields")
        for fr in tqdm(range(2, len(cine))):
            target = cine[0]
            source = cine[fr]
            dofin = compose_dofs[fr]
            dofout = dof_dir.joinpath("ffd_00_to_{:02d}.dof.gz".format(fr))

            if not dofout.exists():
                mirtk.register(
                    str(target),
                    str(source),
                    parin=str(self.ffd_refine_cfg),
                    dofin=str(dofin),
                    dofout=str(dofout)
                )
            refine_dofs[fr] = dofout
        refine_dofs[1] = forward_dofs[1]
        landmark_init_dof = dof_dir.joinpath("landmarks.dof.gz")
        if not landmark_init_dof.exists() or overwrite:
            mirtk.register(
                str(landmark_path),
                str(self.template.landmark),
                model="Rigid",
                dofout=str(landmark_init_dof),
            )

        lv_endo_frame_0 = self.transform_mesh(
            source_mesh=self.template.lv_endo(Phase.ED),
            target_mesh=ED_mesh.lv.endocardium,
            landmark_init_dof=landmark_init_dof,
            output_dir=output_dir.joinpath("lv", "endo"),
            overwrite=overwrite,
        )

        # Transform the mesh
        print("\n ...   Transform the LV endo mesh")
        self.motion_mesh(
            frame_0_mesh=lv_endo_frame_0,
            motion_dofs=refine_dofs,
            output_dir=output_dir.joinpath("lv", "endo"),
            overwrite=overwrite
        )

        lv_epi_frame_0 = self.transform_mesh(
            source_mesh=self.template.lv_epi(Phase.ED),
            target_mesh=ED_mesh.lv.epicardium,
            landmark_init_dof=landmark_init_dof,
            output_dir=output_dir.joinpath("lv", "epi"),
            overwrite=overwrite,
        )

        # Transform the mesh
        print("\n ...   Transform the LV epi mesh")
        self.motion_mesh(
            frame_0_mesh=lv_epi_frame_0,
            motion_dofs=refine_dofs,
            output_dir=output_dir.joinpath("lv", "epi"),
            overwrite=overwrite
        )

        rv_frame_0 = self.transform_mesh(
            source_mesh=self.template.rv(Phase.ED),
            target_mesh=ED_mesh.rv.rv,
            landmark_init_dof=landmark_init_dof,
            output_dir=output_dir.joinpath("rv"),
            overwrite=overwrite,
        )

        # Transform the mesh
        print("\n ...   Transform the RV mesh")
        self.motion_mesh(
            frame_0_mesh=rv_frame_0,
            motion_dofs=refine_dofs,
            output_dir=output_dir.joinpath("rv"),
            overwrite=overwrite
        )
        output_dir.joinpath("seg").mkdir(parents=True, exist_ok=True)
        for fr in tqdm(range(0, len(cine) - 1)):
            # os.system('mirtk transform-image {0}/lvsa_seg_fr00.nii.gz '
            #           '{2}/lvsa_seg_wrap_ffd_fr{1:02d}.nii.gz '
            #           '-dofin {3}/ffd_00_to_{1:02d}.dof.gz '
            #           '-invert -interp NN'
            #           .format(seg_slice_path, fr, pred_path, dof_path))
            mirtk.transform_image(
                str(ed_segmentation),
                str(output_dir.joinpath("seg").joinpath(f"lvsa_{fr}.nii.gz")),
                "-invert", "-v",
                interp="NN",
                dofin=refine_dofs[fr + 1]
            )

    @staticmethod
    def transform_mesh(source_mesh: MeshResource, target_mesh: MeshResource, landmark_init_dof: Path,
                       output_dir: Path, overwrite: bool = False) -> MeshResource:
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir.joinpath("temp")
        output_dir.joinpath("vtk").mkdir(parents=True, exist_ok=True)
        dof_dir = temp_dir.joinpath("dof")
        dof_dir.mkdir(parents=True, exist_ok=True)
        rigid_dof = dof_dir.joinpath("srreg.dof.gz")
        affine_dof = dof_dir.joinpath("sareg.dof.gz")
        nonrigid_dof = dof_dir.joinpath("snreg.dof.gz")
        temp_srreg_vtk = MeshResource(temp_dir.joinpath("srreg.vtk"))
        transformed_vtk = MeshResource(output_dir.joinpath("vtk", "fr00.vtk"))
        if not rigid_dof.exists() or overwrite:
            mirtk.register_points(
                "-t", str(target_mesh),
                "-s", str(source_mesh),
                "-symmetric",
                dofin=str(landmark_init_dof),
                dofout=str(rigid_dof),
            )
        if not temp_srreg_vtk.exists() or overwrite:
            mirtk.transform_points(
                str(source_mesh),
                str(temp_srreg_vtk),
                "-invert",
                dofin=str(rigid_dof),
            )

        if not affine_dof.exists() or overwrite:
            mirtk.register_points(
                "-t", str(temp_srreg_vtk),
                "-s", str(target_mesh),
                "-symmetric",
                model="Affine",
                dofout=str(affine_dof),
            )

        if not nonrigid_dof.exists() or overwrite:
            mirtk.register(
                str(temp_srreg_vtk),
                str(target_mesh),
                "-par", "Point set distance correspondence", "CP",
                ds=20,
                model="FFD",
                dofin=str(affine_dof),
                dofout=str(nonrigid_dof),
            )
        if not transformed_vtk.exists() or overwrite:
            mirtk.transform_points(
                str(temp_srreg_vtk),
                str(transformed_vtk),
                dofin=str(nonrigid_dof),
            )
        return transformed_vtk

    @staticmethod
    def motion_mesh(frame_0_mesh: MeshResource, motion_dofs: dict, output_dir: Path, overwrite: bool = False):
        vtks = [frame_0_mesh]
        for fr in tqdm(range(1, len(motion_dofs.keys()) + 1)):
            vtk = output_dir.joinpath("vtk", "fr{:02d}.vtk".format(fr))
            if not vtk.exists() or overwrite:
                mirtk.transform_points(
                    str(frame_0_mesh),
                    str(vtk),
                    dofin=str(motion_dofs[fr]),
                )
            vtks.append(vtk)

        # Convert vtks to text files
        print("\n ...   Convert vtks to text files")
        output_dir.joinpath("txt").mkdir(parents=True, exist_ok=True)
        txts = []
        for fr in tqdm(range(0, len(motion_dofs.keys()) + 1)):
            txt = output_dir.joinpath("txt", "fr{:02d}.txt".format(fr))
            if not txt.exists() or overwrite:
                mirtk.convert_pointset(
                    str(vtks[fr]),
                    str(txt),
                )
            txts.append(txt)
        return vtks, txts
