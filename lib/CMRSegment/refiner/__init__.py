import mirtk
mirtk.subprocess.showcmd = True
import vtk
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from trimesh.registration import procrustes

from CMRSegment.common.constants import RESOURCE_DIR
from CMRSegment.common.utils import set_affine

from CMRSegment.common.resource import Segmentation, PhaseImage
from CMRSegment.common.data_table import DataTable


class SegmentationRefiner:
    """Use multi-altas registration to refine predicted segmentation"""
    def __init__(self, csv_path: Path, n_atlas: int = None, param_path: Path = None):
        assert csv_path.exists(), "Path to csv file containing list of atlases must exist. "
        data_table = DataTable.from_csv(csv_path)
        label_paths = data_table.select_column("label_path")
        landmarks = []
        atlases = []
        for idx, path in enumerate(label_paths):
            if idx % 2 == 0:
                if Path(path).parent.joinpath("landmarks2.vtk").exists():
                    atlases.append(Path(path))
                    landmarks.append(Path(path).parent.joinpath("landmarks2.vtk"))
        print("Total {} atlases with landmarks...".format(len(atlases)))
        if n_atlas is not None:
            if n_atlas < len(atlases):
                print("Randomly choosing {} atlases...".format(n_atlas))
                indices = np.random.choice(np.arange(len(atlases)), n_atlas, replace=False)
                atlases = np.array(atlases)
                landmarks = np.array(landmarks)
                atlases = atlases[indices].tolist()
                landmarks = landmarks[indices].tolist()
                print("Total {} atlases remained...".format(len(atlases)))

        self.atlases = atlases
        self.landmarks = landmarks
        if param_path is None:
            param_path = RESOURCE_DIR.joinpath("ffd_label_1.cfg")
        self.param_path = param_path
        self.affine_param_path = RESOURCE_DIR.joinpath("segareg_2.txt")

    def select_altases(self, subject_image_path, subject_seg: Path, subject_landmarks: Path, output_dir: Path,
                       n_top: int, force: bool):
        """Select top similar atlases, according to subject segmentation and landmark"""
        nmi = []

        top_similar_atlases = []

        n_atlases = len(self.atlases)

        # if force:
        #     for f in output_dir.glob("shapenmi*.txt"):
        #         f.unlink()
        output_dofs = []
        top_atlas_dofs = []
        top_atlas_landmarks = []

        for i in range(n_atlases):
            try:
                if not output_dir.joinpath(f"shapelandmarks_{i}.dof.gz").exists() or force:

                    # Rigid registration with landmarks
                    poly = vtk.vtkPolyData()
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(str(self.landmarks[i]))
                    reader.Update()
                    atlas_landmark_polydata = reader.GetOutput()
                    atlas_landmark = np.array(atlas_landmark_polydata.GetPoints().GetData())

                    poly = vtk.vtkPolyData()
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(str(subject_landmarks))
                    reader.Update()
                    subject_landmark_polydata = reader.GetOutput()
                    subject_landmark = np.array(subject_landmark_polydata.GetPoints().GetData())

                    matrix, transform, cost = procrustes(
                        a=subject_landmark, b=atlas_landmark, reflection=True, translation=True, scale=False,
                        return_cost=True
                    )  # matrix transform a to b
                    # save matrix to dofs
                    matrix = matrix.tolist()

                    with open(str(output_dir.joinpath(f"shapelandmarks_{i}.txt")), "w") as file:
                        for array in matrix:
                            array = [str(a) for a in array]
                            file.write(" ".join(array) + "\n")
                    mirtk.convert_dof(
                        output_dir.joinpath(f"shapelandmarks_{i}.txt"),
                        output_dir.joinpath(f"shapelandmarks_{i}.dof.gz"),
                        input_format="aladin",
                        output_format="mirtk",
                    )

                    # Affine registration using landmark as initialisation
                    # Split label maps into separate binary masks
                    atlas_label_paths = []
                    subject_label_paths = []
                    output_dir.joinpath("temp_labels").mkdir(parents=True, exist_ok=True)

                    new_atlas_path = output_dir.joinpath("temp_labels", "atlas", f"{i}", self.atlases[i].name)
                    output_dir.joinpath("temp_labels", "atlas", f"{i}").mkdir(parents=True, exist_ok=True)
                    mirtk.calculate_element_wise(
                        str(self.atlases[i]),
                        "-label", 3, 4,
                        set=3,
                        output=str(new_atlas_path),
                    )
                    set_affine(self.atlases[i], new_atlas_path)
                    self.atlases[i] = new_atlas_path

                    mirtk.transform_image(
                        str(new_atlas_path),
                        str(new_atlas_path.parent.joinpath("atlas_label_init.nii.gz")),
                        dofin=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                        target=str(subject_image_path),
                        interp="NN",
                    )
                    mirtk.transform_points(
                        str(self.landmarks[i]),
                        str(new_atlas_path.parent.joinpath(f"atlas_lm_init.vtk")),
                        "-invert",
                        dofin=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                    )

                    mirtk.transform_points(
                        str(subject_landmarks),
                        str(new_atlas_path.parent.joinpath(f"subject_lm_init.vtk")),
                        dofin=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                    )
                    new_atlas_path = new_atlas_path.parent.joinpath("atlas_label_init.nii.gz")
                    for tag, label_path in zip(["atlas", "subject"], [new_atlas_path, subject_seg]):
                        for label in [1, 2, 3]:
                            if tag == "atlas":
                                path = output_dir.joinpath("temp_labels", "atlas", f"{i}",
                                                           Path(label_path.stem).stem + f"_{label}.nii.gz")
                            else:
                                path = output_dir.joinpath(
                                    "temp_labels", tag, Path(label_path.stem).stem + f"_{label}.nii.gz"
                                )
                            output_dir.joinpath("temp_labels", tag).mkdir(parents=True, exist_ok=True)
                            # path = path.stem.joinpath(f"seg_lvsa_SR_ED_{label}.nii.gz")
                            if tag == "atlas":
                                atlas_label_paths.append(path)
                            else:
                                subject_label_paths.append(path)
                            if not path.exists():
                                mirtk.calculate_element_wise(
                                    str(label_path),
                                    opts=[
                                        ("binarize", label, label),
                                        ("out", str(path))
                                    ],
                                )
                            set_affine(label_path, path)

                    mirtk.register(
                        *[str(path) for path in subject_label_paths],  # target
                        *[str(path) for path in atlas_label_paths],  # source
                        dofin=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                        dofout=str(output_dir.joinpath(f"shapeaffine_{i}.dof.gz")),
                        parin=str(self.affine_param_path),
                        model="Affine",
                    )

                if not output_dir.joinpath(f"shapenmi_{i}.txt").exists() or force:
                    mirtk.evaluate_similarity(
                        str(subject_seg),  # target
                        str(self.atlases[i]),  # source
                        Tbins=64,
                        Sbins=64,
                        dofin=str(output_dir.joinpath(f"shapeaffine_{i}.dof.gz")),  # source image transformation
                        table=str(output_dir.joinpath(f"shapenmi_{i}.txt")),
                    )
                output_dofs.append(output_dir.joinpath(f"shapeaffine_{i}.dof.gz"))

                if output_dir.joinpath(f"shapenmi_{i}.txt").exists():
                    similarities = np.genfromtxt('{0}/shapenmi_{1}.txt'.format(str(output_dir), i), delimiter=",")
                    nmi += [similarities[1, 8]]
                else:
                    nmi += [0]
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise e
                continue

        if n_top < n_atlases:
            sortedIndexes = np.array(nmi).argsort()[::-1]
            for i in range(n_top):
                top_similar_atlases += [self.atlases[sortedIndexes[i]]]
                top_atlas_dofs += [output_dofs[sortedIndexes[i]]]
                top_atlas_landmarks += [self.landmarks[sortedIndexes[i]]]
        else:
            top_similar_atlases = self.atlases
            top_atlas_dofs = output_dofs
            top_atlas_landmarks = self.landmarks

        return top_similar_atlases, top_atlas_dofs, top_atlas_landmarks

    def run(self, subject_image: PhaseImage, subject_seg: Segmentation, subject_landmarks: Path, output_dir: Path,
            n_top: int, force: bool) -> Segmentation:
        output_path = output_dir.joinpath(subject_seg.path.stem + "_refined.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if force or not Segmentation(path=output_path, phase=subject_seg.phase).exists():
            tmp_dir = output_dir.joinpath("tmp", str(subject_seg.phase))
            tmp_dir.mkdir(exist_ok=True, parents=True)

            top_atlases, top_dofs, top_lm = self.select_altases(
                subject_image_path=subject_image.path,
                subject_seg=subject_seg.path,
                subject_landmarks=subject_landmarks,
                output_dir=tmp_dir,
                n_top=n_top,
                force=force,
            )
            atlas_labels = []
            phase = str(subject_seg.phase)
            for i, (atlas, dof) in enumerate(zip(top_atlases, top_dofs)):

                label_path = tmp_dir.joinpath(f"seg_affine_{i}_{phase}.nii.gz")
                if not label_path.exists() or force:
                    mirtk.transform_image(
                        str(atlas),
                        str(label_path),
                        dofin=str(dof),  # Transformation that maps atlas to subject
                        target=str(subject_image),
                        interp="NN",
                    )
                    set_affine(subject_image.path, label_path)

                    # Transform points for debugging
                    mirtk.transform_points(
                        str(top_lm[i]),
                        str(tmp_dir.joinpath(f"lm_affine_{i}_{phase}.vtk")),
                        "-invert",
                        dofin=str(dof),
                    )

                if not tmp_dir.joinpath(f"shapeffd_{i}_{str(phase)}.dof.gz").exists() or force:
                    mirtk.register(
                        str(subject_seg),  # target
                        str(atlas),  # source
                        parin=str(self.param_path),
                        dofin=str(dof),
                        dofout=tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
                    )

                label_path = tmp_dir.joinpath(f"seg_{i}_{phase}.nii.gz")
                if not label_path.exists() or force:
                    mirtk.transform_image(
                        str(atlas),
                        str(label_path),
                        dofin=str(tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")),
                        target=str(subject_image),
                        interp="NN",
                    )
                dof = tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
                # Transform points for debugging
                mirtk.transform_points(
                    str(top_lm[i]),
                    str(tmp_dir.joinpath(f"lm_ffd_{i}_{phase}.vtk")),
                    "-invert",
                    dofin=str(dof),
                )
                atlas_labels.append(label_path)

            # apply label fusion
            labels = sitk.VectorOfImage()

            for label_path in atlas_labels:
                label = sitk.ReadImage(str(label_path), imageIO="NiftiImageIO", outputPixelType=sitk.sitkUInt8)
                labels.push_back(label)
            voter = sitk.LabelVotingImageFilter()
            voter.SetLabelForUndecidedPixels(0)
            fused_label = voter.Execute(labels)
            sitk.WriteImage(
                fused_label, str(output_path), imageIO="NiftiImageIO"
            )

        return Segmentation(
            path=output_path,
            phase=subject_seg.phase
        )
