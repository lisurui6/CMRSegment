import mirtk
import subprocess
from pathlib import Path
import numpy as np
from typing import List
import shutil
from CMRSegment.common.constants import RESOURCE_DIR
mirtk.subprocess.showcmd = True
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from CMRSegment.common.resource import Segmentation, PhaseImage, Phase
from CMRSegment.common.data_table import DataTable
import nibabel as nib


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

    def select_altases(self, subject_seg: Path, subject_landmarks: Path, output_dir: Path, n_top: int, force: bool):
        """Select top similar atlases, according to subject segmentation and landmark"""
        nmi = []

        top_similar_atlases = []

        n_atlases = len(self.atlases)

        # if force:
        #     for f in output_dir.glob("shapenmi*.txt"):
        #         f.unlink()
        output_dofs = []
        top_atlas_dofs = []
        for i in range(n_atlases):
            try:
                if not output_dir.joinpath(f"shapelandmarks_{i}.dof.gz").exists() or force:
                    mirtk.register(
                        str(subject_landmarks),
                        str(self.landmarks[i]),
                        model="Affine",
                        dofout=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                    )
                output_dofs.append(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz"))
                if not output_dir.joinpath(f"shapenmi_{i}.txt").exists() or force:
                    mirtk.evaluate_similarity(
                        str(subject_seg),
                        str(self.atlases[i]),
                        Tbins=64,
                        Sbins=64,
                        dofin=str(output_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                        table=str(output_dir.joinpath(f"shapenmi_{i}.txt")),
                    )

                if output_dir.joinpath(f"shapenmi_{i}.txt").exists():
                    similarities = np.genfromtxt('{0}/shapenmi_{1}.txt'.format(str(output_dir), i), delimiter=",")
                    nmi += [similarities[1, 5]]
                else:
                    nmi += [0]
            except:
                nmi += [0]

        if n_top < n_atlases:
            sortedIndexes = np.array(nmi).argsort()[::-1]
            for i in range(n_top):
                top_similar_atlases += [self.atlases[sortedIndexes[i]]]
                top_atlas_dofs += [output_dofs[sortedIndexes[i]]]
        else:
            top_similar_atlases = self.atlases
            top_atlas_dofs = output_dofs

        return top_similar_atlases, top_atlas_dofs

    def run(self, subject_image: PhaseImage, subject_seg: Segmentation, subject_landmarks: Path, output_dir: Path,
            n_top: int, force: bool) -> Segmentation:
        tmp_dir = output_dir.joinpath("tmp", str(subject_seg.phase))
        tmp_dir.mkdir(exist_ok=True, parents=True)

        top_atlases, top_dofs = self.select_altases(
            subject_seg=subject_seg.path,
            subject_landmarks=subject_landmarks,
            output_dir=tmp_dir,
            n_top=n_top,
            force=force,
        )
        atlas_labels = []
        phase = str(subject_seg.phase)
        for i, (atlas, dof) in enumerate(zip(top_atlases, top_dofs)):
            if not tmp_dir.joinpath(f"shapeffd_{i}_{str(phase)}.dof.gz").exists() or force:
                mirtk.register(
                    str(subject_seg),
                    str(atlas),
                    parin=str(self.param_path),
                    dofin=str(dof),
                    dofout=tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
                )

            label_path = tmp_dir.joinpath(f"seg_affine_{i}_{phase}.nii.gz")
            if not label_path.exists() or force:
                mirtk.transform_image(
                    str(atlas),
                    str(label_path),
                    dofin=str(dof),
                    target=str(subject_image),
                    interp="NN",
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
            nim = nib.load(str(label_path))
            label = nim.get_data()
            label[label == 4] = 3
            nim2 = nib.Nifti1Image(label, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            nib.save(nim2, str(label_path))
            atlas_labels.append(label_path)

        # apply label fusion
        labels = sitk.VectorOfImage()

        for label_path in atlas_labels:
            label = sitk.ReadImage(str(label_path), imageIO="NiftiImageIO", outputPixelType=sitk.sitkUInt8)
            labels.push_back(label)
        voter = sitk.LabelVotingImageFilter()
        voter.SetLabelForUndecidedPixels(0)
        fused_label = voter.Execute(labels)
        output_path = output_dir.joinpath(subject_seg.path.stem + "_refined.nii.gz")
        sitk.WriteImage(
            fused_label, str(output_path), imageIO="NiftiImageIO"
        )

        return Segmentation(
            path=output_path,
            phase=subject_seg.phase
        )
