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


class SegmentationCorrector:
    """Use multi-altas registration to correct predicted segmentation"""
    def __init__(self, atlas_dir: Path, param_path: Path = None):
        assert atlas_dir.is_dir(), "Atlas dir needs to a directory"
        atlases = []
        landmarks = []
        for subject in os.listdir(str(atlas_dir)):
            subject_dir = atlas_dir.joinpath(subject)
            atlas = subject_dir.joinpath("seg_lvsa_SR_ED.nii.gz")
            landmark = subject_dir.joinpath("landmarks2.vtk")
            atlases.append(atlas)
            landmarks.append(landmark)
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

        if force:
            for f in output_dir.glob("shapenmi*.txt"):
                f.unlink()
        output_dofs = []
        top_atlas_dofs = []
        for i in range(n_atlases):
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
        tmp_dir = output_dir.joinpath("tmp")
        top_atlases, top_dofs = self.select_altases(
            subject_seg=subject_seg.path,
            subject_landmarks=subject_landmarks,
            output_dir=tmp_dir,
            n_top=n_top,
            force=force,
        )
        tmp_dir.mkdir(exist_ok=True, parents=True)
        atlas_labels = []
        phase = str(subject_seg.phase)
        for i, atlas, dof in enumerate(zip(top_atlases, top_dofs)):
            if not tmp_dir.joinpath(f"shapeffd_{i}_{str(phase)}.dof.gz").exists() or force:
                mirtk.register(
                    str(subject_seg),
                    str(atlas),
                    parin=str(self.param_path),
                    dofin=str(dof),
                    dofout=tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
                )
            if not tmp_dir.joinpath(f"seg_{i}_{phase}.nii.gz").exists() or force:
                mirtk.transform_image(
                    str(atlas),
                    str(tmp_dir.joinpath(f"seg_{i}_{phase}.nii.gz")),
                    dofin=str(tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")),
                    target=str(subject_image),
                    interp="NN",
                )

            atlas_labels.append(tmp_dir.joinpath(f"seg_{i}_{phase}.nii.gz"))

        # apply label fusion
        labels = []
        for label_path in atlas_labels:
            label = sitk.ReadImage(str(label_path), imageIO="NiftiImageIO", outputPixelType=sitk.sitkUInt8)
            labels.append(label)
        voter = sitk.LabelVotingImageFilter()
        voter.SetLabelForUndecidedPixels(0)
        fused_label = voter.Execute(*labels)
        output_path = output_dir.joinpath(subject_seg.path.name + "_fused.nii.gz")
        sitk.WriteImage(
            fused_label, str(output_path), imageIO="NiftiImageIO"
        )

        return Segmentation(
            path=output_path,
            phase=subject_seg.phase
        )


def output_refinement(atlases, DLSeg, param_dir, tmps_dir, dofs_dir, subject_dir, saved_ind, fr, use_mirtk, force):
    atlasUsedNo = len(atlases)
    atlas_labels = []
    for i in range(atlasUsedNo):

        if not dofs_dir.joinpath(f"shapeffd_{i}_{fr}.dof.gz").exists() or force:
            mirtk.register(
                str(DLSeg),
                str(atlases[i]),
                parin=str(param_dir.joinpath("ffd_label_1.cfg")),
                dofin=str(dofs_dir.joinpath(f"shapelandmarks_{saved_ind[i]}.dof.gz")),
                dofout=dofs_dir.joinpath(f"shapeffd_{i}_{fr}.dof.gz")
            )
        if not tmps_dir.joinpath(f"seg_lvsa_SR_{i}_{fr}.nii.gz").exists() or force:
            mirtk.transform_image(
                str(atlases[i]),
                str(tmps_dir.joinpath(f"seg_lvsa_SR_{i}_{fr}.nii.gz")),
                dofin=str(dofs_dir.joinpath(f"shapeffd_{i}_{fr}.dof.gz")),
                target=str(subject_dir.joinpath(f"lvsa_SR_{fr}.nii.gz")),
                interp="NN",
            )

        atlas_labels.append(tmps_dir.joinpath(f"seg_lvsa_SR_{i}_{fr}.nii.gz"))

    # apply label fusion
    labels = []
    for label_path in atlas_labels:
        label = sitk.ReadImage(str(label_path), imageIO="NiftiImageIO", outputPixelType=sitk.sitkUInt8)
        labels.append(label)
    voter = sitk.LabelVotingImageFilter()
    voter.SetLabelForUndecidedPixels(0)
    fused_label = voter.Execute(*labels)
    sitk.WriteImage(fused_label, str(subject_dir.joinpath(f"seg_lvsa_SR_{fr}_fused.nii.gz")), imageIO="NiftiImageIO")


def select_all_atlas(dataset_dir: Path):
    atlases_list, landmarks_list = {}, {}

    for fr in ['ED', 'ES']:

        atlases_list[fr], landmarks_list[fr] = [], []

        for atlas in sorted(os.listdir(str(dataset_dir))):

            atlas_dir = dataset_dir.joinpath(atlas)

            if not atlas_dir.is_dir():
                print('  {0} is not a valid atlas directory, Discard'.format(str(atlas_dir)))
                continue

            if atlas_dir.joinpath(f"seg_lvsa_SR_{fr}.nii.gz").exists():
                atlas_3D_shape = atlas_dir.joinpath(f"seg_lvsa_SR_{fr}.nii.gz")
            else:
                atlas_3D_shape = atlas_dir.joinpath(f"seg_lvsa_SR_{fr}.gipl")

            landmarks = atlas_dir.joinpath("landmarks2.vtk")

            if atlas_3D_shape.exists() and landmarks.exists():
                atlases_list[fr] += [atlas_3D_shape]

                landmarks_list[fr] += [landmarks]

    return atlases_list, landmarks_list


def select_top_similar_atlas(
        atlases: List[Path],
        atlas_landmarks: List[Path],
        subject_landmarks: Path,
        tmps_dir: Path,
        dofs_dir: Path,
        DLSeg: Path,
        n_top_atlases: int,
        force: bool,
):

    nmi = []

    top_similar_atlases = []

    n_atlases = len(atlases)

    # for f in tmps_dir.glob("shapenmi*.txt"):
    #     f.unlink()

    for i in range(n_atlases):
        if not dofs_dir.joinpath(f"shapelandmarks_{i}.dof.gz").exists() or force:
            mirtk.register(
                str(subject_landmarks),
                str(atlas_landmarks[i]),
                model="Affine",
                dofout=str(dofs_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
            )
        if not tmps_dir.joinpath(f"shapenmi_{i}.txt").exists() or force:
            mirtk.evaluate_similarity(
                str(DLSeg),
                str(atlases[i]),
                Tbins=64,
                Sbins=64,
                dofin=str(dofs_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
                table=str(tmps_dir.joinpath(f"shapenmi_{i}.txt")),
            )
        # os.system('cardiacimageevaluation '
        #           '{0} '
        #           '{1} '
        #           '-nbins_x 64 '
        #           '-nbins_y 64 '
        #           '-dofin {2}/shapelandmarks_{4}.dof.gz '
        #           '-output {3}/shapenmi_{4}.txt'
        #           .format(DLSeg, atlases[i], dofs_dir, tmps_dir, i))

        if tmps_dir.joinpath(f"shapenmi_{i}.txt").exists():
            similarities = np.genfromtxt('{0}/shapenmi_{1}.txt'.format(str(tmps_dir), i), delimiter=",")
            nmi += [similarities[1, 5]]
        else:
            nmi += [0]

    if n_top_atlases < n_atlases:

        sortedIndexes = np.array(nmi).argsort()[::-1]

        savedInd = np.zeros(n_top_atlases, dtype=int)

        for i in range(n_top_atlases):
            top_similar_atlases += [atlases[sortedIndexes[i]]]

            savedInd[i] = sortedIndexes[i]
    else:
        top_similar_atlases = atlases
        savedInd = np.arange(n_atlases)

    return top_similar_atlases, savedInd


def main(atlas_dir: Path, data_dir: Path, param_dir: Path, n_similar_atlases: int, force: bool):
    print('Select all the shape atlases for 3D multi-atlas registration')

    atlases_list, landmarks_list = select_all_atlas(atlas_dir)

    for subject in os.listdir(str(data_dir)):
        print('  registering {0}'.format(subject))

        subject_dir = data_dir.joinpath(subject)

        if subject_dir.is_dir():

            tmps_dir = subject_dir.joinpath("atlas", "tmps")
            tmps_dir.mkdir(exist_ok=True, parents=True)

            dofs_dir = subject_dir.joinpath("atlas", "dofs")
            dofs_dir.mkdir(exist_ok=True, parents=True)

            segs_dir = subject_dir.joinpath("segs")

            sizes_dir = subject_dir.joinpath("sizes")

            subject_landmarks = subject_dir.joinpath("landmark.vtk")

            for fr in ['ED', 'ES']:

                DLSeg = subject_dir.joinpath(f"seg_lvsa_SR_{fr}.nii.gz")

                if not DLSeg.exists():
                    print(' segmentation {0} does not exist. Skip.'.format(DLSeg))

                    continue

                top_similar_atlases, saved_ind = select_top_similar_atlas(
                    atlases_list[fr], landmarks_list[fr], subject_landmarks, tmps_dir, dofs_dir, DLSeg, n_similar_atlases, force
                )

                output_refinement(top_similar_atlases, DLSeg, param_dir, tmps_dir, dofs_dir, subject_dir, saved_ind, fr, mirtk, force)

            print('  finish 3D nonrigid-registering one subject {}'.format(subject))
        else:
            print('  {0} is not a valid directory, do nothing'.format(subject_dir))


main(
    atlas_dir=Path(__file__).parent.parent.parent.parent.joinpath("atlas"),
    data_dir=Path(__file__).parent.parent.parent.parent.joinpath("subject"),
    param_dir=RESOURCE_DIR,
    n_similar_atlases=3,
    force=True
)
