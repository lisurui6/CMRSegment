import mirtk
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from CMRSegment.common.resource import PhaseImage, Segmentation, CineImages


class Segmentor:
    def __init__(self, model_path: Path, overwrite: bool = False, use_irtk: bool = False):
        self.model_path = model_path
        self.overwrite = overwrite
        self.use_irtk = use_irtk

    def run(self, image: np.ndarray) -> np.ndarray:
        """Call sess.run()"""
        raise NotImplementedError("Must be implemented by subclasses.")

    def apply(self, image: PhaseImage, output_path: Path) -> Segmentation:
        np_image, predicted = self.execute(image.path, output_path)
        return Segmentation(phase=image.phase, path=output_path)

    def execute(self, phase_path: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Segment a 3D volume cardiac phase from phase_path, save to output_dir"""
        raise NotImplementedError("Must be implemented by subclasses.")


class CineSegmentor:
    def __init__(self, phase_segmentor: Segmentor):
        """ Cine CMR consists in the acquisition of the same slice position at different phases of the cardiac cycle."""
        self.__segmentor = phase_segmentor

    def apply(self, cine: CineImages, output_dir: Path) -> List[Segmentation]:
        segmentations = []
        output_dir.joinpath("segs").mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(tqdm(cine)):
            segmentation = self.__segmentor.apply(
                image, output_path=output_dir.joinpath("segs").joinpath(f"lvsa_{idx}.nii.gz")
            )
            segmentations.append(segmentation)
        nim = nib.load(str(segmentations[-1].path))
        # batch * height * width * channels (=slices)
        segt_labels = np.array([seg.get_data() for seg in segmentations], dtype=np.int32)
        segt_labels = np.transpose(segt_labels, (1, 2, 3, 0))
        images = np.array([np.squeeze(image.get_data(), axis=3) for image in cine], dtype=np.float32)  # b
        images = np.transpose(images, (1, 2, 3, 0))
        nim2 = nib.Nifti1Image(segt_labels, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.joinpath("4D_rview").mkdir(exist_ok=True, parents=True)
        nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dseg.nii.gz")))
        nim2 = nib.Nifti1Image(images, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dimg.nii.gz")))
        return segmentations


class AtlasRefiner:
    def __init__(self, use_mirtk: bool = True):
        self.use_mirtk = use_mirtk
        pass

    def run(self):
        segstring = ''
        ind = 0
        atlasUsedNo = len(atlases)

        for i in range(atlasUsedNo):

            if self.use_mirtk:
                mirtk.register(
                    DLSeg,
                    atlases[i],
                    parin="{}/ffd_label_1.cfg".format(param_dir),
                    dofin="{}/shapelandmarks_{}.dof.gz".format(dofs_dir, savedInd[i]),
                    dofout="{}/shapeffd_{}_{}.dof.gz".format(dofs_dir, i, fr)
                )
                mirtk.transform_image(
                    atlases[i],
                    "{}/seg_lvsa_SR_{}_{}.nii.gz".format(tmps_dir, i, fr),
                    dofin="{}/shapeffd_{}_{}.dof.gz".format(dofs_dir, i, fr),
                    target="{}/lvsa_SR_{}.nii.gz".format(subject_dir, fr),
                    interp="NN",
                )
            else:
                command = 'nreg ' \
                          f'{DLSeg} ' \
                          f'{atlases[i]} ' \
                          f'-parin {param_dir}/segreg.txt ' \
                          f'-dofin {dofs_dir}/shapelandmarks_{savedInd[i]}.dof.gz ' \
                          f'-dofout {dofs_dir}/shapeffd_{i}_{fr}.dof.gz'
                print(command)
                subprocess.call(command, shell=True)
                command = 'transformation ' \
                          f'{atlases[i]} ' \
                          f'{tmps_dir}/seg_lvsa_SR_{i}_{fr}.nii.gz ' \
                          f'-dofin {dofs_dir}/shapeffd_{i}_{fr}.dof.gz ' \
                          f'-target {subject_dir}/lvsa_SR_{fr}.nii.gz -nn'
                print(command)
                subprocess.call(command, shell=True)

            segstring += '{0}/seg_lvsa_SR_{1}_{2}.nii.gz '.format(tmps_dir, i, fr)

            ind += 1

        # apply label fusion
        command = 'combineLabels {0}/seg_lvsa_SR_{1}.nii.gz {2} {3}'.format(subject_dir, fr, ind, segstring)
        subprocess.call(command, shell=True)


import os


def select_all_atlas(dataset_dir: Path):
    atlases_list, landmarks_list = {}, {}

    for fr in ['ED', 'ES']:

        atlases_list[fr], landmarks_list[fr] = [], []

        for atlas in sorted(os.listdir(str(dataset_dir))):

            atlas_dir = dataset_dir.joinpath(atlas)

            if not atlas_dir.is_dir():
                print('  {0} is not a valid atlas directory, Discard'.format(str(atlas_dir)))
                continue

            if atlas_dir.joinpath(f"segmentation_{fr}.nii.gz").exists():
                atlas_3D_shape = atlas_dir.joinpath(f"segmentation_{fr}.nii.gz")
            else:
                atlas_3D_shape = atlas_dir.joinpath(f"segmentation_{fr}.gipl")

            landmarks = atlas_dir.joinpath("landmarks2.vtk")

            if atlas_3D_shape.exists() and landmarks.exists():
                atlases_list[fr] += [atlas_3D_shape]

                landmarks_list[fr] += [landmarks]

    return atlases_list, landmarks_list

from typing import List
import shutil


def select_top_similar_atlas(
        atlases: List[Path],
        atlas_landmarks: List[Path],
        subject_landmarks: Path,
        tmps_dir: Path,
        dofs_dir: Path,
        DLSeg: Path,
        param_dir: Path,
        n_top_atlases: int
):
    landmarks = True

    nmi = []

    top_similar_atlases = []

    n_atlases = len(atlases)

    for f in tmps_dir.glob("shapenmi*.txt"):
        f.unlink()

    for i in range(n_atlases):

        mirtk.register(
            str(subject_landmarks),
            str(atlas_landmarks[i]),
            model="Affine",
            dofout=str(dofs_dir.joinpath(f"shapelandmarks_{i}.dof.gz")),
        )
        mirtk.evaluate_similarity(
            str(DLSeg),
            str(atlases[i]),
            Tbins=64,
            Sbins=64,
            dofin=str(dofs_dir.joinpath("shapelandmarks_{}.dof.gz")),
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
            similarities = np.genfromtxt('{0}/shapenmi_{1}.txt'.format(str(tmps_dir), i))
            nmi += [similarities[3]]
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

