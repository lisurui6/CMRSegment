import mirtk
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple

from CMRSegment.common.subject import Image, Segmentation, Cine


class Segmentor:
    def __init__(self, model_path: Path, overwrite: bool = False, use_irtk: bool = False):
        self.model_path = model_path
        self.overwrite = overwrite
        self.use_irtk = use_irtk

    def run(self, image: np.ndarray) -> np.ndarray:
        """Call sess.run()"""
        raise NotImplementedError("Must be implemented by subclasses.")

    def apply(self, image: Image) -> Segmentation:
        if self.overwrite or not image.resampled.exists():
            if not self.use_irtk:
                mirtk.resample_image(str(image.path), str(image.resampled), '-size', 1.25, 1.25, 2)
            else:
                command = 'resample ' \
                    f'{str(image.path)} ' \
                    f'{str(image.resampled)} ' \
                    '-size 1.25 1.25 2'
                print(command)
                subprocess.call(command, shell=True)
        if self.overwrite or not image.enlarged.exists():
            if not self.use_irtk:
                mirtk.enlarge_image(str(image.resampled), str(image.enlarged), z=20, value=0)
            else:
                command = 'enlarge_image ' \
                    f'{str(image.resampled)} ' \
                    f'{str(image.enlarged)} ' \
                    '-z 20 -value 0'
                print(command)
                subprocess.call(command, shell=True)
        nim = nib.load(str(image.path))
        data = nim.get_data()
        print("Original image shape: {}".format(data.shape))
        nim = nib.load(str(image.resampled))
        data = nim.get_data()
        print("Resampled image shape: {}".format(data.shape))
        nim = nib.load(str(image.enlarged))
        data = nim.get_data()
        print("Enlarged image shape: {}".format(data.shape))

        np_image, predicted = self.execute(image.enlarged, image.segmented)
        return Segmentation(phase=image.phase, path=image.enlarged, image=np_image, predicted=predicted)

    def execute(self, phase_path: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Segment a 3D volume cardiac phase from phase_path, save to output_dir"""
        raise NotImplementedError("Must be implemented by subclasses.")


class CineSegmentor:
    def __init__(self, phase_segmentor: Segmentor):
        """ Cine CMR consists in the acquisition of the same slice position at different phases of the cardiac cycle."""
        self.__segmentor = phase_segmentor

    def apply(self, cine: Cine, output_dir: Path) -> List[Segmentation]:
        segmentations = []
        for idx, phase_path in enumerate(cine):
            image = Image(
                phase=idx, path=phase_path, output_dir=output_dir.joinpath("segs"),
                segmented=output_dir.joinpath("segs").joinpath(f"lvsa_{idx}.nii.gz"),
                resampled=output_dir.joinpath("resampled").joinpath(f"lvsa_{idx}.nii.gz"),
                enlarged=output_dir.joinpath("enlarged").joinpath(f"lvsa_{idx}.nii.gz"),
            )
            segmentation = self.__segmentor.apply(image)
            segmentations.append(segmentation)
        nim = nib.load(str(segmentations[-1].path))
        # batch * height * width * channels (=slices)
        segt_labels = np.array([seg.predicted for seg in segmentations], dtype=np.int32)
        segt_labels = np.transpose(segt_labels, (1, 2, 3, 0))
        images = np.array([seg.image for seg in segmentations], dtype=np.float32)  # b
        images = np.transpose(images, (1, 2, 3, 0))
        nim2 = nib.Nifti1Image(segt_labels, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.joinpath("4D_rview").mkdir(exist_ok=True, parents=True)
        nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dseg.nii.gz")))
        nim2 = nib.Nifti1Image(images, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dimg.nii.gz")))
        return segmentations
