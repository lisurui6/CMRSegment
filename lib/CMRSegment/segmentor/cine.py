import os
import math
from pathlib import Path
import nibabel as nib
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label

from typing import List
from CMRSegment.subject import Subject
from CMRSegment.utils import rescale_intensity
import mirtk
from skimage.exposure import match_histograms
from CMRSegment.segmentor import TF1Segmentor


mirtk.subprocess.showcmd = False


class TF1CineSegmentor(TF1Segmentor):
    def run(self, image: np.ndarray, training: bool = False) -> np.ndarray:
        # Intensity rescaling
        image = rescale_intensity(image, (1, 99))
        # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
        # in the network will result in the same image size at each resolution level.
        X, Y, Z = image.shape
        n_slices = 100
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
        x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
        z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        image = image[:, :, z1_: z2_]
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_ - z1, z2 - z2_)), 'constant')
        # extend batch for network requirement
        image = np.expand_dims(image, axis=0)
        pred_segt = self._sess.run(['pred_segt:0'], feed_dict={'image:0': image, 'training:0': training})
        # Transpose and crop the segmentation to recover the original size
        pred_segt = np.squeeze(pred_segt[0], axis=0).astype(np.int16)
        pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
        return pred_segt

    def execute(self, phase_path: Path, output_dir: Path):
        nim = nib.load(str(phase_path))
        image = nim.get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        pred_segt = self.run(image)
        pred_segt = refined_mask(pred_segt, phase_path, output_dir.joinpath("tmp"))
        #########################################################################
        # map back to original size
        nim2 = nib.Nifti1Image(pred_segt, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(output_dir.joinpath(phase_path.name)))
        return image, pred_segt

    def apply(self, subject: Subject):
        images, segt_labels = [], []
        for phase_name in tqdm(os.listdir(subject.enlarge_phases_dir())):
            phase_path = subject.enlarge_phases_dir().joinpath(phase_name)
            image, pred_segt = self.execute(phase_path, subject.motions_dir())
            segt_labels += [pred_segt]
            images += [image]
        nim = nib.load(str(phase_path))
        segt_labels = np.array(segt_labels, dtype=np.int32)  # batch * height * width * channels (=slices)
        segt_labels = np.transpose(segt_labels, (1, 2, 3, 0))
        images = np.array(images, dtype=np.float32)  # b
        images = np.transpose(images, (1, 2, 3, 0))
        nim2 = nib.Nifti1Image(segt_labels, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(subject.rview_dir().joinpath("4Dseg.nii.gz")))
        nim2 = nib.Nifti1Image(images, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(subject.rview_dir().joinpath("4Dimg.nii.gz")))


def get_labels(seg, label):
    mask = np.zeros(seg.shape, dtype=np.uint8)
    mask[seg[:, :, :] == label] = 1
    return mask


def refined_mask(pred_segt: np.ndarray, phase_path: Path, tmp_dir: Path):
    nim = nib.load(str(phase_path))
    ###########################################################################
    nim2 = nib.Nifti1Image(pred_segt, nim.affine)
    print(nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, str(tmp_dir.joinpath("seg_{}".format(phase_path.name))))
    ###########################################################################
    nim = nib.load(str(tmp_dir.joinpath("seg_{}".format(phase_path.name))))
    lvsa_data = nim.get_data()
    lvsa_data_bin = np.where(lvsa_data > 0, 1, lvsa_data)
    labelled_mask, num_labels = label(lvsa_data_bin)
    refined_mask = lvsa_data.copy()
    minimum_cc_sum = 5000
    for labl in range(num_labels + 1):
        if np.sum(refined_mask[labelled_mask == labl]) < minimum_cc_sum:
            refined_mask[labelled_mask == labl] = 0
    final_mask = np.zeros(refined_mask.shape, dtype=np.uint8)
    lv = get_labels(refined_mask, 1)
    myo = get_labels(refined_mask, 2)
    rv = get_labels(refined_mask, 3)
    kernel = np.ones((5, 5), np.uint8)
    final_mask[lv[:, :, :] == 1] = 1
    final_mask[myo[:, :, :] == 1] = 2
    final_mask[rv[:, :, :] == 1] = 3
    nim2 = nib.Nifti1Image(final_mask[:, :, :], affine=np.eye(4))
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, str(tmp_dir.joinpath("seg_{}".format(phase_path.name))))
    mirtk.header_tool(
        str(tmp_dir.joinpath("seg_{}".format(phase_path.name))),
        str(tmp_dir.joinpath("seg_{}".format(phase_path.name))),
        target=str(phase_path),
    )
    nim = nib.load(str(tmp_dir.joinpath("seg_{}".format(phase_path.name))))
    image = nim.get_data()
    image = np.squeeze(image, axis=-1)
    return image


from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.common.constants import ROOT_DIR, MODEL_DIR
preprocessor = DataPreprocessor(force_restart=False)
model_path = MODEL_DIR.joinpath("3D", "biobank_low2high.ckpt-300")
reference_center = "genscan"
subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", reference_center))
reference_subject = subjects[0]
reference_ed_nim = nib.load(str(reference_subject.ed_path))
reference_ed_image = reference_ed_nim.get_data()
# center = "sheffield"
# , "ukbb", "sheffield"
# "singapore_hcm", "singapore_lvsa", "sheffield",
for center in ["ukbb"]:
    subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", center))
    for subject in subjects:
        # get_segmentation(model_path, subject, reference_ed_image)
        with TF1CineSegmentor(model_path=model_path) as segmentor:
            segmentor.apply(subject)
