import os
import glob
from pathlib import Path
from CMRSegment.common.constants import ROOT_DIR
import shutil
import subprocess
import nibabel as nib
from CMRSegment.common.plot import plot_nii_gz

import sys
import mirtk
from typing import Tuple
from tqdm import tqdm
from CMRSegment.subject import Subject
from multiprocessing import Pool
from functools import partial
from typing import List


mirtk.subprocess.showcmd = True   # whether to print executed commands with arguments


class DataPreprocessor:
    def __init__(self, force_restart: bool = False):
        self.force_restart = force_restart

    def find_subjects(self, data_dir: Path):
        subjects = []
        for subject_dir in sorted(os.listdir(str(data_dir))):
            subject_dir = data_dir.joinpath(subject_dir)

            nii_path = list(subject_dir.glob("*.nii"))
            if not nii_path:
                print("  original nifit image does not exist, use lvsa.nii.gz")
                nii_path = list(subject_dir.glob("*.nii.gz"))
            nii_path = nii_path[0]
            print("nii path: {}".format(nii_path))
            subject = Subject(dir=subject_dir, nii_name=nii_path.name)
            if self.force_restart:
                subject.clean()
            if not subject.tmp_nii_path.exists():
                shutil.copy(str(subject.nii_path), str(subject.tmp_nii_path))
            if not subject.ed_path.exists() or not subject.es_path.exists():
                print(' Detecting ED/ES phases {}...'.format(subject.nii_path))
                mirtk.auto_contrast(str(subject.tmp_nii_path), str(subject.tmp_nii_path))
                mirtk.detect_cardiac_phases(
                    str(subject.tmp_nii_path), output_ed=str(subject.ed_path), output_es=str(subject.es_path)
                )
                print('  Found ED/ES phases ...')

            if not subject.ed_path.exists() or not subject.es_path.exists():
                print(" ED {0} or ES {1} does not exist. Skip.".format(subject.ed_path, subject.es_path))
                continue
            subjects.append(subject)
        return subjects

    def apply(self, subject: Subject):
        print('  co-registering {0}'.format(subject.name))
        if not subject.dir.is_dir():
            print('  {0} is not a valid directory, do nothing'.format(subject.name))
            return
        print("\n ... Split sequence")

        if self.force_restart or len(list(subject.gray_phases_dir().glob("lvsa_*"))) == 0:
            mirtk.split_volume(
                str(subject.tmp_nii_path), "{}/lvsa_".format(str(subject.gray_phases_dir())), "-sequence"
            )
        if self.force_restart or len(list(subject.resample_phases_dir().glob("lvsa_*"))) == 0:
            for fr in tqdm(range(len(os.listdir(str(subject.gray_phases_dir()))))):
                mirtk.resample_image(
                    '{}/lvsa_{}.nii.gz'.format(str(subject.gray_phases_dir()), "{0:0=2d}".format(fr)),
                    '{}/lvsa_{}.nii.gz'.format(str(subject.resample_phases_dir()), "{0:0=2d}".format(fr)),
                    '-size', 1.25, 1.25, 2,
                )

        print("\n ... Enlarge preprocessing generation")
        if self.force_restart or len(list(subject.enlarge_phases_dir().glob("lvsa_*"))) == 0:
            for fr in tqdm(range(len(os.listdir(str(subject.gray_phases_dir()))))):
                mirtk.enlarge_image(
                    '{}/lvsa_{}.nii.gz'.format((str(subject.resample_phases_dir())), "{0:0=2d}".format(fr)),
                    '{}/lvsa_SR_{}.nii.gz'.format(str(subject.enlarge_phases_dir()), "{0:0=2d}".format(fr)),
                    z=20, value=0,
                )
        print('  finish cine preprocessing in subject {0}'.format(subject.name))

    def run(self, data_dir: Path) -> List[Subject]:
        subjects = self.find_subjects(data_dir)
        for subject in subjects:
            self.apply(subject)
        return subjects

    def parallel_run(self,data_dir: Path, n_core: int) -> List[Subject]:
        subjects = self.find_subjects(data_dir)
        pool1 = Pool(processes=n_core)
        # multiprocessing preprocessing
        pool1.map(partial(self.apply), subjects)
        pool1.close()
        pool1.join()
        return subjects


# subjects = preprocessing(ROOT_DIR.joinpath("data", "genscan"))
# subjects = preprocessing(ROOT_DIR.joinpath("data", "sheffield"))
preprocessor = DataPreprocessor(force_restart=False)

for center in ["singapore_hcm", "singapore_lvsa", "ukbb"]:
    subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", center))

