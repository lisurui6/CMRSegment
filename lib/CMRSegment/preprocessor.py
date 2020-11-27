import os
from pathlib import Path

import mirtk
from tqdm import tqdm
from CMRSegment.common.subject import Subject
from multiprocessing import Pool
from functools import partial
from typing import List


class DataPreprocessor:
    def __init__(self, overwrite: bool = False):
        self.overwrite = overwrite

    def find_subjects(self, data_dir: Path):
        subjects = []
        for subject_dir in sorted(os.listdir(str(data_dir))):
            subject_dir = data_dir.joinpath(subject_dir)
            subject = Subject(dir=subject_dir)
            if self.overwrite:
                subject.clean()
            if not subject.ed_path.exists() or not subject.es_path.exists() or not subject.contrasted_nii_path:
                print(' Detecting ED/ES phases {}...'.format(subject.nii_path))
                mirtk.auto_contrast(str(subject.nii_path), str(subject.contrasted_nii_path))
                mirtk.detect_cardiac_phases(
                    str(subject.contrasted_nii_path), output_ed=str(subject.ed_path), output_es=str(subject.es_path)
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
        if self.overwrite or len(subject.gray_phases) == 0:
            print("\n ... Split sequence")
            mirtk.split_volume(
                str(subject.contrasted_nii_path), "{}/lvsa_".format(str(subject.gray_phases_dir())), "-sequence"
            )
        if self.overwrite or len(subject.resample_phases) == 0:
            for gray_phase_path in tqdm(subject.gray_phases):
                mirtk.resample_image(
                    str(gray_phase_path),
                    str(subject.resample_phases_dir().joinpath(gray_phase_path.name)),
                    '-size', 1.25, 1.25, 2,
                )

        if self.overwrite or len(subject.enlarge_phases) == 0:
            print("\n ... Enlarge preprocessing generation")
            for resample_phase_path in tqdm(subject.resample_phases):
                mirtk.enlarge_image(
                    str(resample_phase_path),
                    str(subject.enlarge_phases_dir().joinpath(resample_phase_path.name)),
                    z=20, value=0,
                )
        print('  finish cine preprocessing in subject {0}'.format(subject.name))

    def run(self, data_dir: Path) -> List[Subject]:
        subjects = self.find_subjects(data_dir)
        for subject in subjects:
            self.apply(subject)
        return subjects

    def parallel_run(self, data_dir: Path, n_core: int) -> List[Subject]:
        subjects = self.find_subjects(data_dir)
        pool1 = Pool(processes=n_core)
        # multiprocessing preprocessing
        pool1.map(partial(self.apply), subjects)
        pool1.close()
        pool1.join()
        return subjects
