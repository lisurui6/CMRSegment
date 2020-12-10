import os
from pathlib import Path
from typing import List
import shutil
import mirtk
from CMRSegment.common.subject import Subject


class DataPreprocessor:
    """Find subjects in data_dir, find ED/ES phases, and split into a sequence of phases"""
    def __init__(self, overwrite: bool = False, use_irtk: bool = False):
        self.use_irtk = use_irtk
        self.overwrite = overwrite

    def run(self, data_dir: Path, output_dir: Path) -> List[Subject]:
        subjects = []
        for subject_dir in sorted(os.listdir(str(data_dir))):
            subject = Subject(dir=data_dir.joinpath(subject_dir), output_dir=output_dir.joinpath(subject_dir))
            if self.overwrite:
                subject.clean()
            shutil.copy(str(subject.dir.joinpath("lvsa_ED.nii.gz")), str(subject.output_dir))
            shutil.copy(str(subject.dir.joinpath("lvsa_ES.nii.gz")), str(subject.output_dir))
            print(subject_dir)
            if not subject.ed_path.exists() or not subject.es_path.exists() or not subject.contrasted_nii_path:
                print(' Detecting ED/ES phases {}...'.format(subject.nii_path))
                if not self.use_irtk:
                    mirtk.auto_contrast(str(subject.nii_path), str(subject.contrasted_nii_path))
                    mirtk.detect_cardiac_phases(
                        str(subject.contrasted_nii_path), output_ed=str(subject.ed_path), output_es=str(subject.es_path)
                    )
                else:
                    os.system('autocontrast '
                              f'{str(subject.nii_path)} '
                              f'{str(subject.contrasted_nii_path)} >/dev/nul ')

                    os.system('cardiacphasedetection '
                              f'{str(subject.contrasted_nii_path)} '
                              f'{str(subject.ed_path)} '
                              f'{str(subject.es_path)} >/dev/nul ')
                print('  Found ED/ES phases ...')

            if not subject.ed_path.exists() or not subject.es_path.exists():
                print(" ED {0} or ES {1} does not exist. Skip.".format(subject.ed_path, subject.es_path))
                continue
            subjects.append(subject)
            if self.overwrite or len(subject.gray_phases) == 0:
                print("\n ... Split sequence")
                if not self.use_irtk:
                    mirtk.split_volume(
                        str(subject.contrasted_nii_path), "{}/lvsa_".format(str(subject.gray_phases_dir())), "-sequence"
                    )
                else:
                    os.system(f'splitvolume {str(subject.contrasted_nii_path)} '
                              f'{str(subject.gray_phases_dir())}/lvsa_ -sequence')

        return subjects

    # def parallel_run(self, data_dir: Path, n_core: int) -> List[Subject]:
    #     subjects = self.find_subjects(data_dir)
    #     pool1 = Pool(processes=n_core)
    #     # multiprocessing preprocessing
    #     pool1.map(partial(self.apply), subjects)
    #     pool1.close()
    #     pool1.join()
    #     return subjects