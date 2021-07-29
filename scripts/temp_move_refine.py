from pathlib import Path
import os
from argparse import ArgumentParser
import shutil


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", dest="input_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.is_dir()
    for subdir in os.listdir(str(input_dir)):
        subdir = input_dir.joinpath(subdir)
        ed_path = subdir.joinpath("refine", "seg_lvsa_SR_ED.nii._refined.nii.gz")
        es_path = subdir.joinpath("refine", "seg_lvsa_SR_ES.nii._refined.nii.gz")
        shutil.copy(str(ed_path), str(subdir.joinpath("seg_lvsa_SR_ED.nii._refined.nii.gz")))
        shutil.copy(str(es_path), str(subdir.joinpath("seg_lvsa_SR_ES.nii._refined.nii.gz")))


if __name__ == '__main__':
    main()
