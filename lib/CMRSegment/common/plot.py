from pathlib import Path
from vedo import Plotter


def plot_nii_gz(file_path: Path):
    vp = Plotter()
    image = vp.load(str(file_path))
    vp.show(image)
    vp.close()
