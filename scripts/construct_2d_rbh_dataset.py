import cv2
import numpy as np
from pathlib import Path
from CMRSegment.common.config import DatasetConfig
from CMRSegment.common.data_table import DataTable
from CMRSegment.common.nn.torch.data import Torch2DSegmentationDataset
from tqdm import tqdm
config = DatasetConfig.from_conf(name="RBH_3D_atlases", mode="3D", mount_prefix=Path("/mnt/storage/home/suruli/"))
# config = DatasetConfig.from_conf(name="RBH_3D_atlases", mode="3D", mount_prefix=Path("D:/surui/rbh/"))


def read_dataframe(dataframe_path: Path):
    data_table = DataTable.from_csv(dataframe_path)
    image_paths = data_table.select_column("image_path")
    label_paths = data_table.select_column("label_path")
    image_paths = [Path(path) for path in image_paths]
    label_paths = [Path(path) for path in label_paths]
    return image_paths, label_paths


image_paths, label_paths = read_dataframe(config.dataframe_path)

images = []
labels = []
output_dir = Path(__file__).parent.joinpath("output")
output_dir.joinpath("image").mkdir(parents=True, exist_ok=True)
output_dir.joinpath("label").mkdir(parents=True, exist_ok=True)
pbar = list(zip(image_paths, label_paths))
for image_path, label_path in tqdm(pbar):
    image = Torch2DSegmentationDataset.read_image(image_path, None, None)
    label = Torch2DSegmentationDataset.read_label(label_path, None, None)
    middle_slice_index = image.shape[0] // 2
    image = image[middle_slice_index, :, :]
    label = label[:, middle_slice_index, :, :]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))
    # cv2.imshow("label", label)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    image = image * 255
    label = label * 255
    cv2.imwrite(str(output_dir.joinpath("image", image_path.parent.name + "_" + image_path.name + ".png")), image)
    cv2.imwrite(str(output_dir.joinpath("label", image_path.parent.name + "_" + image_path.name + ".png")), label)
