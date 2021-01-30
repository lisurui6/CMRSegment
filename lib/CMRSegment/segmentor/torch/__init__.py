import torch
import numpy as np
import nibabel as nib
from typing import Tuple
from pathlib import Path
from CMRSegment.segmentor import Segmentor
from CMRSegment.common.nn.torch.data import resize_image, rescale_intensity
from CMRSegment.common.nn.torch import prepare_tensors


class TorchSegmentor(Segmentor):
    def __init__(self, model_path: Path, overwrite: bool = False, resize_size: Tuple = None, device: int = 0,
                 use_irtk: bool = False):
        super().__init__(model_path, overwrite)
        self.model = torch.load(str(model_path))
        self.model.eval()
        if resize_size is None:
            resize_size = (128, 128, 64)
        self.resize_size = resize_size
        self.device = device
        self.use_irtk = use_irtk

    def run(self, image: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image, 0)
        image = prepare_tensors(image, True, self.device)
        predicted = self.model(image)
        predicted = torch.sigmoid(predicted)
        # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = (predicted > 0.5).float()
        # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = predicted.cpu().detach().numpy()
        predicted = np.squeeze(predicted, axis=0)
        # map back to original size
        final_predicted = np.zeros((image.shape[2], image.shape[3], image.shape[4]))
        # print(predicted.shape, final_predicted.shape)

        for i in range(predicted.shape[0]):
            final_predicted[predicted[i, :, :, :] > 0.5] = i + 1
        # image = nim.get_data()
        final_predicted = np.transpose(final_predicted, [1, 2, 0])
        return final_predicted

    def execute(self, phase_path: Path, output_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        nim = nib.load(str(phase_path))
        image = nim.get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        # resized_image = resize_image(image, (self.resize_size[0], self.resize_size[1], self.resize_size[2]), 0)
        image = np.transpose(image, (2, 0, 1))
        print("Segmenting image with shape: {}".format(image.shape))
        image = rescale_intensity(image, (1.0, 99.0))
        image = np.expand_dims(image, 0)
        predicted = self.run(image)
        # predicted = resize_image(predicted, image.shape, 0)
        nim2 = nib.Nifti1Image(predicted, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(output_path))
        return image, predicted
