from CMRSegment.common.nn.torch.experiment import Experiment
import numpy as np
import torch


def read_image(dataset, idx):
    import math
    image = dataset.read_image(dataset.image_paths[idx], dataset.feature_size, dataset.n_slices)
    X, Y, Z = image.shape
    n_slices = 96
    X2, Y2 = int(math.ceil(X / 32.0)) * 32, int(math.ceil(Y / 32.0)) * 32
    x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
    x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
    z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    image = image[:, z1_: z2_]
    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_ - z1, z2 - z2_)), 'constant')
    image = torch.from_numpy(image).float()
    return image


class FCN3DExperiment(Experiment):
    def inference(self, epoch: int):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)

                image = val.read_image(val.image_paths[idx], val.feature_size, val.n_slices)
                label = val.get_label_tensor_from_index(idx)
                # image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem),
                    self.config.gpu, self.config.device,
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)

                image = val.read_image(val.image_paths[idx], val.feature_size, val.n_slices)
                label = val.get_label_tensor_from_index(idx)
                # image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem),
                    self.config.gpu, self.config.device,
                )
