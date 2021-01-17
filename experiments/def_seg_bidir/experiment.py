
import torch
import numpy as np
from CMRSegment.common.nn.torch import prepare_tensors
from CMRSegment.common.nn.torch.experiment import Experiment


class DefSegExperiment(Experiment):

    def inference(self, epoch: int):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                # template = val.get_label_tensor_from_index(np.random.randint(0, len(val)))
                template = val.template
                template = torch.from_numpy(template).float()
                template = torch.unsqueeze(template, 0)
                template = prepare_tensors(template, self.config.gpu, self.config.device)

                template_image = val.template_image
                template_image = torch.from_numpy(template_image).float()
                template_image = torch.unsqueeze(template_image, 0)
                template_image = prepare_tensors(template_image, self.config.gpu, self.config.device)

                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    (image, template, template_image), label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem)
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                # template = val.get_label_tensor_from_index(np.random.randint(0, len(val)))
                template = val.template
                template = torch.from_numpy(template).float()
                template = torch.unsqueeze(template, 0)
                template = prepare_tensors(template, self.config.gpu, self.config.device)

                template_image = val.template_image
                template_image = torch.from_numpy(template_image).float()
                template_image = torch.unsqueeze(template_image, 0)
                template_image = prepare_tensors(template_image, self.config.gpu, self.config.device)

                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    (image, template, template_image), label, image_path, self.network,
                    output_dir.joinpath(val.name, image_path.parent.stem),
                )
