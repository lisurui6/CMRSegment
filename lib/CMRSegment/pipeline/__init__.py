from tqdm import tqdm
from pathlib import Path
from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor import CineSegmentor
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor

from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.pipeline.config import PipelineConfig


class CMRPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, data_dir: Path):
        preprocessor = DataPreprocessor(overwrite=self.config.overwrite, use_irtk=self.config.use_irtk)
        if self.config.extract:
            mesh_extractor = MeshExtractor(
                iso_value=self.config.extract_config.iso_value,
                blur=self.config.extract_config.blur,
                overwrite=self.config.overwrite
            )
        if self.config.coregister:
            coregister = Coregister(
                template_dir=self.config.coregister_config.template_dir,
                param_dir=self.config.coregister_config.param_dir,
                overwrite=self.config.overwrite
            )
        if self.config.segment:
            if not self.config.segment_config.torch:
                from CMRSegment.segmentor.tf1.HR import TF13DSegmentor

                hr_segmentor = TF13DSegmentor(
                    model_path=self.config.segment_config.model_path, overwrite=self.config.overwrite
                )
                hr_segmentor.__enter__()
            else:
                from CMRSegment.segmentor.torch import TorchSegmentor
                hr_segmentor = TorchSegmentor(
                    model_path=self.config.segment_config.model_path, overwrite=self.config.overwrite,
                    use_irtk=self.config.use_irtk,
                )
            if self.config.segment_config.segment_cine:
                cine_segmentor = CineSegmentor(phase_segmentor=hr_segmentor)
        subjects = preprocessor.run(data_dir=data_dir, output_dir=self.config.output_dir)
        for ed_image, es_image, cine, output_dir in subjects:
            if self.config.segment and self.config.segment_config.segment_cine:
                cine_segmentor.apply(cine, output_dir=output_dir)

            for phase_image in [ed_image, es_image]:
                if self.config.segment:
                    segmentation = hr_segmentor.apply(
                        phase_image, output_path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                    )
                if self.config.extract:
                    landmark_path = extract_landmarks(
                        segmentation.path, output_path=output_dir.joinpath("landmark.vtk"), labels=[2, 3]
                    )
                    mesh = mesh_extractor.run(segmentation, output_dir.joinpath("mesh"))
                if self.config.coregister:
                    coregister.run(
                        mesh, segmentation, landmark_path, output_dir=output_dir.joinpath("registration")
                    )
