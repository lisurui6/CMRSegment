from pathlib import Path
from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor import CineSegmentor
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor

from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor
from CMRSegment.common.subject import Image, Phase, Cine
from CMRSegment.pipeline.config import PipelineConfig


class CMRPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, data_dir: Path):
        preprocessor = DataPreprocessor(overwrite=self.config.overwrite)
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
            hr_segmentor = TF13DSegmentor(
                model_path=self.config.segment_config.model_path, overwrite=self.config.overwrite
            )
            hr_segmentor.__enter__()
            cine_segmentor = CineSegmentor(phase_segmentor=hr_segmentor)
        subjects = preprocessor.run(data_dir=data_dir, output_dir=self.config.output_dir)
        for subject in subjects:
            if self.config.segment:
                cine = Cine(dir=subject.gray_phases_dir())
                cine_segmentor.apply(cine, output_dir=subject.output_dir.joinpath("segmentation", "phases"))

            for phase, phase_path in zip([Phase.ED, Phase.ES], [subject.ed_path, subject.es_path]):
                image = Image(
                    path=phase_path, phase=phase,
                    output_dir=subject.output_dir.joinpath("segmentation", "ED_ES")
                )
                if self.config.segment:
                    segmentation = hr_segmentor.apply(image)
                if self.config.extract:
                    landmark_path = extract_landmarks(
                        segmentation.path, output_path=subject.landmark_path, labels=[2, 3]
                    )
                    mesh = mesh_extractor.run(segmentation, subject.output_dir.joinpath("mesh"))
                if self.config.coregister:
                    coregister.run(
                        mesh, segmentation, landmark_path, output_dir=subject.output_dir.joinpath("registration")
                    )
