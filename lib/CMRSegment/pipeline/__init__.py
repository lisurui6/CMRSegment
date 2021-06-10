from tqdm import tqdm
from pathlib import Path
from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor import CineSegmentor
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor

from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.pipeline.config import PipelineConfig
from CMRSegment.common.resource import Segmentation, PhaseMesh
from CMRSegment.motion_tracker import MotionTracker
from CMRSegment.segmentor.atlas import SegmentationCorrector


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
            if self.config.segment_config.refine:
                segmentation_corrector = SegmentationCorrector(
                    atlas_dir=self.config.segment_config.refine_atlas_dir,
                    param_path=None,
                )
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
        if self.config.track_motion:
            motion_tracker = MotionTracker(
                template_dir=self.config.motion_tracker_config.template_dir,
                param_dir=self.config.motion_tracker_config.param_dir
            )
        subjects = preprocessor.run(data_dir=data_dir, output_dir=self.config.output_dir)
        for ed_image, es_image, cine, output_dir in subjects:
            if self.config.segment and self.config.segment_config.segment_cine:
                print("Segmenting all {} cine images...".format(len(cine)))
                cine_segmentations = cine_segmentor.apply(cine, output_dir=output_dir)
            else:
                cine_segmentations = [
                    Segmentation(
                        path=output_dir.joinpath("segs").joinpath(f"lvsa_{idx}.nii.gz"), phase=idx
                    ) for idx in range(len(cine))
                ]
                if self.config.segment_config.refine:
                    corrected_segmentations = []
                    for i, segmentation, image in enumerate(zip(cine_segmentations, cine)):
                        landmark_path = extract_landmarks(
                            segmentation.path,
                            output_path=output_dir.joinpath("segs", "landmarks", f"landmark_{i}.vtk"),
                            labels=[2, 3]
                        )

                        output = segmentation_corrector.run(
                            subject_seg=segmentation,
                            subject_image=image,
                            output_dir=output_dir.joinpath("segs"),
                            subject_landmarks=landmark_path,
                            n_top=3,
                            force=True,
                        )
                        corrected_segmentations.append(output)
                    cine_segmentations = corrected_segmentations

            meshes = []
            for phase_image in [ed_image, es_image]:
                if self.config.segment:
                    print("Segmenting {} image...".format(phase_image.phase))
                    segmentation = hr_segmentor.apply(
                        phase_image, output_path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                    )
                else:
                    segmentation = Segmentation(
                        phase=phase_image.phase, path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                    )
                if self.config.extract:
                    print("Extracting {} segmentation...".format(phase_image.phase))
                    landmark_path = extract_landmarks(
                        segmentation.path, output_path=output_dir.joinpath("landmark.vtk"), labels=[2, 3]
                    )
                    mesh = mesh_extractor.run(segmentation, output_dir.joinpath("mesh"))
                else:
                    mesh = PhaseMesh.from_dir(output_dir.joinpath("mesh"), phase=phase_image.phase)
                    landmark_path = output_dir.joinpath("landmark.vtk")
                if self.config.segment_config.refine:
                    segmentation = segmentation_corrector.run(
                        subject_image=phase_image,
                        subject_seg=segmentation,
                        subject_landmarks=landmark_path,
                        output_dir=output_dir,
                        n_top=self.config.segment_config.refine_n_top,
                        force=True,
                    )

                if self.config.coregister:
                    print("Coregistering")
                    coregister.run(
                        mesh, segmentation, landmark_path, output_dir=output_dir.joinpath("registration")
                    )
                meshes.append(mesh)

            if self.config.track_motion:
                print("Tracking motion")
                motion_tracker.run(
                    cine=cine,
                    cine_segmentations=cine_segmentations,
                    landmark_path=landmark_path,
                    ED_mesh=meshes[0],
                    output_dir=output_dir.joinpath("motion"),
                    overwrite=self.config.overwrite,
                )
