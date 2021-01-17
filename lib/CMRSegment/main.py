from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor
from CMRSegment.segmentor import CineSegmentor
from CMRSegment.common.constants import LIB_DIR, MODEL_DIR, ROOT_DIR
from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor
from CMRSegment.common.subject import Image, Phase, Cine
import mirtk

# mirtk.subprocess.showcmd = True   # whether to print executed commands with arguments

preprocessor = DataPreprocessor(overwrite=False)
model_path = MODEL_DIR.joinpath("3D", "biobank_low2high.ckpt-300")

# center = "sheffield"
# , "ukbb", "sheffield"
# "singapore_hcm", "singapore_lvsa", "sheffield",
mesh_extractor = MeshExtractor()
coregister = Coregister(
    template_dir=ROOT_DIR.joinpath("input", "params"),
    param_dir=ROOT_DIR.joinpath("input", "params"),
)
with TF13DSegmentor(model_path=model_path) as hr_segmentor:
    cine_segmentor = CineSegmentor(phase_segmentor=hr_segmentor)
    for center in ["genscan"]:
        subjects = preprocessor.run(data_dir=LIB_DIR.parent.joinpath("data", center), output_dir=ROOT_DIR.joinpath("output"))
        for subject in subjects:
            # get_segmentation(model_path, subject, reference_ed_image)
            cine = Cine(dir=subject.gray_phases_dir())
            cine_segmentor.apply(cine, output_dir=subject.output_dir.joinpath("segmentation", "phases"))

            ed_es_segmentations = []
            for phase, phase_path in zip([Phase.ED, Phase.ES], [subject.ed_path, subject.es_path]):
                image = Image(
                    path=phase_path, phase=phase, output_dir=subject.output_dir.joinpath("segmentation", "ED_ES")
                )
                segmentation = hr_segmentor.apply(image)
                # ed_es_segmentations.append(segmentation)
                landmark_path = extract_landmarks(
                    segmentation.path, output_path=subject.landmark_path, labels=[2, 3]
                )
                mesh = mesh_extractor.run(segmentation, subject.output_dir.joinpath("mesh"))
                coregister.run(
                    mesh, segmentation, landmark_path, output_dir=subject.output_dir.joinpath("registration")
                )
