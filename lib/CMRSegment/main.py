from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor
from CMRSegment.segmentor.tf1.cine import TF1CineSegmentor
from CMRSegment.common.constants import LIB_DIR, MODEL_DIR, ROOT_DIR
from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor
from CMRSegment.common.subject import Image, Phase
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
    segareg_path=ROOT_DIR.joinpath("input", "params", "segareg.txt"),
    segreg_path=ROOT_DIR.joinpath("input", "params", "segreg.txt"),
    spnreg_path=ROOT_DIR.joinpath("input", "params", "spnreg.txt"),
)
with TF1CineSegmentor(model_path=model_path) as cine_segmentor:
    with TF13DSegmentor(model_path=model_path) as hr_segmentor:
        for center in ["genscan"]:
            subjects = preprocessor.run(data_dir=LIB_DIR.parent.joinpath("data", center))
            for subject in subjects:
                # get_segmentation(model_path, subject, reference_ed_image)
                images = []
                segmentations = []
                for idx, phase_path in enumerate(subject.enlarge_phases):
                    image = Image(path=phase_path, phase=idx, output_dir=subject.output_dir.joinpath("segmentation", "phases"))
                    segmentation = cine_segmentor.apply(image)
                    segmentations.append(segmentation)
                cine_segmentor.save_cine(segmentations, subject.rview_dir())

                ed_es_segmentations = []
                for fr, phase_path in zip(["ED", "ES"], [subject.ed_path, subject.es_path]):
                    image = Image(path=phase_path, phase=Phase[fr], output_dir=subject.output_dir.joinpath("segmentation"))
                    segmentation = hr_segmentor.apply(image)
                    ed_es_segmentations.append(segmentation)
                landmark_path = extract_landmarks(ed_es_segmentations[0].path, output_path=subject.landmark_path, labels=[2, 3])
                ed_es_meshs = []
                for segmentation in ed_es_segmentations:
                    mesh = mesh_extractor.run(segmentation, subject.output_dir.joinpath("mesh"))
                    ed_es_meshs.append(mesh)
                for mesh, segmentation in zip(ed_es_meshs, ed_es_segmentations):
                    coregister.run(mesh, segmentation, landmark_path, output_dir=subject.output_dir.joinpath("registration"))
                # meshGeneration(subject, ROOT_DIR.joinpath("input", "params"))
