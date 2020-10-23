from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor
from CMRSegment.segmentor.tf1.cine import TF1CineSegmentor
from CMRSegment.segmentor.tf1.LR import TF12DSegmentor
from CMRSegment.common.constants import ROOT_DIR, MODEL_DIR
from CMRSegment.landmark_extraction import extract_landmarks
from CMRSegment.coregister.meshfitting import meshGeneration


preprocessor = DataPreprocessor(force_restart=False)
model_path = MODEL_DIR.joinpath("3D", "biobank_low2high.ckpt-300")

# center = "sheffield"
# , "ukbb", "sheffield"
# "singapore_hcm", "singapore_lvsa", "sheffield",
for center in ["genscan"]:
    subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", center))
    for subject in subjects:
        # get_segmentation(model_path, subject, reference_ed_image)
        # with TF1CineSegmentor(model_path=model_path) as segmentor:
        #     segmentor.apply(subject)

        # with TF13DSegmentor(model_path=model_path) as segmentor:
        #     segmentor.apply(subject)

        extract_landmarks(subject, labels=[2, 3])
        meshGeneration(subject, ROOT_DIR.joinpath("input", "params"))
