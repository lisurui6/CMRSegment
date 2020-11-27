from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor
from CMRSegment.segmentor.tf1.cine import TF1CineSegmentor
from CMRSegment.common.constants import LIB_DIR, MODEL_DIR, ROOT_DIR
from CMRSegment.extractor.landmark import extract_landmarks
from CMRSegment.coregister.meshfitting import meshGeneration
from CMRSegment.coregister import Coregister
from CMRSegment.extractor.mesh import MeshExtractor
from CMRSegment.subject import Image, Phase, Artifact

preprocessor = DataPreprocessor(force_restart=False)
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
for center in ["genscan"]:
    subjects = preprocessor.run(data_dir=LIB_DIR.parent.joinpath("data", center))
    for subject in subjects:
        # get_segmentation(model_path, subject, reference_ed_image)
        with TF1CineSegmentor(model_path=model_path) as segmentor:
            images = []
            segmentations = []
            for idx, phase_path in enumerate(subject.enlarge_phases):
                image = Image(path=phase_path, phase=idx, output_dir=subject.motions_dir())
                segmentation = segmentor.apply(image)
                segmentations.append(segmentation)
            segmentor.save_cine(segmentations, subject.rview_dir())

        with TF13DSegmentor(model_path=model_path) as segmentor:
            ed_es_segmentations = []
            for fr, phase_path in zip(["ED", "ES"], [subject.ed_path, subject.es_path]):
                image = Image(path=phase_path, phase=Phase[fr])
                segmentation = segmentor.apply(image)
                ed_es_segmentations.append(segmentation)

        landmark_path = extract_landmarks(subject, labels=[2, 3])
        ed_es_meshs = []
        for segmentation in ed_es_segmentations:
            mesh = mesh_extractor.run(segmentation, subject.vtks_dir())
            ed_es_meshs.append(mesh)
        subject.vtks_dir().joinpath("registration").mkdir(exist_ok=True, parents=True)
        for mesh, segmentation in zip(ed_es_meshs, ed_es_segmentations):
            coregister.run(mesh, segmentation, landmark_path, output_dir=subject.vtks_dir().joinpath("registration"))
        # meshGeneration(subject, ROOT_DIR.joinpath("input", "params"))
