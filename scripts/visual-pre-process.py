from CMRSegment.common.plot import plot_nii_gz
from CMRSegment.common.constants import LIB_DIR

# plot enlarge phases
# data_dir = ROOT_DIR.joinpath("data")
# for center in ["genscan", "sheffield", "singapore_hcm", "singapore_lvsa", "ukbb"]:
#     subject = Subject(dir=data_dir.joinpath(center, "1"), nii_name="lvsa_.nii.gz")
#     phase_path = subject.enlarge_phases_dir().joinpath("lvsa_SR_00.nii.gz")
#     print(phase_path)
#     plot_nii_gz(phase_path)
plot_nii_gz(LIB_DIR.joinpath("data", "ukbb", "1", "4d_rview", "4Dimg.nii.gz"))
