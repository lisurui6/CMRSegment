from CMRSegment.preprocessor import DataPreprocessor
from CMRSegment.common.constants import LIB_DIR
import nibabel as nib
from matplotlib import pyplot as plt

preprocessor = DataPreprocessor(overwrite=False)
# center = "sheffield"
# # , "ukbb", "sheffield"
# for center in ["genscan", "ukbb", "sheffield", "singapore_hcm", "singapore_lvsa"]:
#     subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", center))
#     for subject in subjects:
#         ed_image = nib.load(str(subject.ed_path))
#         ed_image = ed_image.get_data()
#         ed_image = rescale_intensity(ed_image)
#         print(center, np.mean(ed_image), np.max(ed_image))
#         print(ed_image.shape, ed_image.flatten().shape)
#         plt.hist(ed_image.flatten(), bins=128, density=True)
#         plt.title("{} hist after rescale".format(center))
#         plt.savefig("{}.png".format(center))
#         plt.show()


from skimage.exposure import match_histograms
reference_center = "genscan"
subjects = preprocessor.run(data_dir=LIB_DIR.joinpath("data", reference_center))
reference_subject = subjects[0]
reference_ed_nim = nib.load(str(reference_subject.ed_path))
reference_ed_image = reference_ed_nim.get_data()
# reference_ed_image = rescale_intensity(reference_ed_image)
for target_center in ["singapore_lvsa"]:
    subjects = preprocessor.run(data_dir=LIB_DIR.joinpath("data", target_center))
    target_subject = subjects[0]
    target_ed_nim = nib.load(str(target_subject.ed_path))
    target_ed_image = target_ed_nim.get_data()
    # target_ed_image = rescale_intensity(target_ed_image)
    matched = match_histograms(target_ed_image, reference_ed_image, multichannel=False)
    plt.figure()
    plt.hist(reference_ed_image.flatten(), bins=128, density=True)
    plt.title("rerference {}".format(reference_center))
    plt.figure()
    plt.hist(target_ed_image.flatten(), bins=128, density=True)
    plt.title("target {}".format(target_center))
    plt.figure()
    plt.hist(matched.flatten(), bins=128, density=True)
    plt.title("matched")
    plt.show()
    nim2 = nib.Nifti1Image(matched, affine=target_ed_nim.affine)
    nim2.header['pixdim'] = target_ed_nim.header['pixdim']
    nib.save(nim2, './matched_r_{}_t_{}.nii.gz'.format(reference_center, target_center))
