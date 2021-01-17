import os
from multiprocessing import Pool
from functools import partial
from CMRSegment.common.subject import Subject
import mirtk
from pathlib import Path
import shutil

mirtk.subprocess.showcmd = True


def evaluate_registration(target: Path, source: Path, dof: Path):
    mirtk.info(str(dof))
    mirtk.transform_points(str(target), str(target.parent.joinpath("{}_temp.vtk".format(target.stem))), dofin=str(dof))
    mirtk.evaluate_distance(str(target.parent.joinpath("{}_temp.vtk".format(target.stem))), str(source))
    # target.parent.joinpath("temp.vtk").unlink()


def meshGeneration(subject: Subject, template_dir: Path):
   
    print("\n ... Mesh Generation - step [1] -")
    for phase, segmented_path, segmented_LR_path in zip(
        ["ED", "ES"], subject.segmented_ed_es, subject.segmented_LR_ed_es
    ):
        # extract meshes of lvendo, lvepi, lvmyo, rv and rveip at ED and ES, respectively
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 3, 4,
            set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("RV_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 3, 4,
            set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_RVepi_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_RVepi_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("RVepi_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-map", 3, 0, 4, 0,
            output=str(subject.tmps_dir().joinpath("vtk_LV_{}.nii.gz".format(phase))),
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 1, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVendo_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVendo_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 1, 2, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVepi_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVepi_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )
        mirtk.calculate_element_wise(
            str(segmented_path),
            "-label", 2, set=255, pad=0,
            output=str(subject.tmps_dir().joinpath("vtk_LVmyo_{}.nii.gz".format(phase))),
        )
        mirtk.extract_surface(
            str(subject.tmps_dir().joinpath("vtk_LVmyo_{}.nii.gz".format(phase))),
            str(subject.vtks_dir().joinpath("LVmyo_{}.vtk".format(phase))),
            isovalue=120, blur=2,
        )

    # use landmark to initialise the registration
    mirtk.register(
        str(subject.landmark_path),
        str(template_dir.joinpath("landmarks2.vtk")),
        model="Rigid",
        dofout=str(subject.dofs_dir().joinpath("landmarks.dof.gz")),
    )
    
    ###############################################################################

    print("\n ... Mesh Generation - step [2] -")
    for fr in ['ED', 'ES']:

        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.dofs_dir().joinpath("landmarks.dof.gz")),
            dofout=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr)))
        )

        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            "-t", str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )

        mirtk.register_points(
            "-t", str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            "-s", str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-symmetric",
            dofin=str(subject.tmps_dir().joinpath("{}.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )

        ###########################################################################
        # os.system('ptransformation '
        #           '{0}/RV_{2}.vtk '
        #           '{0}/N_RV_{2}.vtk '
        #           '-dofin {1}/rv_{2}_rreg.dof.gz >/dev/nul '
        #           .format(vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )

        # os.system('ptransformation '
        #           '{0}/RVepi_{2}.vtk '
        #           '{0}/N_RVepi_{2}.vtk '
        #           '-dofin {1}/rv_{2}_rreg.dof.gz >/dev/nul '
        #           .format(vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("RVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv_{}_rreg.dof.gz".format(fr))),
        )
        # os.system('ptransformation '
        #           '{0}/LVendo_{2}.vtk '
        #           '{0}/N_LVendo_{2}.vtk '
        #           '-dofin {1}/lv_{2}_rreg.dof.gz >/dev/nul '
        #           .format(vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        # os.system('ptransformation '
        #           '{0}/LVepi_{2}.vtk '
        #           '{0}/N_LVepi_{2}.vtk '
        #           '-dofin {1}/lv_{2}_rreg.dof.gz >/dev/nul '
        #           .format(vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        # os.system('ptransformation '
        #           '{0}/LVmyo_{2}.vtk '
        #           '{0}/N_LVmyo_{2}.vtk '
        #           '-dofin {1}/lv_{2}_rreg.dof.gz >/dev/nul '
        #           .format(vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(subject.vtks_dir().joinpath("LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVmyo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        # os.system('transformation '
        #           '{0}/vtk_RV_{1}.nii.gz '
        #           '{0}/N_vtk_RV_{1}.nii.gz '
        #           '-dofin {0}/lv_{1}_rreg.dof.gz '
        #           '-invert >/dev/nul '
        #           .format(tmps_dir, fr))

        mirtk.transform_image(
            str(subject.tmps_dir().joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            "-invert",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        # os.system('transformation '
        #           '{0}/vtk_LV_{1}.nii.gz '
        #           '{0}/N_vtk_LV_{1}.nii.gz '
        #           '-dofin {0}/lv_{1}_rreg.dof.gz '
        #           '-invert >/dev/nul '
        #           .format(tmps_dir, fr))
        mirtk.transform_image(
            str(subject.tmps_dir().joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            "-invert",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_rreg.dof.gz".format(fr))),
        )
        ###########################################################################
        #affine

        # os.system('areg '
        #           '{0}/vtk_RV_{3}.nii.gz '
        #           '{1}/N_vtk_RV_{3}.nii.gz '
        #           '-dofout {1}/rv_{3}_areg.dof.gz '
        #           '-parin {2}/segareg.txt >/dev/nul '
        #           .format(template_dir, tmps_dir, param_dir, fr))

        mirtk.smooth_image(
            str(template_dir.joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
            1,
            "-float"
        )
        mirtk.register(
            str(subject.tmps_dir().joinpath("smoothed_template_vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            model="Affine",
            dofout=str(subject.tmps_dir().joinpath("rv_{}_areg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segareg.txt")),
        )
        # os.system('areg '
        #           '{0}/vtk_LV_{3}.nii.gz '
        #           '{1}/N_vtk_LV_{3}.nii.gz '
        #           '-dofout {1}/lv_{3}_areg.dof.gz '
        #           '-parin {2}/segareg.txt >/dev/nul '
        #           .format(template_dir, tmps_dir, param_dir, fr))
        mirtk.register(
            str(template_dir.joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            model="Affine",
            dofout=str(subject.tmps_dir().joinpath("lv_{}_areg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segareg.txt")),
        )
        #non-rigid
        # os.system('nreg '
        #           '{0}/vtk_RV_{3}.nii.gz '
        #           '{1}/N_vtk_RV_{3}.nii.gz '
        #           '-dofin {1}/rv_{3}_areg.dof.gz '
        #           '-dofout {1}/rv_{3}_nreg.dof.gz '
        #           '-parin {2}/segreg.txt >/dev/nul '
        #           .format(template_dir, tmps_dir, param_dir, fr))

        mirtk.register(
            str(template_dir.joinpath("vtk_RV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_RV_{}.nii.gz".format(fr))),
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("rv_{}_areg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv_{}_nreg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segreg.txt")),
        )
        # os.system('snreg '
        #           '{0}/RV_{3}.vtk '
        #           '{1}/N_RV_{3}.vtk '
        #           '-dofin {2}/rv_{3}_nreg.dof.gz '
        #           '-dofout {2}/rv{3}ds8.dof.gz '
        #           '-ds 8 -symmetric >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))
        mirtk.register(
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            # "-symmetric",
            "-par", "Point set distance correspondence", "CP",
            ds=8,
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("rv_{}_nreg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("rv{}ds8.dof.gz".format(fr))),
        )
        # # os.system('nreg '
        # #           '{0}/vtk_LV_{3}.nii.gz '
        # #           '{1}/N_vtk_LV_{3}.nii.gz '
        # #           '-dofin {1}/lv_{3}_areg.dof.gz '
        # #           '-dofout {1}/lv_{3}_nreg.dof.gz '
        # #           '-parin {2}/segreg.txt >/dev/nul '
        # #           .format(template_dir, tmps_dir, param_dir, fr))
        mirtk.register(
            str(template_dir.joinpath("vtk_LV_{}.nii.gz".format(fr))),
            str(subject.tmps_dir().joinpath("N_vtk_LV_{}.nii.gz".format(fr))),
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_areg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv_{}_nreg.dof.gz".format(fr))),
            parin=str(template_dir.joinpath("segreg.txt")),
        )
        # os.system('msnreg 2 '
        #           '{0}/LVendo_{3}.vtk '
        #           '{0}/LVepi_{3}.vtk '
        #           '{1}/N_LVendo_{3}.vtk '
        #           '{1}/N_LVepi_{3}.vtk '
        #           '-dofin {2}/lv_{3}_nreg.dof.gz '
        #           '-dofout {2}/lv{3}final.dof.gz '
        #           '-ds 4 -symmetric >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))
        mirtk.register(
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            # "-symmetric",
            "-par", "Energy function", "PCD(T o P(1:2:end), P(2:2:end))",
            model="FFD",
            dofin=str(subject.tmps_dir().joinpath("lv_{}_nreg.dof.gz".format(fr))),
            dofout=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            ds=4,
        )
        # same number of points
        # os.system('cardiacsurfacemap '
        #           '{0}/LVendo_{3}.vtk '
        #           '{1}/N_LVendo_{3}.vtk '
        #           '{2}/lv{3}final.dof.gz '
        #           '{1}/F_LVendo_{3}.vtk >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))

        mirtk.match_points(
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVendo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
        )

        # os.system('cardiacsurfacemap '
        #           '{0}/LVepi_{3}.vtk '
        #           '{1}/N_LVepi_{3}.vtk '
        #           '{2}/lv{3}final.dof.gz '
        #           '{1}/F_LVepi_{3}.vtk >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))
        mirtk.match_points(
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_LVepi_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
        )
        # os.system('ptransformation '
        #           '{0}/LVmyo_{3}.vtk '
        #           '{1}/F_LVmyo_{3}.vtk '
        #           '-dofin {2}/lv{3}final.dof.gz >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))
        mirtk.transform_points(
            str(template_dir.joinpath("LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("lv{}final.dof.gz".format(fr))),
        )

        # os.system('cardiacsurfacemap '
        #           '{0}/RV_{3}.vtk '
        #           '{1}/N_RV_{3}.vtk '
        #           '{2}/rv{3}ds8.dof.gz '
        #           '{1}/C_RV_{3}.vtk >/dev/nul '
        #           .format(template_dir, vtks_dir, tmps_dir, fr))
        mirtk.match_points(
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RV_{}.vtk".format(fr))),
            dofin=str(subject.tmps_dir().joinpath("rv{}ds8.dof.gz".format(fr))),
            output=str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
        )
        ###########################################################################
        # os.system('cp {0}/F_LVendo_{1}.vtk {0}/S_LVendo_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/F_LVepi_{1}.vtk {0}/S_LVepi_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/F_LVmyo_{1}.vtk {0}/S_LVmyo_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/F_LVmyo_{1}.vtk {0}/C_LVmyo_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/F_LVmyo_{1}.vtk {0}/W_LVmyo_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/C_RV_{1}.vtk {0}/S_RV_{1}.vtk'.format(vtks_dir, fr))
        # os.system('cp {0}/C_RV_{1}.vtk {0}/W_RV_{1}.vtk'.format(vtks_dir, fr))

        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVendo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVepi_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_LVmyo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))
        )

        shutil.copy(
            str(subject.vtks_dir().joinpath("F_LVmyo_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("W_LVmyo_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))
        )
        shutil.copy(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))
        )

        print("\n ... Mesh Generation - step [3] -")

        ###########################################################################
        # compute the quantities of the heart with respect to template
        # os.system('cardiacwallthickness '
        #           '{0}/F_LVendo_{1}.vtk '
        #           '{0}/F_LVepi_{1}.vtk '
        #           '-myocardium '
        #           '{0}/W_LVmyo_{1}.vtk >/dev/nul '
        #           .format(vtks_dir, fr))

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("F_LVendo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("F_LVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )

        # os.system('cardiacenlargedistance '
        #           '{0}/S_LVendo_{2}.vtk '
        #           '{0}/S_LVepi_{2}.vtk '
        #           '{1}/LVendo_{2}.vtk '
        #           '{1}/LVepi_{2}.vtk '
        #           '-myocardium '
        #           '{0}/S_LVmyo_{2}.vtk >/dev/nul '
        #           .format(vtks_dir, template_dir, fr))
        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_LVendo_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVendo_{}.vtk".format(fr))),
            name="WallThickness",
        )
        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_LVepi_{}.vtk".format(fr))),
            str(template_dir.joinpath("LVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )
        
        # os.system('DiscreteCurvatureEstimator '
        #           '{0}/C_LVmyo_{1}.vtk '
        #           '{0}/FC_LVmyo_{1}.vtk >/dev/nul '
        #           .format(vtks_dir, fr))
        #
        # os.system('cardiaccurvature '
        #           '{0}/FC_LVmyo_{1}.vtk '
        #           '{0}/C_LVmyo_{1}.vtk >/dev/nul '
        #           '-smooth 64'
        #           .format(vtks_dir, fr))
        # mirtk.decimate_surface(
        #     str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
        #     str(subject.vtks_dir().joinpath("FC_LVmyo_{}.vtk".format(fr))),
        # )
        mirtk.calculate_surface_attributes(
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            smooth_iterations=64,
        )
        
        # os.system('DiscreteCurvatureEstimator '
        #           '{0}/C_RV_{1}.vtk '
        #           '{0}/FC_RV_{1}.vtk >/dev/nul '
        #           .format(vtks_dir, fr))
        #
        # os.system('cardiaccurvature '
        #           '{0}/FC_RV_{1}.vtk '
        #           '{0}/C_RV_{1}.vtk '
        #           '-smooth 64 >/dev/nul '
        #           .format(vtks_dir, fr))

        # mirtk.decimate_surface(
        #     str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
        #     str(subject.vtks_dir().joinpath("FC_RV_{}.vtk".format(fr))),
        # )
        mirtk.calculate_surface_attributes(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            smooth_iterations=64,
        )

        # os.system('sevaluation '
        #           '{0}/S_RV_{2}.vtk '
        #           '{1}/RV_{2}.vtk '
        #           '-scalar '
        #           '-signed >/dev/nul '
        #           .format(vtks_dir, template_dir, fr))

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))),
            str(template_dir.joinpath("RV_{}.vtk".format(fr))),
            "-normal",
            name="WallThickness",
        )

        # os.system('cardiacwallthickness '
        #           '{0}/W_RV_{1}.vtk '
        #           '{0}/N_RVepi_{1}.vtk >/dev/nul '
        #           .format(vtks_dir, fr))

        mirtk.evaluate_distance(
            str(subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))),
            str(subject.vtks_dir().joinpath("N_RVepi_{}.vtk".format(fr))),
            name="WallThickness",
        )
        
        ###########################################################################
#        os.system('vtk2txt {0}/C_RV_{1}.vtk {2}/rv_{1}_curvature.txt'.format(vtks_dir, fr, subject_dir))
#        os.system('vtk2txt {0}/W_RV_{1}.vtk {2}/rv_{1}_wallthickness.txt'.format(vtks_dir, fr, subject_dir))
#        os.system('vtk2txt {0}/S_RV_{1}.vtk {2}/rv_{1}_signeddistances.txt'.format(vtks_dir, fr, subject_dir))
#        os.system('vtk2txt {0}/W_LVmyo_{1}.vtk {2}/lv_myo{1}_wallthickness.txt'.format(vtks_dir, fr, subject_dir))
#        os.system('vtk2txt {0}/C_LVmyo_{1}.vtk {2}/lv_myo{1}_curvature.txt'.format(vtks_dir, fr, subject_dir))
#        os.system('vtk2txt {0}/S_LVmyo_{1}.vtk {2}/lv_myo{1}_signeddistances.txt'.format(vtks_dir, fr, subject_dir))
        if fr == 'ED':
            fr_ = 'ed'
        if fr == 'ES':
            fr_ = 'es'
        # os.system('vtk2txt {0}/C_RV_{1}.vtk {2}/rv_{3}_curvature.txt'.format(vtks_dir, fr, subject_dir, fr_))
        # os.system('vtk2txt {0}/W_RV_{1}.vtk {2}/rv_{3}_wallthickness.txt'.format(vtks_dir, fr, subject_dir, fr_))
        # os.system('vtk2txt {0}/S_RV_{1}.vtk {2}/rv_{3}_signeddistances.txt'.format(vtks_dir, fr, subject_dir, fr_))
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("C_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_curvature.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("W_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_wallthickness.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("S_RV_{}.vtk".format(fr))),
            str(subject.dir.joinpath("rv_{}_signeddistances.txt".format(fr_))),
        )
        ############################################################################################################
        # os.system('vtk2txt {0}/W_LVmyo_{1}.vtk {2}/lv_myo{3}_wallthickness.txt'.format(vtks_dir, fr, subject_dir, fr_))
        # os.system('vtk2txt {0}/C_LVmyo_{1}.vtk {2}/lv_myo{3}_curvature.txt'.format(vtks_dir, fr, subject_dir, fr_))
        # os.system('vtk2txt {0}/S_LVmyo_{1}.vtk {2}/lv_myo{3}_signeddistances.txt'.format(vtks_dir, fr, subject_dir, fr_))
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("W_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_wallthickness.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("C_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_curvature.txt".format(fr_))),
        )
        mirtk.convert_pointset(
            str(subject.vtks_dir().joinpath("S_LVmyo_{}.vtk".format(fr))),
            str(subject.dir.joinpath("lv_myo{}_signeddistances.txt".format(fr_))),
        )


def apply_PC(subject, data_dir, param_dir, template_dir, mirtk):
    
    print('  co-registering {0}'.format(subject))    
    
    subject_dir = os.path.join(data_dir, subject)
        
    if os.path.isdir(subject_dir):
        
        tmps_dir = '{0}/tmps'.format(subject_dir)

        vtks_dir = '{0}/vtks'.format(subject_dir)

        dofs_dir = '{0}/dofs'.format(subject_dir)

        meshGeneration(subject_dir, template_dir, param_dir, tmps_dir, vtks_dir, dofs_dir)
        
        print('  finish generating meshes from segmentations in {0}'.format(subject))
        
    else:  
        print('  {0} is not a valid directory, do nothing'.format(subject_dir))


def meshCoregstration(dir_0, dir_2, dir_3, coreNo, parallel, mirtk):
               
    print('Generate meshes from segmentations running on {0} cores'.format(coreNo))
    
    pool = Pool(processes = coreNo) 
    
    # partial only in Python 2.7+
    pool.map(partial(apply_PC, 
                     data_dir=dir_0,  
                     param_dir=dir_2, 
                     template_dir=dir_3,
                     mirtk=mirtk), 
                     sorted(os.listdir(dir_0)))  

    pool.close()     
    pool.join()

