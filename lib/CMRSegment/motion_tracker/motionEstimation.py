#!/usr/bin/python
import os, glob
from   multiprocessing import Pool
from   functools       import partial
from   tqdm            import tqdm
#############################
import multiprocessing as mp
import multiprocessing.pool as mpool
from multiprocessing.util import debug
#############################


# Perform motion tracking for cine MR
def track_cine(data_dir, par_dir, tamplate_dir):
        
    motion_dir = os.path.join(data_dir, 'motion')

    n_frame = len(glob.glob('{0}/lvsa_??.nii.gz'.format(motion_dir)))

    print("\n ...  Inter-frame motion estimation")

    ###################################  
    # Inter-frame motion estimation   #
    ###################################

    for fr in tqdm(range(1, n_frame)):
        target = '{0}/lvsa_{1:02d}.nii.gz'.format(motion_dir, fr-1)
        source = '{0}/lvsa_{1:02d}.nii.gz'.format(motion_dir, fr)
        par = '{0}/ffd_motion.cfg'.format(par_dir)
        dof = '{0}/ffd_{1:02d}_to_{2:02d}.dof.gz'.format(motion_dir, fr-1, fr)

        if not os.path.exists(dof):
            os.system('mirtk register '
                      '{0} '
                      '{1} '
                      '-parin {2} '
                      '-dofout {3} >/dev/nul '
                      .format(target, source, par, dof))
    
    #############################################
    # Compose inter-frame transformation fields #
    #############################################

    print("\n ...  Compose inter-frame transformation fields")
    for fr in tqdm(range(2, n_frame)):
        dofs = ''
        for k in range(1, fr+1):
            dof = '{0}/ffd_{1:02d}_to_{2:02d}.dof.gz'.format(motion_dir, k-1, k)
            dofs += dof + ' '
            dof_out = '{0}/ffd_comp_00_to_{1:02d}.dof.gz'.format(motion_dir, fr)
            if not os.path.exists(dof_out):
                os.system('mirtk compose-dofs '
                          '{0} '
                          '{1} >/dev/nul '
                          .format(dofs, dof_out))

    ##########################################################################################################################
    # Refine motion fields                                                                                                   #
    # Composition of inter-frame motion fields can lead to accumulative errors. At this step, we refine the motion fields    #   
    # by re-registering the n-th frame with the ED frame.                                                                    #
    ##########################################################################################################################


    print("\n ...  Refine motion fields")
    for fr in tqdm(range(2, n_frame)):
        target = '{0}/lvsa_00.nii.gz'.format(motion_dir)
        source = '{0}/lvsa_{1:02d}.nii.gz'.format(motion_dir, fr)
        par = '{0}/ffd_refine.cfg'.format(par_dir)
        dofin = '{0}/ffd_comp_00_to_{1:02d}.dof.gz'.format(motion_dir, fr)
        dof = '{0}/ffd_00_to_{1:02d}.dof.gz'.format(motion_dir, fr)
        if not os.path.exists(dof):
            os.system('mirtk register '
                      '{0} '
                      '{1} '
                      '-parin {2} '
                      '-dofin {3} '
                      '-dofout {4} >/dev/nul '
                      .format(target, source, par, dofin, dof))


    ##########################################################################
    # Obtain the RV mesh with the same number of points as the template mesh #
    ##########################################################################

    os.system('prreg '
              '{2}/landmarks2.vtk '
              '{0}/landmarks.vtk '
              '-dofout {1}/landmarks.dof.gz >/dev/nul '
              .format(tamplate_dir, motion_dir, data_dir))
   

    print("\n ...  Mesh transformation, last step")


    ###########
    # LV endo #    
    ###########

    os.system('srreg '
              '{2}/vtks/F_LVendo_ED.vtk '
              '{0}/LVendo_ED.vtk '
              '-dofin {1}/landmarks.dof.gz '
              '-dofout {1}/LV_srreg.dof.gz '
              '-symmetric >/dev/nul '
              .format(tamplate_dir, motion_dir, data_dir))
    
    os.system('mirtk transform-points '
              '{0}/LVendo_ED.vtk '
              '{1}/LV_ED_srreg_endo.vtk '
              '-dofin {1}/LV_srreg.dof.gz '
              '-invert >/dev/nul '
              .format(tamplate_dir, motion_dir))

    os.system('sareg '
              '{0}/LV_ED_srreg_endo.vtk '
              '{1}/vtks/F_LVendo_ED.vtk '
              '-dofout {0}/LV_sareg.dof.gz '
              '-symmetric >/dev/nul '
              .format(motion_dir, data_dir))


    os.system('snreg '
              '{0}/LV_ED_srreg_endo.vtk '
              '{1}/vtks/F_LVendo_ED.vtk '
              '-dofin {0}/LV_sareg.dof.gz '
              '-dofout {0}/LV_snreg.dof.gz '
              '-ds 20 -symmetric >/dev/nul '
              .format(motion_dir, data_dir))
    

    os.system('mirtk transform-points '
              '{0}/LV_ED_srreg_endo.vtk '
              '{0}/LV_endo_fr00.vtk '
              '-dofin {0}/LV_snreg.dof.gz >/dev/nul '
              .format(motion_dir))
    
    # Transform the mesh
    print("\n ...   Transform the LV endo mesh")
    for fr in tqdm(range(1, n_frame)):
        os.system('mirtk transform-points '
                  '{0}/LV_endo_fr00.vtk '
                  '{0}/LV_endo_fr{1:02d}.vtk '
                  '-dofin {0}/ffd_00_to_{1:02d}.dof.gz >/dev/nul '
                  .format(motion_dir, fr))
    

    # Convert vtks to text files
    print("\n ...   Convert  LV endo vtks to text files")
    for fr in tqdm(range(0, n_frame)):
        os.system('vtk2txt '
                  '{0}/LV_endo_fr{1:02d}.vtk '
                  '{0}/LV_endo_fr{1:02d}.txt >/dev/nul '
                  .format(motion_dir, fr))


    ##########
    # LV epi #
    ##########

    os.system('srreg '
            '{2}/vtks/F_LVepi_ED.vtk '
            '{0}/LVepi_ED.vtk '
            '-dofin {1}/landmarks.dof.gz '
            '-dofout {1}/LV_srreg_epi.dof.gz '
            '-symmetric >/dev/nul '
            .format(tamplate_dir, motion_dir, data_dir))
    


    os.system('mirtk transform-points '
            '{0}/LVepi_ED.vtk '
            '{1}/LV_ED_srreg_epi.vtk '
            '-dofin {1}/LV_srreg_epi.dof.gz '
            '-invert >/dev/nul '
            .format(tamplate_dir, motion_dir))

    

    os.system('sareg '
            '{0}/LV_ED_srreg_epi.vtk '
            '{1}/vtks/F_LVepi_ED.vtk '
            '-dofout {0}/LV_sareg_epi.dof.gz '
            '-symmetric >/dev/nul '
            .format(motion_dir, data_dir))


    os.system('snreg '
            '{0}/LV_ED_srreg_epi.vtk '
            '{1}/vtks/F_LVepi_ED.vtk '
            '-dofin {0}/LV_sareg_epi.dof.gz '
            '-dofout {0}/LV_snreg_epi.dof.gz '
            '-ds 20 -symmetric >/dev/nul '
            .format(motion_dir, data_dir))
    

    os.system('mirtk transform-points '
            '{0}/LV_ED_srreg_epi.vtk '
            '{0}/LV_epi_fr00.vtk '
            '-dofin {0}/LV_snreg_epi.dof.gz >/dev/nul '
            .format(motion_dir))
    
    # Transform the mesh
    print("\n ...   Transform the LV epi mesh")
    for fr in tqdm(range(1, n_frame)):
        os.system('mirtk transform-points '
                '{0}/LV_epi_fr00.vtk '
                '{0}/LV_epi_fr{1:02d}.vtk '
                '-dofin {0}/ffd_00_to_{1:02d}.dof.gz >/dev/nul '
                .format(motion_dir, fr))
    

    # Convert vtks to text files
    print("\n ...   Convert LV epi vtks to text files")
    for fr in tqdm(range(0, n_frame)):
        os.system('vtk2txt '
                '{0}/LV_epi_fr{1:02d}.vtk '
                '{0}/LV_epi_fr{1:02d}.txt >/dev/nul '
                .format(motion_dir, fr))

    ######
    # RV #
    ######

    os.system('srreg '
            '{2}/vtks/C_RV_ED.vtk '
            '{0}/RV_ED.vtk '
            '-dofin {1}/landmarks.dof.gz '
            '-dofout {1}/RV_srreg.dof.gz '
            '-symmetric >/dev/nul '
            .format(tamplate_dir, motion_dir, data_dir))
    


    os.system('mirtk transform-points '
            '{0}/RV_ED.vtk '
            '{1}/RV_ED_srreg.vtk '
            '-dofin {1}/RV_srreg.dof.gz '
            '-invert >/dev/nul '
            .format(tamplate_dir, motion_dir))

    os.system('sareg '
            '{0}/RV_ED_srreg.vtk '
            '{1}/vtks/C_RV_ED.vtk '
            '-dofout {0}/RV_sareg.dof.gz '
            '-symmetric >/dev/nul '
            .format(motion_dir, data_dir))


    os.system('snreg '
            '{0}/RV_ED_srreg.vtk '
            '{1}/vtks/C_RV_ED.vtk '
            '-dofin {0}/RV_sareg.dof.gz '
            '-dofout {0}/RV_snreg.dof.gz '
            '-ds 20 -symmetric >/dev/nul '
            .format(motion_dir, data_dir))
    

    os.system('mirtk transform-points '
            '{0}/RV_ED_srreg.vtk '
            '{0}/RV_fr00.vtk '
            '-dofin {0}/RV_snreg.dof.gz >/dev/nul '
            .format(motion_dir))
    
    # Transform the mesh
    print("\n ...   Transform the mesh")
    for fr in tqdm(range(1, n_frame)):
        os.system('mirtk transform-points '
                '{0}/RV_fr00.vtk '
                '{0}/RV_fr{1:02d}.vtk '
                '-dofin {0}/ffd_00_to_{1:02d}.dof.gz >/dev/nul '
                .format(motion_dir, fr))
    

    # Convert vtks to text files
    print("\n ...   Convert vtks to text files")
    for fr in tqdm(range(0, n_frame)):
        os.system('vtk2txt '
                '{0}/RV_fr{1:02d}.vtk '
                '{0}/RV_fr{1:02d}.txt >/dev/nul '
                .format(motion_dir, fr))
    


def apply_PC(subject, data_dir, param_dir, template_dir):
    
    print('  co-registering {0}'.format(subject))    
    
    subject_dir = os.path.join(data_dir, subject)
        
    if os.path.isdir(subject_dir):
        
        track_cine(subject_dir, param_dir, template_dir)
        
        print('  finish motion tracking in subject {0}'.format(subject))
        
    else:  
        print('  {0} is not a valid directory, do nothing'.format(subject_dir))


def motionTracking(dir_0, dir_2, dir_3, coreNo, parallel):
               
    #if parallel:
    
    print('Motion tracking running on {0} cores'.format(coreNo))
    
    pool = Pool(processes = coreNo) 
    
    # partial only in Python 2.7+
    pool.map(partial(apply_PC, 
                     data_dir=dir_0,  
                     param_dir=dir_2, 
                     template_dir=dir_3), 
                     sorted(os.listdir(dir_0)))   
    pool.close()     
    pool.join() 
