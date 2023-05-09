
import numpy as np
import os
import pandas as pd
import glob
import nibabel as nib
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
from scripts import design_matrices as DM
from scripts import contrasts

def main(root_dir = None, timestamps_path = None, dir_to_save= None, contrast_type = None, parser = True, compute_DM = True, dot_with = 'NPS'):

    """
    Arguments
    --------

    root_dir : String, Default = None
        directory to all the subject's fmri volumes. This script is built assuming that root dir is a dir with a folder for each participant.
    timestamps_path : Path to all the timestamps in a folder. The script identify which timestamps to choose based on its name. Can be .mat or .csv
    dir_to_save : Directory to save the statistical contrast maps. It will only be used if contrast_type is not 'None'.
    contrast_type : Type of contrast to compute. Choices=['all_shocks','each_shocks','neut_shocks', 'suggestions']. By default None.
    parser : True by default. If False, parser.args will be overlooked and user will be able to manually provide the necessary arguments while calling main().
    compute_DM : True by default but if different than True, a path to the design matrices and the concatenated fmri timeseries is expected.
    dir_to_save_DM : Path where the design matrices and fmri timeseries will be saved. Only taken into account if compute_DM = True.
    ####second_level : Bool., by default True.
        Runs a second level GLM

    Example
    -------
    main(dir_to_save= dir_to_save,compute_DM = True, contrast_type = None)

    """

    #--------Parser--------
    #Argument parser
    if __name__ == "__main__":
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument("--root_dir", type=str) #dir to the subjects' files containing the fmri data
        parser.add_argument("--dir_to_save", type=str) #path to save the output
        parser.add_argument("--timestamps_path_root", type=str) #path to the timestamps files
        parser.add_argument("--many_runs", type=str)
        parser.add_argument('--contrast_type', type=str, choices=['all_shocks','each_shocks','suggestions'], default='all_shocks')
        args = parser.parse_args()

    if root_dir != None: # if no root directory is provided, it's expected to receive a path to design matrices as 'compute_DM'
        ls_subj_name = [subject for subject in os.listdir(root_dir)]
        ls_subj_path  = glob.glob(os.path.join(root_dir,'*'))
    else:
        ls_subj_name = [subject for subject in os.listdir(compute_DM)]#Assuming compute DM is a path with the same structure as root_dir
        ls_subj_path  = glob.glob(os.path.join(compute_DM,'*'))
    contrast_paths = []

    #---creating directory---
    if contrast_type != None: #make a result directory for the contrasts
        path_contrast_results = os.path.join(dir_to_save,contrast_type) #creating a root dir to save all the contrast
        if os.path.exists(path_contrast_results) is False:
            os.mkdir(path_contrast_results)
        print(path_contrast_results)
    # Main loop
    for subj_path in ls_subj_path:
        subj_name = os.path.basename(os.path.normpath(subj_path))
        print('At : ' + subj_name)

        # Get design matrix and timeseries
        if compute_DM != True:
            paths_design_matrices = glob.glob(os.path.join(compute_DM, subj_name,'DM*csv'))#assuming that the design matrix has DM in its filename
            #design_matrices = DM.load_pkl_to_pd(paths_design_matrices)
            design_matrices = pd.read_csv(paths_design_matrices[0],index_col = [0])
            fmri_time_series = glob.glob(os.path.join(compute_DM, subj_name, '*fmri*'))#assuming that the 4D timeseries contains 'fmri' in its name
            conditions = 'hyper_hypo'
        else:
            DM.check_if_empty(root_dir)
            save_DM = os.path.join(dir_to_save,'DM_timeseries')
            if os.path.exists(save_DM) is False:
                os.mkdir(save_DM)
            design_matrix, fmri_time_series, conditions = DM.compute_DM(subj_path,timestamps_path,3, save = save_DM) #3 is the TR

        #-----Contrast-----
        if contrast_type == 'each_shocks':
            beta_map, contrast_path = contrasts.glm_contrast_1event(design_matrices,
             fmri_time_series, path_contrast_results,subj_name, run_name = conditions)
        elif contrast_type == 'neut_shocks':
            beta_map, contrast_path = contrasts.glm_contrast_N_shocks(design_matrices,
             fmri_time_series, path_contrast_results,subj_name, run_name = conditions)
        elif contrast_type == 'suggestions':
            pass#***in developpement***
        elif contrast_type == 'all_shocks': #default = all_shocks
            beta_map, contrast_path = contrasts.glm_contrast_all_shocks(design_matrices,
             fmri_time_series, path_contrast_results, subj_name, run_name = conditions)
        else:
            print('WARNING : skipping contrast')

        if contrast_type != None : #keep track of the contrasts' paths to save them
            contrast_paths.append(contrast_path)
        print('contrast_paths lenght : ', len(contrast_paths))


    #Second level analysis

#main path to data, change according to environment
#root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test'
#timestamps_root_path = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\SPM_multiple_condition_files_TxT'
#dir_to_save = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_csv'
#compute_DM = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_all_runs'

#elm
#root_dir = r'/data/rainville/dylan_projet_ivado_decodage/Nii'
#dir_to_save = r'/data/rainville/dylan_projet_ivado_decodage/results_GLM/'
#compute_DM = r'/data/rainville/dylan_projet_ivado_decodage/results_GLM/SPM_DM_timeseries'

timestamps_root_path = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\UM_H22\PSY3008\times_stamps'
compute_DM = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_timeseries'
dir_to_save = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM'
main(dir_to_save= dir_to_save,compute_DM = compute_DM, contrast_type = 'each_shocks')

