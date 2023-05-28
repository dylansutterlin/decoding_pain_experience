
#design_matrix = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test\APM_03_H1\DM_HYPER_APM_03_H1.npy'
import pandas as pd
import os
import numpy as np
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, mean_img
import nibabel as nib
from nilearn.plotting import plot_design_matrix


####manual test#######
#design_matrix = pd.read_csv(r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock\APM_02_H2\DM_HYPER_APM_02_H2.csv')
#subj_result_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_mean_shock'
#subj_name = 'APM_02_H2'
#all_runs_fmri_img = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock\APM_02_H2\APM_02_H2_HYPER_concat_fmri.nii'
######################

#-----------------------------
def glm_contrast_1event(design_matrices,all_runs_fmri_img, dir_to_save, subj_name, run_name = None, output_type = 'effect_size'):
    """
    #====================
    #Function that takes a design matrix, a path to save, a subject name, a run_name name and a 4D nii file to conpute contrast
    #This function is supposed to be used in a loop over many subject file. Therefor, it has as arguments a path to save, a subject's
    #name and a run name to save the file under a name that will take into account such information.

    #Arguments
    #----------
    #design_matrices_paths : A list with all the DM path. Takes csv files
    #dir_to_save : a path where you want to save the contrast
    #subj_name : name of the subject
    #fmri_img : a 4D nii file containing data
    #run_name : *optionnal. A string containing the name of the actual run that will eb added to the name of the saved contrast file.
        #If not specified, no run string will be included in the saved contrast name
    output_type : String, default = 'effect_size' i.e. beta
        Argument (named output_type) to insert in fmri_glm.compute_contrast function.
    """

    all_runs_fmri_img_name = subj_name + '_concat_fmri.nii'

    #-------Model--------
    print('==============')
    print('COMPUTING GLM for subject ' + subj_name)

    fmri_glm = FirstLevelModel(t_r=3,
                               noise_model='ar1',
                               standardize=False,
                               slice_time_ref = 0.5,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass= 0.00233645)
    fmri_glm = FirstLevelModel(t_r=3,
                               noise_model='ar1',
                               standardize=False,
                               slice_time_ref = 0.5,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass= 0.00233645)

    fmri_glm = fmri_glm.fit(all_runs_fmri_img, design_matrices = design_matrices)

    #==============
    #identity matrix having shape of number of regressor x number of regressor in the Design matrix
    #Each column will serve to encode a 1 in a specifi columns of interest
    #none contrast is a dictionnary having design matrix column name as key name. The values of each
    #vector is a 'number of regressor'(design_matrix.shape[1]) long.
    #the vector in each key is 0 or 1 and serve to encode the contrast in further steps
    identity_matrix = np.eye((design_matrices.shape)[1],(design_matrices.shape)[1])
    null_matrix = np.zeros(design_matrices.shape)

    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrices.columns)])
    contrast_vector = np.zeros(((design_matrices.shape)[1]))#ones will be added to this vector as we specify which regressor we want to contrast

    #list of all the regressors/keys to keep track of the regressors we've added to contrast
    ls_keys = []
    key_list = list(none_contrasts)
    indx = 0
    #---prep saving----
    subj_result_path = os.path.join(dir_to_save,subj_name)
    if os.path.exists(subj_result_path) is False:
        os.mkdir(subj_result_path)
    else :
        pass
    for keys in none_contrasts:
        #if the key is str, in order to exclude the drifts and other parameters
        if type(keys) is str:
            string_interest = 'shock' #In this case every key that has the string 'shock' will be attributed a 1 in that column

            if string_interest in keys:
                actual_key_name = key_list[indx]
                ls_keys.append(keys)#keep track of the regressors for which contrast has been done
                #contrast_vector += identity_matrix[:, indx] #sum of the contrast vector with the identity matrix column to stack the ones
                                                            #in the contrast vector for all regressors of interest
                contrast_vector = identity_matrix[:, indx]
                #--------computing contrast---------
                print('==============')
                print('COMPUTING CONTRAST')
                print('With the CONTRAST VECTOR : {} '.format(contrast_vector))
                stat_map = fmri_glm.compute_contrast(
                     contrast_vector,stat_type = 't', output_type=output_type)#compute the contrasts with contrast vector

                #---------Plot option----------
                #plotting.plot_stat_map(
                #    stat_map, threshold=3.0, display_mode='z',
                #    cut_coords=3, black_bg=True, title=actual_key_name)
                #plotting.show()

                #-------SAVING OUTPUT-------
                if run_name == None:#control if a run_name string has been provided to include in the name of the file to save
                    name_to_save = 'stat_map_' + subj_name + '_' + actual_key_name
                    contrast_path = os.path.join(subj_result_path, name_to_save)
                    nib.save(stat_map, contrast_path)

                if run_name != None:
                    name_to_save = 'stat_map_' + subj_name + '_' + run_name + actual_key_name
                    contrast_path = os.path.join(subj_result_path, name_to_save)
                    nib.save(stat_map, contrast_path)
                print('Have saved stat_map as a : {} , having shape : {} , under name : {}'.format(type(stat_map),stat_map.shape, name_to_save))
        indx +=1
    return stat_map,contrast_path

#--------------------------
def glm_contrast_N_shocks(design_matrices,all_runs_fmri_img, dir_to_save, subj_name, run_name = None, output_type = 'effect_size'):
    """
    #====================
    #Function that takes a design matrix, a path to save, a subject name, a run_name name and a 4D nii file to conpute contrast
    #This function is supposed to be used in a loop over many subject file. Therefor, it has as arguments a path to save, a subject's
    #name and a run name to save the file under a name that will take into account such information.

    #Arguments
    #----------
    #design_matrices_paths : A list with all the DM path. Takes csv files
    #dir_to_save : a path where you want to save the contrast
    #subj_name : name of the subject
    #fmri_img : a 4D nii file containing data
    #run_name : *optionnal. A string containing the name of the actual run that will eb added to the name of the saved contrast file.
        #If not specified, no run string will be included in the saved contrast name
    output_type : String, default = 'effect_size' i.e. beta
        Argument (named output_type) to insert in fmri_glm.compute_contrast function.
    """
    fmri_glm = FirstLevelModel(t_r=3,
                               noise_model='ar1',
                               standardize=False,
                               slice_time_ref = 0.5,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass= 0.00233645)
    from nilearn.plotting import plot_design_matrix
    plot_design_matrix(design_matrices)
    import matplotlib.pyplot as plt
    plt.show()

    all_runs_fmri_img_name = subj_name + '_concat_fmri.nii'
    #-------Model--------
    print('==============')
    print('COMPUTING GLM for subject ' + subj_name)
    

    fmri_glm = fmri_glm.fit(all_runs_fmri_img, design_matrices = design_matrices)
    print('has fit')
    #==============
    identity_matrix = np.eye((design_matrices.shape)[1],(design_matrices.shape)[1])
    null_matrix = np.zeros(design_matrices.shape)
    #print(null_matrix.shape), print('NULL MATRIX')

    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrices.columns)])
    #print('NONE CONTRAST  {}'.format(none_contrasts))
    #print('lenght of NONE CONTRAST  {}'.format(len(none_contrasts)))
    contrast_vector = np.zeros(((design_matrices.shape)[1]))#ones will be added to this vector as we specify which regressor we want to contrast

    #list of all the regressors/keys to keep track of the regressors we've added to contrast
    ls_keys = []
    key_list = list(none_contrasts)
    indx = 0
    for keys in none_contrasts:

        #if the key is str, in order to exclude the drifts and other parameters
        if type(keys) is str:
            string_interest1 = 'N_'
            string_interest2 = 'shock'

            if string_interest1 in keys and string_interest2 in keys:
                actual_key_name = key_list[indx]
                ls_keys.append(keys)
                contrast_vector += identity_matrix[:, indx]
        indx += 1
    print(ls_keys, ' : All the keys added to contrast')
    #--------computing contrast---------
    print('==============')
    print('COMPUTING CONTRAST')
    print('With the CONTRAST VECTOR : {} '.format(contrast_vector))
    stat_map = fmri_glm.compute_contrast(contrast_vector, stat_type = 't', output_type= output_type)#compute the contrasts with contrast vector
    print('has done contrast')
     #---------Plot option----------
    #plotting.plot_stat_map(
    #    stat_map, threshold=3.0, display_mode='z',
    #    cut_coords=3, black_bg=True, title=actual_key_name)
    #plotting.show()

     #-------SAVING OUTPUT-------
    subj_result_path = os.path.join(dir_to_save,subj_name)

    if os.path.exists(subj_result_path) is False:
                   os.mkdir(subj_result_path)
    else :
        pass

    if run_name == None:#control if a run_name string has been provided to include in the name of the file to save
        name_to_save = 'stat_map_' + subj_name  + '_Neut_shocks'
        contrast_path = os.path.join(subj_result_path, name_to_save)
        nib.save(stat_map, contrast_path)

    if run_name != None:
        name_to_save = 'stat_map_' + subj_name + '_' + run_name + '_Neut_shocks'
        contrast_path = os.path.join(subj_result_path, name_to_save)
        nib.save(stat_map, contrast_path)
    print('Have saved stat_map as a : {} , having shape : {} , under name : {}'.format(type(stat_map),stat_map.shape, name_to_save))

    return stat_map,contrast_path

#-----------------------------
def glm_contrast_all_shocks(glm_model = None, all_runs_fmri_img, save_to, subj_name, run_name = None, stat_type = 't', output_type = 'effect_size', verbose = True):
    """

    #Function that takes a design matrix, a path to save, a subject name, a run_name name and a 4D nii file to conpute contrast
    #This function is supposed to be used in a loop over many subject file. Therefor, it has as arguments a path to save, a subject's
    #name and a run name to save the file under a name that will take into account such information.

    #Arguments
    #----------
    #design_matrices_paths : A list with all the DM path. Takes csv files
    #dir_to_save : a path where you want to save the contrast
    #subj_name : name of the subject
    #fmri_img : a 4D nii file containing data
    #run_name : *optionnal. A string containing the name of the actual run that will eb added to the name of the saved contrast file.
        #If not specified, no run string will be included in the saved contrast name
    output_type : String, default = 'effect_size' i.e. beta
        Argument (named output_type) to insert in fmri_glm.compute_contrast function.
    """


    if glm_model == None:
        
    all_runs_fmri_img_name = subj_name + '_concat_fmri.nii'
    #-------Model--------

    fmri_glm = FirstLevelModel().fit(all_runs_fmri_img, design_matrices = design_matrix)

    #==============
    identity_matrix = np.eye((design_matrix.shape)[1],(design_matrix.shape)[1])
    null_matrix = np.zeros(design_matrix.shape)
    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrix.columns)])
    contrast_vector = np.zeros(((design_matrix.shape)[1])) # ones will be added to this vector as we specify which regressor we want to contrast

    # list of all the regressors/keys to keep track of the regressors we've added to contrast
    ls_keys = []
    key_list = list(none_contrasts)

    for indx, keys in enumerate(none_contrasts):
        #if the key is str, in order to exclude the drifts and other parameters
        if type(keys) is str:
            string_interest = 'shock' #In this case every key that has the string 'shock' will be attributed a 1 in that column
            if string_interest in keys:#if the keys contains the word 'shock'
                actual_key_name = key_list[indx]
                ls_keys.append(keys)#extract the actual regressor/key name with the index position
                contrast_vector += identity_matrix[:, indx] #sum of the contrast vector with the identity matrix column to stack the ones
                                                            #in the contrast vector for all regressors of interest

    #--------computing contrast---------
    stat_map = fmri_glm.compute_contrast(
         contrast_vector,  stat_type = stat_type, output_type= output_type)#compute the contrasts with contrast vector

    #---------Plot option----------
    #plotting.plot_stat_map(
        #stat_map, threshold=3.0, display_mode='z',
        #cut_coords=3, black_bg=True, title=actual_key_name)
    #plotting.show()

    #-------SAVING OUTPUT-------
    if os.path.exists(os.path.join(save_to, 'glm_all_shocks', subj_name)) is False:
        os.mkdir(os.path.join(save_to, 'glm_all_shocks', subj_name))
    else :
        pass

    if run_name == None:#control if a run_name string has been provided to include in the name of the file to save
        name_to_save = stat_type + '_map_' + subj_name  + '_all_shocks'
        contrast_path = os.path.join(save_to, 'glm_all_shocks', subj_name, name_to_save)
        nib.save(stat_map, contrast_path)

    if run_name != None:
        name_to_save = stat_type + '_map_' + subj_name + '_' + run_name + '_all_shocks'
        contrast_path = os.path.join(save_to, subj_name, name_to_save)
        nib.save(stat_map, contrast_path)

    if verbose:
        print('==============')
        print('COMPUTING GLM and contrast for subject ' + subj_name)
        print('Have saved stat_map as a : {} , having shape : {} , under name : {}'.format(type(stat_map),stat_map.shape, name_to_save))

    return stat_map, contrast_path


def change_events_name(matrix,str_to_keep):

    for col in matrix.columns:
        if str_to_keep in col:
            new_col = col[6:-6]
            matrix.rename(columns = {col:new_col}, inplace = True)

    return matrix
