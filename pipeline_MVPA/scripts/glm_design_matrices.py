
import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import pickle
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, mean_img

def compute_DM(subj_path, timestamps_path,tr, file_str = 'sw*', save_to = None, runs = True, verbose = True):

    """
    Arguments
    ---------
    subj_path : Path to folders where the functionnal files are located
    timestamps_path : path to all the timestamps
    tr : In seconds
    file_str : str
                str used to extract fmri data in folder. Use '*' if all files in folder are to be used
    runs : True by default. If False, the code needs to be adapted to a single run by changing the «str_analgesia ='Analgesia'»
    save : None by default, if not None, it's expected that a path to the location to save in provided. Then an inside folder will be created to save DM and timeseries

    Description
    -----------
    A function that returns all the data needed to compute a design matrix using the nilearn function make_first_level_design_matrix.

    """

    subj_name = os.path.basename(os.path.normpath(subj_path))#extract last part of subj_path to get subject's name
    # Movement regressor file
    mvmnt_reg_path = glob.glob(os.path.join(subj_path, '*nuisreg*'))[0]
    df_mvmnt_reg_full = pd.read_csv(mvmnt_reg_path, sep= '\s+', header=None)# full because we'll split it later according to condition

    # --File names in 'subj_path' that contains the fMRI volumes of each runs--
    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'

    if runs:
        runs_fmri_imgs = []
        runs_timestamps =[]
        runs_confounds = []
        design_matrices = []
        cond_ls = [i for i in os.listdir(subj_path) if str_analgesia in i or str_hyper in i ]
        cond_ls.sort()

        # --iterates over runs to stack volumes, timestamps, confounds... --
        for condition_file in cond_ls: # Controls for each runs
            condition = condition_file[3:] # *initial folder name has '00-' before condition name, hence 3 characters

            #-------Extracting fMRI volumes-------
            run_path = os.path.join(subj_path,condition_file) # path to the data such as : /subj_01/02-Analgesia/<all nii files>
            subj_volumes = glob.glob(os.path.join(subj_path,condition_file, file_str)) # extracting all the nii files that start with sw
            runs_fmri_imgs.append(subj_volumes)

            #-------Extracting timestamps--------
            timestamps = get_timestamps(condition, subj_name, timestamps_path, return_df =True).sort_values(by=['onset'])
            runs_timestamps.append([timestamps])

            #-------movement regressors--------
            if condition == 'Analgesia':
                 df_mvmnt_reg = split_reg_upper(df_mvmnt_reg_full,len(subj_volumes))
            elif condition == 'Hyperalgesia':
                df_mvmnt_reg = split_reg_lower(df_mvmnt_reg_full,len(subj_volumes)) #splitting either the first half or lower half of the mvmnt regressor df according to condition (analg/hyper)
            runs_confounds.append(df_mvmnt_reg)

        # --Concatenate fmri imgs, timestamps and confunds for all runs to compute GLM--
        fmri_imgs = concat_imgs(runs_fmri_imgs)
        print('fmri_imgs shapes', fmri_imgs.shape)

        # --Adjust timestamps from different runs--
        timestamps = runs_timestamps[0] # starts with the 1st df to ajust timing in other dfs
        for i, df in enumerate(runs_timestamps[1:], start = 1): # starts at the 2d df
            max_time = timestamps['onset'].max()
            # Modifies the ongoing timestamps with adjusted time
            df['onset'] = df['onset'].apply(lambda x : x + max_time)
            timestamps = pd.concat([timestamps,df])

        # --Confounds--
        confounds = pd.concat([ele for ele in runs_confounds])

        conditions = '_'.join([str(n[3:]) for n in cond_ls])
        DM_name = 'DM_' + subj_name + conditions + '.pkl'
        design_matrix = create_DM(fmri_imgs, timestamps, DM_name, confounds, subj_name, tr)

        #-----Save-----
        if save_to != None:
            if os.path.exists(os.path.join(save_to, subj_name)) is False: #make the subj_path_to_save
                    os.mkdir(os.path.join(save_to, subj_name))
            else :
                pass
            design_matrix.to_pickle(os.path.join(save_to, subj_name, DM_name))
            fmri_img_name = subj_name + '_fmri_time_series.nii'
            nib.save(fmri_imgs, os.path.join(save_to, subj_name ,fmri_img_name))

    return  design_matrix, fmri_imgs, conditions


def create_DM(fmri_imgs, timestamps, DM_name, confounds, subj_name, tr, save = None, verbose = False):

    """
    Description
    -----------
    A function that computes a design matrix using the nilearn function make_first_level_design_matrix.
    It returns a pandas design matrix and a 4D time series of the subject nii data.
    """

    # TIMESTAMPS
    if type(timestamps) is str: # e.i. if a path is provided instead of a df
        timestamps = pd.read_excel(timestamps, header=None)
        timestamps = pd.DataFrame.transpose(events)
        timestamps.rename(columns = {0:'onset', 1:'duration', 2:'trial_type'}, inplace = True)


    n_scans = fmri_imgs.shape[3] # pos '3' as shape is 4 dim
    #frame_times = np.arange(n_scans) * tr
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    design_matrix = make_first_level_design_matrix(
                frametimes,
                timestamps,
                hrf_model='spm',
                drift_model='cosine',
                high_pass= 0.00233645,
                add_regs = confounds) #array of shape(n_frames, n_add_reg)

    if verbose:
    #--------Prints for info--------
        print('COMPUTING design matrix under name : ' + DM_name)
        print('SHAPE of TIMESPTAMPS is {0}'.format(timestamps.shape))
        print('SHAPE of MOVEMENT REGRESSORS: {} '.format(confounds.shape))
        print('SHAPE OF fMRI TIMESERIES : {} '.format(fmri_imgs.shape))
        print('SHAPE OF DESIGN MATRIX : {} '.format(design_matrix.shape))

    #--------plot option--------
        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(design_matrix)
        import matplotlib.pyplot as plt
        #plt.show()

    return design_matrix

############################################
#TO FIX with module
############################################

import numpy as np
import os
import pandas as pd
from os.path import exists
import scipy.io

def get_subj_data(data_dir, prefix = None):

    """
    -------------Fonction description----------

    Function that takes a directory of file and runs through all the file in it whilst it only list the path of the file starting
    with the prefix given as argument

    Example : in a directory you have many nii files, but you only want to use a sub-sample of them, e.g. the ones starting with the prefix swaf*

    -----------------Variables-----------------
    data_dir: path to a directory containing all files
    prefix : prefix of the file of interest that you want to list the path
    if prefix = None, all the files of the dir will be put in the list

    --------------------------------------------
    """


    #if prefix = None, all the files of the dir will be put in the list
    if prefix == None:

        ls_volumes_all = os.listdir(data_dir)

        #Crée une liste avec seulement les fichiers commencant par sw
        all_list = [x for x in ls_volumes_all]
        #print(swaf_list)

        #joindre le path avec les noms des volumes dans une liste
        #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
        subject_data = [os.path.join(data_dir, name) for name in all_list]


        return subject_data

    if prefix != None:
        #Extraction des volumes pour un sujet dans une liste
        ls_volumes_all = os.listdir(data_dir)

        #Crée une liste avec seulement les fichiers commencant par sw
        swaf_list = [x for x in ls_volumes_all if x.startswith(prefix)]
        #print(swaf_list)

        #joindre le path avec les noms des volumes dans une liste
        #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
        subject_data = [os.path.join(data_dir, name) for name in swaf_list]


        return subject_data




def get_timestamps(condition_file, subj_name, timestamps_path_root, return_df=None):

    """
    Parameters
    ----------

    condition_file : name of the folder containing run or condition name to extract the proper timestamps file
    timestamps_path_root : root path to the timestamps
    subj_name : subject's name, e.g. APM_02_H2 to identify the particular cubjects (with different timestamps)
    return_df : if True, the function returns a pandas dataframe with the timestamps. If None, the path to timestamps will be returned

    Returns
    -------
    timestamps_path : a path to the timestamps file or if return_df =True, a pandas dataFrame which is named df_timestamps
    """


    #Read the file

#======================================

    if 'Hyperalgesia' in condition_file: #need to return the right timestamps

        #TIMESTAMPS
        if subj_name == 'APM_02_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx')# csv option

        elif subj_name == 'APM_05_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.xlsx')

        elif subj_name == 'APM_17_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.xlsx')

        elif subj_name == 'APM_20_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.xlsx')

            #timestamps HYPER for all other subjects H2
        else :
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_HYPER.mat'),simplify_cells =True)#.mat option
            else :
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx')

        #if we are in the Analgesia/hypoalgesia condition
    elif 'Analgesia' in condition_file:

        if subj_name == 'APM_02_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.xlsx')

        elif subj_name == 'APM_05_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.xlsx')

        elif subj_name == 'APM_17_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.xlsx')

        elif subj_name == 'APM_20_H2':
            if return_df:
                timestamps =scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.xlsx')

        #timestamps HYPO/ANA for other 'normal' subjects
        else :
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx')

    #-----return------
    #if the return is supposed to be a dataframe
    if return_df :

        df_timestamps = pd.concat([pd.DataFrame(timestamps['onsets']),pd.DataFrame(timestamps['durations']),pd.DataFrame(timestamps['names'])], axis = 1)
        df_timestamps.columns = ['onset', 'duration','trial_type']

        df_timestamps['onset'] = df_timestamps['onset'].astype(np.float64)
        df_timestamps['duration'] = df_timestamps['duration'].astype(np.float64)
        df_timestamps['trial_type'] = df_timestamps['trial_type'].astype(np.str)

        return df_timestamps
    #else return the path
    else:

        return timestamps_path



def split_reg_upper(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)
    #-------------Fonction description----------

    #function that takes a matrix and split it horizontally at the index given as argument (target_lenght). Returns the **UPPER** part
    #of where the matrix was split.

    #-----------------Variables-----------------
    #matrix_to_split : a matrix
    #target_lenght : the index at which you want the split matrix
    #--------------------------------------------

     #on split horizontalement la matrice en une tranche de
     split_matrix = matrix_to_split.iloc[0:target_lenght, :]

     return split_matrix


def split_reg_lower(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split) selon le nombre de volumes qu'on donne en argument (target_lenght)
    #-------------Function description----------

    #function that takes a matrix and split it horizontally at the index given as argument (target_lenght). Returns the **LOWER** part
    #of where the matrix was split.

    #-----------------Variables-----------------
    #matrix_to_split : a matrix
    #target_lenght : the index at which you want the split matrix
    #--------------------------------------------


    #inverse_lenght = len(matrix_to_split) - target_lenght

    split_matrix = matrix_to_split.iloc[- target_lenght :]

    return split_matrix

def if_str_in_file(condition_file, subj_name):

    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'
    # defining design matrix name and mouvement regessors according to condition
    if str_analgesia in condition_file:
        condition = 'HYPO'
        DM_name = 'DM_HYPO_' + subj_name + '.pkl' #'.csv' #Initializing the name under which the design matrix will be saved

    else:
        condition = 'HYPER'
        DM_name = 'DM_HYPER_' + subj_name + '.pkl' #.csv'

    return condition, DM_name


def check_if_empty(dir):

    for folder in os.listdir(dir):
        ls = os.listdir(os.path.join(dir,folder))
        if len(ls) == 0:
            raise


def load_pkl_to_pd(ls_pkl_paths):
    #--load design matrices from pkl to pandas-----
    ls_pandas = []
    for i in range(len(ls_pkl_paths)):
        tmp = pd.read_pickle(ls_pkl_paths[i])
        ls_pandas.append(tmp)

    return ls_pandas

