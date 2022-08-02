
import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, mean_img

def compute_DM(subj_path,timestamps_root_path,tr, save = None, runs = True):

    """
    Arguments
    ---------
    subj_path : Path to folders where the functionnal files are located
    timestamps_root_path : path to all the timestamps
    tr : In seconds
    runs : True by default. If False, the code needs to be adapted to a single run by changing the «str_analgesia ='Analgesia'»
    save : None by default, if not None, it's expected that a path to the location to save in provided. Then an inside folder will be created to save DM and timeseries

    Description
    -----------
    A function that returns all the data needed to compute a design matrix using the nilearn function make_first_level_design_matrix.

    """

    subj_name = os.path.basename(os.path.normpath(subj_path))#extract last part of subj_path to get subject's name
    #--------------
    #looking for the regressors' file, which starts with 'APM' and read it as csv
    movement_reg_name = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = os.path.join(subj_path,movement_reg_name[0])#movement_reg_name[0] because there's only one item in the list
    df_mvmnt_reg_full = pd.read_csv(mvmnt_reg_path, sep= '\s+', header=None)#full because we'll split it later according to condition

    #--------------
    #file names that contains the fMRI volumes.
    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'

    if runs:
        design_matrices = []
        all_fmri_timeseries = []
        conditions = []
        DM_names = []
        #In that loop, a Timestamps,a DM name,a mouvement regressors dataframe, a DM and statistical maps will be generated and saved
        for condition_file in [i for i in os.listdir(subj_path) if str_analgesia in i or str_hyper in i ]: #controls for each runs

            #-------Extracting fMRI volumes-------
            data_path = os.path.join(subj_path,condition_file) #path to the data such as : /subj_01/02-Analgesia/<all nii files>
            subj_volumes= glob.glob(os.path.join(data_path,'sw*'))#extracting all the nii files that start with sw
            print('=========================')
            print('{} NII files in path {} for subject {}'.format(len(subj_volumes),data_path,subj_name))
            print('lenght of movement regesssor df : {}'.format(len(df_mvmnt_reg_full)))

            #-------Extracting timestamps--------
            timestamps = get_timestamps(data_path, subj_name,timestamps_root_path,return_df =True)
            timestamps.sort_values(by=['onset'])

            #----condition and design matrix name-----
            condition, DM_name = if_str_in_file(condition_file,subj_name)#checks if str_analgesia or str_hyper is in condition_file

            #-------movement regessors--------
            if condition == 'HYPO':
                 df_mvmnt_reg = split_reg_upper(df_mvmnt_reg_full,len(subj_volumes))
            elif condition == 'HYPER':
                df_mvmnt_reg = split_reg_lower(df_mvmnt_reg_full,len(subj_volumes)) #splitting either the first half or lower half of the mvmnt regressor df according to condition (analg/hyper)

            #-------compute DM-------
            design_matrix, fmri_time_series = create_DM(subj_volumes, timestamps, DM_name, df_mvmnt_reg, subj_name, tr)
            #-------append--------
            design_matrices.append(design_matrix)
            all_fmri_timeseries.append(fmri_time_series)
            conditions.append(condition)
            DM_names.append(DM_name)

        #-----Save-----
        if save != None:
            subj_path_save = os.path.join(save,subj_name)
            if os.path.exists(subj_path_save) is False: #make the subj_path_to_save
                    os.mkdir(subj_path_save)
            else :
                pass

            for i in range(len(design_matrices)): #will save all the element in the design matrix, timeseries and conditions lists
                design_matrix = design_matrices[i]
                design_matrix.to_csv(os.path.join(subj_path_save,DM_names[i]), index = False)

                fmri_time_series = all_fmri_timeseries[i]
                fmri_img_name = subj_name + '_' + conditions[i] + '_fmri_time_series.nii.gz'
                nib.save(fmri_time_series, os.path.join(subj_path_save,fmri_img_name))

    return design_matrices, all_fmri_timeseries, conditions


def create_DM(subject_data, timestamps, DM_name, df_mvmnt_reg, subj_name, tr, save = None):

    """
    Description
    -----------
    A function that computes a design matrix using the nilearn function make_first_level_design_matrix.
    It returns a pandas design matrix and a 4D time series of the subject nii data.
    """
    print('COMPUTING design matrix under name : ' + DM_name)

    fmri_time_series = concat_imgs(subject_data)#Extraction of subject'volumes (4D nii file)
    #////////////TIMESTAMPS////////////////
    if type(timestamps) is str:

        timestamps = pd.read_excel(timestamps, header=None)
        #formatage des type des entrées et insertion de titres
        timestamps = pd.DataFrame.transpose(events)
        timestamps.rename(columns = {0:'onset', 1:'duration', 2:'trial_type'}, inplace = True)

    timestamps['onset'] = timestamps['onset'].astype(np.float64)
    timestamps['duration'] = timestamps['duration'].astype(np.float64)
    timestamps['trial_type'] = timestamps['trial_type'].astype(np.str)

    n_scans = len(subject_data)
    frame_times = np.arange(n_scans) * tr

    design_matrix = make_first_level_design_matrix(
                frame_times,
                timestamps,
                hrf_model='spm',
                drift_model='cosine',
                high_pass=.00233645,
                add_regs = df_mvmnt_reg) #array of shape(n_frames, n_add_reg)

    #--------Prints for info--------
    print('SHAPE of TIMESPTAMPS is {0}'.format(timestamps.shape))
    print('SHAPE of MOVEMENT REGRESSORS: {} '.format(df_mvmnt_reg.shape))
    print('SHAPE OF fMRI TIMESERIES : {} '.format(fmri_time_series.shape))
    print('SHAPE OF DESIGN MATRIX : {} '.format(design_matrix.shape))

    #--------plot option--------
    #from nilearn.plotting import plot_design_matrix
    #plot_design_matrix(design_matrix)
    #import matplotlib.pyplot as plt
    #plt.show()

    return design_matrix, fmri_time_series

##############################################33
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

        #Extraction des volumes pour un sujet dans une liste
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




def get_timestamps(data_path, subj_name, timestamps_path_root, return_df=None):

    """
    Parameters
    ----------

    data_path : path to fmri data in order to know in which conditions the subject is
    timestamps_path_root : root path to the timestamps
    subj_name : subject's name, e.g. APM_02_H2 to identify the particular cubjects (with different timestamps)
    return_df : if True, the function returns a pandas dataframe with the timestamps. If None, the path to timestamps will be returned

    Returns
    -------
    timestamps_path : a path to the timestamps file or if return_df =True, a pandas dataFrame which is named df_timestamps
    """


    #Read the file

#======================================

    #condition file is e.g. 02-Analgesia, it's the file that contains the fMRI volumes
    condition_file = os.path.basename(os.path.normpath(data_path))

    if 'Hyperalgesia' in condition_file: #need to return the right timestamps

        #TIMESTAMPS
        if subj_name == 'APM_02_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx')#csv option

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

            #timestamps HYPER pour les sujets normaux dans H2
        else :#For all other subjects
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
    #-------------Fonction description----------

    #function that takes a matrix and split it horizontally at the index given as argument (target_lenght). Returns the **LOWER** part
    #of where the matrix was split.

    #-----------------Variables-----------------
    #matrix_to_split : a matrix
    #target_lenght : the index at which you want the split matrix
    #--------------------------------------------


    #inverse_lenght = len(matrix_to_split) - target_lenght

    split_matrix = matrix_to_split.iloc[- target_lenght :]

    return split_matrix

def if_str_in_file(condition_file,subj_name):


    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'
#defining design matrix name and mouvement regessors according to condition
    if str_analgesia in condition_file:
        condition = 'HYPO'
        DM_name = 'DM_HYPO_' + subj_name + '.csv' #Initializing the name under which the design matrix will be saved

    else:
        condition = 'HYPER'
        DM_name = 'DM_HYPER_' + subj_name + '.csv'

    return condition,DM_name

def check_if_empty(dir):

    for folder in os.listdir(dir):
        ls = os.listdir(os.path.join(dir,folder))
        if len(ls) == 0:
            raise
