import numpy as np
import os
import pandas as pd
from os.path import exists
import glob as glob
import scipy.io
from sklearn.utils import Bunch


def load_data(data_dir, events_dir, key_str="APM"):
    """
    -------------Fonction description----------

    Function that takes a directory of file and runs through all the file in it whilst it only list the path of the file starting
    with the key_str given as argument

    Example : in a directory you have many nii files, but you only want to use a sub-sample of them, e.g. the ones starting with the key_str swaf*

    -----------------Variables-----------------
    data_dir: path to a directory containing all files
    key_str : prefix of the file of interest that you want to list the path
    if prefix = None, all the files of the dir will be put in the list

    --------------------------------------------
    """
    conditions = ["Analgesia", "Hyperalgesia"]
    data = Bunch(
        subjects=[sub for sub in os.listdir(data_dir) if key_str in sub],
        func_Ana=[
            glob.glob(os.path.join(data_dir, sub, "*" + conditions[0], "sw*"))
            for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
        ],
        func_Hyper=[
            glob.glob(os.path.join(data_dir, sub, "*" + conditions[1], "sw*"))
            for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
        ],
        anat=[
            glob.glob(os.path.join(data_dir, sub, "*MEMPRAGE", "*.nii"))
            for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
            if key_str in sub
        ],
        all_confounds=[
            pd.read_csv(paths, sep="\s+", header=None)
            for paths in [
                glob.glob(os.path.join(data_dir, sub, "*nuisreg*"))[0]
                for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
            ]
        ],
        events=Bunch(
            Analgesia=[
                get_timestamps(
                    data_dir, sub, events_dir, conditions[0], return_path=True
                )
                for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
            ],
            Hyperalgesia=[
                get_timestamps(
                    data_dir, sub, events_dir, conditions[1], return_path=True
                )
                for sub in [subj for subj in os.listdir(data_dir) if key_str in subj]
            ],
        ),
    )
    return data


def get_timestamps(
    data_path, subj_name, timestamps_path_root, condition_file, return_path=None
):
    """
    Parameters
    ----------

    data_path : path to fmri data in order to know in which conditions the subject is
    timestamps_path_root : root path to the timestamps
    subj_name : subject's name, e.g. APM_02_H2 to identify the particular cubjects (with different timestamps)
    return_path : if True, the function returns a pandas dataframe with the timestamps. If None, the path to timestamps will be returned

    Returns
    -------
    timestamps_path : a path to the timestamps file or if return_path =True, a pandas dataFrame which is named df_timestamps
    """

    # Read the file

    # ======================================
    if "Hyperalgesia" in condition_file:  # need to return the right timestamps
        # TIMESTAMPS
        if subj_name == "APM_02_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.mat",
                    ),
                    simplify_cells=True,
                )  # .mat option
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx",
                )  # csv option

        elif subj_name == "APM_05_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.mat",
                    ),
                    simplify_cells=True,
                )  # .mat option
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.xlsx",
                )

        elif subj_name == "APM_17_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.mat",
                    ),
                    simplify_cells=True,
                )  # .mat option
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.xlsx",
                )

        elif subj_name == "APM_20_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.mat",
                    ),
                    simplify_cells=True,
                )  # .mat option
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.xlsx",
                )

            # timestamps HYPER pour les sujets normaux dans H2
        else:  # For all other subjects
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_HYPER.mat",
                    ),
                    simplify_cells=True,
                )  # .mat option
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx",
                )

        # if we are in the Analgesia/hypoalgesia condition
    elif "Analgesia" in condition_file:
        if subj_name == "APM_02_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.mat",
                    ),
                    simplify_cells=True,
                )
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.xlsx",
                )

        elif subj_name == "APM_05_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.mat",
                    ),
                    simplify_cells=True,
                )
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.xlsx",
                )

        elif subj_name == "APM_17_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.mat",
                    ),
                    simplify_cells=True,
                )
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.xlsx",
                )

        elif subj_name == "APM_20_H2":
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.mat",
                    ),
                    simplify_cells=True,
                )
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root,
                    r"ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.xlsx",
                )

        # timestamps HYPO/ANA for other 'normal' subjects
        else:
            if return_path is False:
                timestamps = scipy.io.loadmat(
                    os.path.join(
                        timestamps_path_root,
                        r"ASTREFF_Model6_TxT_model3_multicon_ANA.mat",
                    ),
                    simplify_cells=True,
                )
            else:
                timestamps_path = os.path.join(
                    timestamps_path_root, r"ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx"
                )

    # -----return------
    # if the return is supposed to be a dataframe
    if return_path is False:
        df_timestamps = pd.concat(
            [
                pd.DataFrame(timestamps["onsets"]),
                pd.DataFrame(timestamps["durations"]),
                pd.DataFrame(timestamps["names"]),
            ],
            axis=1,
        )
        df_timestamps.columns = ["onset", "duration", "trial_type"]

        return df_timestamps
    # else return the path
    else:
        return timestamps_path


def split_conf_matrix(matrix_to_split, indices_lenght, run_name):
    if run_name == "Analgesia":
        split_matrix = matrix_to_split.iloc[0:indices_lenght, :]
    elif run_name == "Hyperalgesia":
        split_matrix = matrix_to_split.iloc[-indices_lenght:, :]

    return split_matrix


def if_str_in_file(condition_file, subj_name):
    str_analgesia = "Analgesia"
    str_hyper = "Hyperalgesia"
    # defining design matrix name and mouvement regessors according to condition
    if str_analgesia in condition_file:
        condition = "HYPO"
        DM_name = (
            "DM_HYPO_" + subj_name + ".csv"
        )  # Initializing the name under which the design matrix will be saved

    else:
        condition = "HYPER"
        DM_name = "DM_HYPER_" + subj_name + ".csv"

    return condition, DM_name


# def exctract_files(path, str_t = None):
# for file in [i for i in os.listdir(path) if type in i or str_hyper in i ]:
