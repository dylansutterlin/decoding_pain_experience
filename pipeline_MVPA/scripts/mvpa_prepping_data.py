import numpy as np
import pandas as pd
import nibabel as nib
import glob
import os
from nilearn.masking import apply_mask
from nilearn.maskers import NiftiMasker
from nilearn.image import resample_img
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch


# desired template : subject_data .func .anat .events
def extract_data(path, extract_str="beta*", folder_per_participant=True):
    """
    Function that returns a list containing the paths of all the files in path, even its subfolder based
    on the filter extract_str.

    arguments
    ---------
    path : Str. ; Path to all the file data
    extract_str : Str. ; Will select only the file that satisfies the wildcard
    folder_per_participant : Bool. ; Assume that the path to data contains a folder for each participant containing all activation maps for this participant.
                            this is taken into account when encoding the gr, e.i. whichs maps are assign to which sub.
    """

    path, dirs, files = next(os.walk(path))
    # want to return a list of all the nii file from different folders

    data = []
    gr = []
    group_indx = 1
    for dir in dirs:
        tmp_folder = glob.glob(os.path.join(path, dir, extract_str))
        for item in tmp_folder:
            data.append(item)
            gr.append(
                group_indx
            )  # gr reflects which path or trial is associated with each participant (1 to n participants)
        group_indx += 1
    filenames = [os.path.basename(os.path.normpath(file)) for file in data]

    if folder_per_participant != True:  # manualy getting gr to work
        names = set(
            [w[0:6] for w in filenames]
        )  # get substrings of filenames to obtain e.g. 'APM_02', 'APM_05'..
        new_gr = np.zeros(len(data), dtype=int)

        for i_gr, subj_name in enumerate(names, 1):
            for i_files, file in enumerate(data):
                if subj_name in file:
                    new_gr[i_files] = i_gr
        gr = new_gr.tolist()

    return data, gr, filenames


def keep_sub_data(data, gr, sub_data):
    """
    Function to keep a part only of the data and leave the rest unused. E.g. to build a model only on specific condition.

    arguments
    ---------
    data : List ; list containing the paths of all the files
    sub_data : List or str. ; List of string that are comprised in filenames. The filenames containing those str. will be kept
    """
    filt_data = []
    new_gr = []
    if sub_data[0] == "exception_ANAHYPER":
        to_exclude = ["_N_"]
        for idx, img in enumerate(data):
            res = [ele for ele in to_exclude if (ele not in img)]
            if res:
                filt_data.append(img)
                new_gr.append(gr[idx])
    else:
        for idx, img in enumerate(data):
            res = [
                ele for ele in sub_data if (ele in img)
            ]  # checks if element 'ele' of the sub_data list is in the img's filename
            if res:
                # if res if not empty, meaning that img path contain an element in sub_data, e.g. 'ANA' or 'N_ANA'
                filt_data.append(img)
                new_gr.append(gr[idx])
    filenames = [os.path.basename(os.path.normpath(file)) for file in filt_data]
    return filt_data, new_gr, filenames


def y_transformer(y, func=np.log1p):
    """
    Transform y using a specified function

    Parameters
    ----------
    y: variable to transform
    func: list of numpy transformations to apply to the variable

    Returns
    ----------
    df_y: DataFrame containing y and the transformed y according to the specified transformations
    """
    df_y = pd.DataFrame(y.tolist(), columns=["y"])

    for element in func:
        transformer = FunctionTransformer(element, validate=True)
        Y_transformed = transformer.fit_transform(y.reshape(-1, 1))
        Y_transformed = Y_transformed[:, 0]
        df_y[
            str(element).replace("<ufunc '", "").replace("'>", "")
        ] = Y_transformed.tolist()

    return df_y


def extract_signal(
    data, mask="whole-brain-template", runs=None, smoothing_fwhm=None, standardize=True
):
    """
    Apply a mask to extract the signal from the data and save the mask
    in a html format

    Parameters
    ----------
    data: list of Niimg-like objects
    mask: strategy to compute the mask. By default the gray matter is extracted based on the MNI152 brain mask
    standardize: strategy to standardize the signal. The signal is z-scored by default

    Returns
    ----------
    masker_all: mask of the data
    masker_gm: array containing the extracted signal

    See also NifitMasker documentation
    """
    masker_all = NiftiMasker(
        mask_strategy=mask,
        standardize=standardize,
        smoothing_fwhm=smoothing_fwhm,
        verbose=0,
    )

    masker_gm = masker_all.fit_transform(data)
    print("mean: ", round(masker_gm.mean(), 2), "\nstd: ", round(masker_gm.std(), 2))
    print("Signal size: ", masker_gm.shape)

    report = masker_all.generate_report()
    report.save_as_html("masker_report.html")

    return masker_all, masker_gm


def extract_signal_from_mask(data, mask):
    """
    Apply a pre-computed mask to extract the signal from the data

    Parameters
    ----------
    data: Niimg-like object
    mask: mask to apply to the data

    Returns
    ----------
    signal: extracted signal from mask

    See also nilearn masking documentation
    """
    affine = data[0].affine
    resample_mask = resample_img(mask, affine)
    signal = apply_mask(data, resample_mask, ensure_finite=True)
    # print(signal.shape, type(signal))

    return signal


def encode_classes(data, gr):
    # Y data
    y_colnames = ["filename", "target", "conditions", "run", "group"]
    df_target = pd.DataFrame(columns=y_colnames)
    index = 0
    for file in data:
        # filename col
        filename = os.path.basename(os.path.normpath(file))  # get file name from path
        df_target.loc[
            index, "filename"
        ] = filename  # add file to coord (index,'filnames')

        # encoding classes associated with each file in data
        if "ANA" in filename:
            if "N_ANA" in filename:
                target = 1  # hypo neutral
                cond = "N_Hypo"
                run = "1-Hypo"
            else:  # Hypo
                target = 2
                cond = "Hypo"
            run = "1-Hypo"
        else:  # hyper
            if "N_HYPER" in filename:
                target = 3
                cond = "N_HYPER"
            else:
                target = 4
                cond = "HYPER"
            run = "Hyper"
            # print('attributed : ', target, 'as target and :', cond, 'as condition')
            # print('-----------')
        df_target.loc[index, "target"] = target
        df_target.loc[index, "conditions"] = cond

        index += 1
    df_target["group"] = gr
    cond_target = ["1 = N_ANA", "2 = HYPO", "3 = N_HYPER", "4 = HYPER"]
    df_target.loc[index, "run"] = run

    return df_target, cond_target


def encode_manip_classes(data, gr):
    """
    Function to encode encode binary classes based on experimental manipulation. It is designed to encode the experimental condition or modulation (1) and
    the neutral condition (2) based on substring in fMRI images' filename
    argument
    Return a dataframe with Y info and cond_target which is a list of strings of what condition is attributed to each target
    --------
    data : List; List containing all the path to fMRI activation maps
    gr : List; List of int. to keep tract of all the files that are from the same participant. e.g. [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2...]
    """

    # Y data
    y_colnames = ["filename", "target", "conditions", "run", "group"]
    df_target = pd.DataFrame(columns=y_colnames)

    index = 0
    for file in data:
        # filename col
        filename = os.path.basename(os.path.normpath(file))  # get file name from path
        df_target.loc[
            index, "filename"
        ] = filename  # add file to coord (index,'filnames')

        # encoding classes associated with each file in data
        if "ANA" in filename:
            if "N_ANA" in filename:
                target = 2  # hypo neutral
                cond = "Neutral"

            else:  # Hypo
                target = 1
                cond = "Hypo"
            run = "1-Hypo"
        else:  # hyper
            if "N_HYPER" in filename:
                target = 2
                cond = "Neutral"
            else:
                target = 1
                cond = "Hyper"
            run = "2-Hyper"
            # print('attributed : ', target, 'as target and :', cond, 'as condition')
            # print('-----------')

        df_target.loc[index, "target"] = target
        df_target.loc[index, "conditions"] = cond
        df_target.loc[index, "run"] = run
        index += 1
    df_target["group"] = gr
    cond_target = ["1 = HYPO/HYPER", "2 = Neutrals"]

    return df_target, cond_target


def encode_runs_as_classes(data, gr):
    # Y data
    y_colnames = ["filename", "target", "conditions", "run", "group"]
    df_target = pd.DataFrame(columns=y_colnames)
    index = 0
    for file in data:
        # filename col
        filename = os.path.basename(os.path.normpath(file))  # get file name from path
        df_target.loc[
            index, "filename"
        ] = filename  # add file to coord (index,'filnames')

        # encoding classes associated with each file in data
        if "ANA" in filename:
            if "N_ANA" in filename:
                target = 1  # hypo neutral
                cond = "Neutral"

            else:  # Hypo
                target = 1
                cond = "hypo"
            run = "1-Hypo"
        else:  # hyper
            if "N_HYPER" in filename:
                target = 2
                cond = "Neutral"
            else:
                target = 2
                cond = "hyper"
            run = "2-Hyper"
        df_target.loc[index, "target"] = target
        df_target.loc[index, "conditions"] = cond
        df_target.loc[index, "run"] = run
        index += 1
    df_target["group"] = gr
    cond_target = ["1 = HYPO_run", "2 = HYPER_run"]

    return df_target, cond_target


def train_test_iso_split(data, X, Y, train_samp, random_state=30):
    """
    provided a list of conditions, e.g. ['ANA', 'N_ANA'] as training sample 'train_samp', this function will return X_train and Y_train exclusively composed
    of the conditions in train_samp. It will also return X_test and Y_test arrays composed of the remaining conditions in 'data'. This
    function should be used if you want to train on a isolation of the data and test on the other. Note that the conditions in train_samp
    need to be comprised in the filenames in 'data'
    """
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    y_train_gr_idx = []
    idx = np.arange(0, X.shape[0], 1, dtype=int)
    count = 0
    for file, train_idx, test_idx in zip(data, idx, idx):
        # print(X.shape)
        # print(file)
        # print(train_idx)
        # print(test_idx)

        res = [
            ele for ele in train_samp if (ele in file)
        ]  # Bool result of wether the file contains at least one condition from train_samp
        if res:
            X_train.append(X[train_idx])
            Y_train.append(Y[train_idx])
            y_train_gr_idx.append(test_idx)
        else:  # 'ANA' or 'N_ANA' in file
            X_test.append(X[test_idx])
            Y_test.append(Y[test_idx])
        # print(X_train, X_test, Y_train, Y_test)

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(Y_train),
        np.array(Y_test),
        np.array(y_train_gr_idx),
    )
