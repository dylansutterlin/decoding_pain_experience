import numpy as np
import os
import pandas as pd
import glob
import nibabel as nib
from nilearn.plotting import plot_design_matrix
from nilearn.image import concat_imgs, mean_img
from nilearn.maskers import NiftiMasker
import matplotlib.pyplot as plt
from scripts import glm_design_matrices as glm
from scripts import glm_manip_data as manip_data
from nilearn.glm.first_level import FirstLevelModel

# from scripts import glm_contrasts as stats


def main(
    data_dir,
    run_names=None,
    events_dir=None,
    dir_to_save=None,
    save_design_matrix=True,
    contrast_type=None,
    parser=True,
    compute_DM=True,
    dot_with="NPS",
    max_iter=None,
    verbose=True,
):
    """
    Arguments
    --------

    data_dir : String, path to the root directory to all the subject's fmri volumes. This script is built assuming that root dir is a dir with a folder for each participant
        directory to all the subject's fmri volumes. This script is built assuming that root dir is a dir with a folder for each participant.


    events_dir : Path to all the timestamps in a folder. The script identify which timestamps to choose based on its name. Can be .mat or .csv
    dir_to_save : Directory to save the statistical contrast maps. It will only be used if contrast_type is not 'None'.
    contrast_type : Type of contrast to compute. Choices=['all_shocks','each_shocks','neut_shocks', 'suggestions']. By default None.
    parser : True by default. If False, parser.args will be overlooked and user will be able to manually provide the necessary arguments while calling main().
    compute_DM : True by default but if != True, a path to the design matrices and the concatenated fmri timeseries is expected.
    dir_to_save_DM : Path where the design matrices and fmri timeseries will be saved. Only taken into account if compute_DM = True.
    ####second_level : Bool., by default True.
        Runs a second level GLM

    Example
    -------
    main(dir_to_save= dir_to_save,compute_DM = True, contrast_type = None)

    """

    # --------Parser--------
    # Argument parser
    if __name__ == "__main__":
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--dir_to_save", type=str)
        parser.add_argument("--events_dir", type=str)
        parser.add_argument("--many_runs", type=str)
        parser.add_argument(
            "--contrast_type",
            type=str,
            choices=["all_shocks", "each_shocks", "suggestions"],
            default="all_shocks",
        )
        args = parser.parse_args()

    # --------Variables--------
    if run_names is None:
        run_names = ["run1"]
    else:
        run_names = ["Analgesia", "Hyperalgesia"]
    if events_dir != None:
        data = manip_data.load_data(data_dir, events_dir)
    results = dict()
    # else:
    ##design_matrices, timeseries = load_DM_series()

    # ---Creating saving directory---
    if contrast_type != None:
        results_path = os.path.join(dir_to_save, "contrast_" + contrast_type)
        if os.path.exists(results_path) is False:
            os.mkdir(results_path)
    if save_design_matrix == True:
        save_design_matrix = os.path.join(dir_to_save, "design_matrices_TxT")
        if os.path.exists(save_design_matrix) is False:
            os.mkdir(save_design_matrix)

    # First level model
    for i, run in enumerate(range(0, len(run_names)), start=0):
        # --GLM parameters--
        masker = NiftiMasker(standardize="psc", smoothing_fwhm=6)
        fmri_glm = FirstLevelModel(
            t_r=3,
            noise_model="ar1",
            standardize=False,
            slice_time_ref=0.5,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=0.00233645,
        )

        # ---!! always starting with Analgeria --> adapt timestamps and confounds. events

        fmri_timeseries = data.get(
            "func_{}".format(run_names[i][:3])
        )  # func_Ana / func_Hyper
        breakpoint()
        events = data.events.get(run_names[i])
        confounds = manip_data.split_conf_matrix(
            data.all_confounds[i], len([fmri_timeseries[i]]), run_names[i]
        )
        results[run_names[i]] = {
            "timeseries": fmri_timeseries,
            "events": events,
            "confounds": confounds,
        }

        for idx, sub in enumerate(data.subjects):
            # -- GLM fit  for each subject --
            X = masker.fit_transform(fmri_timeseries[idx])  # should add confound ?
            fmri_glm = fmri_glm.fit(
                masker.inverse_transform(X),
                events=events[idx],
                confounds=confounds[idx],
            )
            design_matrix = fmri_glm.design_matrices_[0]
            results[run_names[i]][sub] = {
                "design_matrix": design_matrix,
                "glm": fmri_glm,
            }

            # --Quality check--
            mean_img = X
            plot_glass_brain(mean_img, title="mean_img_{}".format(run_names[i]))
            plotting.show()
            from nilearn.plotting import plot_design_matrix

            plot_design_matrix(design_matrix)
            import matplotlib.pyplot as plt

            plt.show()

            breakpoint()

            # -- save design matrix --
            if save_design_matrix != False:
                if not os.path.exists(os.path.join(save_design_matrix, sub)):
                    os.makedirs(os.path.join(save_design_matrix, sub))
                design_matrix.to_csv(
                    os.path.join(
                        save_design_matrix,
                        sub,
                        "design_matrix_{}.csv".format(run_names[i]),
                    )
                )
                # save.plot_design_matrix(design_matrix)

            contrast_type = None
            # -- compute contrast --
            if contrast_type != None:
                # -- compute contrast --
                if contrast_type == "all_shocks":
                    contrast_matrix = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                elif contrast_type == "each_shocks":
                    contrast_matrix = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                elif contrast_type == "suggestions":
                    contrast_matrix = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                contrast = fmri_glm.compute_contrast(
                    contrast_matrix, output_type="z_score"
                )
                # -- save contrast maps --
                if os.path.exists(results_path) is False:
                    os.mkdir(results_path)
                save_path = os.path.join(results_path, sub)
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                contrast.to_filename(
                    os.path.join(save_path, "contrast_{}.nii.gz".format(run_names[i]))
                )

            # -- save contrast maps --
            glm.check_if_empty(results_path)
            save_DM = os.path.join(dir_to_save, "DM_timeseries")
            if os.path.exists(results_path) is False:
                os.mkdir(save_DM)
            design_matrix, fmri_time_series, conditions = glm.compute_DM(
                subj_path, events_dir, 3, save_to=save_DM
            )  # 3 is the TR
            paths_design_matrices = glob.glob(
                os.path.join(compute_DM, subj_name, "DM*csv")
            )  # Assumes that the design matrix file has DM in its filename
            # design_matrices = DM.load_pkl_to_pd(paths_design_matrices)
            design_matrices = pd.read_csv(paths_design_matrices[0], index_col=[0])
            fmri_time_series = glob.glob(
                os.path.join(compute_DM, subj_name, "*fmri*")
            )  # assuming that the 4D timeseries contains 'fmri' in its name
            conditions = "hyper_hypo"


"""
        else:
            print("WARNING : skipping contrast")
            
        if verbose:
            print("Run  : {} : ".format(run_names[i])

        if contrast_type != None:  # keep track of the contrasts' paths to save them
            contrast_paths.append(contrast_path)

    # Second level analysis

"""
# Local
data = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test"
timestamps_root_path = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\SPM_multiple_condition_files_TxT"
# compute_DM = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_GLM\SPM_DM_timeseries'
dir_to_save = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_GLM\test_res_GLM"
# server
# data = r'/home/p1226014/projects/def-rainvilp/p1226014/data/desmarteaux2021/Nii'
# timestamps_root_path = r'/home/p1226014/projects/def-rainvilp/p1226014/data/desmarteaux2021/All_run_timestamps'
# dir_to_save = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/py_all_shocks'

main(
    data_dir=data,
    dir_to_save=dir_to_save,
    events_dir=timestamps_root_path,
    run_names=["Analgesia", "Hyperalgesia"],
    compute_DM=True,
    contrast_type="all_shocks",
    max_iter=None,
)
