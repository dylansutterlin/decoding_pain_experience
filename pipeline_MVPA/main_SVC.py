import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scripts import mvpa_prepping_data as prepping_data
from scripts import mvpa_building_model as building_model
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit


def main_svc(save_path, data_input, subj_folders = True, sub_data = False,  which_train_data = False, kfold = 5, n_components_pca = 0.90, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'], cov_corr = True, binary = False, binary_fct = 'modulation', verbose = True):


    """
    This function serves to run a linear SVC on fmri data.
    arguments
    ---------

    data_input : String; Path to fmri activation maps. It will extract the path of all imgs in path and its subfolders that has 'beta*' in name

    subj_folders : Bool; Changes the way path to data files are extracted. If False, e.i. no sub01/*, sub02/*, etc. but e.g. cond1/*sub01, cond1/*sub02, an extra step is needed
		         to assign the same group id to each map of the same subject

    sub_data :  Bool or string; By default, the model is built with all the data (sub_data = False), but list of string can be given to select only a sample of data to build model.
                E.g. sub_data = ['cond1', 'cond2'], only the data filename containing 'cond1' and 'cond2 will included in the model (train and test). If = 'exception_ANAHYPER'

    which_train_data : Bool or string; By default, all the data is used to train/test the model (= False), but if a list of string is specified, the data having those strings in
                        their name will be selected to train the model and the rest of the data will be used to test the model. If which_train_data != False, binary = True automatically
                        E.g. which_train_data = ['cond1', 'cond2'], training is done with those conditions and the model is tested on 'cond3' and 'cond4'
                        *if which_train_data != False, be aware of the binary_fct

    kfold : Int.; Number of folds to perform on the test set. If kfold = 0, kfold-cross-validation will be skipped. Default = 5

    n_components_pca : Float; Pourcentage of variance to keep in the principal component analysis for dimensianality reduction. If set to 0, no PCA will be applied. Default = 0.90

    classes : String; Names of the conditions of the classes to classify. By default =  ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'].

    cov_corr : Bool; A covariance correction is applied on the final model's coefficients to reproject them in 3D brain space. The correction is from Haufe et al. (2014). Default = True

    binary : Bool; The model is initially built to be a multiclass model, but if binary = True, Y can be encoded bynarily according to binary_fct

    binary_fct : string; Only used in case binary = True. Different functions of class encoding are available. The default encoding function is based on manipulation vs neutral condition
                Default = 'modulation' Other choices = ['runs', ]
                Refer to scripts/mvpa_prepping.py data for functions' descritption

    verbose : String; Wether to print output description or not. Default = True.
    """

    # Extract data as paths to files based on sting in arg. e.g. '*.nii'
    data, gr, files = prepping_data.extract_data(data_input, extract_str = '*.hdr', folder_per_participant = subj_folders)

    if sub_data != False: # Will filter 'data' based on strings in 'sub_data', e.g. sub_data = ['hyper','hypo]use only a sub-part of data
        data, gr, files = prepping_data.keep_sub_data(data,gr, sub_data)

    # Y data
    # Endodes target binary or multiclass. binary_fct may change according to model.
    if binary or which_train_data != False: # controls for the case binary = True,
			                    # and which_train_data = ['c1', 'c2'] so you don't want to encode Y as binary
        if binary_fct == 'modulation':
            df_target, cond_target  = prepping_data.encode_manip_classes(data, gr) # ['filename', 'target', 'condition', 'group'] as col

        elif binary_fct == 'runs':
            df_target, cond_target = prepping_data.encode_runs_as_classes(data, gr)

        binary = True # if which_train_data != False and binary = False : to make sure model is binary

    else: # regular multiclass case otherwise
        df_target, cond_target = prepping_data.encode_classes(data, gr)
    Y = np.array(df_target['target'])

    # X data / vectorize fMRI activation maps
    masker, extract_X = prepping_data.extract_signal(data, mask = 'whole-brain-template', standardize = True) # extract_X is a (N obs. x N. voxels) structure
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X_vec = stand_X.T

    # Ordering according to group
    XYgr = pd.concat([pd.DataFrame(X_vec), pd.DataFrame({'files': files, 'Y': Y, 'gr' : gr})], axis = 1) # [[X_vec],[files],[Y],[gr]] of dim [n x m features + 3]
    XYgr_ordered = XYgr.sort_values(by=XYgr.columns[-1]) # -1 : last col. of XYgr e.i. 'gr'
    X = XYgr_ordered.iloc[:,:-3].to_numpy() # X part of XYgr
    Y = np.array(XYgr_ordered['Y'])
    gr = np.array(XYgr_ordered['gr'])
    data = XYgr_ordered['files'] # 'data' is a list of filenames instead of paths** To  reuse paths, change 'files' to 'path' column of XYgr

    # Saving test for matlab script
    save_test = False
    if save_test:
        np.savez_compressed('X_data.npz', X = X)
        np.savez_compressed('Y_data.npz', Y = Y)
        np.savez_compressed('gr_data.npz', gr = gr)
        np.savez_compressed('Xfiles.npz', files = files)
    ###############

    # PCA
    if n_components_pca < 1:
        pca = PCA(n_components = n_components_pca)

    # Split
    if which_train_data != False: # if = ['c1', 'c2'] these cond will be used for training
        X_train, X_test, Y_train, Y_test, y_train_gr_idx = prepping_data.train_test_iso_split(data,X,Y, which_train_data)
        split_gr = [gr[ele] for ele in y_train_gr_idx] # previoulsy : gr[:len(Y_train)
        binary = True                                  # split the group vector according to the split applied to X and Y
    else:
        #GroupShuffleSplit for train / validation set
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=30)
        train_index, test_index = next(gss.split(X, Y, gr))

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        gr_train = (np.array(gr)[train_index]).tolist()
        split_gr = gr_train
        # Previous method
        #ss = ShuffleSplit(n_splits=1, test_size=0.30, random_state=30)
        #X_train, X_test, Y_train, Y_test = train_test_split(X,Y,shuffle = False, test_size=0.30, random_state=30) #old precedure
        #gr[:len(Y_train)]

    #K_FOLD models on (1 - test_size)% train set
    if kfold > 0:
        dict_fold_results = building_model.train_test_models(X_train,Y_train, split_gr, kfold, binary = binary, verbose = verbose)

    # Final model
    model_clf = SVC(kernel="linear", probability = True, decision_function_shape = 'ovo')
    final_model = model_clf.fit(pca.fit_transform(X_train), list(Y_train))
    X_test = pca.transform(X_test)
    PCA_var = np.array(pca.explained_variance_ratio_)
    PC_values = np.arange(pca.n_components_) + 1

    # Metrics
    Y_pred = final_model.predict(X_test)
    final_row_metrics, cm, cr = building_model.compute_metrics_y_pred(Y_test, Y_pred, verbose) # cm : confusion matrix and cr : classification report

    if binary:
        final_y_score = final_model.predict_proba(X_test)[:, 1] # Only  takes the second row of the output
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score))
    else:
        final_y_score =  final_model.predict_proba(X_test)
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score), multi_class = 'ovo')
    final_decision_func = final_model.decision_function(pca.transform(X))

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision']
    df_ypred_metrics = pd.DataFrame(columns = metrics_colnames)
    df_ypred_metrics.loc[0] = final_row_metrics

    dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func,PCA_var_final = PCA_var,PC_val_final = PC_values)

    # Covariance matrix of X_test,Y_test
    if cov_corr:
        cov_x = np.cov(X_test.transpose().astype(np.float64))
        cov_y = np.cov(Y_test.transpose().astype(np.float64))
        #cov_mat = np.cov(X_test.transpose().astype(float),wide_Y_test.transpose().astype(float), rowvar = False, dtype = np.float64)

    # saving coeff., dict_final_results, final_model and fold_results
    #os.chdir(save_path)
    contrast_counter = 1
    for weights in final_model.coef_: # W is the weight vector

        (masker.inverse_transform(pca.inverse_transform(weights))).to_filename(os.path.join(save_path, f"coeffs_whole_brain_{contrast_counter}.nii.gz"))
        if cov_corr:
            # correction from Eqn 6 (Haufe et al., 2014)
            A = np.matmul(cov_x, weights)*(1/cov_y) # j'ai enlevé weights.transpose()
            print('A.shape : ')
            print(A.shape)
            print(masker.inverse_transform(pca.inverse_transform(A)).shape)
            # reproject to nii
            (masker.inverse_transform(pca.inverse_transform(A))).to_filename(os.path.join(save_path, f"eq6_adj_coeff_whole_brain_{contrast_counter}.nii.gz"))
        contrast_counter += 1

    with open(os.path.join(save_path, 'final_results.pickle'), 'wb') as handle:
        pickle.dump(dict_final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('final_results.pickle', 'rb') as handle:
    #    b = pickle.load(handle)

    filename_model = os.path.join(save_path, "final_model_SVC.pickle")
    pickle_out = open(filename_model,"wb")
    pickle.dump(final_model, pickle_out)
    pickle_out.close()

    if kfold > 0:
        with open(os.path.join(save_path, 'kfold_results.pickle'), 'wb') as handle:
            pickle.dump(dict_fold_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #with open('kfold_results.pickle', 'rb') as handle:
        #    b = pickle.load(handle)

    np.savez_compressed(os.path.join(save_path, 'XY_data_split.npz'),df_target = df_target, X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
    #np.savez_compressed('cov_matrix.npz', cov_mat=cov_mat)
    main_args = f'kfold = {kfold}, n_components_pca  = {n_components_pca}, sub_data = {sub_data}, which_train_data = {which_train_data}, classes = {classes}, cov_corr = {cov_corr}, binary = {binary}, binary_fct = {binary_fct}'
    with open(os.path.join(save_path, 'main_args.txt'), 'w') as main_args_file:
        main_args_file.write(''.join(cond_target) + ' / ' + main_args)

    if verbose:
        print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X_vec.shape)
        print('final_roc_auc_ovo: {}  :'.format(final_roc_auc_ovo))
        print('weights shape : {} '.format(weights.shape))

        print('covmat_X.shape : {}, covmat_Y.shape : {}'.format(cov_x.shape,cov_y.shape))
        print('Fold metrics  : {}'.format(dict_fold_results['df_fold_metrics']))
        print(main_args, cond_target)

# py maps
#data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/each_shocks'
# SPM maps
data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/contrast_singEvent_SPM'
save_out = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/mvpa '

main_svc(save_out, data_input, subj_folders=False, sub_data = ['exception_ANAHYPER'], which_train_data = False, kfold = 6, n_components_pca = .90, binary = True, binary_fct = 'runs')

#  which_train_data = ['oANA', 'oHYPER'] to train on hyper-hypo and test on rest
