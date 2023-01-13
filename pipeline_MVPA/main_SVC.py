import numpy as np
import os
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scripts import mvpa_prepping_data as prepping_data
from scripts import mvpa_building_model as building_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, permutation_test_score


def main_svc(data_input, save_path, kfold = 5, n_components_pca = 0.90, sub_data = False, which_train_data = False, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'], cov_corr = True, binary = False, binary_fct = 'modulation'):
    """
    This function serves to run a linear SVC on fmri data.
    arguments
    ---------
    data_input : String; Path to fmri activation maps. It will extract the path of all imgs in path and its subfolders that has 'beta*' in name
    save_path : String; path to where results will be saved
    kfold : Int.; Number of folds to perform on the test set. If kfold = 0, kfold-cross-validation will be skipped. Default = 5
    n_components_pca : Float; Pourcentage of variance to keep in the principal component analysis for dimensianality reduction. If set to 0, no PCA will be applied. Default = 0.90
    sub_data :  Bool or string; By default, the model is built with all the data (sub_data = False), but list of string can be given to select only a sub-part of data to build model.
                E.g. sub_data = ['cond1', 'cond2'], only the data filename containing those strings will be selected to build the model (train and test).
    which_train_data : Bool or string; By default, all the data is used to train/test the model (= False), but if a list of string is specified, the data having those strings in
                        their name will be selected to train the model and the rest of the data will be used to test the model. If which_train_data != False, binary = True automatically
                        E.g. which_train_data = ['cond1', 'cond2'], training is done with those conditions and the model is tested on 'cond3' and 'cond4'
                        *if which_train_data != False, be aware of the binary_fct
    **classes : String; Names of the conditions of the classes to classify. By default =  ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'].
    cov_corr : Bool; A covariance correction is applied on the final model's coefficients to reproject them in 3D brain space. The correction is from Haufe et al. (2014). Default = True
    binary : Bool; The model is initially built to be a multiclass model, but if binary = True, Y can be encoded bynarily according to binary_fct
    binary_fct : string; Only used in case binary = True. Different functions of class encoding are available. The default encoding function is based on manipulation vs neutral condition
                Default = 'modulation' Other choices = ['runs', ]
                Refer to scripts/mvpa_prepping.py data for functions' descritption
    """

    #extract data from path input
    data, gr = prepping_data.extract_data(data_input, extract_str = 'beta*')

    if sub_data != False: # For case where we need to use only a sub-part of data
        data, gr = prepping_data.keep_sub_data(data, sub_data)

    # Y data
    if binary or which_train_data != False: # controls for the case where binary = True, and which_train_data = ['', ''] so you don't want to encode Y as binary
        if binary_fct == 'modulation':
            df_target, cond_target  = prepping_data.encode_manip_classes(data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col

        elif binary_fct == 'runs':
            df_target, cond_target = prepping_data.encode_runs_as_classes(data, gr)
        binary = True # if which_train_data is not False but binary = False to make sure model is binary
    else: # regular multiclass case otherwise
        df_target = prepping_data.encode_classes(data, gr)
    Y = np.array(df_target['target'])

    # X data / vectorize fMRI activation maps
    masker, extract_X = prepping_data.extract_signal(data, mask = 'whole-brain-template', standardize = True) # extract_X is a (N obs. x N. voxels) structure. It will serve as X
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    check = np.isnan(X)
    print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X.shape)

    # PCA
    if n_components_pca < 1:
        pca = PCA(n_components = n_components_pca)
        #X = pca.fit_transform(X)

    # Split
    if which_train_data != False:
        X_train, X_test, Y_train, Y_test, y_train_gr_idx = prepping_data.train_test_iso_split(data,X,Y, which_train_data)
        split_gr = [gr[ele] for ele in y_train_gr_idx] #gr[:len(Y_train)] # split the group vector according to the split applied to X and Y
        binary = True
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=30)
        split_gr = gr[:len(Y_train)]

    print(type(X_train), type(X_test), type(Y_train), type(Y_test))
    print(X_train, X_test, Y_train, Y_test)
    print(split_gr)
    print('X_train.shape : {}, Y_train.shape : {}, X_test.shape : {}, Y_test.shape : {}'.format(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape))

    #K_FOLD MODELS
    if kfold > 0:
        dict_fold_results = building_model.train_test_models(X_train,Y_train, split_gr, kfold, binary = binary)

    # FINAL MODEL
    print('--Fitting final model--')
    model_clf = SVC(kernel="linear", probability = True, decision_function_shape = 'ovo')
    final_model = model_clf.fit(pca.fit_transform(X_train), list(Y_train))
    X_test = pca.transform(X_test)

    # Metrics
    final_results = dict()
    Y_pred = final_model.predict(X_test)
    final_row_metrics, cm, cr = building_model.compute_metrics_y_pred(Y_test, Y_pred) # cm : confusion matrix and cr : classification report
    if binary:
        final_y_score = final_model.predict_proba(X_test)[:, 1] # Here we only  take the second row of the output
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score))
    else:
        final_y_score =  final_model.predict_proba(X_test)
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score), multi_class = 'ovo')
    print('final_roc_auc_ovo: {}  :'.format(final_roc_auc_ovo))
    final_decision_func = final_model.decision_function(pca.transform(X))

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision']
    df_ypred_metrics = pd.DataFrame(columns = metrics_colnames)
    df_ypred_metrics.loc[0] = final_row_metrics

    dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func)

    # covariance matrix of X_test,Y_test
    wide_Y_test = building_model.hot_split_Y_test(Y_test,len(classes))
    if cov_corr:
        cov_x = np.cov(X_test.transpose().astype(np.float64))
        cov_y = np.cov(Y_test.transpose().astype(np.float64))
        print(cov_x.shape)
        print(cov_y.shape)
        #cov_mat = np.cov(X_test.transpose().astype(float),wide_Y_test.transpose().astype(float), rowvar = False, dtype = np.float64)
        print('cov_x and y shape')
        print(cov_x.shape)
        print(cov_y.shape)
        print(cov_y)

    # saving coeff., dict_final_results, final_model and fold_results
    #os.chdir(save_path)
    contrast_counter = 1
    for weights in final_model.coef_: # W is the weight vector

        (masker.inverse_transform(pca.inverse_transform(weights))).to_filename(f"coeffs_whole_brain_{contrast_counter}.nii.gz")
        print('weights shape')
        print(weights.shape)
        if cov_corr:
            # correction from Eqn 6 (Haufe et al., 2014)
            A = np.matmul(cov_x, weights)*(1/cov_y) # j'ai enlevé weights.transpose()
            print('A.shape : ')
            print(A.shape)
            print(masker.inverse_transform(pca.inverse_transform(A)).shape)
            # reproject to nii
            (masker.inverse_transform(pca.inverse_transform(A))).to_filename(f"eq6_adj_coeff_whole_brain_{contrast_counter}.nii.gz")
        contrast_counter += 1

    with open('final_results.pickle', 'wb') as handle:
        pickle.dump(dict_final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('final_results.pickle', 'rb') as handle:
        b = pickle.load(handle)

    filename_model = "final_model_SVC.pickle"
    pickle_out = open(filename_model,"wb")
    pickle.dump(final_model, pickle_out)
    pickle_out.close()

    if kfold > 0:
        with open('kfold_results.pickle', 'wb') as handle:
            pickle.dump(dict_fold_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('kfold_results.pickle', 'rb') as handle:
            b = pickle.load(handle)

    np.savez_compressed('XY_data_split.npz', X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
    #np.savez_compressed('cov_matrix.npz', cov_mat=cov_mat)

    main_args = f'kfold = {kfold}, n_components_pca  = {n_components_pca}, sub_data = {sub_data}, which_train_data = {which_train_data}, classes = {classes}, cov_corr = {cov_corr}, binary = {binary}'
    print(main_args, cond_target)
    with open('main_args.txt', 'w') as main_args_file:
        main_args_file.write(''.join(cond_target) + ' / ' + main_args)

#data_input = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\GLM_each_shock_4sub'
#save_out = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_decoding\test_4sub\test_final'


data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/each_shocks'
#data_input = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\test_each_shock'

save_out = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/mvpa '

main_svc(data_input, save_out, kfold = 5,n_components_pca = .90, which_train_data = ['oANA', 'oHYPER'], sub_data = False, binary = True, binary_fct = 'runs')

#  which_train_data = ['oANA', 'oHYPER'] to train on hyper-hypo and test on rest


























    


























































    














