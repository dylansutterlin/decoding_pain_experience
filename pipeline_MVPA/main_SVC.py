import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from scripts import mvpa_prepping_data as prepping_data
from scripts import mvpa_building_model as building_model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from nilearn.decoding import Decoder
from sklearn.metrics import roc_auc_score,roc_curve, accuracy_score, confusion_matrix, classification_report
#test modif for git

def main_svc(save_path, data_input, subj_folders = True, sub_data = False,  which_train_data = False, test_size = 0.30, n_splits = 5, split_proced = 'GSS', grid_search = True, n_components_pca = 0.90, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'], cov_corr = True, binary = False, binary_func = 'modulation', rand_seed = 30, verbose = True):


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
                        *if which_train_data != False, be aware of the binary_func

    n_splits : Int.; Number of folds to perform on the test set. If n_splits = 0, kfold-cross-validation will be skipped. Default = 5

    split_proced : Str; Procedure to use to split data. Default = 'GSS' /GroupShuffleSplit (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
                        Other Choices : 'LOGO' /Leave One Group Out (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut)

    n_components_pca : Float; Pourcentage of variance to keep in the principal component analysis for dimensianality reduction. If set to 0, no PCA will be applied. Default = 0.90

    classes : String; Names of the conditions of the classes to classify. By default =  ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'].

    cov_corr : Bool; A covariance correction is applied on the final model's coefficients to reproject them in 3D brain space. The correction is from Haufe et al. (2014). Default = True

    binary : Bool; The model is initially built to be a multiclass model, but if binary = True, Y can be encoded bynarily according to binary_func

    binary_func : string; Only used in case binary = True. Different functions of class encoding are available. The default encoding function is based on manipulation vs neutral condition
                Default = 'modulation' Other choices = ['runs', ]
                Refer to scripts/mvpa_prepping.py data for functions' descritption

    rand_seed : Int; Random seed. Default = 30

    test_size : Float; Default = 0.30

    verbose : String; Wether to print output description or not. Default = True.
    """

    # Extract data as paths to files based on sting in arg. e.g. '*.nii'
    data, gr, files = prepping_data.extract_data(data_input, extract_str = '*.hdr', folder_per_participant = subj_folders)

    if sub_data != False: # Will filter 'data' based on strings in 'sub_data', e.g. sub_data = ['hyper','hypo]use only a sub-part of data
        data, gr, files = prepping_data.keep_sub_data(data,gr, sub_data)

    # Y data
    # Endodes target binary or multiclass. binary_func may change according to model.
    if binary or which_train_data != False: # controls for the case binary = True,
			                    # and which_train_data = ['c1', 'c2'] so you don't want to encode Y as binary
        if binary_func == 'modulation':
            df_target, cond_target  = prepping_data.encode_manip_classes(data, gr) # ['filename', 'target', 'condition', 'group'] as col

        elif binary_func == 'runs':
            df_target, cond_target = prepping_data.encode_runs_as_classes(data, gr)
        binary = True # if which_train_data != False and binary = False : to make sure model is binary

    else: # regular multiclass case otherwise
        df_target, cond_target = prepping_data.encode_classes(data, gr)

    Y = np.array(df_target['target'])

    load_X = False
    if load_X == False:
    # X data / vectorize fMRI activation maps
        masker, extract_X = prepping_data.extract_signal(data, mask = 'background', smoothing_fwhm = None, standardize = True) # extract_X is a (N obs. x N. voxels) structure
        stand_X = StandardScaler().fit_transform(extract_X.T)
        X_vec = stand_X.T
    else:
        X_vec = np.load('X_vec.npz')['X']
        print(X_vec.shape)
    print(masker)
    #np.savez_compressed('X_vec.npz', X = X_vec)

    # Ordering according to group
    XYgr = pd.concat([pd.DataFrame(X_vec), pd.DataFrame({'files': data, 'Y': Y, 'gr' : gr})], axis = 1) # [[X_vec],[files],[Y],[gr]] of dim [n x m features + 3]
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

    #------Splitting procedures-------
    if which_train_data != False: # if = ['c1', 'c2'] these cond will be used for training
        X_train, X_test, Y_train, Y_test, y_train_gr_idx = prepping_data.train_test_iso_split(data,X,Y, which_train_data)
        split_gr = [gr[ele] for ele in y_train_gr_idx] # previoulsy : gr[:len(Y_train)
        binary = True                                  # split the group vector according to the split applied to X and Y
    else:   # --Choices--
        if split_proced == 'GSS':
            test_procedure = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state = 42)
            inner_procedure = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state = 42)
        elif split_proced == 'LOGO':
            test_procedure = LeaveOneGroupOut()
        # ---Train-Test Sets---
        train_index, test_index = next(test_procedure.split(X, Y, gr))
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        gr_train = (np.array(gr)[train_index]).tolist()
        cond_train = df_target['conditions'][train_index]
        cond_test = df_target['conditions'][test_index]
        split_gr = gr_train

    #K_FOLD models on (1 - test_size)% train set
    if n_splits > 0:
        #dict_fold_results = building_model.train_test_models(X_train,Y_train, split_gr, n_splits, binary = binary, test_size = test_size, random_seed = rand_seed, verbose = verbose)
        dict_fold_results = building_model.train_test_models(X, Y, gr, n_splits, split_proced = split_proced, binary = binary, test_size = test_size, random_seed = rand_seed, verbose = False)


    decoder_test = False
    if decoder_test == True:
    # Simple decoder from nilearn
        decoder = Decoder(
        estimator='svc',
        mask = masker,
        standardize=True,
        screening_percentile=5,
        scoring="accuracy",
        cv=inner_procedure,
        )
        decoder.fit(data, df_target['conditions'], groups=gr)
        decoder_cv_scores = decoder.cv_scores_

    #--------Hyperparameter tuning-----------

    final_model = SVC(kernel="linear", probability = True, decision_function_shape = 'ovo')
    pca = PCA(n_components = n_components_pca)
    if grid_search :
        #--- Inner loop---
        # Choice : GridSearchCV
        param_grid = [{'C': [0.00001,0.0001, 0.001, 0.01, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf'],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'gamma' : ['scale', 'auto']}]
        best_score = []
        best_estim = []
        # Nested CV with parameter optimizati
        print('--------------- \nIn grid Search')
        GS = GridSearchCV(estimator=final_model, param_grid=param_grid, cv=inner_procedure)
        GS.fit(X_train,cond_train, groups = gr_train) # K fold on X_train which corresponds to 1- test_size of data
        best_score.append(GS.best_score_)
        #best_estim.append(GS.best_estimator_)
        GS_params = GS.best_params_

        # ---Final model with best params from GridSearchCV--
        final_model = SVC(C = GS_params['C'], kernel = GS_params["kernel"], gamma = GS_params['gamma'], probability = True, decision_function_shape = 'ovo')
        pipe = Pipeline([('scaler',StandardScaler()),('reduce_dim', pca), ('model', final_model)])
        scores = cross_val_score(pipe, X,  df_target['conditions'], groups = gr, cv=inner_procedure)
        pipe.fit(X_train, list(Y_train))
        final_predictions = pipe.predict(X_test) # Unbiased, test set not used in CV for hypertunning
        accuracy = accuracy_score(list(Y_test), list(final_predictions))
        dict_fold_results = building_model.train_test_models(X, Y, gr, n_splits, default_pipe = pipe, split_proced = split_proced, binary = binary, test_size = test_size, random_seed = rand_seed, verbose = verbose)

        #n_sv = pipe['model'].n_support_
    else:
        # ----------Train/test on X using 'cross_val_score' with inner_procedure('e.g. GSS,LOGO')-----------
        pipe = Pipeline([('scaler',StandardScaler()),('reduce_dim', pca), ('model', final_model)])

        dict_fold_results = building_model.train_test_models(X, Y, gr, n_splits, default_pipe = pipe, split_proced = split_proced, binary = binary, test_size = test_size, random_seed = rand_seed, verbose = verbose)

        scores = cross_val_score(pipe, X,  df_target['conditions'], groups = gr, cv=inner_procedure)
        print('Cross val scores : ', scores)


        if n_components_pca < 1:
           # final_model = model_clf.fit(pca.fit_transform(X_train), list(Y_train))
            #X_test = pca.transform(X_test)
            PCA_var = np.array(pipe['reduce_dim'].explained_variance_ratio_)
            PC_values = np.arange(pipe['reduce_dim'].n_components) + 1
            n_iter = pipe['model'].n_iter_
            n_sv = pipe['model'].n_support_
            n_features = pipe['model'].n_features_in_
            idx_sv = pipe['model'].support_
        else:
            final_model = model_clf.fit(X_train, list(Y_train))
            n_iter = final_model.n_iter_
            #params = final_model.get_params()
            n_sv = final_model.n_support_
            n_features = final_model.n_features_in_
            idx_sv = final_model.support_


    # Metrics
    Y_pred = pipe.predict(X_test)
    final_row_metrics, cm, cr = building_model.compute_metrics_y_pred(Y_test, Y_pred, verbose) # cm : confusion matrix and cr : classification report

    if binary:
        final_y_score = pipe.predict_proba(X_test)[:, 1] # Only  takes the second row of the output
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score))
    else:
        final_y_score =  final_model.predict_proba(X_test)
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score), multi_class = 'ovo')
    final_decision_func = final_model.decision_function(pca.transform(X))

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision']
    df_ypred_metrics = pd.DataFrame(columns = metrics_colnames)
    df_ypred_metrics.loc[0] = final_row_metrics

    if grid_search:
        dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func)
    else:
        dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func,PCA_var_final = PCA_var,PC_val_final = PC_values)

    # Covariance matrix of X_test,Y_test
    cov_corr = False
    if cov_corr:
        cov_x = np.cov(X_test.transpose().astype(np.float64))
        cov_y = np.cov(Y_test.transpose().astype(np.float64))
        #cov_mat = np.cov(X_test.transpose().astype(float),wide_Y_test.transpose().astype(float), rowvar = False, dtype = np.float64)

    # saving coeff., dict_final_results, final_model and fold_results
    #os.chdir(save_path)
    if (final_model.get_params())['kernel'] == 'linear':
        contrast_counter = 1
        for weights in final_model.coef_: # W is the weight vector
            (masker.inverse_transform(pipe['reduce_dim'].inverse_transform(weights))).to_filename(os.path.join(save_path, f"coeffs_whole_brain_{contrast_counter}.nii.gz"))
            if cov_corr:
                # correction from Eqn 6 (Haufe et al., 2014)
                A = np.matmul(cov_x, weights)*(1/cov_y) # j'ai enlevé weights.transpose()
                print(masker.inverse_transform(pca.inverse_transform(A)).shape)
                (masker.inverse_transform(pipe['reduce_dim'].inverse_transform(A))).to_filename(os.path.join(save_path, f"eq6_adj_coeff_whole_brain_{contrast_counter}.nii.gz"))
            contrast_counter += 1

    with open(os.path.join(save_path, 'final_results.pickle'), 'wb') as handle:
        pickle.dump(dict_final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    filename_model = os.path.join(save_path, "final_model_SVC.pickle")
    pickle_out = open(filename_model,"wb")
    pickle.dump(final_model, pickle_out)
    pickle_out.close()

    if n_splits > 0:
        with open(os.path.join(save_path, 'kfold_results.pickle'), 'wb') as handle:
            pickle.dump(dict_fold_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.savez_compressed(os.path.join(save_path, 'XY_data_split.npz'),df_target = df_target, X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
    #np.savez_compressed('cov_matrix.npz', cov_mat=cov_mat)
    main_args = f'kfold = {n_splits}, n_components_pca  = {n_components_pca}, sub_data = {sub_data}, which_train_data = {which_train_data}, classes = {classes}, cov_corr = {cov_corr}, binary = {binary}, binary_func = {binary_func}'
    with open(os.path.join(save_path, 'main_args.txt'), 'w') as main_args_file:
        main_args_file.write(''.join(cond_target) + ' / ' + main_args)

    if verbose:
        #print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X_vec.shape)
        #print('final_roc_auc_ovo: {}  :'.format(final_roc_auc_ovo))
        #print('weights shape : {} '.format(weights.shape))
        #print('n features in fit', n_features)
        #print('number of support vectors for each class:  {}  :'.format(n_sv))
        #print('n iteration for aoptimization', n_iter)
        #print('covmat_X.shape : {}, covmat_Y.shape : {}'.format(cov_x.shape,cov_y.shape))
        #print('------------ main_svc arguments and target conditions ------------ : \n{} \n{}  :'.format(main_args, cond_target))
        if grid_search:
            print('---Best score : ', best_score, '\nBest params :', GS_params)
            print('Unbiased final accuracy on test set (not used in hyper params tunning :)', accuracy)
            print('Confusion matrix from final predition (test set) :\n', confusion_matrix(list(Y_test),final_predictions))
            #print(classification_report(list(Y_test),final_predictions))
            print('Cross val scores with best params from Grid Search: ', scores)
        print('Fold metrics for final model : {}'.format(dict_fold_results['df_fold_metrics']))
        print('Confusion matrices per folds: {}  :\n'.format(dict_fold_results['confusion_matrix']))
        if decoder_test:
            print('Nilearn Decoder cv scores :  {}'.format(decoder_cv_scores))


# SPM maps
data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/contrast_singEvent_SPM'
save_out = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/mvpa/svc/GSS'
#save = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_decoding\test_4sub'

stdoutOrigin=sys.stdout
sys.stdout = open("out_main.txt", "w")

ls_save = ['bin_HyperAna','bin_HyperNhyper','bin_AnaNana', 'bin_interRun']
sub_data = [['exception_ANAHYPER'], ['HYPER,N_HYPER'],['ANA', 'N_ANA'], False]
bin_func = ['runs', 'modulation', 'modulation', 'runs']
ls_split = ['GSS','GSS', 'GSS','GSS']
for save_id, sub, func, method in zip(ls_save,sub_data,bin_func, ls_split):
    save = os.path.join(r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/mvpa/svc/GSS', save_id)
    if os.path.exists(save) == False:
        os.mkdir(save)
    print('---------In {} loop'.format(sub))
    main_svc(save, data_input, subj_folders = False, sub_data = sub, which_train_data = False, test_size = 0.20, n_splits = 5, split_proced = method, n_components_pca = 0.80, binary = True, binary_func = func)
os.chdir(save_out)
sys.stdout.close()
sys.stdout=stdoutOrigin
#  which_train_data = ['oANA', 'oHYPER'] to train on hyper-hypo and test on rest

#main_svc(save, data_input, rand_seed = 34, subj_folders = False, sub_data = ['exception_ANAHYPER'], which_train_data = False, test_size = 0.20, n_splits = 5, split_proced = 'GSS', n_components_pca = 0.90, binary = True, binary_func = 'runs')
