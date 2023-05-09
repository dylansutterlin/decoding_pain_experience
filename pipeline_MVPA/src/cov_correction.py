import os
import numpy as np
import glob
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.maskers import NiftiMasker 
import pickle

def cov_correction(data):

    os.chdir(data)
    coefs = glob.glob(os.path.join(data, 'coefs*'))
    # load
    with open('masker_to_fit_X.pickle', 'rb') as handle:
        masker = pickle.load(handle)
    # Load test data
    xy = np.load('XY_data_split.npz', allow_pickle = True)
    Y_test = xy['Y_test']
    X_test = xy['X_test']

    # covariance
    cov_x = np.cov(X_test.transpose().astype(np.float64))
    cov_y = np.cov(Y_test.transpose().astype(np.float64))
    print(cov_x.shape)
    print(cov_y.shape)

    # load W, weight vector
    contrast_counter = 1
    for file in coefs:
        print(file)
        img = nib.load(file)
        W = apply_mask(img, masker)
        print('w shape', W.shape)

        # correction from Eqn 6 (Haufe et al., 2014)
        A = np.matmul(cov_x, W)*(1/cov_y)
        print('A.shape : ')
        print(A.shape)
        # reproject to nii
        nl_masker = NiftiMasker(mask_img = masker)

        (nl_masker.inverse_transform(A)).to_filename(f"corrCoeff_whole_brain_{contrast_counter}.nii.gz")

        contrast_counter += 1


#data = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_decoding\model_SVC_2folds'
data = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/model1' 
cov_correction(data)




