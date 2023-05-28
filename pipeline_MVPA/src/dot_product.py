
import os
import numpy as np
import pickle
import nibabel as nib
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn import image
import glob

def keep_sub_data(data, sub_data):
    '''
    Function to keep a part only of the data and leave the rest unused. E.g. to build a model only on specific condition.

    arguments
    ---------
    data : List ; list containing the paths of all the files
    sub_data : List or str. ; List of string that are comprised in filenames. The filenames containing those str. will be kept
    '''
    filt_data = []
    idx = 0
    for img in data:
        res = [ele for ele in sub_data if (ele in img)] #checks if element 'ele' of the sub_data list is in the img's filename
        if res:
        # if res if not empty, meaning that img path contain an element in sub_data, e.g. 'ANA' or 'N_ANA'
            filt_data.append(img)
        idx += 1

    return filt_data


def dot(path_to_img, path_output, to_dot_with = 'nps', dot_id = None, resample_to_mask=True, img_format = '*.nii',participant_folder = True, filter_data = False):
    """
    Parameters
    ----------
    path_to_img : string
        path to the files (nii, hdr or img) on which the signature will be applied
    path_output : string 
        path to save the ouput of the function (i.e., dot product and related file)
    to_dot_with : string, default = 'nps'
         signature or fmri images to apply on the data, the maks,e.g. NPS. The paths to the signature files are defined inside the function.
         Can also be a *list* of paths to images.
    dot_id : string, default = None
        name of the experimental condition or id to include in output filename
    img_format : string, default = '.nii'
        change according to the format of fmri images, e.g. '*.hdr' or '*.img'
    participant_folder : string, default = True
        If True, it's assumed that the path_to_img contains a folder for each participant that contain the fMRI image(s) that will be used to compute smt product. If False,
        it is assumed that all the images that will be used to compute dot product are directly in path_to_img
    filter_data : Bool. or list; False by default, if not False, it is exêcted that a list od string is given as arg. This will filter the data used for
                    dot product based on the string present in the filename. E.g. 'filename1','filename2' filter_data = ['2'], will noly keep filename2 for dot
    Returns
    -------
    dot_array : numpy array
    subj_array : list
    """
    pwd = os.getcwd()
    print(pwd)
    #Define signature path

    if to_dot_with == "nps":
        mask_path = os.path.join(pwd,r"signatures/weights_NSF_grouppred_cvpcr.hdr")
    if to_dot_with == "siips":
        mask_path = "nonnoc_v11_4_137subjmap_weighted_mean.nii"
    if to_dot_with == "vps":
        mask_path = "bmrk4_VPS_unthresholded.nii"

    if resample_to_mask:
        resamp = "img_to_mask"
    if resample_to_mask == False:
        resamp = "mask_to_img"

    #----------Formatting files--------

    #returns a list with all the paths of images in the provided argument 'path_to_img'
    if participant_folder:
        fmri_imgs = glob.glob(os.path.join(path_to_img,'*',img_format)) #e.g : path/*all_subjects'_folders/*.nii
    else:
        fmri_imgs = glob.glob(os.path.join(path_to_img,img_format)) #e.g : path/*.nii
        #fmri_imgs = glob.glob(os.path.join(path,'*','*.nii'))

    if filter_data:
        fmri_imgs = list(keep_sub_data(fmri_imgs, filter_data))


    #fmri_imgs = list(filter(lambda x: "hdr" in x, data_path))
    fmri_imgs.sort(reverse=True)
    fmri_imgs.sort()

    if type(to_dot_with) is list:
        to_dot_with.sort(reverse=True)
        to_dot_with.sort()
        print('NOTICE : \'Mask\' is a list of fmri images')
    else :
        mask = nib.load(mask_path) #Load the mask as a nii file
        print('mask shape :  {} '.format(mask.shape))
    #------------------------

    #Initializing empty arrays for results)
    dot_array = np.array([])
    subj_array = []
    masks_names =[] #to save either the mask name or the list of images'names

    #----------Main loop--------------

    #For each map/nii file in the provided path
    indx = 0
    for maps in fmri_imgs:
        if type(to_dot_with) is list:#i.e. if a list of images was given instead of a unique mask. Masks_names was only defined in this specific case
            mask = nib.load(to_dot_with[indx])
            mask_name = os.path.basename(os.path.normpath(to_dot_with[indx])) #mask_name

        else:
            mask_name = to_dot_with #by default nps

        #load ongoing image
        img = nib.load(maps)

        #saving the file/subject's name in a list
        subj = os.path.basename(os.path.normpath(maps))
        subj_array.append(subj)

        #if image is not the same shape as the mask, the image will be resample
        #mask is the original mask to dot product with the provided images
        if img.shape != mask.shape:
        #---------Resampling--------
            print('Resampling : ' + os.path.basename(os.path.normpath(maps)))
            #resampling img to mask's shape
            if resample_to_mask:
                resampled = image.resample_to_img(maps,mask)
                print('image has been resample to mask\'s shape : {} '.format(resampled.shape))
            else:
                resampled = image.resample_to_img(mask,maps)
                print('Mask has been resample to image\'s shape : {} '.format(resampled.shape))

        else:#if input and image are the same size
            if resample_to_mask:
                resampled = img
            else:
                resampled = mask

        #---------fitting images to 1 dimension------------
        #making mask and image a vector in order to dot product

        #Fitting the masker of the 'mask' for which we want to dot product the beta maps
        masker_all = NiftiMasker(mask_strategy="whole-brain-template")

        #masker of the initial mask provided, in our case the mask is called NPS
        if resample_to_mask:
            masker_NPS = masker_all.fit_transform(mask)
        else:
            masker_NPS = masker_all.fit_transform(img)

        #fitting temporary masker for ongoing beta map :'maps'
        masker_tmp = masker_all.fit_transform(resampled)

        #---------Dot product---------
        print(subj,' dot with : ', mask_name)
        print(f'Computing dot product : {indx + 1}/{len(fmri_imgs)}')
        print('---------------------------')
        #dot product of the image's masker with the mask(NPS)'s masker
        dot_res = np.dot(masker_tmp,masker_NPS.T)
        print(dot_res)
        #storing the result in array
        dot_array = np.append(dot_array,dot_res)
        masks_names.append(mask_name)

        indx += 1

    if type(to_dot_with) is list:
        to_dot_with = 'aslist'

    if dot_id == None:
        df_res = pd.concat([pd.DataFrame(dot_array.T, columns = [f'dot_results_{resamp}_{to_dot_with}']),
            pd.DataFrame(subj_array, columns = ['files']),
            pd.DataFrame(masks_names, columns = ['masks'])],axis=1)
        if path_output is False: # if no output path have been entered, e.i. = False, results will be saved as pickle in pwd
            with open(f'dot_{resamp}_{to_dot_with}.pickle', 'wb') as handle:
                pickle.dump(df_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            df_res.to_csv(os.path.join(path_output,f'results_{resamp}_{to_dot_with}.csv'))

    else:
        df_res = pd.concat([pd.DataFrame(dot_array.T, columns = [f'dot_results_{resamp}_{to_dot_with}_{dot_id}']),
            pd.DataFrame(subj_array, columns = ['files']),
            pd.DataFrame(masks_names, columns = ['masks'])], axis=1)
        if path_output is False:
            with open(f'results_{resamp}_{to_dot_with}_{dot_id}.pickle', 'wb') as handle:
                pickle.dump(df_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            df_res.to_csv(os.path.join(path_output,f'results_{resamp}_{to_dot_with}_{dot_id}.csv'))

    return dot_array, subj_array


##Arguments of the function

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--to_dot_with", type=str, choices=['nps','siips','vps']) #to_dot_with on which to compute the dot product
    parser.add_argument("--path_to_img", type=str) #path to beta maps
    parser.add_argument("--path_output", type=str) #path to save the output
    parser.add_argument("--condition", type=str, default=None) #specify the experimental condition
    parser.add_argument("--resample_to_mask", type=bool, default=True) #how to resample the data. If true signature is resampled to data, otherwise data is resampled to signature
    args = parser.parse_args()

    #dot_array, subj_array = dot(args.path_to_data, args.path_output, args.to_dot_with, args.condition, args.resample_to_mask)


#example
path_to_img = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_GLM\all_shocks'
path_output = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\dot_product'

#path_output = r'/data/rainville/dylan_projet_ivado_decodage/results/dot/each_shock'
#path_jeni = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\099_TxT_Individual_N-SHOCKS_files'
#ls_neut_shocks = glob.glob(os.path.join(path_jeni,'*.hdr'))#have to give a list of images and not the path
#dot(path_to_img= path_to_img, path_output=path_output, to_dot_with=ls_neut_shocks,dot_id = 'neutShocks_py',participant_folder=True)
#imgs_jeni = glob.glob(os.path.join(path_jeni,'*.hdr'))

dot(path_to_img= path_to_img, path_output=False, to_dot_with='nps',dot_id = 'all_shock_py', participant_folder = True, filter_data = False)

