import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib

#lung_path = 'E:/data/Resize_nature_wulab'
lung_path = '/Data/Resize_nature_wulab'

#save_path  =  'E:/data/Histology/nature'
save_path  =  '/Data/Histology/nature'
os.makedirs(save_path, exist_ok=True)

ct_list = [i for i in os.listdir(lung_path) if i.endswith('ct.nii.gz')]
bias = 1024
for i,ct in enumerate(ct_list):
    print('{} .  {}'.format(i+1, ct))
    whole_path_ct = os.path.join(lung_path, ct)
    whole_path_seg = os.path.join(lung_path, ct[:-9]+'lung.nii.gz')

    # load lung and normalize 
    lung_h = nib.load(whole_path_ct)
    lung = lung_h.get_fdata()
    lung = lung + bias
    #lung = normalize(lung)

    #load seg
    seg_h = nib.load(whole_path_seg)
    seg = seg_h.get_fdata()

    # lung* seg
    lung_seg = lung*seg
    lung_seg = lung_seg - bias

    img_NIFTI = nib.Nifti1Image(lung_seg, lung_h.affine )
    img_NIFTI.to_filename(os.path.join(save_path,ct))
    