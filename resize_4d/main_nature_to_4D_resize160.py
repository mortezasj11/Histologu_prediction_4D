import numpy as np
import os
import nibabel as nib
from skimage.transform import resize


lung_path = '/DataS/Nifti/Lung_deeplearning/Resize'                           #-v /rsrch1/ip_rsrch/wulab/Wu:/DataS
save_path = '/Data/Histology/Nature_Gemini_4d_Apr2022/nature_4d_4_15_2022'    #-v /rsrch1/ip_rsrch/wulab/Mori:/Data
os.makedirs(save_path, exist_ok=True)


def ct_400_200(img, min_v=-400, max_v=200):
    img = np.clip(img, min_v, max_v)
    img = (img-min_v)/(max_v-min_v)
    img = img.astype(float)
    return img

def ct_1000_200(img, min_v=-1000, max_v=200):
    img = np.clip(img, min_v, max_v)
    img = (img-min_v)/(max_v-min_v)
    img = img.astype(float)
    return img

if __name__=='__main__':
    ct_list = [i for i in os.listdir(lung_path) if i.endswith('ct.nii.gz')]
    bias = 4000
    for i,ct_name in enumerate(ct_list):
        
        print('{}/{}.  {}'.format(i+1, len(ct_list),ct_name))
        whole_path_ct        = os.path.join(lung_path, ct_name)
        whole_path_seg       = os.path.join(lung_path, ct_name[:-9]+'lung.nii.gz')
        whole_path_seg_Tumor = os.path.join(lung_path, ct_name[:-9]+'manu.nii.gz')

        #print(whole_path_ct)
        #print(whole_path_seg)

        # load lung and seg
        seg_h = nib.load(whole_path_seg)
        seg = seg_h.get_fdata()
        
        ct_h = nib.load(whole_path_ct)
        ct = ct_h.get_fdata()

        seg_h_t = nib.load(whole_path_seg_Tumor)
        seg_t = seg_h_t.get_fdata()

        # 1st ct normalized -1000 150
        ct_n = ct_1000_200(ct)

        # 2nd lung*seg normalized -100 100
        ct_bias = ct + bias
        ct_seg = ct_bias*seg
        ct_seg_tumor = ct_bias*seg_t
        ct_seg = ct_seg - bias
        ct_seg_tumor = ct_seg_tumor - bias

        x,y,z = 160,160,160
        ct_seg_250_150 = ct_400_200( ct_seg )
        ct_seg_250_150_resize = resize(ct_seg_250_150, (x,y,z), order=0, preserve_range=True)
        
        # 3rd lung*seg normalized -1000 150
        ct_seg_1000_150 = ct_1000_200( ct_seg )
        ct_seg_1000_150_resize =  resize(ct_seg_1000_150, (x,y,z), order=0, preserve_range=True)

        ct_seg_tumor_250_150 = ct_400_200( ct_seg_tumor )
        ct_seg_tumor_250_150_resize =  resize(ct_seg_tumor_250_150, (x,y,z), order=0, preserve_range=True)

        # Resizing
        out_no_resize = np.zeros((x,y,z,4))
        out_no_resize[:,:,:,0] = resize(ct_n, (x,y,z), order=0, preserve_range=True)   # whole CT -1000 200
        out_no_resize[:,:,:,1] = ct_seg_250_150_resize                                 # Lung seg -400 200
        out_no_resize[:,:,:,2] = ct_seg_tumor_250_150_resize                           # Tumor seg -400 200
        out_no_resize[:,:,:,3] = ct_seg_1000_150_resize                                # Lung seg  -1000 200

        # Saving
        save_path_name = os.path.join(save_path,ct_name)
        img_NIFTI = nib.Nifti1Image(out_no_resize, ct_h.affine )
        img_NIFTI.to_filename(save_path_name)



