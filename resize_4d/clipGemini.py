import os
import nibabel as nib
import numpy as np

path_lung = '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/Mask_johof'              # CT_0012_0000_seg.nii.gz    ===    t_seg
path_tumor = '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/Tumor_seg'              # 0011_RTS_P.nii.gz
path_CT = '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/CT'                        # CT_0012_0000.nii.gz

path_save = '/Data/Histology/Nature_Gemini_4d_Apr2022/Gemini_clipped_ctTumorLung_Johof'
os.makedirs(path_save, exist_ok=True)

for t_seg in os.listdir(path_lung):
    if t_seg[3:7] + '_RTS_P.nii.gz' in os.listdir(path_tumor):
        whole_path_lung = os.path.join(path_lung, t_seg)
        whole_seg_l = nib.load(whole_path_lung)
        seg_l = whole_seg_l.get_fdata()

        a = np.zeros((seg_l.shape[2],)) 
        b = np.zeros((seg_l.shape[2],)) 
        for i in range(0, seg_l.shape[2]):
            a[i] = np.sum(seg_l[:,:,i])

        clip_min = max(0, np.nonzero(a > 0)[0][0] - 10)
        clip_max = min(seg_l.shape[2], np.nonzero(a > 0)[0][-1] + 10)

        whole_ct = nib.load(    os.path.join(path_CT,  t_seg[:-11] + '.nii.gz')     )
        ct = whole_ct.get_fdata()
        whole_seg_t = nib.load(    os.path.join(path_tumor,  t_seg[3:7] + '_RTS_P.nii.gz')     )
        seg_t = whole_seg_t.get_fdata()

        #
        ct_new = ct[:,:,clip_min:clip_max]
        seg_t_new = seg_t[:,:,clip_min:clip_max]
        seg_l_new = seg_l[:,:,clip_min:clip_max]

        if np.abs(clip_max - seg_l.shape[2])>10:
            print(t_seg)
        ct_NIFTI = nib.Nifti1Image(ct_new, whole_ct.affine, whole_ct.header)
        ct_NIFTI.to_filename(os.path.join(path_save, t_seg))

        segl_NIFTI = nib.Nifti1Image(seg_l_new, whole_ct.affine)
        segl_NIFTI.to_filename(os.path.join(path_save, t_seg[:7]+'_lung.nii.gz'))

        segt_NIFTI = nib.Nifti1Image(seg_t_new, whole_ct.affine)
        segt_NIFTI.to_filename(os.path.join(path_save, t_seg[:7]+'_tumor.nii.gz'))
    