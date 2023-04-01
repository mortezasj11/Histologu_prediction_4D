from lungmask import mask
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import pathlib
import torch

if __name__ == '__main__':
   
    root_dir =  '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/'  #pathlib.Path.cwd()  
    #nifty_root = os.path.join(root_dir, "Nifti_files", "Input")
    nifty_root = os.path.join(root_dir, "CT")
    out_root = os.path.join(root_dir, "Mask_johof")
    
    image_names = sorted([ele for ele in os.listdir(nifty_root) if ele.endswith(".nii.gz")])
    image_paths = [os.path.join(nifty_root, ele) for ele in image_names]
    #patient_list = sorted([os.path.split(x)[1] for x in subdir_lists[1:]])
    #print(patient_list)
    
    # Load different kind of models
    #model = mask.get_model('unet','LTRCLobes')
    #model = mask.get_model('unet','R231')

    for ind, cur_img_path in enumerate(image_paths):
            file_name = os.path.basename(cur_img_path).split('.', 1)[0]
            file_name2=file_name.split('_',1)[1]
            print("Segmenting {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))
            #image = nib.load(cur_img_path).get_fdata().astype(np.float32)
            #CT_path = os.path.join(nifty_root, patient, 'CT.nii')
            #CT_Seg_path = os.path.join(out_root, file_name,'_lungMask.nii') 
            CT_Seg_path = os.path.join(out_root, file_name2+'_lm.nii.gz')
            #print(CT_path)
        
            #input_image = sitk.ReadImage(CT_path)
            input_image = sitk.ReadImage(image_paths[ind])
            segmentation = mask.apply_fused(input_image)  # default model is U-net(R231)
            #print(segmentation.type())
            #model = mask.get_model('unet','LTRCLobes')
            #segmentation = mask.apply(input_image, model)
            #segmentation = mask.apply_fused(input_image)
            #new_image = nib.Nifti1Image(segmentation,affine=np.eye(4))
            out = sitk.GetImageFromArray(segmentation)
            out.CopyInformation(input_image) 
            sitk.WriteImage(out, CT_Seg_path) 
            # segmentation.WriteImage(nifty_root, 'CT_Seg.nii')
            # new_image.imwrite(nifty_root, 'CT_Seg.nii')
            print('All good')
print('Complted !!!')