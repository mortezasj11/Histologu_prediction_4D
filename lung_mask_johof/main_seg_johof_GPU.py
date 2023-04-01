from lungmask import mask
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import pathlib
import torch

if __name__ == '__main__':
   
    root_dir =  '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/'  #pathlib.Path.cwd()  
    nifty_root = os.path.join(root_dir, "CT")
    out_root = os.path.join(root_dir, "Mask_johof")
    
    image_names = sorted([ele for ele in os.listdir(nifty_root) if ele.endswith(".nii.gz")])
    image_paths = [os.path.join(nifty_root, ele) for ele in image_names]

    # Load different kind of models
    model = mask.get_model('unet','R231')  #model = mask.get_model('unet','LTRCLobes')
    model.to(torch.device('cuda')) 

    for ind, cur_img_path in enumerate(image_paths):
            file_name = os.path.basename(cur_img_path).split('.', 1)[0]
            file_name2=file_name.split('_',1)[1]
            print("Segmenting {} {:3d}/{:3d}".format(file_name, ind+1, len(image_paths)))
            #image = nib.load(cur_img_path).get_fdata().astype(np.float32)
            #input_image = sitk.ReadImage(image_paths[ind])
            input_image = nib.load(cur_img_path).get_fdata().astype(np.float32)
            breakpoint()
            segmentation = mask.apply_fused(torch.from_numpy(input_image).to(torch.device('cuda')), model)  # default model is U-net(R231)
            #segmentation = mask.apply(torch.from_numpy(input_image).to(torch.device('cuda')), model)  # default model is U-net(R231)

            out = sitk.GetImageFromArray(segmentation)
            out.CopyInformation(input_image) #new_image = nib.Nifti1Image(segmentation,affine=np.eye(4))
            CT_Seg_path = os.path.join(out_root, file_name2+'_lm.nii.gz')
            sitk.WriteImage(out, CT_Seg_path) 

print('Complted !!!')