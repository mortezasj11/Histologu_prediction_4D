# You install git here in Dockerfile
    # RUN apt-get update && apt-get install -y --no-install-recommends \
    #   tzdata build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev python3-openslide \
    #   curl wget git sudo vim htop ca-certificates \
    #   && rm -rf /var/lib/apt/lists/*

# You add these
    # WORKDIR /home/${USER_NAME}
    # RUN pip install git+https://github.com/JoHof/lungmask

# The code will use GPU itself! Speciify one gpu in docker run '"device=3"'
    # docker run -it --gpus '"device=3"' --rm --shm-size=192G --user $(id -u):$(id -g) --cpuset-cpus=30-39 -v /rsrch1/ip/msalehjahromi/data:/Data --name lungmask simplelung:Mori

from lungmask import mask
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import pathlib
from joblib import Parallel, delayed
#import multiprocessing

def johof(i):
    file_name  = os.path.basename(image_paths[i]).split('.', 1)[0]
    print("Segmenting {} {:3d}/{:3d}".format(file_name, i+1, len(image_paths)))
    CT_Seg_path = os.path.join(out_root, file_name+'_seg.nii.gz')
    input_image = sitk.ReadImage(image_paths[i])
    segmentation = mask.apply_fused(input_image)  # default model is U-net(R231)     model = mask.get_model('unet','R231')
    out = sitk.GetImageFromArray(segmentation)
    out.CopyInformation(input_image) 
    sitk.WriteImage(out, CT_Seg_path)


if __name__ == '__main__':

    root_dir =  '/Data/Histology/Nature_Gemini_4d_Apr2022/GEMINI_WithTumorSeg/'
    nifty_root = os.path.join(root_dir, "CT")

    out_root = os.path.join(root_dir, "Mask_johof")
    os.makedirs(out_root, exist_ok=True)

    image_paths = sorted([os.path.join(nifty_root, ele) for ele in os.listdir(nifty_root) if (ele.endswith(".nii.gz") or ele.endswith(".nii") )])

    # Do it once to only download the JOHOF model
    segmentation = mask.apply_fused(  sitk.ReadImage(image_paths[0])  )

    inputs = range(len(image_paths))
    num_cores = 5   #on new gpu machine 5, on old 3 or 4, test it.
    results = Parallel(n_jobs=num_cores)(delayed(johof)(i) for i in inputs)

    print('All Completed !!!')

