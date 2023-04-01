
# Copy the Histology folder

# docker run -it --rm --gpus all --shm-size=200G --user $(id -u):$(id -g) --cpuset-cpus=51-74 \
-v /rsrch1/ip/XXXXXXX/XXXXXXX/Histology:/home/XXXXXXX/Histology \
-v /rsrch1/ip/msalehjahromi/data:/Data \
--name aname XXXXXX:latest


# I myself need to look at the architecture, I rewrite the test part:
# You two run with diff params:


    i. lr (schedule line 275)
    ii. Data Augmentation (200)
        i. rand_p ... ()
    iii. loss ( ~ 270 )   713-319-224
        i. loss_BCE = torch.nn.CrossEntropyLoss()  Add the weight here
        ii. Look for other losses, ask Pping and Jia
        iii.  t_loss = loss_func( t_loss_label, t_loss_img) + lambda * torch.log10(l1_reg)/1e8
                i. we can start with lambda 0, we expect that we overfit (highest acc in training).
                ii. we can bring the l1_norm to loss_func 
                ii. somple weight loss for two of the losses lambda1*loss_AE + (1-lambda1)*loss_BCE


cd /home/msalehjahromi/miniconda/envs/py385/lib/python3.8/site-packages/monai/data
vim utils.py
Esc, Shift+R, typing, Esc, :wq , Enter




Double check the data csv, combination and ... train_files
checking train_files , val_files, test_files was done!

Different Scheduling of lr should be applied:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=lr_decay,  min_lr=1e-5)


MultiLabel_Acc function need to be double checked.
Calculating accuracy need to be double checked.
Step 6 needs to be rewrite


os.environ["CUDA_VISIBLE_DEVICES"] = '2'