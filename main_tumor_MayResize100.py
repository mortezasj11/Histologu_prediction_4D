import os
import numpy as np
from tqdm import trange
from datetime import datetime
from utils import plot_confusion_matrix, plt2arr, plot_ROC, plot_age, Tensorboard, plot_histogram
from sklearn.metrics import confusion_matrix,balanced_accuracy_score, accuracy_score,roc_curve, auc
from sklearn.preprocessing import label_binarize
from resnet_model import ResNet, BasicBlock
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas as pd

import monai
from monai.data import DataLoader, decollate_batch, list_data_collate, Dataset, CacheDataset
from monai.transforms import (RandSpatialCropd,SpatialCropd, Resized, AsChannelFirstd,Compose,LoadImaged,Lambdad,RandAffined,RandAdjustContrastd,RandZoomd,EnsureTyped)
#from monai.apps import DecathlonDataset   # load the image data
#from monai.config import print_config
#from monai.losses import DiceLoss
#from monai.metrics import DiceMetric
#from monai.networks.nets import UNet, AutoEncoder, VarAutoEncoder
#from monai.utils import set_determinism
#from monai.visualize import plot_2d_or_3d_image, GradCAM, CAM
#from torchinfo import summary

#from fastai.layers import MSELossFlat, CrossEntropyFlat
#import glob as glob
#import Fn_model
#set_determinism(seed=123)



class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
    def forward(self, L0,L1,L2 ):
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * L0 + self.log_vars[0]
        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * L1 + self.log_vars[1]
        precision0 = torch.exp(-self.log_vars[2])
        loss2 = precision0 * L2 + self.log_vars[2]
        #precision1 = torch.exp(-self.log_vars[3])
        #loss3 = precision1 * L3 + self.log_vars[3]
        return loss0 + loss1 + loss2 

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ViT_model(nn.Module):
    def __init__(self,drop_rate, in_channels):
        super(ViT_model, self).__init__()
        self.backbone = monai.networks.nets.ViT(in_channels=in_channels, img_size=(160,160,160), patch_size = [20,20,20] ,pos_embed='conv', classification=True, num_classes=2, dropout_rate=drop_rate, spatial_dims=3)
    def forward(self, x):  # define network
        label,_ = self.backbone(x)  # _ is a list, len(_)=12,    len(_[1][0])=3       len(_[1][0][0])=217
        return label


class Dense_model(nn.Module):
    def __init__(self,drop_rate, in_channels, dense='dense121'):
        super(Dense_model, self).__init__()
        if dense =='dense121':
            self.backbone = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1, dropout_prob=drop_rate)
            last_layer_n = 1024
        elif dense == 'dense201':
            self.backbone = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=in_channels, out_channels=1, dropout_prob=drop_rate)
            last_layer_n = 1920
        self.backbone.class_layers.out = Identity()
        # self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=3, num_classes=1024)
        self.hist_drop_layer0 = nn.Dropout(p=drop_rate)
        self.fc1 = torch.nn.Linear(last_layer_n, 32)
        self.drop_layer1 = nn.Dropout(p=drop_rate)
        self.fc2 = torch.nn.Linear(32, 2)
        self.drop_layer2 = nn.Dropout(p=0.1)
        self.fc_smoking = torch.nn.Linear(last_layer_n, 2)
        self.fc2_gender = torch.nn.Linear(last_layer_n, 2)
        self.fc3_age = torch.nn.Linear(last_layer_n, 1)

    def forward(self, x):  # define network
        encoded = self.backbone(x)
        encoded = self.hist_drop_layer0(encoded)
        label32 = self.fc1(encoded)
        label32 = self.drop_layer1(label32)
        hist = self.fc2(label32)                 #label16 = self.drop_layer2(label16)
        hist = self.drop_layer2(hist)                   #label = torch.softmax(label,dim=1)
        smoking = self.fc_smoking(encoded)
        gender = self.fc2_gender(encoded)
        age = 7*torch.sigmoid(self.fc3_age(encoded))-3.5
        return hist, age, smoking, gender

def focal_loss_from_bce_loss(bce_loss, alpha=1, gamma=1.5):
    pt = torch.exp(-bce_loss)
    F_loss = alpha * (1-pt)**gamma * bce_loss
    return F_loss


def main(hparams):
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams['GPU_number']
    #torch.cuda.empty_cache()
    for i in hparams:
        print(i,': ',hparams[i])
    model_train = True
    model_test = False
    rand_p = hparams['rand_p']  # data augmentation random proportion
    lr = hparams['lr']
    drop_rate = hparams['drop_rate'] # model dropout rate
    max_epochs = hparams['max_epochs']     # epoch number
    train_batch = hparams['train_batch']
    val_batch = 2
    test_batch = 2
    skip_epoch_model = 100 # not saving the model for initial fluctuation
    Coefs = hparams['class_coef']
    ######################################################################################################
    print('-------------------start, step 1: Loading csv files --------------------------')

    files_dir = '/Data/Histology/Nature_Gemini_4d_Apr2022/GeminiNature_4D'
    train_os = pd.read_csv('/Data/Histology/Nature_Gemini_4d_Apr2022/TrainValTest/NatureGeminiAll_NoTobacco/train.csv')
    val_os   = pd.read_csv('/Data/Histology/Nature_Gemini_4d_Apr2022/TrainValTest/NatureGeminiAll_NoTobacco/val.csv')
    test_os  =  pd.read_csv('/Data/Histology/Nature_Gemini_4d_Apr2022/TrainValTest/NatureGeminiAll_NoTobacco/val.csv')

    list_path_t = [os.path.join( files_dir, i+'.nii.gz') for i in train_os["ID"].tolist()]
    list_hist_t = train_os["Histology012"].tolist()
    list_age_t = train_os["Age"].tolist()
    #list_smoking_t = train_os["smoking"].tolist()
    list_gender_t = train_os["gender"].tolist()
    train_files = [{"input": in_img, "histology": hist, "age":age,  "gender": gender}
                for in_img, hist, age, gender in
                zip(list_path_t, list_hist_t, list_age_t,  list_gender_t)]

    list_path_v = [os.path.join( files_dir, i+'.nii.gz') for i in val_os["ID"].tolist()]
    list_hist_v = val_os["Histology012"].tolist()
    list_age_v = val_os["Age"].tolist()
    #list_smoking_v = val_os["smoking"].tolist()
    list_gender_v = val_os["gender"].tolist()
    val_files = [{"input": in_img, "histology": hist, "age":age,  "gender": gender}
                for in_img, hist, age,  gender in 
                zip(list_path_v, list_hist_v, list_age_v,  list_gender_v)]

    list_path_s = [os.path.join( files_dir, i+'.nii.gz') for i in test_os["ID"].tolist()]
    list_hist_s = test_os["Histology012"].tolist()
    list_age_s = test_os["Age"].tolist()
    #list_smoking_s = test_os["smoking"].tolist()
    list_gender_s = test_os["gender"].tolist()
    test_files = [{"input": in_img, "histology": hist, "age":age,  "gender": gender}
                for in_img, hist, age,  gender in
                zip(list_path_s, list_hist_s, list_age_s, list_gender_s)]

    ######################################################################################################
    print('-------------------start, step 2: define the Augmentations--------------------------')
    if hparams["Aug"].lower()=='no':
        rand_p_affine = 0
        rand_p_contrast = 0.1
        rotate_range, shear_range, translate_range = None, None, None
    elif hparams["Aug"].lower()=='simple':
        print('{} Aug is being used.'.format(hparams["Aug"]))
        rotate_range, shear_range, translate_range = ((-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)), None, None
        rand_p_affine = rand_p
        rand_p_contrast = 0.1
    else:
        print('{} Aug is being used.'.format(hparams["Aug"]))
        rotate_range, shear_range, translate_range =((-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)), (-0.05, 0.05, -0.05, 0.05,-0.05, 0.05), (0.05, 0.05, 0.05)
        rand_p_affine = rand_p
        rand_p_contrast = 0.1

    train_transforms = Compose(
            [LoadImaged(keys=["input"]),
            Lambdad(keys=["input"], func=lambda x: x[:,:,:,0:hparams['in_channels']]), #20:hparams['in_channels']
            AsChannelFirstd(keys=["input"], channel_dim=-1),
            RandAdjustContrastd(keys=["input"], prob=rand_p_contrast),
            RandZoomd(keys="input", prob=rand_p, min_zoom=0.80, max_zoom=1.20),
            EnsureTyped(keys=["input"]),
            Resized(keys=["input"], spatial_size=[128,128,128]),
            RandSpatialCropd(keys=["input"],roi_size=[100,100,100]),
            RandAffined(keys=["input"], prob=rand_p_affine, rotate_range=rotate_range, shear_range=shear_range, translate_range=translate_range)])
            #AddChanneld(keys=["input"]),
            #Resized(keys=["input"], spatial_size=[160,160,160]),  # augment, flip, rotate, intensity ...
            #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
            #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
            #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(1, 2)),
            #RandFlipd(keys=["input"], prob=rand_p),
        
    val_transforms = Compose(
        [LoadImaged(keys=["input"]),
        Lambdad(keys=["input"], func=lambda x: x[:,:,:,0:hparams['in_channels']]),  #2:hparams['in_channels']
        AsChannelFirstd(keys=["input"], channel_dim=-1),
        Resized(keys=["input"], spatial_size=[128,128,128]),
        RandSpatialCropd(keys=["input"], roi_size=[100,100,100]),
        EnsureTyped(keys=["input"])])
        #AddChanneld(keys=["input"]),
        #Resized(keys=["input"], spatial_size=[160,160,160]),

    ######################################################################################################
    print('-------------------start, step 3: define data loader--------------------------')

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms) #, cache_num=50, num_workers=20)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=10,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(), drop_last=True)

    # create a validation data
    val_ds = Dataset(data=val_files, transform=val_transforms) #, cache_num=40, num_workers=10)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=True, num_workers=10, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    # create a test data
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=True, num_workers=10, collate_fn=list_data_collate,
                             pin_memory=torch.cuda.is_available())

    ######################################################################################################
    print('-------------------start, step 4: define NET Optimizer Loss--------------------------')

    # model
    if hparams["model"].lower()=="resnet": # To be completed ... 
        model = ResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512])
    else:
        model = Dense_model(drop_rate, in_channels=hparams['in_channels'], dense= hparams['model']) #monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = model.cuda()

    # each losses
    loss_CE_hist    = torch.nn.CrossEntropyLoss(weight=hparams['hist_weight'].cuda(),reduction=hparams["reduction"])
    loss_CE_smoking = torch.nn.CrossEntropyLoss(weight=hparams['smoking_weight'].cuda(),reduction=hparams["reduction"])
    loss_CE_gender  = torch.nn.CrossEntropyLoss(weight=hparams['gender_weight'].cuda(),reduction=hparams["reduction"])
    loss_MSE        = torch.nn.MSELoss()

    # loss wrapper
    n_loss = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = MultiTaskLossWrapper(n_loss).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=lr_decay,  min_lr=1e-5)  #Comes with scheduler.step(val_epoch_loss)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams['schedular_step_size'], gamma=hparams['schedular_gamma'])  #Comes with scheduler.step() 
    
    ######################################################################################################
    print('-------------------start, step 5: model training--------------------------')

    if model_train:

        # Tensorboard
        time_stamp = "{0:%Y-%m-%d-T%H-%M-%S}".format(datetime.now())
        writer =SummaryWriter(log_dir='runs_NG_all/' + time_stamp +"__"+"_Focal_"+ str(hparams["Focal_loss"]) +hparams['name'] + hparams["Aug"] + "__" + str(hparams["rand_p"]*100) + "__" + str(hparams["GPU_number"]))
        
        # start a typical PyTorch training
        best_metric_acc_b = 0.65

        t = trange(max_epochs, desc=f"densenet survival -- epoch 0, avg loss: inf", leave=True)
        for epoch in t:
            #torch.cuda.empty_cache()
            model.train()
            step = 0
            t.set_description(f"epoch {epoch} started")
            for train_data in train_loader:
                # Reading True labels and input
                inputs = train_data["input"].cuda()
                label_hist_true = train_data["histology"].unsqueeze(1).cuda()
                label_age_true = train_data["age"].unsqueeze(1).cuda().float()
                #label_smoking_true = train_data["smoking"].unsqueeze(1).cuda()
                label_gender_true = train_data["gender"].unsqueeze(1).cuda()
                # Predinct
                optimizer.zero_grad()
                label_hist_pred, label_age_pred, label_smoking_pred, label_gender_pred = model(inputs)
                # Making list for true and pred labels
                if step == 0:
                    # true label lists
                    label_hist_true_list = label_hist_true
                    label_age_true_list = label_age_true
                    #label_smoking_true_list = label_smoking_true
                    label_gender_true_list = label_gender_true
                    # pred label lists
                    label_hist_pred_list = label_hist_pred
                    label_age_pred_list = label_age_pred
                    label_smoking_pred_list   = label_smoking_pred
                    label_gender_pred_list = label_gender_pred
                else:
                    # true label lists
                    label_hist_true_list = torch.cat((label_hist_true_list, label_hist_true),0)
                    label_age_true_list  = torch.cat((label_age_true_list, label_age_true),0)
                    #label_smoking_true_list  = torch.cat((label_smoking_true_list, label_smoking_true),0)
                    label_gender_true_list  = torch.cat((label_gender_true_list, label_gender_true),0)
                    # pred label lists
                    label_hist_pred_list  = torch.cat((label_hist_pred_list, label_hist_pred),0)
                    label_age_pred_list  = torch.cat((label_age_pred_list, label_age_pred),0)
                    #label_smoking_pred_list  = torch.cat((label_smoking_pred_list, label_smoking_pred),0)
                    label_gender_pred_list  = torch.cat((label_gender_pred_list, label_gender_pred),0)

                # L1 regularization 
                l1_reg = None
                for W in model.parameters():
                    if l1_reg is None:
                        l1_reg = torch.abs(W).sum()
                    else:
                        l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
                # each loss
                loss_hist    = Coefs['hist_coed']    * loss_CE_hist   (   label_hist_pred    , label_hist_true.view(-1)    )
                #loss_smoking = Coefs['smoking_coef'] * loss_CE_smoking(   label_smoking_pred , label_smoking_true.view(-1) )
                loss_gender  = Coefs['gender_coef']  * loss_CE_gender (   label_gender_pred  , label_gender_true.view(-1)  )
                loss_age     = Coefs['age_coef']     * loss_MSE       (   label_age_pred     , label_age_true              )

                if hparams["Focal_loss"]:
                    loss_hist    = focal_loss_from_bce_loss(loss_hist   )
                    #loss_smoking = focal_loss_from_bce_loss(loss_smoking)
                    loss_gender  = focal_loss_from_bce_loss(loss_gender )

                # loss form with wrapper
                if hparams['l1_loss_mode']=='mul':
                    loss = loss_func(loss_hist, loss_age, loss_gender)*torch.log10(l1_reg)
                elif hparams['l1_loss_mode']=='add':
                    loss = loss_func(loss_hist, loss_age, loss_gender) + hparams['lambda_l1']*l1_reg

                # Making list for losses
                loss_hist_list = loss_hist.unsqueeze(0) if step == 0 else torch.cat((loss_hist_list, loss_hist.unsqueeze(0)), dim=0)
                #loss_smoking_list = loss_smoking.unsqueeze(0) if step == 0 else torch.cat((loss_smoking_list, loss_smoking.unsqueeze(0)), dim=0)
                loss_gender_list = loss_gender.unsqueeze(0) if step == 0 else torch.cat((loss_gender_list, loss_gender.unsqueeze(0)), dim=0)
                loss_age_list = loss_age.unsqueeze(0) if step == 0 else torch.cat((loss_age_list, loss_age.unsqueeze(0)), dim=0)
                loss_list = loss.unsqueeze(0) if step == 0 else torch.cat((loss_list, loss.unsqueeze(0)),dim=0)

                # updating ...
                loss.backward()
                optimizer.step()
                step += 1
            scheduler.step()


            #'----------------------------   Tensorboard train   ------------------------------'
            if epoch%hparams["tensorbord_val_interval"]==0:
                # input
                if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                    writer.add_images('Train/z_Im',  inputs[0,0,:,:,80].unsqueeze(0).unsqueeze(0), epoch)


                # losses
                writer.add_scalar("Train/Loss_Hist",         loss_hist_list.mean().item(),     epoch)
                writer.add_scalar("Train/Loss_Age",          loss_age_list.mean().item(),      epoch)
                #writer.add_scalar("Train/Loss_Smoking",    loss_smoking_list.mean().item(),    epoch)
                writer.add_scalar("Train/Loss_Gender",  loss_gender_list.mean().item(),        epoch)
                writer.add_scalar("Train/Loss_Overall",      loss_list.mean().item(),          epoch)


                # hist
                t_hist_tensorboard = Tensorboard(label_hist_pred_list, label_hist_true_list)
                _,t_acc_hist_b = t_hist_tensorboard.accuracy()
                t_two_hist_cm = t_hist_tensorboard.confusion_matrix(classes_name = ["AD/other", "SQ"])
                t_image_hist_roc = t_hist_tensorboard.roc(title='ROC_Hist')
                writer.add_scalar("Train/Acc_Hist",          t_acc_hist_b.item(),              epoch)
                if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                    writer.add_images('Train/CM_Hist',               t_two_hist_cm,                epoch) 
                    writer.add_images('Train/ROC_Hist',           t_image_hist_roc,                epoch)


                # age
                t_image_age = plot_age(label_age_pred_list.squeeze(), label_age_true_list.squeeze())
                #t_image_age = plot_histogram(label_age_pred_list.squeeze(), label_age_true_list.squeeze())
                _, t_fig = plt2arr(t_image_age)
                if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                    writer.add_images('Train/Age',           t_fig[:,0:3,:,:],                 epoch)


                # smoking
                #t_smoking_tensorboard = Tensorboard(label_smoking_pred_list, label_smoking_true_list)
                #_,t_acc_smoking_b = t_smoking_tensorboard.accuracy()
                #t_two_smoking_cm = t_smoking_tensorboard.confusion_matrix(classes_name = ["Cu/Fo", "Never"])
                #t_image_smoking_roc = t_smoking_tensorboard.roc(title='ROC_Smoke')
                #writer.add_scalar("Train/Acc_Smoking",       t_acc_smoking_b.item(),          epoch)
                #writer.add_images('Train/CM_Smoking',         t_two_smoking_cm,               epoch)
                #writer.add_images('Train/ROC_Smoking',          t_image_smoking_roc,          epoch)


                # gender
                t_gender_tensorboard = Tensorboard(label_gender_pred_list, label_gender_true_list)
                _,t_acc_gender_b = t_gender_tensorboard.accuracy()
                t_two_gender_cm = t_gender_tensorboard.confusion_matrix(classes_name = ["Female", "Male"])
                t_image_gender_roc = t_gender_tensorboard.roc(title='ROC_Gender')
                writer.add_scalar("Train/Acc_Gender",       t_acc_gender_b.item(),          epoch)
                if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                    writer.add_images('Train/CM_Gender',         t_two_gender_cm,               epoch)
                    writer.add_images('Train/ROC_Gender',          t_image_gender_roc,          epoch)


                # hyperparameters
                writer.add_scalar("hparam/initial_lr",           hparams['lr'],                  epoch)
                writer.add_scalar("hparam/rand_p",               hparams['rand_p'],              epoch)
                writer.add_scalar("hparam/drop_rate",            hparams['drop_rate'],           epoch)
                writer.add_scalar("hparam/max_epochs",           hparams['max_epochs'],          epoch)
                writer.add_scalar("hparam/train_batch",          hparams['train_batch'],         epoch)
                writer.add_scalar("hparam/schedular_step_size",  hparams['schedular_step_size'], epoch)
                writer.add_scalar("hparam/schedular_gamma",      hparams['schedular_gamma'],     epoch)
                writer.add_scalar("hparam/lambda_l1",            hparams['lambda_l1'],           epoch)
                writer.add_scalar("hparam/in_channels",          hparams['in_channels'],         epoch)
                writer.add_scalar("hparam/L1 loss",              l1_reg.item(),                  epoch)
                writer.add_scalar("hparam/lr",             optimizer.param_groups[0]['lr'],      epoch)
                writer.add_scalar("hparam/age_coef",    hparams['class_coef']['age_coef'],       epoch)
                writer.add_scalar("hparam/gender_coef", hparams['class_coef']['gender_coef'],    epoch)
                # weight histogram
                #for name, weight in model.named_parameters():
                    #writer.add_histogram(name,weight, epoch)
                    #writer.add_histogram(f'{name}.grad',weight.grad, epoch)


        #################################    VALIDATION    ###################################
            if epoch % hparams["tensorbord_val_interval"]==0:
                model.eval()

                with torch.no_grad():
                    step_val= 0
                    for val_data in val_loader:
                        inputs_val = val_data["input"].cuda()
                        label_hist_true_val = val_data["histology"].unsqueeze(1).cuda()
                        label_age_true_val = val_data["age"].unsqueeze(1).cuda().float()
                        #label_smoking_true_val = val_data["smoking"].unsqueeze(1).cuda()
                        label_gender_true_val = val_data["gender"].unsqueeze(1).cuda()

                        label_hist_pred_val, label_age_pred_val, label_smoking_pred_val, label_gender_pred_val = model(inputs_val)

                        if step_val == 0:
                            # true label lists
                            label_hist_true_list_val = label_hist_true_val
                            label_age_true_list_val = label_age_true_val
                            #label_smoking_true_list_val = label_smoking_true_val
                            label_gender_true_list_val = label_gender_true_val
                            # pred label lists
                            label_hist_pred_list_val = label_hist_pred_val
                            label_age_pred_list_val = label_age_pred_val
                            #label_smoking_pred_list_val   = label_smoking_pred_val
                            label_gender_pred_list_val = label_gender_pred_val

                        else:
                            # true label lists
                            label_hist_true_list_val = torch.cat((label_hist_true_list_val, label_hist_true_val),0)
                            label_age_true_list_val  = torch.cat((label_age_true_list_val, label_age_true_val),0)
                            #label_smoking_true_list_val  = torch.cat((label_smoking_true_list_val, label_smoking_true_val),0)
                            label_gender_true_list_val  = torch.cat((label_gender_true_list_val, label_gender_true_val),0)
                            # pred label lists
                            label_hist_pred_list_val  = torch.cat((label_hist_pred_list_val, label_hist_pred_val),0)
                            label_age_pred_list_val  = torch.cat((label_age_pred_list_val, label_age_pred_val),0)
                            #label_smoking_pred_list_val  = torch.cat((label_smoking_pred_list_val, label_smoking_pred_val),0)
                            label_gender_pred_list_val  = torch.cat((label_gender_pred_list_val, label_gender_pred_val),0)

                        # each loss
                        loss_hist_val    = Coefs['hist_coed']    * loss_CE_hist(   label_hist_pred_val    , label_hist_true_val.view(-1)    )
                        #loss_smoking_val = Coefs['smoking_coef'] * loss_CE_smoking(   label_smoking_pred_val , label_smoking_true_val.view(-1) )
                        loss_gender_val  = Coefs['gender_coef']  * loss_CE_gender(   label_gender_pred_val  , label_gender_true_val.view(-1)  )
                        loss_age_val     = Coefs['age_coef']     * loss_MSE(   label_age_pred_val    , label_age_true_val     )

                        if hparams["Focal_loss"]:
                            loss_hist_val = focal_loss_from_bce_loss(loss_hist_val)
                            #loss_smoking_val = focal_loss_from_bce_loss(loss_smoking_val)
                            loss_gender_val = focal_loss_from_bce_loss(loss_gender_val)

                        # loss wrapper
                        if hparams['l1_loss_mode']=='mul':
                            loss_val = loss_func(loss_hist_val, loss_age_val,  loss_gender_val)*torch.log10(l1_reg)

                        elif hparams['l1_loss_mode']=='add':
                            loss_val = loss_func(loss_hist_val, loss_age_val,  loss_gender_val) + hparams['lambda_l1']*l1_reg
                        
                        # Making list for losses
                        loss_hist_list_val = loss_hist_val.unsqueeze(0) if step_val == 0 else torch.cat((loss_hist_list_val, loss_hist_val.unsqueeze(0)), dim=0)
                        #loss_smoking_list_val = loss_smoking_val.unsqueeze(0) if step_val == 0 else torch.cat((loss_smoking_list_val, loss_smoking_val.unsqueeze(0)), dim=0)
                        loss_gender_list_val = loss_gender_val.unsqueeze(0) if step_val == 0 else torch.cat((loss_gender_list_val, loss_gender_val.unsqueeze(0)), dim=0)
                        loss_age_list_val = loss_age_val.unsqueeze(0) if step_val == 0 else torch.cat((loss_age_list_val, loss_age_val.unsqueeze(0)), dim=0)
                        loss_list_val = loss_val.unsqueeze(0) if step_val == 0 else torch.cat((loss_list_val, loss_val.unsqueeze(0)),dim=0)
                        # updating ... 
                        step_val += 1
                        #torch.cuda.empty_cache()

                    #'----------------------------   Tensorboard val   ------------------------------'
                    # input  Val
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val/z_Im',   inputs_val[0,0,:,:,80].unsqueeze(0).unsqueeze(0),  epoch)
                    # losses
                    writer.add_scalar("Val/Loss_Hist",         loss_hist_list_val.mean().item(),     epoch)
                    writer.add_scalar("Val/Loss_Age",          loss_age_list_val.mean().item(),      epoch)
                    #writer.add_scalar("Val/Loss_Smoking",    loss_smoking_list_val.mean().item(),    epoch)
                    writer.add_scalar("Val/Loss_Gender",  loss_gender_list_val.mean().item(),        epoch)
                    writer.add_scalar("Val/Loss_Overall",      loss_list_val.mean().item(),          epoch)

                    # hist
                    t_hist_tensorboard_val = Tensorboard(label_hist_pred_list_val, label_hist_true_list_val)
                    _,t_acc_hist_b_val = t_hist_tensorboard_val.accuracy()
                    t_two_hist_cm_val = t_hist_tensorboard_val.confusion_matrix(classes_name = [ "AD/other","SQ"])
                    t_image_hist_roc_val = t_hist_tensorboard_val.roc(title='ROC_Hist')
                    writer.add_scalar("Val/Acc_Hist",          t_acc_hist_b_val.item(),              epoch)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val/CM_Hist',               t_two_hist_cm_val,                epoch)
                        writer.add_images('Val/ROC_Hist',           t_image_hist_roc_val,                epoch)

                    # age
                    t_image_age_val = plot_age(label_age_pred_list_val.squeeze(), label_age_true_list_val.squeeze())
                    #t_image_age_val = plot_histogram(label_age_pred_list_val.squeeze(), label_age_true_list_val.squeeze())
                    _, v_fig = plt2arr(t_image_age_val)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val/Age',           v_fig[:,0:3,:,:],                 epoch)

                    # smoking
                    # t_smoking_tensorboard_val = Tensorboard(label_smoking_pred_list_val, label_smoking_true_list_val)
                    # _,t_acc_smoking_b_val = t_smoking_tensorboard_val.accuracy()
                    # t_two_smoking_cm_val = t_smoking_tensorboard_val.confusion_matrix(classes_name = ["Cu/Fo", "Never"])
                    # t_image_smoking_roc_val = t_smoking_tensorboard_val.roc(title='ROC_Smoke')
                    # writer.add_scalar("Val/Acc_Smoking",       t_acc_smoking_b_val.item(),          epoch)
                    # writer.add_images('Val/CM_Smoking',         t_two_smoking_cm_val,               epoch)
                    # writer.add_images('Val/ROC_Smoking',          t_image_smoking_roc_val,          epoch)

                    # gender
                    t_gender_tensorboard_val = Tensorboard(label_gender_pred_list_val, label_gender_true_list_val)
                    _,t_acc_gender_b_val = t_gender_tensorboard_val.accuracy()
                    t_two_gender_cm_val = t_gender_tensorboard_val.confusion_matrix(classes_name = ["Female", "Male"])
                    t_image_gender_roc_val = t_gender_tensorboard_val.roc(title='ROC_Gender')
                    writer.add_scalar("Val/Acc_Gender",       t_acc_gender_b_val.item(),          epoch)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val/CM_Gender',         t_two_gender_cm_val,               epoch)
                        writer.add_images('Val/ROC_Gender',          t_image_gender_roc_val,          epoch)



            ################################# TEST ###################################

            if model_test and epoch % hparams["tensorbord_val_interval"]==0:
                model.eval()
                with torch.no_grad():
                    step_test= 0
                    for test_data in test_loader:

                        inputs_test = test_data["input"].cuda()
                        label_hist_true_test = test_data["histology"].unsqueeze(1).cuda()
                        label_age_true_test = test_data["age"].unsqueeze(1).cuda()
                        #label_smoking_true_test = test_data["smoking"].unsqueeze(1).cuda()
                        label_gender_true_test = test_data["gender"].unsqueeze(1).cuda()

                        label_hist_pred_test, label_hist_pred_test, label_age_pred_test, label_smoking_pred_test, label_gender_pred_test = model(inputs_test)

                        if step_test == 0:
                            # true label lists
                            label_hist_true_list_test = label_hist_true_test
                            label_age_true_list_test = label_age_true_test
                            #label_smoking_true_list_test = label_smoking_true_test
                            label_gender_true_list_test = label_gender_true_test
                            # pred label lists
                            label_hist_pred_list_test = label_hist_pred_test
                            label_age_pred_list_test = label_age_pred_test
                            #label_smoking_pred_list_test   = label_smoking_pred_test
                            label_gender_pred_list_test = label_gender_pred_test
                        else:
                            # true label lists
                            label_hist_true_list_test = torch.cat((label_hist_true_list_test, label_hist_true_test),0), 
                            label_age_true_list_test  = torch.cat((label_age_true_list_test, label_age_true_test),0)
                            #label_smoking_true_list_test  = torch.cat((label_smoking_true_list_test, label_smoking_true_test),0)
                            label_gender_true_list_test  = torch.cat((label_gender_true_list_test, label_gender_true_test),0)
                            # pred label lists
                            label_hist_pred_list_test  = torch.cat((label_hist_pred_list_test, label_hist_pred_test),0)
                            label_age_pred_list_test  = torch.cat((label_age_pred_list_test, label_age_pred_test),0)
                            #label_smoking_pred_list_test  = torch.cat((label_smoking_pred_list_test, label_smoking_pred_test),0)
                            label_gender_pred_list_test  = torch.cat((label_gender_pred_list_test, label_gender_pred_test),0)


                        # each loss
                        loss_hist_test    = Coefs['hist_coed']    * loss_CE_hist(   label_hist_pred_test    , label_hist_true_test.view(-1)    )
                        #loss_smoking_test = Coefs['smoking_coef'] * loss_CE_smoking(   label_smoking_pred_test , label_smoking_true_test.view(-1) )
                        loss_gender_test  = Coefs['gender_coef']  * loss_CE_gender(   label_gender_pred_test  , label_gender_true_test.view(-1)  )
                        loss_age_test     = Coefs['age_coef']     * loss_MSE(   label_age_pred_test    , label_age_true_test     )

                        if hparams["Focal_loss"]:
                            loss_hist_test = focal_loss_from_bce_loss(loss_hist_test)
                            #loss_smoking_test = focal_loss_from_bce_loss(loss_smoking_test)
                            loss_gender_test = focal_loss_from_bce_loss(loss_gender_test)

                        # loss wrapper
                        if hparams['l1_loss_mode']=='mul':
                            loss_test = loss_func(loss_hist_test, loss_age_test,  loss_gender_test)*torch.log10(l1_reg)
                        elif hparams['l1_loss_mode']=='add':
                            loss_test = loss_func(loss_hist_test, loss_age_test,  loss_gender_test) + hparams['lambda_l1']*l1_reg
                        
                        # Making list for losses
                        loss_hist_list_test = loss_hist_test.unsqueeze(0) if step_test == 0 else torch.cat((loss_hist_list_test, loss_hist_test.unsqueeze(0)), dim=0)
                        #loss_smoking_list_test = loss_smoking_test.unsqueeze(0) if step_test == 0 else torch.cat((loss_smoking_list_test, loss_smoking_test.unsqueeze(0)), dim=0)
                        loss_gender_list_test = loss_gender_test.unsqueeze(0) if step_test == 0 else torch.cat((loss_gender_list_test, loss_gender_test.unsqueeze(0)), dim=0)
                        loss_age_list_test = loss_age_test.unsqueeze(0) if step_test == 0 else torch.cat((loss_age_list_test, loss_age_test.unsqueeze(0)), dim=0)
                        loss_list_test = loss_test.unsqueeze(0) if step_test == 0 else torch.cat((loss_list_test, loss_test.unsqueeze(0)),dim=0)
                        
                        # updating ... 
                        step_test += 1
                        #torch.cuda.empty_cache()

                    #'----------------------------   Tensorboard test   ------------------------------'
                    # input  test
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val_2/z_Im',   inputs_test[0,0,:,:,80].unsqueeze(0).unsqueeze(0),  epoch)

                    # losses
                    writer.add_scalar("Val_2/Loss_Hist",         loss_hist_list_test.mean().item(),     epoch)
                    writer.add_scalar("Val_2/Loss_Age",          loss_age_list_test.mean().item(),      epoch)
                    #writer.add_scalar("Val_2/Loss_Smoking",    loss_smoking_list_test.mean().item(),    epoch)
                    writer.add_scalar("Val_2/Loss_Gender",  loss_gender_list_test.mean().item(),        epoch)
                    writer.add_scalar("Val_2/Loss_Overall",      loss_list_test.mean().item(),          epoch)

                    # hist
                    t_hist_tensorboard_test = Tensorboard(label_hist_pred_list_test, label_hist_true_list_test)
                    _,t_acc_hist_b_test = t_hist_tensorboard_test.accuracy()
                    t_two_hist_cm_test = t_hist_tensorboard_test.confusion_matrix(classes_name = ["AD/other", "SQ"])
                    t_image_hist_roc_test = t_hist_tensorboard_test.roc(title='ROC_Hist')
                    writer.add_scalar("Val_2/Acc_Hist",          t_acc_hist_b_test.item(),              epoch)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val_2/CM_Hist',               t_two_hist_cm_test,                epoch)
                        writer.add_images('Val_2/ROC_Hist',           t_image_hist_roc_test,                epoch)

                    # age
                    t_image_age_test = plot_age(label_age_pred_list_test.squeeze(), label_age_true_list_test.squeeze())
                    _, s_fig = plt2arr(t_image_age_test)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Train/Age',           s_fig[:,0:3,:,:],                 epoch)

                    # smoking
                    # t_smoking_tensorboard_test = Tensorboard(label_smoking_pred_list_test, label_smoking_true_list_test)
                    # _,t_acc_smoking_b_test = t_smoking_tensorboard_test.accuracy()
                    # t_two_smoking_cm_test = t_smoking_tensorboard_test.confusion_matrix(classes_name = ["Cu/Fo", "Never"])
                    # t_image_smoking_roc_test = t_smoking_tensorboard_test.roc(title='ROC_Smoke')
                    # writer.add_scalar("Val_2/Acc_Smoking",       t_acc_smoking_b_test.item(),          epoch)
                    # writer.add_images('Val_2/CM_Smoking',         t_two_smoking_cm_test,               epoch)
                    # writer.add_images('Val_2/ROC_Smoking',          t_image_smoking_roc_test,          epoch)

                    # gender
                    t_gender_tensorboard_test = Tensorboard(label_gender_pred_list_test, label_gender_true_list_test)
                    _,t_acc_gender_b_test = t_gender_tensorboard_test.accuracy()
                    t_two_gender_cm_test = t_gender_tensorboard_test.confusion_matrix(classes_name = ["Female", "Male"])
                    t_image_gender_roc_test = t_gender_tensorboard_test.roc(title='ROC_Gender')
                    writer.add_scalar("Val_2/Acc_Gender",       t_acc_gender_b_test.item(),          epoch)
                    if hparams['Add_images']==True and str(epoch) in ['1', '5', '20', '80', '200']:
                        writer.add_images('Val_2/CM_Gender',         t_two_gender_cm_test,               epoch)
                        writer.add_images('Val_2/ROC_Gender',          t_image_gender_roc_test,          epoch)

            # model save
            if epoch > skip_epoch_model:
                if t_acc_hist_b_val > best_metric_acc_b:
                    torch.save(model.state_dict(), hparams["name"] + str(np.round(t_acc_hist_b_val*100)) + ".pth")

        writer.close()


if __name__ == "__main__":
    hparams = { 'lr' : 8e-4,
                'rand_p' : 0.34,
                'drop_rate' : 0.34,
                'max_epochs' : 200,
                'train_batch' : 25,
                'hist_weight' : torch.tensor([801/999, 198/999 ]),    # 198 801
                'gender_weight' : torch.tensor([0.55, 0.45 ]),        # 434 565
                'smoking_weight' : torch.tensor([191/999, 808/999 ]), # 808 191
                'class_coef' : {'hist_coed':1.0, 'smoking_coef':1.0, 'gender_coef':5.0, 'age_coef':100},
                'reduction':'sum',       # "none", 'mean', "sum"
                'schedular_step_size':5, # every 'schedular_step_size' epochs the lr will be multiplied by 'schedular_gamma'
                'schedular_gamma':0.8,   # every 'schedular_step_size' epochs the lr will be multiplied by 'schedular_gamma'
                'l1_loss_mode':'mul',    # 'mul' or 'add'
                'lambda_l1':2e-7,        # only when l1_loss_mode is on 'add'
                'tensorbord_val_interval':1,
                'in_channels': 3,
                'Focal_loss':False,
                'model':'dense121', #'dense201'          # 'Dense' 'resnet' 'ViT'  to be completed ... 
                'Add_images':False,
                'Aug': 'NOTsimple',        # 'no', 'simple' or anything else
                'name':'_Resize_',
                'GPU_number': '5'}


    main(hparams)
    # Notes
        #apparentlen lr of 0.8e-4 is good and 3e-2 is too much. 
    # Why Freezing:
        #1st guess: Augmentation!!
        #loss_Wrapper?
        #empty_cache?
        #Multiplying loss?
        #dense201 
        #could it be because of the batch_size????????



