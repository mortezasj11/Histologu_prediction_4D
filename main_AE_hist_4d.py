import os
import shutil
import tempfile
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from datetime import datetime
from utils import plot_confusion_matrix, plt2arr
from sklearn.metrics import confusion_matrix

import monai
from monai.apps import DecathlonDataset   # load the image data
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, list_data_collate, Dataset, CacheDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, AutoEncoder, VarAutoEncoder
from monai.transforms import (
    Activations,
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Lambdad,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandRotate90d,
    RandZoomd,
    RandShiftIntensityd,
    RandGaussianSmoothd,
    RandAxisFlipd,
    RandSpatialCropd,
    Rand3DElasticd,
    Resized,
    Spacingd,
    EnsureTyped,
    EnsureType,
    RandRotate90d
)
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image, GradCAM, CAM

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.autograd import Variable
from torchvision import models
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
#from fastai.layers import MSELossFlat, CrossEntropyFlat

import pandas as pd
import glob as glob
from sklearn.metrics import accuracy_score
import Fn_model

#set_determinism(seed=123)

######################################################################################################
def MultiLabel_Acc(Pred,Y):
    Pred = Pred.cpu().numpy()
    Y = Y.cpu().numpy()
    acc = None
    for i in range(len(Y[1,:])):
       if i == 0:
           acc = accuracy_score(Y[:,i],Pred[:,i])
       else:
           acc = np.concatenate((acc,accuracy_score(Y[:,i],Pred[:,i])),axis=None)

    return(acc)

def model_eval_gpu(model,datafile,val_transforms):
    data_loader = DataLoader(Dataset(data=datafile, transform=val_transforms), batch_size=1, shuffle=False, num_workers=10, collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available())

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        train_labelpred = None
        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            t_label = batch_data["histology"].cuda()
            output4,output5,feature  = model(inputs)

            if step == 0:
                train_label = t_label
                train_labelpred = output4
                all_feature = feature
            else:
                train_label = torch.cat((train_label, t_label),0)
                train_labelpred = torch.cat((train_labelpred, output4),0)
                all_feature = torch.cat((all_feature,feature),0)

            step += 1

            #t_label_pred = (train_labelpred >= 0.)
            _, t_label_pred = torch.max(train_labelpred,1)

            
            #t_acc = MultiLabel_Acc(t_label_pred, train_label)
            t_acc = accuracy_score(t_label_pred, train_label)

    print(f"\n model evaluation" f"\n Accuracy: {t_acc:.4f} ")

    return train_labelpred, t_label_pred, t_acc, all_feature

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, L0,L1 ):

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * L0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * L1 + self.log_vars[1]

        #precision2 = torch.exp(-self.log_vars[2])
        #loss2 = precision2 * L2 + self.log_vars[1]

        return loss0 + loss1


def main():
    ###********************************************************
    ###hyperparameter setting
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.empty_cache()
    verbose = False
    model_train = True
    model_pred = False
    model_visual = False
    model_test = False

    BEST_MODEL_NAME = "AE_multitask_11_1_model1-rep1"

    rand_p = 0.1  # data augmentation random proportion
    lr = 0.4e-3
    lr_decay = 0.6
    drop_rate = 0.1  # model dropout rate

    ### ME, Change from 5 to 2
    n_loss = 2

    max_epochs = 250     # epoch number
    train_batch = 4
    val_batch = 4
    test_batch = 4
    skip_epoch_model = 50 # not saving the model for initial fluctuation

    ###********************************************************
    ######################################################################################################

    files_dir = '/Data/Histology/Nature_4D_Organized/TrainValTest/GEMINI_Nature'

    train_os = pd.read_csv('/Data/Histology/Nature_4D_Organized/TrainValTest/train.csv')
    val_os   = pd.read_csv('/Data/Histology/Nature_4D_Organized/TrainValTest/val.csv')
    test_os  = pd.read_csv('/Data/Histology/Nature_4D_Organized/TrainValTest/test.csv')

    #train_os.sort_values('ID')
    #val_os.sort_values('ID')
    #test_os.sort_values('ID')
    
    #D_convert = {1:0, 2:0, 4:1, 5:1}

    list_path_t = [os.path.join( files_dir, i+'.nii.gz') for i in train_os["ID"].tolist()]
    list_Malig_t = train_os["Histology012"].tolist()
    train_files = [{"input": in_img, "histology": df15}
                   for in_img, df15 in
                   zip(list_path_t, list_Malig_t)]

    list_path_v = [os.path.join( files_dir, i+'.nii.gz') for i in val_os["ID"].tolist()]
    list_Malig_v = val_os["Histology012"].tolist()
    val_files = [{"input": in_img, "histology": df15}
                   for in_img, df15 in
                   zip(list_path_v, list_Malig_v)]

    list_path_s = [os.path.join( files_dir, i+'.nii.gz') for i in test_os["ID"].tolist()]
    list_Malig_s = test_os["Histology012"].tolist()
    test_files = [{"input": in_img, "histology": df15}
                   for in_img, df15 in
                   zip(list_path_s, list_Malig_s)]

    breakpoint()
    
    # checked
    ######################################################################################################
    print('-------------------start, step 2: define the transforms--------------------------')

    train_transforms = Compose(
            [
                LoadImaged(keys=["input"]),
                AsChannelFirstd(keys=["input"], channel_dim=-1),
                #Lambdad(keys=["input"], func=lambda x: x[:,:,:,0]),
                #AddChanneld(keys=["input"]),
                Resized(keys=["input"], spatial_size=[128,128,128]),  # augment, flip, rotate, intensity ...
                RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
                RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
                RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(1, 2)),
                RandAdjustContrastd(keys=["input"], prob=rand_p),
                RandFlipd(keys=["input"], prob=rand_p),
                RandZoomd(keys="input", prob=rand_p),
                EnsureTyped(keys=["input"]),
            ]
        ) #torch.Size([1, 3, 330, 330, 324]) without Resize
        
    val_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            AsChannelFirstd(keys=["input"], channel_dim=-1),
            #AddChanneld(keys=["input"]),
            Resized(keys=["input"], spatial_size=[128,128,128]),
            EnsureTyped(keys=["input"]),
        ]
    )
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
        pin_memory=torch.cuda.is_available(), drop_last=True
    )

    # create a validation data loadertest_ae.py
    val_ds = Dataset(data=val_files, transform=val_transforms) #, cache_num=40, num_workers=10)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=10, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    # load the test data
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, num_workers=10, collate_fn=list_data_collate,
                             pin_memory=torch.cuda.is_available())

    ######################################################################################################
    print('-------------------start, step 4: define NET--------------------------')

    # create Net, Loss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transfer learning
    #model = Dense_model(drop_rate) #monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    model = Fn_model.AutoEncoder_New(
        dimensions=3,
        in_channels=3,
        out_channels=3,
        num_res_units=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        inter_channels=(256,256),
        inter_dilations=(2,2),
        dropout=drop_rate)

    model = model.cuda()
    loss_func = MultiTaskLossWrapper(n_loss).to(device)
    loss_AE = torch.nn.L1Loss()
    loss_BCE = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=lr_decay,  min_lr=1e-5)  Comes with scheduler.step(val_epoch_loss)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)  #Comes with scheduler.step() 
    

    # visualize the NET
    examples = iter(train_loader)
    example_data = examples.next()

    #loss_MSE = torch.nn.MSELoss()
    #loss_BCE = torch.nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor([0.9,0.3,0.2,4,4,1.1,3,4,3,3,4,3,7,2,2,4,3,9,19,7]))
    #loss_BCE = torch.nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor([0.6]))
    #optimizer = torch.optim.Adamax(model.parameters(), lr)
    # print(example_data["OS"],example_data["time"])
    #summary(model, input_size=example_data["input"].shape)
    #writer.add_graph(model,example_data["input"].cuda())

    ######################################################################################################
    print('-------------------start, step 5: model training--------------------------')

    if model_train:

        time_stamp = "{0:%Y-%m-%d-T%H-%M-%S/}".format(datetime.now()) + BEST_MODEL_NAME
        writer = SummaryWriter(log_dir='runs/' + time_stamp + '__randp_p5__NoL1_Using_precision_posWeight_p6')

        # start a typical PyTorch training
        val_interval = 1
        best_metric_pfs = -1
        epoch_loss_values = list()
        val_epoch_loss_values = list()
        test_epoch_loss_values = list()

        t = trange(max_epochs,
            desc=f"densenet survival -- epoch 0, avg loss: inf", leave=True)
        
        for epoch in t:
            model.train()
            epoch_loss = 0
            step = 0
            train_label = None
            train_labelpred = None
            t.set_description(f"epoch {epoch + 1} started")

            for batch_data in train_loader:
                inputs = batch_data["input"].cuda()
                t_histology = batch_data["histology"].cuda()
                t_histology = t_histology.unsqueeze(1)
                t_label = t_histology

                optimizer.zero_grad()
                output4,output5,_ = model(inputs)

                torch.cuda.empty_cache()
                #################################define survival loss###################################
                if step == 0:
                    train_label = t_label
                    train_labelpred = output4
                else:
                    train_label = torch.cat((train_label, t_label),0)
                    train_labelpred = torch.cat((train_labelpred, output4),0)

                #######add L1 regularization + multitask learning
                l1_reg = None
                for W in model.parameters():
                    if l1_reg is None:
                        l1_reg = torch.abs(W).sum()
                    else:
                        l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

                #################################define auxiliary loss###################################

                t_loss_label = loss_BCE(output4, t_label.view(-1))
                # t_loss_img = loss_MSE(output5,inputs)
                t_loss_img = loss_AE(output5, inputs)
                #
                # t_loss = t_loss_os + w1 * t_loss_pfs + w2 * t_loss_age + w3 * t_loss_label + w4 * l1_reg

                #t_loss = loss_func(t_loss_pfs,t_loss_age,t_loss_label)*torch.log10(l1_reg)*t_loss_os
                '''ME
                t_loss = loss_func(t_loss_os, t_loss_pfs, t_loss_age, t_loss_label, t_loss_img) * torch.log10(l1_reg)
                '''
                ### Me added
                t_loss = loss_func( t_loss_label, t_loss_img) + 1e-8 * torch.log10(l1_reg)/1e8
                #t_loss = t_loss_label*t_loss_img*torch.log10(l1_reg)/1e5
                
                #t_loss = t_loss_label*t_loss_img*torch.log10(l1_reg)/1e5
                ###
                ###################################################################################################
                del inputs
                del t_histology

                if verbose:
                    print(f"\n training epoch: {epoch}, step: {step}")
                    print(f"\n L1 loss: {l1_reg:.4f}, total loss: {t_loss:.4f}")
                torch.cuda.empty_cache()
                ###################################################################################################
                step += 1
                t_loss.backward()
                optimizer.step()
                epoch_loss += t_loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("Train/overall loss: step", t_loss.item(), epoch_len * epoch + step)

                writer.add_scalar("L1 loss: step", l1_reg.item(), epoch_len * epoch + step)
                writer.add_scalar("Train/label loss: step", t_loss_label.item(), epoch_len * epoch + step)
                writer.add_scalar("Train/AE loss: step", t_loss_img.item(), epoch_len * epoch + step)

            with torch.no_grad():
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                #t_label_pred = (train_labelpred >= 0.)
                _, t_label_pred = torch.max(train_labelpred,1)
                #t_acc = MultiLabel_Acc(t_label_pred, train_label)
                t_acc = accuracy_score(t_label_pred.cpu().numpy(), train_label.cpu().numpy())


                writer.add_scalar("Train/accuracy: epoch", t_acc.item(), epoch)
                cm = confusion_matrix(train_label.cpu().detach().numpy(), t_label_pred.cpu().detach().numpy() )
                figure = plot_confusion_matrix(cm, class_names=['Ad','Sq','O'])
                _, torch_im = plt2arr(figure)
                writer.add_images('Train/CM', torch_im[:,0:3,:,:], epoch)

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_label = None
                    val_labelpred = None
                    val_epoch_loss = 0
                    val_step = 0
                    for val_data in val_loader:
                        v_inputs = val_data["input"].cuda()
                        v_histology = val_data["histology"].cuda()
                        v_histology = v_histology.unsqueeze(1)
                        v_label = v_histology

                        val_output4,val_output5,_ = model(v_inputs)

                        #################################define survival loss###################################
                        if val_step == 0:
                            val_label = v_label
                            val_labelpred = val_output4
                        else:
                            val_label = torch.cat((val_label,v_label),0)
                            val_labelpred = torch.cat((val_labelpred,val_output4),0)
                        #################################define auxiliary loss###################################

                        v_loss_label = loss_BCE(val_output4, v_label.view(-1))
                        # v_loss_img = loss_MSE(val_output5, v_inputs)
                        v_loss_img = loss_AE(val_output5, v_inputs)

                        ###################################################################################################
                        val_step += 1
                        ### ME added
                        v_loss = loss_func(v_loss_label, v_loss_img)
                        ###
                        #v_loss = loss_func(v_loss_pfs, v_loss_age, v_loss_label) * v_loss_os
                        val_epoch_len = len(val_ds) // val_loader.batch_size + 1
                        writer.add_scalar("Val/overall loss: step", v_loss.item(), val_epoch_len * epoch + val_step)

                        writer.add_scalar("Val/label loss: step", v_loss_label.item(), val_epoch_len * epoch + val_step)
                        writer.add_scalar("Val/AE loss: step", v_loss_img.item(),
                                          val_epoch_len * epoch + val_step)

                        torch.cuda.empty_cache()

                    val_epoch_loss /= val_step
                    #scheduler.step(val_epoch_loss)
                    scheduler.step()
                    val_epoch_loss_values.append(val_epoch_loss)
                    writer.add_scalar("learning rate: epoch", optimizer.param_groups[0]['lr'], epoch + 1)

                    #v_label_pred = (val_labelpred >= 0.)
                    _, v_label_pred = torch.max(val_labelpred,1)
                    #v_acc = MultiLabel_Acc(v_label_pred, val_label)
                    v_acc = accuracy_score(v_label_pred.cpu().numpy(), val_label.cpu().numpy())

                    writer.add_scalar("Val/accuracy: epoch", v_acc.item(), epoch)

                    cm_v = confusion_matrix(val_label.cpu().detach().numpy(), v_label_pred.cpu().detach().numpy() )
                    figure_v = plot_confusion_matrix(cm_v, class_names=['Ad','Sq','O'])
                    _, torch_im_v = plt2arr(figure_v)
                    writer.add_images('Val/CM', torch_im_v[:,0:3,:,:], epoch)
                    ###

                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    plot_2d_or_3d_image(v_inputs, epoch + 1, writer, index=0, tag="input image")

                    #################################test the model###################################
                    test_label = None
                    test_labelpred = None
                    test_epoch_loss = 0
                    test_step = 0
                    if model_test:
                        for test_data in test_loader:
                            s_inputs = test_data["input"].cuda()
                            s_histology = test_data["histology"].cuda()

                            s_histology = s_histology.unsqueeze(1)

                            s_label = s_histology
                            #s_label = torch.cat((s_Gender, s_ADC, s_Smoker, s_LMet, s_AMet, s_BoMet, s_BrMet, s_LNMet), 1)

                            s_output4, s_output5,_ = model(s_inputs)

                            #################################define survival loss###################################
                            if test_step == 0:

                                test_label = s_label
                                test_labelpred = s_output4
                            else:

                                test_label = torch.cat((test_label, s_label), 0)
                                test_labelpred = torch.cat((test_labelpred, s_output4), 0)


                            #################################define auxiliary loss###################################
                            s_loss_label = loss_BCE(s_output4, s_label.view(-1))
                            #s_loss_img = loss_MSE(s_output5, s_inputs)
                            s_loss_img = loss_AE(s_output5, s_inputs)

                            ###################################################################################################
                            test_step += 1

                            ### ME Added
                            s_loss = loss_func(s_loss_label,s_loss_img)
                            ###

                            #s_loss = loss_func(s_loss_pfs, s_loss_age, s_loss_label) * s_loss_os
                            test_epoch_len = len(test_ds) // test_loader.batch_size + 1
                            writer.add_scalar("Test/overall loss: step", s_loss.item(), test_epoch_len * epoch + test_step)

                            writer.add_scalar("Test/label loss: step", s_loss_label.item(), test_epoch_len * epoch + test_step)
                            writer.add_scalar("Test/AE loss: step", s_loss_img.item(),
                                            test_epoch_len * epoch + test_step)

                            torch.cuda.empty_cache()

                        test_epoch_loss /= test_step
                        test_epoch_loss_values.append(test_epoch_loss)

                        #s_label_pred = (test_labelpred >= 0.)
                        _, s_label_pred = torch.max(test_labelpred,1)

                        #s_acc = MultiLabel_Acc(s_label_pred, test_label)
                        s_acc = accuracy_score(s_label_pred.cpu().numpy(), test_label.cpu().numpy())

                        ### ME Added
                        writer.add_scalar("Test/accuracy: epoch", s_acc.item(), epoch)

                        cm = confusion_matrix(test_label.cpu().detach().numpy(), s_label_pred.cpu().detach().numpy() )
                        figure = plot_confusion_matrix(cm, class_names=['Ad','Sq','O'])
                        _, torch_im = plt2arr(figure)
                        writer.add_images('Test/CM', torch_im[:,0:3,:,:], epoch)
                        ###

                        ### SHould be revised later

                    if epoch > skip_epoch_model:

                        metric1 = v_acc
                        if metric1 > best_metric_pfs:
                            best_metric_pfs = metric1
                            # best_metric_epoch = epoch + 1
                            # os.chdir('runs/'+time_stamp)
                            torch.save(model.state_dict(), BEST_MODEL_NAME + "_PFS.pth")

                        #     print(f"\n epoch {epoch + 1} saved new best metric model")
                        # print(
                        #     f"\n current epoch: {epoch + 1} current loss: {val_epoch_loss:.4f}"
                        #     f"\n best loss: {best_metric:.4f} at epoch {best_metric_epoch}"
                        # )
        writer.close()


    ######################################################################################################
    print('-------------------start, step 6: model evaluation--------------------------')

    if model_pred:
        # test PFS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_PFS.pth"))

        _, pfs_risk_train, os_train, ostime_train, pfs_train, pfstime_train, ID_train, age_train, bn_feature_train = model_eval_gpu(model,
                                                                                                                  train_files,
                                                                                                                  val_transforms)
        _, pfs_risk_val, os_val, ostime_val, pfs_val, pfstime_val, ID_val, age_val, bn_feature_val = model_eval_gpu(model, val_files,
                                                                                                    val_transforms)
        _, pfs_risk_test, os_test, ostime_test, pfs_test, pfstime_test, ID_test, age_test, bn_feature_test = model_eval_gpu(model,
                                                                                                           test_files,
                                                                                                           val_transforms)

        # test OS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_OS.pth"))

        os_risk_train, _, _, _, _, _, _, _, _ = model_eval_gpu(model, train_files, val_transforms)
        os_risk_val, _, _, _, _, _, _, _, _ = model_eval_gpu(model, val_files, val_transforms)
        os_risk_test, _, _, _, _, _, _, _, _ = model_eval_gpu(model, test_files, val_transforms)

        os_train, ostime_train, pfs_train, pfstime_train, ID_train = os_train.unsqueeze(1), ostime_train.unsqueeze(
            1), pfs_train.unsqueeze(1), pfstime_train.unsqueeze(1), ID_train.unsqueeze(1)
        os_val, ostime_val, pfs_val, pfstime_val, ID_val = os_val.unsqueeze(1), ostime_val.unsqueeze(
            1), pfs_val.unsqueeze(1), pfstime_val.unsqueeze(1), ID_val.unsqueeze(1)
        os_test, ostime_test, pfs_test, pfstime_test, ID_test = os_test.unsqueeze(1), ostime_test.unsqueeze(
            1), pfs_test.unsqueeze(1), pfstime_test.unsqueeze(1), ID_test.unsqueeze(1)

        # save the model prediction risk
        pred_train_save = torch.cat((os_risk_train, os_train, ostime_train, pfs_risk_train, pfs_train, pfstime_train, ID_train, age_train), 1)
        pred_val_save = torch.cat((os_risk_val, os_val, ostime_val, pfs_risk_val, pfs_val, pfstime_val, ID_val, age_val), 1)
        pred_test_save = torch.cat((os_risk_test, os_test, ostime_test, pfs_risk_test, pfs_test, pfstime_test, ID_test, age_test), 1)

        pred_train_save = pred_train_save.cpu().numpy()
        pred_val_save = pred_val_save.cpu().numpy()
        pred_test_save = pred_test_save.cpu().numpy()

        pred_train_save = pd.DataFrame(pred_train_save)
        pred_val_save = pd.DataFrame(pred_val_save)
        pred_test_save = pd.DataFrame(pred_test_save)

        pred_train_save.to_csv(BEST_MODEL_NAME + "_train.csv")
        pred_val_save.to_csv(BEST_MODEL_NAME + "_val.csv")
        pred_test_save.to_csv(BEST_MODEL_NAME + "_test.csv")

        # save the bottleneck features
        pred_f_train = torch.cat((ID_train, bn_feature_train), 1)
        pred_f_val = torch.cat((ID_val, bn_feature_val), 1)
        pred_f_test = torch.cat((ID_test, bn_feature_test), 1)

        pred_f_train = pred_f_train.cpu().numpy()
        pred_f_val = pred_f_val.cpu().numpy()
        pred_f_test = pred_f_test.cpu().numpy()

        pred_f_train = pd.DataFrame(pred_f_train)
        pred_f_val = pd.DataFrame(pred_f_val)
        pred_f_test = pd.DataFrame(pred_f_test)

        pred_f_train.to_csv(BEST_MODEL_NAME + "_feature_train.csv")
        pred_f_val.to_csv(BEST_MODEL_NAME + "_feature_val.csv")
        pred_f_test.to_csv(BEST_MODEL_NAME + "_feature_test.csv")

    if model_visual:
        model.load_state_dict(torch.load(BEST_MODEL_NAME + ".pth"))

        for name, _ in model.named_modules(): print(name)

        check_ds = Dataset(data=train_files, transform=val_transforms)
        check_loader = DataLoader(check_ds, batch_size=2, shuffle=True,num_workers=10, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        inputs = check_data["input"].cuda()
        breakpoint()
        cam1 = GradCAM(nn_module=model, target_layers="backbone.class_layers.relu")
        cam2 = CAM(nn_module=model, target_layers="backbone.class_layers.relu", fc_layers="backbone.class_layers.out")
        result = cam(x=inputs)

if __name__ == "__main__":
    main()
