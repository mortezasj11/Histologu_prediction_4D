import os
import shutil
import tempfile
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from datetime import datetime
from utils import plot_confusion_matrix, plt2arr, plot_ROC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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
    Affine,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Lambdad,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
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
#from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image, GradCAM, CAM
#from torchinfo import summary
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
def model_eval_gpu(model,datafile,val_transforms):
    data_loader = DataLoader(Dataset(data=datafile, transform=val_transforms), batch_size=1, shuffle=False, num_workers=10, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        train_labelpred = None
        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            t_label = batch_data["histology"].cuda()
            output4  = model(inputs)

            if step == 0:
                train_label = t_label
                train_labelpred = output4

            else:
                train_label = torch.cat((train_label, t_label),0)
                train_labelpred = torch.cat((train_labelpred, output4),0)
            step += 1
            _, t_label_pred = torch.max(train_labelpred,1)
            t_acc = accuracy_score(t_label_pred, train_label)
    print(f"\n model evaluation" f"\n Accuracy: {t_acc:.4f} ")
    return train_labelpred, t_label_pred, t_acc

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
        return loss0 + loss1

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Dense_model(nn.Module):
    def __init__(self,drop_rate, in_channels):
        super(Dense_model, self).__init__()
        self.backbone = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=1, dropout_prob=drop_rate)
        self.backbone.class_layers.out = Identity()
        # self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=3, num_classes=1024)
        self.fc1 = torch.nn.Linear(1024, 2)

    def forward(self, x):  # define network
        encoded = self.backbone(x)
        label = self.fc1(encoded)
        return label

######################################################################################################
print('-------------------start, step 2: Hyper Parameters--------------------------')
 
hparams = { 'lr' : 1e-4,
            'rand_p' : 0.2,
            'drop_rate' : 0.5,
            'max_epochs' : 200,
            'train_batch' : 4,
            'weight' : torch.tensor([594/647, 153/647 ]), #153 594
            'schedular_step_size':10, 
            'schedular_gamma':0.9,
            'lambda_l1':3e-8,
            'in_channels': 3,
            'Adam':False,
            'GPU_number': '0' }

def main():
    ###********************************************************
    ###hyperparameter setting
    os.environ["CUDA_VISIBLE_DEVICES"] = hparams['GPU_number']
    for i in hparams:
        print(i,': ',hparams[i])
    torch.cuda.empty_cache()
    verbose = False
    model_train = True
    model_pred = False
    model_visual = False
    model_test = False

    BEST_MODEL_NAME = "DenseBinary"

    rand_p = hparams['rand_p']  # data augmentation random proportion
    lr = hparams['lr']
    #lr_decay = 0.03
    drop_rate = hparams['drop_rate'] # model dropout rate
    max_epochs = hparams['max_epochs']     # epoch number
    train_batch = hparams['train_batch']
    val_batch = 4
    test_batch = 4
    skip_epoch_model = 100 # not saving the model for initial fluctuation

    ###********************************************************
    ######################################################################################################

    files_dir = '/Data/Histology/Nature_Gemini_4d_Feb2022/GeminiNature_4D'

    train_os = pd.read_csv('/Data/Histology/Nature_Gemini_4d_Feb2022/TrainValTest/OnlyNature2/train.csv')
    val_os   = pd.read_csv('/Data/Histology/Nature_Gemini_4d_Feb2022/TrainValTest/OnlyNature2/val.csv')
    test_os  =  pd.read_csv('/Data/Histology/Nature_Gemini_4d_Feb2022/TrainValTest/OnlyNature2/val.csv')

    #train_os.sort_values('ID')
    #val_os.sort_values('ID')
    #test_os.sort_values('ID')

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

    ######################################################################################################
    print('-------------------start, step 2: define the transforms--------------------------')

    train_transforms = Compose(
            [
                LoadImaged(keys=["input"]),
                Lambdad(keys=["input"], func=lambda x: x[:,:,:,0:hparams['in_channels']]),
                AsChannelFirstd(keys=["input"], channel_dim=-1),
                #AddChanneld(keys=["input"]),
                #Resized(keys=["input"], spatial_size=[160,160,160]),  # augment, flip, rotate, intensity ...
                #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
                #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
                #RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(1, 2)),
                #RandAdjustContrastd(keys=["input"], prob=rand_p),
                #RandFlipd(keys=["input"], prob=rand_p),
                RandZoomd(keys="input", prob=rand_p, min_zoom=0.85, max_zoom=1.15),
                EnsureTyped(keys=["input"]),
                RandAffined(keys=["input"], prob=rand_p, rotate_range=(np.pi, np.pi, np.pi), shear_range=(-5, 5, -5, 5,-5, 5), translate_range=(0.05, 0.05, 0.05))
            ]
        )
        
    val_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            Lambdad(keys=["input"], func=lambda x: x[:,:,:,0:hparams['in_channels']]),
            AsChannelFirstd(keys=["input"], channel_dim=-1),
            #AddChanneld(keys=["input"]),
            #Resized(keys=["input"], spatial_size=[160,160,160]),
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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dense_model
    model = Dense_model(drop_rate, in_channels=hparams['in_channels']) #monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = model.cuda()

    #loss_func = MultiTaskLossWrapper(n_loss).to(device)
    loss_CE = torch.nn.CrossEntropyLoss(weight=hparams['weight'].cuda())
    if hparams["Adam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr)
    else:
        optimizer = torch.optim.LBFGS(model.parameters(), lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    # scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=lr_decay,  min_lr=1e-5)  Comes with scheduler.step(val_epoch_loss)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams['schedular_step_size'], gamma=hparams['schedular_gamma'])  #Comes with scheduler.step() 
    
    # visualize the NET
    #examples = iter(train_loader)
    #example_data = examples.next()
    #summary(model, input_size=example_data["input"].shape)

    ######################################################################################################
    print('-------------------start, step 5: model training--------------------------')

    if model_train:
        time_stamp = "{0:%Y-%m-%d-T%H-%M-%S/}".format(datetime.now()) + BEST_MODEL_NAME
        writer = SummaryWriter(log_dir='runs_BN/' + time_stamp + '__letssee')

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
                t_label = t_histology.unsqueeze(1)

                def closure():
                    optimizer.zero_grad()
                    output4 = model(inputs)

                    #######add L1 regularization + multitask learning
                    l1_reg = None
                    for W in model.parameters():
                        if l1_reg is None:
                            l1_reg = torch.abs(W).sum()
                        else:
                            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
                    t_loss_label = loss_CE(output4, t_label.view(-1))
                    t_loss = t_loss_label + hparams['lambda_l1'] * torch.log10(l1_reg)
                    t_loss.backward()
                    return t_loss

                optimizer.step(closure)

            writer.add_scalar("hparam/initial_lr",           hparams['lr'],                  epoch)
            writer.add_scalar("hparam/rand_p",               hparams['rand_p'],              epoch)
            writer.add_scalar("hparam/drop_rate",            hparams['drop_rate'],           epoch)
            writer.add_scalar("hparam/max_epochs",           hparams['max_epochs'],          epoch)
            writer.add_scalar("hparam/train_batch",          hparams['train_batch'],         epoch)
            writer.add_scalar("hparam/loss_weight_0",        hparams['weight'][0],           epoch)
            writer.add_scalar("hparam/loss_weight_1",        hparams['weight'][1],           epoch)
            writer.add_scalar("hparam/schedular_step_size",  hparams['schedular_step_size'], epoch)
            writer.add_scalar("hparam/schedular_gamma",      hparams['schedular_gamma'],     epoch)
            writer.add_scalar("hparam/lambda_l1",            hparams['lambda_l1'],           epoch)
            writer.add_scalar("hparam/in_channels",          hparams['in_channels'],         epoch)
            writer.add_scalar("hparam/lr",             optimizer.param_groups[0]['lr'],      epoch)

              #################################    VALIDATION    ###################################

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
                        val_output4 = model(v_inputs)

                        if val_step == 0:
                            val_label = v_label
                            val_labelpred = val_output4
                        else:
                            val_label = torch.cat((val_label,v_label),0)
                            val_labelpred = torch.cat((val_labelpred,val_output4),0)

                        v_loss_label = loss_CE(val_output4, v_label.view(-1))
                        v_loss_label_vec = v_loss_label.unsqueeze(0) if val_step == 0 else torch.cat((v_loss_label_vec, v_loss_label.unsqueeze(0)),dim=0)

                        val_step += 1
                        v_loss = v_loss_label
                        val_epoch_len = len(val_ds) // val_loader.batch_size + 1
                        #writer.add_scalar("Val_1/overall loss: step", v_loss.item(), val_epoch_len * epoch + val_step)
                        #writer.add_scalar("Val_1/label loss: step", v_loss_label.item(), val_epoch_len * epoch + val_step)
                        torch.cuda.empty_cache()

                    writer.add_scalar("Val_1/LabelLoss", v_loss_label_vec.mean().item(),  epoch )

                    val_epoch_loss /= val_step
                    scheduler.step()
                    val_epoch_loss_values.append(val_epoch_loss)
                    
                    _, v_label_pred = torch.max(val_labelpred,1)
                    v_acc = accuracy_score(v_label_pred.cpu().numpy(), val_label.cpu().numpy())
                    
                    cm_v = confusion_matrix(val_label.cpu().detach().numpy(), v_label_pred.cpu().detach().numpy() )
                    figure_v = plot_confusion_matrix(cm_v, class_names=['Sq','Ad_Other'])
                    _, torch_im_v = plt2arr(figure_v)

                    writer.add_scalar("Val_1/Accuracy", v_acc.item(),                     epoch)

                    figure_v2 = plot_confusion_matrix(cm_v, class_names=['Sq','Ad_Other'],normalize=False)
                    _, torch_im_v2 = plt2arr(figure_v2)
                    plt.close()
                    v_two_cm = torch.cat((torch_im_v[:,0:3,:,:],torch_im_v2[:,0:3,:,:]), dim =3)
                    writer.add_images('Val_1/CM', v_two_cm, epoch)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    plot_2d_or_3d_image(v_inputs, epoch + 1, writer, index=0, tag="input image")


                    # ROC Val
                    y_t = np.array([ [1,0] if l==0 else [0,1] for l in val_label.cpu().numpy() ])
                    y_p = val_labelpred.cpu().numpy()
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for jj in range(2):
                        fpr[jj], tpr[jj], _ = roc_curve(y_t[:, jj], y_p[:, jj])
                        roc_auc[jj] = auc(fpr[jj], tpr[jj])
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_t.ravel(), y_p.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    figure3 = plot_ROC(fpr[0], tpr[0],roc_auc[0])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_1/ROC0', torch_im2[:,0:3,:,:], epoch)
                    figure3 = plot_ROC(fpr[1], tpr[1],roc_auc[1])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_1/ROC1', torch_im2[:,0:3,:,:], epoch)
                    figure3 = plot_ROC(fpr["micro"], tpr["micro"],roc_auc["micro"])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_1/ROC_micro', torch_im2[:,0:3,:,:], epoch)

            ################################# TEST ###################################
            test_label = None
            test_labelpred = None
            test_epoch_loss = 0
            test_step = 0
            if model_test:
                model.eval()
                with torch.no_grad():
                    for test_data in test_loader:
                        s_inputs = test_data["input"].cuda()
                        s_histology = test_data["histology"].cuda()
                        s_histology = s_histology.unsqueeze(1)
                        s_label = s_histology
                        s_output4 = model(s_inputs)

                        test_label = s_label if test_step == 0 else torch.cat((test_label, s_label), 0)
                        test_labelpred = s_output4 if test_step == 0 else torch.cat((test_labelpred, s_output4), 0)

                        s_loss_label = loss_CE(s_output4, s_label.view(-1))
                        s_loss_label_vec = s_loss_label.unsqueeze(0) if test_step == 0 else torch.cat((s_loss_label_vec, s_loss_label.unsqueeze(0)),dim=0)
                        test_step += 1
                        s_loss = s_loss_label

                        #s_loss = loss_func(s_loss_pfs, s_loss_age, s_loss_label) * s_loss_os
                        test_epoch_len = len(test_ds) // test_loader.batch_size + 1

                        #writer.add_scalar("Val_2/label loss: step", s_loss_label.item(), test_epoch_len * epoch + test_step)
                        torch.cuda.empty_cache()
                    writer.add_scalar("Val_2/LabelLoss", s_loss_label_vec.mean().item(), epoch )

                    test_epoch_loss /= test_step
                    test_epoch_loss_values.append(test_epoch_loss)

                    _, s_label_pred = torch.max(test_labelpred,1)
                    s_acc = accuracy_score(s_label_pred.cpu().numpy(), test_label.cpu().numpy())
                    writer.add_scalar("Val_2/Accuracy", s_acc.item(), epoch)
                    cm = confusion_matrix(test_label.cpu().detach().numpy(), s_label_pred.cpu().detach().numpy() )
                    figure_s = plot_confusion_matrix(cm, class_names=['Sq','Ad_Other'])
                    _, torch_im_s = plt2arr(figure_s)
                    #writer.add_images('Val_2/CM', torch_im_s[:,0:3,:,:], epoch)

                    figure_s2 = plot_confusion_matrix(cm, class_names=['Sq','Ad_Other'],normalize=False)
                    _, torch_im_s2 = plt2arr(figure_s2)
                    plt.close()
                    s_two_cm = torch.cat((torch_im_s[:,0:3,:,:],torch_im_s2[:,0:3,:,:]), dim =3)
                    writer.add_images('Val_2/CM_NN',s_two_cm, epoch)

                    # ROC Val
                    y_t = np.array([ [1,0] if l==0 else [0,1] for l in test_label.cpu().numpy() ])
                    y_p = test_labelpred.cpu().numpy()
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for jj in range(2):
                        fpr[jj], tpr[jj], _ = roc_curve(y_t[:, jj], y_p[:, jj])
                        roc_auc[jj] = auc(fpr[jj], tpr[jj])
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_t.ravel(), y_p.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    figure3 = plot_ROC(fpr[0], tpr[0],roc_auc[0])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_2/ROC0', torch_im2[:,0:3,:,:], epoch)
                    figure3 = plot_ROC(fpr[1], tpr[1],roc_auc[1])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_2/ROC1', torch_im2[:,0:3,:,:], epoch)
                    figure3 = plot_ROC(fpr["micro"], tpr["micro"],roc_auc["micro"])
                    _, torch_im2 = plt2arr(figure3)
                    writer.add_images('Val_2/ROC_micro', torch_im2[:,0:3,:,:], epoch)

            #################################  SAVE  ###################################
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
                #     f"\n best loss: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.close()

    ######################################################################################################
    print('-------------------start, step 6: model evaluation--------------------------')

    if model_pred:

        # test OS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_OS.pth"))

        t_out, t_label_pred, t_acc = model_eval_gpu(model, train_files, val_transforms)
        v_out, v_label_pred, v_acc = model_eval_gpu(model, val_files, val_transforms)
        s_out, s_label_pred, s_acc = model_eval_gpu(model, test_files, val_transforms)

        unsqueeze = True
        if unsqueeze:
            t_out, t_label_pred, t_acc = t_out.unsqueeze(1), t_label_pred.unsqueeze(1), t_acc.unsqueeze(1)
            v_out, v_label_pred, v_acc = v_out.unsqueeze(1), v_label_pred.unsqueeze(1), v_acc.unsqueeze(1)
            s_out, s_label_pred, s_acc = s_out.unsqueeze(1), s_label_pred.unsqueeze(1), s_acc.unsqueeze(1)
        # save the model prediction risk

        pred_train_save = t_labelpred.cpu().numpy()
        pred_val_save = t_label_pred.cpu().numpy()
        pred_test_save = s_acc.cpu().numpy()

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
        #cam1 = GradCAM(nn_module=model, target_layers="backbone.class_layers.relu")
        #cam2 = CAM(nn_module=model, target_layers="backbone.class_layers.relu", fc_layers="backbone.class_layers.out")
        #result = cam(x=inputs)

if __name__ == "__main__":
    main()

