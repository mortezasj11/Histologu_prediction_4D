import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import torch
from matplotlib import rcParams
import torch
from sklearn.metrics import confusion_matrix,balanced_accuracy_score, accuracy_score,roc_curve, auc

class Tensorboard():
    def __init__(self, label_x_pred_list, label_x_true_list) -> None:
        self.label_x_pred_list = label_x_pred_list
        _, label_x_pred_binary = torch.max(label_x_pred_list,1)
        self.pred_binary = label_x_pred_binary.cpu().numpy()
        self.true_binary = label_x_true_list.cpu().numpy()

    def accuracy(self):
        acc_x = accuracy_score(self.pred_binary, self.true_binary)
        acc_x_b = balanced_accuracy_score(self.true_binary, self.pred_binary)
        return acc_x, acc_x_b

    def confusion_matrix(self, classes_name = ["0", "1"]):
        cm_x = confusion_matrix(self.true_binary, self.pred_binary)
        torch_im1 = plot_confusion_matrix(cm_x, class_names=classes_name)
        torch_im2 = plot_confusion_matrix(cm_x, class_names=classes_name, normalize=False)
        two_x_cm = torch.cat((torch_im1[:,0:3,:,:],torch_im2[:,0:3,:,:]), dim =3)
        return two_x_cm

    def roc(self, title = None):
        x_true_hard = np.array([ [1,0] if l==0 else [0,1] for l in self.true_binary ])
        x_pred_hard = torch.softmax(self.label_x_pred_list,dim=1).cpu().detach().numpy() 
        fpr_x, t_tpr_x, x_roc_auc = dict(), dict(), dict()
        fpr_x["micro"], t_tpr_x["micro"], _ = roc_curve(x_true_hard.ravel(), x_pred_hard.ravel())
        x_roc_auc["micro"] = auc(fpr_x["micro"], t_tpr_x["micro"])
        image_x_roc = plot_ROC(fpr_x["micro"], t_tpr_x["micro"],x_roc_auc["micro"], title = title)
        return image_x_roc[:,0:3,:,:]

def plot_confusion_matrix(cm, class_names, normalize=True):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    plt.close('all')
    rcParams['lines.linewidth'] = 2
    # Normalize the confusion matrix.
    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names) #plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Normalize the confusion matrix.
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout(pad=2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #plt2arr
    figure.canvas.draw()
    rgba_buf = figure.canvas.buffer_rgba()
    (w,h) = figure.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    fig_array = torch.unsqueeze(torch.from_numpy(np.transpose(rgba_arr,(2,0,1))),0)

    plt.close('all')
    return fig_array


def plt2arr(fig, draw=True):
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr, torch.unsqueeze(torch.from_numpy(np.transpose(rgba_arr,(2,0,1))),0)



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


def plot_ROC(fpr, tpr, roc_auc, title = None):
    plt.close('all')
    figure = plt.figure(figsize=(3, 3))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout(pad=2)

    #plt2arr
    figure.canvas.draw()
    rgba_buf = figure.canvas.buffer_rgba()
    (w,h) = figure.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    fig_array = torch.unsqueeze(torch.from_numpy(np.transpose(rgba_arr,(2,0,1))),0)
    plt.close('all')

    return fig_array


def plot_age(age_pred, age_true):
    age_pred = age_pred.cpu().detach().numpy()
    age_true = age_true.cpu().detach().numpy()
    plt.close('all')
    figure = plt.figure(figsize=(3, 3))
    plt.plot(age_pred,age_true,'.')
    plt.grid()
    plt.xlim([-3.5, 3.5])
    plt.ylim([-3.5, 3.5])
    plt.xlabel("age_pred")
    plt.ylabel("age_true")
    plt.title("Age")
    plt.tight_layout(pad=2)
    return figure

def plot_histogram(age_pred, age_true):
    age_pred = age_pred.cpu().detach().numpy()
    age_true = age_true.cpu().detach().numpy()
    plt.close('all')
    figure = plt.figure(figsize=(3, 3))
    plt.hist2d(age_pred,age_true, bins=25, cmap='plasma')
    cb = plt.colorbar()
    cb.set_label('Number of entries')
    # Add title and labels to plot.
    plt.title('Heatmap of Age')
    #plt.grid()
    plt.xlim([0.30, .90])
    plt.ylim([0.30, .90])
    plt.xlabel("age_pred")
    plt.ylabel("age_true")
    plt.title("Age")
    plt.tight_layout(pad=2)
    return figure

if __name__=='__main__':
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix([1, 0, 1, 0], [1, 0, 1, 1])
    np_im = plot_confusion_matrix(cm, class_names=['Ad','Sq','O'])
    print(np_im.shape)
    plt.imshow( torch.from_numpy(np_im) )
    plt.show()