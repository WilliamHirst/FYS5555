# Import headers 
import pandas as pd
import numpy as np
from numpy import random

# XGBoost
from xgboost import XGBClassifier

# Sklearn tools 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV 

# Keras tools
import tensorflow as tf
from tensorflow.keras import optimizers
import keras_tuner as kt

# Loading and dumping models.
import joblib
from joblib import dump, load
import os

# Plotting settings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

plt.style.use("bmh")
sns.color_palette("hls", 1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def color_cycle(num_color):
    """ get color from matplotlib
        color cycle
        use as: color = color_cycle(3) """
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color[num_color]


#--- Plot commands ---#
# plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
# plt.xlabel(r"$x$", fontsize=14)
# plt.ylabel(r"$y$", fontsize=14)
# plt.legend(fontsize = 13)
# plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
# plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")

def timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def ask(q):
    state = True
    
    while state == True:
        answ = input(q)
        if answ == "y":
            isYes = True
            state = False
        elif answ == "n":
            isYes = False
            state = False
    return isYes



def splitData(X, Y, split):
    X,Y = shuffle(X,Y, random_state=2)

    x_s = X[Y == 1]
    x_b = X[Y == 0]
    y_s = Y[Y == 1] 
    y_b = Y[Y == 0]


    rng = np.random.default_rng(seed=2)
    indx = rng.choice(len(x_b), len(x_s), replace=False)
    re_indx = [i for i in range(len(x_b)) if i not in indx]

    split_indx  = int(len(x_s)*split)
   
    X_train = np.concatenate((x_s[:,:-1], x_b[indx,:-1]), axis=0)[split_indx:]
    Y_train = np.concatenate((y_s, y_b[indx]))[split_indx:]
    W_train = np.concatenate((x_s[:,-1], x_b[indx,-1]))[split_indx:]
    W_train *= len(X_train)/len(Y)

    X_val = shuffle(X_train, random_state=2)[0:split_indx]
    Y_val = shuffle(Y_train, random_state=2)[0:split_indx]
    W_val = shuffle(W_train, random_state=2)[0:split_indx]
    W_val *= len(X_val)/len(Y)

    W_test = x_b[re_indx,-1] 
    X_test = x_b[re_indx,:-1]
    Y_test = y_b[re_indx]
    W_test *= len(X_test)/len(Y)


    return X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, W_test

def plotHistoBS(y_b, y_s, w_b, w_s, name, title,  nrBins = 15):
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    n, bins, patches = plt.hist(y_b, bins = np.linspace(0,1.,nrBins), facecolor='blue', alpha=0.2,label="Background", weights = w_b)#,density=True)#, density=True)
    n, bins, patches = plt.hist(y_s, bins = np.linspace(0,1.,nrBins), facecolor='red', alpha=0.2, label="Signal", weights = w_s)#,density=True)#, density=True)
    plt.xlabel('XGBoost output',fontsize=14)
    plt.ylabel('Events',fontsize=14)
    plt.title(title,fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.yscale('log')
    if ask("Save Image? [y/n]"):
        plt.savefig(f"{name}", bbox_inches="tight")
    plt.show()

def plotHistoB(y_b, w_b, name, title,  nrBins = 15):
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    n, bins, patches = plt.hist(y_b, bins = np.linspace(0,1.,nrBins), facecolor='blue', alpha=0.6,label="Background", weights = w_b)#,density=True)#, density=True)
    plt.xlabel('XGBoost output',fontsize=14)
    plt.ylabel('Events',fontsize=14)
    plt.title(title,fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.yscale('log')
    if ask("Save Image? [y/n]"):
        plt.savefig(f"{name}", bbox_inches="tight")
    plt.show()


def plotRoc(Y, Y_pred, title):
    fpr, tpr, thresholds = roc_curve(Y,Y_pred[:,1], pos_label=1)
    roc_auc = auc(fpr,tpr)
    lw = 2

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (1-area = %0.2e)' % (1.-roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()