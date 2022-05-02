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

def get_Random_Sample(X, Y, N):
    indx = random.sample(range(len(X)), N)
    return X[indx], Y[indx]
