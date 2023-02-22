import sys
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pdb
import numpy as np
from sklearn import metrics

def readGOPToDF(df, gop_file, method):
    temp_list = []
    with open(gop_file, 'r') as in_file:
        for line in in_file:
            line = line.strip()
            fields = line.split(' ')
            if len(fields) != 5:
                sys.exit("wrong line in the input GOP files")
            temp_list.append([fields[1], round(float(fields[2]),3), fields[3], method])
    return df.append(pd.DataFrame(temp_list, columns=('phoneme','score','label', 'method')))
            
def plot(df,outFile):
    methods = df['method'].unique()
    all_phonemes = df['phoneme'].unique()
    fig, axs = plt.subplots(len(all_phonemes), len(methods), figsize=(20, 4*len(all_phonemes)))
    df["label"] = np.where(df['label']=='C', 0, 1 )
    for row,phoneme in enumerate(all_phonemes):
        for col,mtd in enumerate(methods):
            data_true = df.loc[(df["phoneme"] == phoneme) & (df["method"] == mtd) & (df["label"] == 1), ['score', 'label']].to_numpy()
            data_false = df.loc[(df["phoneme"] == phoneme) & (df["method"] == mtd) & (df["label"] == 0), ['score','label']].to_numpy()
            ax = axs[row][col]
            plot_labels = []
            add_label(ax.violinplot(data_true[:,0],vert=False, quantiles=[0.25,0.5,0.75], points=500, positions=[0]), "Sub or Del({})".format(data_true[:,0].shape[0]), plot_labels)
            add_label(ax.violinplot(data_false[:,0],vert=False, quantiles=[0.25,0.5,0.75], points=100, positions=[1]), "Correct({})".format(data_false[:,0].shape[0]), plot_labels)
            ax.set_xlim([-70, 5])
            ax.set_xlim([-70, 5])
            ax.set_title(mtd + ', Gop for phoneme: ' + phoneme)
            auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
            auc_artist, = plt.plot([], [])
            ax.legend(*zip(*(plot_labels+[(auc_artist, "AUC = {}".format(auc_value))])), loc=2)
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3)
            

def add_label(violin, method, labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), method))

if __name__ == "__main__":
    if len(sys.argv) <= 1 :
        sys.exit("this script takes <labeled GOP file 1> <labeled GOP file 2> <labeled GOP file 3>... <output-file> as arguments. It plots the GOP distributions for each phoneme")

    df = pd.DataFrame(columns=('phoneme','score','label', 'method'))
    methods =  ['GMM-mono', 'GMM-mono-frame', 'DNN-mono', 'DNN-tri']
    assert(len(methods) == len(sys.argv) - 2)
    for i in range(1,len(sys.argv)-1):
        df = readGOPToDF(df, sys.argv[i], methods[i-1])
        print("read one GOP")

    plot(df, sys.argv[-1])
