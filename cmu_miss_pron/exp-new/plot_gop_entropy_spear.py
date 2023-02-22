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
import json
from scipy.stats import spearmanr

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


def plot(df, json_dict, outFile, alpha, beta):
    methods = df['method'].unique()
    all_phonemes = df['phoneme'].unique()
    corr_df = pd.DataFrame(columns=('phoneme','AUC_t','L', 'AUC_stud',  'method'))
    fig, axs = plt.subplots(len(all_phonemes), len(methods)+1, figsize=(30, 5*len(all_phonemes)))
    df["label"] = np.where(df['label']=='C', 0, 1 )
    spear_diff_list = []
    for row,phoneme in enumerate(all_phonemes):
        auc_t_list = []
        auc_stud_list = []
        L_list = []
        compute_corr = False
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
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel("GOP-score")
            auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
            auc_artist, = plt.plot([], [])
            auc_label = (auc_artist, "AUC = {}".format(auc_value))
            auc_stud_list.append(auc_value)
            if phoneme in json_dict[mtd]["phonemes"].keys(): 
                compute_corr = True
                #p:(closest_phoneme, mean_diff, auc_value, entropy, count_of_real, count_of_error)
                entropy = json_dict[mtd]["phonemes"][phoneme][3]
                auc_teacher = json_dict[mtd]["phonemes"][phoneme][2]
                L = round(np.power(entropy, alpha)*np.power(auc_teacher, beta), 3)
                #L = np.log(L)
                L_list.append(L)
                auc_t_list.append(auc_teacher)
                json_artist, = plt.plot([], [])
                json_label = (json_artist, "E={}, A={}, L={}".format(entropy, auc_teacher, L))
                
                ax.legend(*zip(*(plot_labels+[auc_label, json_label])), loc=2)
                corr_df.loc[len(corr_df.index)] = [phoneme, auc_teacher, L, auc_value, mtd]
            else:
               ax.legend(*zip(*(plot_labels+[auc_label])), loc=2)
        #print the spearman correlation
        if compute_corr:
            ax = axs[row][-1]
            ax.set_title("Spearman's coeffcient for {}".format(phoneme))
            auc_coef = str(round(spearmanr(auc_t_list, auc_stud_list)[0],3)) + ' ' + str(round(spearmanr(auc_t_list, auc_stud_list)[1],3))
            L_coef = str(round(spearmanr(L_list, auc_stud_list)[0],3)) + ' ' + str(round(spearmanr(L_list, auc_stud_list)[1],3))
            spear_diff_list.append(round(spearmanr(L_list, auc_stud_list)[0],3) - round(spearmanr(auc_t_list, auc_stud_list)[0],3))
            ax.plot(auc_t_list, auc_stud_list, label = "AUC-T to AUC-S, Spearmanr: {}".format(auc_coef))
            ax.plot(L_list, auc_stud_list, label = "L to AUC-S, Spearmanr: {}".format(L_coef))
            ax.set_ylabel("AUC-student")
            ax.legend()
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)
    return sum(spear_diff_list)/len(spear_diff_list)

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

def read_json(path):
    with open(path,"r") as injson:
        return json.load(injson)

def print_corr(corr_df):
    #columns=('phoneme','AUC_t','L', 'AUC_stud',  'method')
    print(corr_df)
    print("overall correlation between AUC Teacher and AUC Student:")
    print(spearmanr(corr_df['AUC_t'], corr_df['AUC_stud']))
    print("overall correlation between L and AUC Student:")
    print(spearmanr(corr_df['L'], corr_df['AUC_stud']))
    methods = corr_df['method'].unique()
    for mtd in methods:
        print("Method: {} correlation between AUC Teacher and AUC Student:".format(mtd))
        print(spearmanr(corr_df.loc[corr_df["method"]==mtd, "AUC_t"], corr_df.loc[corr_df["method"]==mtd, "AUC_stud"]))
        print("Method: {} correlation between L and AUC Student:".format(mtd))
        print(spearmanr(corr_df.loc[corr_df["method"]==mtd, "L"], corr_df.loc[corr_df["method"]==mtd, "AUC_stud"]))
    pairs = [(methods[0], methods[1])]
    for pair in pairs:
        print("correlation between the diff of the AUC-Teacher and the diff of AUC-student the first two methods")
        diff_auc_t = corr_df.loc[corr_df["method"]==pair[0], "AUC_t"].to_numpy() - corr_df.loc[corr_df["method"]==pair[1], "AUC_t"].to_numpy()
        diff_auc_stud = corr_df.loc[corr_df["method"]==pair[0], "AUC_stud"].to_numpy() - corr_df.loc[corr_df["method"]==pair[1], "AUC_stud"].to_numpy() 
        #relative diff
        #diff_auc_t = diff_auc_t / corr_df.loc[corr_df["method"]==pair[1], "AUC_t"].to_numpy()
        #diff_auc_stud = diff_auc_stud / corr_df.loc[corr_df["method"]==pair[1], "AUC_stud"].to_numpy()
        print(spearmanr(diff_auc_t, diff_auc_stud))
        print("correlation between the diff of the L and the diff of AUC-student the first two methods")
        diff_L = corr_df.loc[corr_df["method"]==pair[0], "L"].to_numpy() - corr_df.loc[corr_df["method"]==pair[1], "L"].to_numpy()
        #relative diff
        #diff_L = diff /corr_df.loc[corr_df["method"]==pair[1], "L"].to_numpy()
        print(spearmanr(diff_L, diff_auc_stud))

        


if __name__ == "__main__":
    if len(sys.argv) <= 1 :
        sys.exit("this script takes <labeled GOP file 1> <json file1> <labeled GOP file 2> <json file2>... <output-file> as arguments. It plots the GOP distributions for each phoneme")

    df = pd.DataFrame(columns=('phoneme','score','label', 'method'))
    #methods =  ['GMM-mono',  'DNN-tri']
    methods =  ['GMM-mono-align', 'GMM-mono-frame', 'DNN-mono',  'DNN-tri']
    json_dict = { mtd:None for mtd in methods}
    assert(len(methods) == (len(sys.argv) - 2)/2)
    for i,mtd in enumerate(methods):
        df = readGOPToDF(df, sys.argv[2*i+1], mtd)
        json_dict[mtd] = read_json(sys.argv[2*i+2])
        print("read one GOP")

    spear_average = plot(df, json_dict, sys.argv[-1], 0.5, 3.5)
    print("spear coef diff average = {} ".format(spear_average))
