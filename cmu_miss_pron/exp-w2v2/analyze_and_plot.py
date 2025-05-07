import sys
import re
import pandas as pd
sys.path.append('/home/xinweic/tools-ntnu/edit-distance')
import edit 
import pdb
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
import json
import os

exclude_token = ["SIL","SPN", "sil"]
#exclude_token = ["SIL","SPN", "sil", "<pad>"]
re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*')
#re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*|<pad>')
#store all the sub pairs
#for each phone in the GOP file:
    #label "S" if it's a substitution error based on the other input files.
    #label "D" if it's a deletion error based on the other input files.
    #label "C" if it's the same in both files

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 'C' else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        rvalue = metrics.roc_auc_score(labels, -array[:, 0])
        return round(rvalue,3)


def labelError(GOP_file, error_list, tran_file, index, pooled):
    gop_df = readGOPToDF(GOP_file, index, pooled=pooled)
    tran_df = readTRANToDF(tran_file)
    df = pd.DataFrame(columns=('phoneme','score','label', 'uttid'))
    tran_list = tran_df['uttid'].unique()
    gop_list = gop_df['uttid'].unique()
    extended = []
    sub_list = []
    #outer loop is the cano df because it is extracted from the gop file and they should have the same number of uttids
    for index, row in gop_df.iterrows(): 
        uttid = row['uttid']
        # if uttid == "fabm2as2":
        #     pdb.set_trace()
        #print("processing {}".format(uttid))
        if uttid not in tran_list:
            print("warning: uttid {0} can't be found in the transcription".format(uttid))
            continue
        cano_seq = [ k for k,v in row['seq-score']]
        tran_df_filtered = tran_df.loc[tran_df["uttid"] == uttid, "seq"]
        if len(tran_df_filtered) != 1:
            sys.exit("duplicate uttids detected in the transcription file, check the input")
        tran_seq = tran_df_filtered.tolist()[0] #[0] converts 2d list to 1d
        dist, labels = edit.edit_dist(cano_seq, tran_seq)
        sub_list += edit.get_sub_pair_list(cano_seq, tran_seq, labels)
        if dist == 0 or uttid not in error_list:
            labels_resized = ['C'] * len(labels)
        else:
            labels_resized = [ label for idx, label in enumerate(labels) if label != 'I']
            if len(labels_resized) != len(cano_seq):
                sys.exit("length of edit distance not maching the gop")

        extended += [ pair + (labels_resized[idx], uttid) for idx, pair in enumerate(row['seq-score']) ]
    df = pd.concat([df, pd.DataFrame(extended, columns=['phoneme','score','label', 'uttid'])])
    return df
    # #pdb.set_trace()
    # #json
    # #p:(auc_value, frequent_sub, mean, std, count_of_del, count_of_sub, total_count)
    # out_form = { \
    #             'phonemes':{},
    #             'summary': {"average-mean": None, "average-std": None, "average-AUC": None}}
    
    # #pdb.set_trace()
    # p_replace_set = np.append(df['phonemes'].unique(), '*')
    # pair_dict = {phoneme_out: { phoneme_in: 0 for phoneme_in in p_replace_set} for phoneme_out in p_replace_set}
    # for pair in sub_list:
    #     l,r = pair.split(' -> ')
    #     if l not in p_replace_set or r not in p_replace_set:
    #         continue
    #     pair_dict[l][r] += 1

    # total_mean=0
    # total_std=0
    # total_auc=0
    # num_phonemes = 0
    # for phoneme in df['phonemes'].unique():
    #     if phoneme in exclude_token:
    #         continue
    #     data_false = df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'C'), ['scores', 'labels']].to_numpy()
    #     data_true = df.loc[(df["phonemes"] == phoneme) & (df["labels"] != 'C'), ['scores','labels']].to_numpy()
    #     auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
    #     total_auc += auc_value
    #     sorted_items = sorted(pair_dict[phoneme].items(), key=lambda kv: kv[1], reverse=True)
    #     freq_sub =  sorted_items[0][0] if sorted_items[0][0]!='*' else sorted_items[1][0]
    #     mean = df.loc[(df["phonemes"] == phoneme),"scores"].mean()
    #     total_mean += mean
    #     std = df.loc[(df["phonemes"] == phoneme),"scores"].std()
    #     total_std += std
    #     num_del = len(df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'D')])
    #     num_sub = len(df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'S')])
    #     num_total = len(df.loc[df["phonemes"] == phoneme])
    #     out_form["phonemes"][phoneme]=(auc_value, freq_sub, mean, std, num_del, num_sub, num_total)
    #     num_phonemes += 1

    # out_form["summary"]["average-mean"]=total_mean/num_phonemes
    # out_form["summary"]["average-std"]=total_std/num_phonemes
    # out_form["summary"]["average-AUC"]=total_auc/num_phonemes

    # return out_form,out_form["summary"]["average-AUC"]

def readGOPToDF(ark_file, index, pooled=False):
    in_file = open(ark_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq-score'))
    isNewUtt = True
    seq_score = []
    for line in in_file:
        line = line.strip()
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 1:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            isNewUtt = False
            continue
        if line == '':
            df.loc[len(df.index)] = [uttid, seq_score]
            isNewUtt = True
            seq_score = []
            continue
        if len(fields) != 4:
            continue
        cur_match = re_phone.match(fields[1])
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            sys.exit("non legal phoneme found in the gop file")
        if pooled:
            fi=3
        else:
            fi=2
        cur_score_list = fields[fi].split(",")
        assert len(cur_score_list) >= 1 and len(cur_score_list) <= 8
        if cur_phoneme not in exclude_token:
            if index != "mean":
                seq_score.append((cur_phoneme, float(cur_score_list[index]))) 
            else:
                gop_list = [ float(x) for x in cur_score_list]
                seq_score.append((cur_phoneme, np.mean(gop_list)))
                #seq_score.append((cur_phoneme, np.mean(np.log((gop_list)))))
    return df

def readTRANToDF(tran_file):
    in_file=open(tran_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq'))
    for line in in_file:
        uttid,seq = line.split()
        df.loc[len(df.index)] = [uttid, seq.split(';')]
    return df

def add_label(violin, method, labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), method))
    
def plot_dist(df, out_path):
    all_phonemes = df['phoneme'].unique()
    fig, axs = plt.subplots(len(all_phonemes), 1, figsize=(20, 5*len(all_phonemes)))
    #df["label"] = np.where(df['label']=='C', 0, 1 )
    for row,phoneme in enumerate(all_phonemes): 
        #pdb.set_trace()
        data_true = df.loc[(df["phoneme"] == phoneme) & (df["label"] == "C"), ['score', 'label']].to_numpy()
        data_false = df.loc[(df["phoneme"] == phoneme) & (df["label"] != "C"), ['score','label']].to_numpy()
        ax = axs[row]
        #plot_labels = []
        #pdb.set_trace()
        ax.violinplot([data_true[:,0].astype(np.float32), data_false[:,0].astype(np.float32)], vert=True, showmeans=True)
        #add_label(ax.violinplot(data_true[:,0],vert=False, quantiles=[0.25,0.5,0.75], points=500, positions=[0]), "Sub or Del({})".format(data_true[:,0].shape[0]), plot_labels)
        #add_label(ax.violinplot(data_false[:,0],vert=False, quantiles=[0.25,0.5,0.75], points=100, positions=[1]), "Correct({})".format(data_false[:,0].shape[0]), plot_labels)
        #ax.set_xlim([-70, 5])
        #ax.set_xlim([-70, 5])
        ax.set_xticks([1,2], labels=["CorrPron","MisPron"])
        auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
        ax.set_title(f'Gop for phoneme: {phoneme}, AUC = {auc_value}')
       
        ax.legend()
    out_file = out_path+".png"
    plt.savefig(out_file)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP file> <error-uttid-list>  <transcribed phoneme-seq file> <out-png>. It labels the phonemes in the GOP file and output distributions of correct and wrong phonemes")

    index = 0
    pooled = True
    #index = "mean"
    utt_list = []
    with open(sys.argv[2]) as ifile:
            for line in ifile:
                line = line.strip()
                fields = line.split()
                if len(fields) != 1:
                    sys.exit("wrong input line")
                utt_list.append(fields[0])
    df = labelError(sys.argv[1], utt_list, sys.argv[3], index, pooled)
    plot_dist(df, sys.argv[4])

 

