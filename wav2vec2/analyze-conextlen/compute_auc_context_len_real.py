import sys
import re
import os
from sklearn import metrics
sys.path.append('/home/xinweic/tools-ntnu/edit-distance')
import edit 
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import torch
from pathlib import Path
import pdb
import matplotlib.pyplot as plt
import json
import seaborn as sns
from sklearn import metrics

exclude_token = ["SIL","SPN", "sil"]
exclude_token_tran = ["SIL","SPN", "sil", "+INHALE+", "+EXHALE+", "+RUSTLE", "+SMACK+", "+NOISE+"]
re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = array[:, 1]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        rvalue = metrics.roc_auc_score(labels, -array[:, 0])
        return round(rvalue,3)
    
    

## plot the error and correct for each phoneme   
# df = pd.DataFrame(data_vec, columns=('uttid','context-len','token', "isFalse", "gop"))
def compute_AUC(in_df):
    #token_list = sorted(in_df['token'].unique()[:5])
    context_list = sorted(in_df['context-len'].unique())
    ## change value -1 of full-context-len to the the x-axis value 
    full_context_x = len(context_list)
    in_df.loc[in_df['context-len'] == -1, "context-len"] = full_context_x
    
    context_list_new = sorted(in_df['context-len'].unique())

    for con_len in context_list_new:
        to_auc = in_df.loc[in_df['context-len'] == con_len, ["gop","isFalse"]].to_numpy()
        auc_value = auc_cal(to_auc)
        print(f"AUC for context len {con_len} is {auc_value}")

    print("done with the experiments")
    


def readGOPToDF(ark_file):
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
        if len(fields) != 3:
            continue
        cur_match = re_phone.match(fields[1])
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            sys.exit("non legal phoneme found in the gop file")
        cur_score = float(fields[2]) 
        if cur_phoneme not in exclude_token:
            seq_score.append((cur_phoneme, cur_score)) 
    return df

def readTRANToDF(tran_file):
    in_file=open(tran_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq'))
    for line in in_file:
        uttid,seq = line.split()
        seq_filtered = [ item for item in seq.split(';') if item not in exclude_token_tran]
        df.loc[len(df.index)] = [uttid, seq_filtered]
    return df

    
    
def full_context_stastics(in_df):
    correct = in_df.loc[ (in_df['isFalse'] == 0) & (in_df['context-len'] == -1), "gop"].to_numpy()
    wrong = in_df.loc[ (in_df['isFalse'] == 1) & (in_df['context-len'] == -1), "gop"].to_numpy()
    print(f"{correct.shape[0]} of correct phonemes:{(correct.mean(), correct.std())}, {wrong.shape[0]} of wrong phonemes:{(wrong.mean(), wrong.std())}")
    ##per phoneme
    # for p in token_list:
    #     correct_df = in_df.loc[(in_df['token'] == p) & (in_df['isFalse'] == 0) & (in_df['context-len'] == -1), "gop"].to_numpy()
    #     wrong_df = in_df.loc[(in_df['token'] == p) & (in_df['isFalse'] == 1) & (in_df['context-len'] == -1), "gop"].to_numpy()
        #print(f"{p}: {correct_df.shape[0]} of correct:{(correct_df.mean(), correct_df.std())}, {wrong_df.shape[0]} of wrong:{(wrong_df.mean(), wrong_df.std())}")
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) < 5:
        sys.exit("this script takes 4 argument, <context-gop> <original-gop> <error-uttid-list> <transcribed phoneme-seq file>") 
    
    error_list = []
    #read error list    
    with open(sys.argv[3]) as ifile:
            for line in ifile:
                line = line.strip()
                fields = line.split()
                if len(fields) != 1:
                    sys.exit("wrong input line")
                error_list.append(fields[0])
    
    gop_df = readGOPToDF(sys.argv[2])
    tran_df = readTRANToDF(sys.argv[4])            
    norm = False
    
    ##label error
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    tran_list = tran_df['uttid'].unique()
    gop_list = gop_df['uttid'].unique()
    extended = []
    sub_list = []
    #outer loop is the cano df because it is extracted from the gop file and they should have the same number of uttids
    for index, row in gop_df.iterrows(): 
        uttid = row['uttid']
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

        extended += [ pair + (labels_resized[idx], uttid+"-"+str(idx)+"-"+pair[0]) for idx, pair in enumerate(row['seq-score']) ]
    df = pd.concat([df, pd.DataFrame(extended, columns=['phonemes','scores','labels', 'uttid'])])
    
    ##read context-len-gop to df
    field_len = 5
    data_vec_occ = []

    with open(sys.argv[1], "r") as ifile2:
        for line in ifile2:
            line = line.strip()
            fields = line.split(",")
            if len(fields) != field_len:
                break
            if fields[1] == "full":
                context_len = -1
            else:
                context_len = int(fields[1])
            permute = fields[2].split("->")
            if len(permute) != 2:
                break
            else:
                token_from, token_to = permute
                ##skip synthesized misprons
                if token_from != token_to:
                    continue
            #check phoneme corrrectness
            uttid=fields[0]
            label = df.loc[df["uttid"]==uttid, "labels"].to_string(index=False)
            if label == "C":
                isFalse = 0
            else:
                isFalse = 1
            gop = round(float(fields[3]),5)
            occ = max(1, round(float(fields[4]),5))
            if norm:
                data_vec_occ.append((fields[0], context_len, token_to, isFalse, round(gop/occ,5)))
            else:
                data_vec_occ.append((fields[0], context_len, token_to, isFalse, round(gop,5)))
                
            pdb.set_trace()

    pdb.set_trace()
    df_occ = pd.DataFrame(data_vec_occ, columns=('uttid','context-len','token', "isFalse", "gop"))
    full_context_stastics(df_occ)
    compute_AUC(df_occ)

   
       

    
    

   







    
  
