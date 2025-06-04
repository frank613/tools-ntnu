import sys
import re
import os
from sklearn import metrics
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
    if len(sys.argv) < 2:
        sys.exit("this script takes 1 argument") 
    norm = False
    ##read csv to df
    field_len = 5
    data_vec_occ = []

    with open(sys.argv[1], "r") as ifile:
        for line in ifile:
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
            if token_from == token_to:
                isFalse = 0
            else:
                isFalse = 1
            gop = round(float(fields[3]),5)
            occ = max(1, round(float(fields[4]),5))
            if norm:
                data_vec_occ.append((fields[0], context_len, token_to, isFalse, round(gop/occ,5)))
            else:
                data_vec_occ.append((fields[0], context_len, token_to, isFalse, round(gop,5)))

    df_occ = pd.DataFrame(data_vec_occ, columns=('uttid','context-len','token', "isFalse", "gop"))
    full_context_stastics(df_occ)
    compute_AUC(df_occ)

   
       

    
    

   







    
  
