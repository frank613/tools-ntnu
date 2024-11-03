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


re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))


## plot the error and correct for each phoneme   
# df = pd.DataFrame(data_vec, columns=('uttid','context-len','token', "isFalse", "gop"))
def plot_it(out_file, in_df):
    plt.rcParams['font.size'] = 50
    token_list = sorted(in_df['token'].unique()[:5])
    context_list = sorted(in_df['context-len'].unique())
    fig, axes = plt.subplots(len(token_list),1,figsize=(15*len(token_list), 70), sharex="col",layout="constrained")

    for i in range(len(token_list)):
        axes_id = i
        mean_correct = []
        mean_error = []
        std_correct = []
        std_error = []
        ##context from 0 -> limit
        for context_len in context_list[1:]:
            #correct gop
            correct_gop = in_df.loc[(in_df['context-len'] == context_len) & (in_df['token'] == token_list[i]) & (in_df['isFalse'] == 0), ["gop"]].to_numpy()
            mean_correct.append(np.mean(correct_gop))
            std_correct.append(np.std(correct_gop))
            error_gop = in_df.loc[(in_df['context-len'] == context_len) & (in_df['token'] == token_list[i]) & (in_df['isFalse'] == 1), ["gop"]].to_numpy()
            mean_error.append(np.mean(error_gop))
            std_error.append(np.std(error_gop))
        ## and the full context  
        correct_gop = in_df.loc[(in_df['context-len'] == -1) & (in_df['token'] == token_list[i]) & (in_df['isFalse'] == 0), ["gop"]].to_numpy()
        mean_correct.append(np.mean(correct_gop))
        std_correct.append(np.std(correct_gop))
        error_gop = in_df.loc[(in_df['context-len'] == -1) & (in_df['token'] == token_list[i]) & (in_df['isFalse'] == 1), ["gop"]].to_numpy()
        mean_error.append(np.mean(error_gop))
        std_error.append(np.std(error_gop))
        
        #pdb.set_trace()
        x_points = context_list[1:] + [context_list[-1] + 2]
        axes[axes_id].errorbar(x_points, mean_correct, yerr=std_correct, fmt='-o', label="correct phoneme")
        axes[axes_id].errorbar(x_points, mean_error, yerr=std_error, fmt='-o', label="substituted phoneme")
        
        axes[axes_id].set_aspect("auto")
        axes[axes_id].set_title("Phoneme " + token_list[i])
        axes[axes_id].legend()
        
    plt.xlabel('Context length')
    plt.xticks(x_points, [ i for i in x_points[:-1]] + ["full"])
    plt.savefig(out_file)
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) < 3:
        sys.exit("this script takes 2 arguments") 
    
    ##read csv to df
    data_vec = []
    with open(sys.argv[1], "r") as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split(",")
            if len(fields) != 5:
                break
            if fields[1] == "full":
                context_len = -1
            else:
                context_len = int(fields[1])
            permute = fields[3].split("->")
            if len(permute) != 2:
                break
            else:
                token_from, token_to = permute
            if token_from == token_to:
                isFalse = 0
            else:
                isFalse = 1
            data_vec.append((fields[0], context_len, token_to, isFalse, round(float(fields[-1]),5)))
    df = pd.DataFrame(data_vec, columns=('uttid','context-len','token', "isFalse", "gop"))
            
    plot_it(sys.argv[2], df)
    
    

   







    
  
