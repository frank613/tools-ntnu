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


re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))


## plot the error and correct for each phoneme   
# df = pd.DataFrame(data_vec, columns=('uttid','context-len','token', "isFalse", "gop"))
def plot_it(out_file, in_df):
    #token_list = sorted(in_df['token'].unique()[:5])
    context_list = sorted(in_df['context-len'].unique())
    ## change value -1 of full-context-len to the the x-axis value 
    full_context_x = len(context_list)
    in_df.loc[in_df['context-len'] == -1, "context-len"] = full_context_x
 
    #fig, axes = plt.subplots(len(token_list),1,figsize=(15*len(token_list), 70), sharex=True, sharey=True, layout="constrained")
    #fig, axes = plt.subplots(1,1,figsize=(30, 30), sharex=True, sharey=True, layout="constrained")
    fig, axes = plt.subplots(1,1, sharex=True, sharey=True, layout="constrained")
    #plt.rcParams['font.size'] = 50
    plt.xlabel('Context length')
    plt.ylabel('GOP')
    plt.xticks(context_list, [ i for i in context_list[:-1]] + ["full"])
    all_df = in_df
    g = sns.lineplot(data=all_df, x="context-len", y="gop", hue="isFalse", ax=axes)
    g.legend(loc='center right', labels=['Correct', '_c1', 'Subsitution', '_s1']) 
    g.set_aspect("auto")
    g.set_title("The GOP-SD-AF value at different context length")
        

    # for i in range(len(token_list)):
    #     axes_id = i    
    #     all_df = in_df.loc[(in_df['token'] == token_list[i])]
    #     g = sns.lineplot(data=all_df, x="context-len", y="gop", hue="isFalse", ax=axes[axes_id])
    #     g.legend(loc='center right', labels=['Correct', '_c1', 'Subsitution', '_s1']) 
    #     g.set_aspect("auto")
    #     g.set_title("Phoneme " + token_list[i])
        
        # ### plot wides
        # #pdb.set_trace()
        # ### correct
        # correct_df = in_df.loc[(in_df['token'] == token_list[i]) & (in_df['isFalse'] == 0)]  
        # correct_wide = correct_df.pivot_table(index="context-len", columns="uttid", values="gop")
        # correct_to_plot = correct_wide.iloc[:,:10]
        # #correct_to_plot = correct_wide
        # g = sns.lineplot(data=correct_to_plot, ax=axes[axes_id,1],  legend=False)
        # g.set_aspect("auto")
        # g.set_title("Phoneme " + token_list[i])
        # ##
        # #pdb.set_trace()
        # wrong_df = in_df.loc[(in_df['token'] == token_list[i]) & (in_df['isFalse'] == 1)]  
        # wrong_wide = wrong_df.pivot_table(index="context-len", columns="uttid", values="gop")
        # #pdb.set_trace()
        # wrong_to_plot = wrong_wide.iloc[:,:20]
        # #wrong_to_plot = wrong_wide
        # g = sns.lineplot(data=wrong_to_plot, ax=axes[axes_id,2], legend=False)
        # g.set_aspect("auto")
        # g.set_title("Phoneme " + token_list[i])
        
    fig.savefig(out_file)
    
    
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
    if len(sys.argv) < 3:
        sys.exit("this script takes 2 arguments") 
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
    plot_it(sys.argv[2], df_occ)

   
       

    
    

   







    
  
