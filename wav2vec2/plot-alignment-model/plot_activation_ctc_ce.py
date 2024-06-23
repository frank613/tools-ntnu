import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
import torch
from pathlib import Path
import pdb
import matplotlib.pyplot as plt
import json


re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')
    
def plot_it(dict_ctc, dict_ce, out_file):

    plt.rcParams['font.size'] = 50
    ##heat_map
    ctc_post = np.array(dict_ctc["post_mat"])
    ce_post = np.array(dict_ce["post_mat"])
    #fig, axes = plt.subplots(2,1,figsize=(70, 70))
    fig, axes = plt.subplots(2,1,figsize=(70, 70), sharex="col",layout="constrained")
    #fig, axes = plt.subplots(2,1, sharex="col",)


    im_ctc = axes[0].imshow(np.transpose(ctc_post), origin="lower")
    im_ce = axes[1].imshow(np.transpose(ce_post), origin="lower")

    ##authentic alignment
    aut_ali = dict_ce["align-seq"]
    y_vec = []
    x_vec = []
    p_vec = []
    for p,pid,s,e in aut_ali:
        p_vec.append(p)
        y_vec.append(pid)
        x_vec.append(s)
        
    axes[1].step(x_vec, y_vec, "o-y", where='post', label='authentic alignment')
    for x,y,p in zip(x_vec,y_vec,p_vec):
        axes[1].text(x, y+1, p, color="white", fontsize=30)
    #axes[1].plot(x_vec, y_vec, "o-y", label='authentic alignment')

    aut_ali_ctc = dict_ctc["align-seq"]
    y_vec = []
    x_vec = []
    p_vec = []
    for p,pid,s,e in aut_ali_ctc:
        p_vec.append(p)
        y_vec.append(pid)
        x_vec.append(s)
    axes[0].step(x_vec, y_vec, "y", where='post', label='authentic alignment')
    for x,y,p in zip(x_vec,y_vec,p_vec):
        axes[0].text(x, y+1, p, color="white", fontsize=30)

    ##alignment
    ce_ali = dict_ce["path_pid"]
    x_vec = range(len(ce_ali))
    axes[1].step(x_vec, ce_ali, "o-r", where='post', label='forced-alignment')

    ctc_ali = dict_ctc["path_pid"]
    x_vec = range(len(ctc_ali))
    axes[0].step(x_vec, ctc_ali, "o-r", where='post', label='forced-alignment')

    #axes[3,0].autoscale(False)
    #axes[0,0].autoscale(False)
    # Create colorbar
    #cbar = fig.colorbar(im_ctc)
    #cbar.ax.set_ylabel("ctc", rotation=-90, va="bottom")
    
    #cbar = fig.colorbar(im_ce)
    #cbar.ax.set_ylabel("ce", rotation=-90, va="bottom")
    # Show all ticks a09.9nd label them with the respective list entries
    #axes[1,0].set_xticks(np.arange(posterior_matrix.shape[1]), labels=np.arange(posterior_matrix.shape[1]))
    #axes[1,0].set_yticks(np.arange(posterior_matrix.shape[0]), labels=labels)

#    # full_path
#    #full_path_height = len(full_path) - 1  - np.array(full_path[1:])
#    full_path_height = np.array(full_path[1:])
#    axes[1,0].step(np.arange(posterior_matrix.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
#    axes[1,0].grid()
#    axes[1,0].legend(loc=2)
#    
#    axes[2,0].step(np.arange(reduced_a.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
#    axes[2,0].grid()
#    axes[2,0].legend(loc=2)
#    
#    axes[3,0].step(np.arange(reduced_b.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
#    axes[3,0].grid()
#    axes[3,0].legend(loc=2)
#    fig.tight_layout()
#    #plt.show()
#    out_file = "./out-plot" + "/" + out_fname + ".png"
#    os.makedirs(os.path.dirname(out_file), exist_ok=True)
#    
    axes[0].set_aspect("auto")
    axes[0].set_title("CTC")
    axes[1].set_aspect("auto")
    axes[1].set_title("Cross-Entropy")
    axes[0].legend()
    axes[1].legend()
#    axes[2,0].set_aspect("auto") 
#    axes[3,0].set_aspect("auto") 
   
    fig.supxlabel('num of frames')
    fig.supylabel('activations for each token')
    #plt.xlabel('xlabel', fontsize=50)
    #plt.ylabel('ylabel', fontsize=50)
    plt.savefig(out_file)
    
    

def decode_post(post_mat, p_tokenizer): 
    ##plot decode path using "maximum t" strategy
    ids = torch.argmax(post_mat, 0)
    return p_tokenizer.convert_ids_to_tokens(ids)
    

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <ctc-json> <ce-json> <out.png>.\n \
            , it takes two jsons and plot paths for comprison") 
    #step 0, read the files
    f_ctc = open(sys.argv[1], "r")
    f_ce = open(sys.argv[2], "r")
    dict_ctc = json.load(f_ctc)
    dict_ce = json.load(f_ce)  
    
    plot_it(dict_ctc, dict_ce, sys.argv[3])
    
    

   







    
  
