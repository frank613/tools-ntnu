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
    
def plot_it(out_file, dict_list):

    plt.rcParams['font.size'] = 50
    fig, axes = plt.subplots(len(dict_list),1,figsize=(25*len(dict_list), 70), sharex="col",layout="constrained")
    for i in range(len(dict_list)):
        ##heat_map
        axes_id = len(dict_list)-1-i
        axes_id = i
        ctc_post = np.array(dict_list[i]["post_mat"])
        im_ctc = axes[axes_id].imshow(np.transpose(ctc_post), origin="lower")
   
        ##authentic alignment
        aut_ali = dict_list[i]["align-seq"]
        y_vec = []
        x_vec = []
        p_vec = []
        for p,pid,s,e in aut_ali:
            p_vec.append(p)
            y_vec.append(pid)
            x_vec.append(s)
        
        axes[axes_id].step(x_vec, y_vec, "o-y", where='post', label='authentic alignment')
        for x,y,p in zip(x_vec,y_vec,p_vec):
            axes[axes_id].text(x, y+1, p, color="white", fontsize=30)
        #axes[1].plot(x_vec, y_vec, "o-y", label='authentic alignment')


        ##alignment
        ce_ali = dict_list[i]["path_pid"]
        x_vec = range(len(ce_ali))
        axes[axes_id].step(x_vec, ce_ali, "o-r", where='post', label='forced-alignment')

    
        axes[axes_id].set_aspect("auto")
        axes[axes_id].set_title("CTC-" + str(i))
        axes[axes_id].legend()


   
    fig.supxlabel('num of frames')
    fig.supylabel('activations for each token')

    plt.savefig(out_file)
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) < 3:
        sys.exit("this script takes at least 2 arguments <out.png> <ctc-json1> <ctc-json2> <ctc-json3> ...") 
    
    list_dicts = []
    for i in range(len(sys.argv) - 2):
        with open(sys.argv[2+i], "r") as fr:
            list_dicts.append(json.load(fr))    
    plot_it(sys.argv[1], list_dicts)
    
    

   







    
  
