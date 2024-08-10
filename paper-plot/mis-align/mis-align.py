import sys
import re
import os
from math import ceil
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
import matplotlib as mpl
from scipy.interpolate import make_interp_spline
import json

##sh ae m
p1_l = 10
p1_r = 20
p2_l = 21
p2_r = 50
p3_l = 51
p3_r = 100
array_len = p3_r + 10

trans_prob = 0


def plot_it(out_file):

    plt.rcParams['font.size'] = 50
    fig, axes = plt.subplots(4,2,figsize=(25*4, 45*2), sharex="col",layout="constrained")
    ##plot the acitvations
    ##CE
    act_ce_1 = np.full(array_len, 0)
    act_ce_2 = np.full(array_len, 0)
    act_ce_3 = np.full(array_len, 0)
    
    act_ce_1[p1_l:p1_r+1] = 1
    act_ce_2[p2_l:p2_r+1] = 1
    act_ce_3[p3_l:p3_r+1] = 1

    act_ce_1[p1_l] = act_ce_1[p1_l] - trans_prob 
    act_ce_1[p1_r] = act_ce_1[p1_r] - trans_prob 
    act_ce_2[p1_r] = act_ce_2[p1_r] + trans_prob 
    act_ce_2[p2_l] = act_ce_2[p2_l] - trans_prob 
    act_ce_1[p2_l] = act_ce_1[p2_l] + trans_prob 
    act_ce_2[p2_r] = act_ce_2[p2_r] - trans_prob 
    act_ce_3[p2_r] = act_ce_3[p2_r] + trans_prob 
    act_ce_3[p3_l] = act_ce_3[p3_l] - trans_prob 
    act_ce_2[p3_l] = act_ce_2[p3_l] + trans_prob 
    act_ce_3[p3_r] = act_ce_3[p3_r] - trans_prob 

    ##interpolation
    CE_X = np.arange(array_len) 
    spline1 = make_interp_spline(CE_X, act_ce_1)
    CE_y1 = spline1(CE_X)
    spline2 = make_interp_spline(CE_X, act_ce_2)
    CE_y2 = spline2(CE_X)
    spline3 = make_interp_spline(CE_X, act_ce_3)
    CE_y3 = spline3(CE_X)

    ##CTC
    act_ctc_1 = np.full(array_len, 0)
    act_ctc_2 = np.full(array_len, 0)
    act_ctc_3 = np.full(array_len, 0)
    act_ctc_pad = np.full(array_len, 1)
    
    a1 = ceil((p1_l+p1_r)/2)
    a2 = ceil((p2_l+p2_r)/2)
    a3 = ceil((p3_l+p3_r)/2)
    act_ctc_1[a1] = 1
    act_ctc_2[a2] = 1
    act_ctc_3[a3] = 1
    act_ctc_pad[a1] = 0
    act_ctc_pad[a2] = 0
    act_ctc_pad[a3] = 0

    act_ctc_pad[a1-1] = act_ctc_pad[a1-1] - trans_prob 
    act_ctc_1[a1-1] = act_ctc_1[a1-1] + trans_prob 
    act_ctc_pad[a1+1] = act_ctc_pad[a1+1] - trans_prob 
    act_ctc_1[a1+1] = act_ctc_1[a1+1] + trans_prob 

    act_ctc_pad[a2-1] = act_ctc_pad[a2-1] - trans_prob 
    act_ctc_2[a2-1] = act_ctc_2[a2-1] + trans_prob 
    act_ctc_pad[a2+1] = act_ctc_pad[a2+1] - trans_prob 
    act_ctc_2[a2+1] = act_ctc_2[a2+1] + trans_prob 

    act_ctc_pad[a3-1] = act_ctc_pad[a3-1] - trans_prob 
    act_ctc_3[a3-1] = act_ctc_3[a3-1] + trans_prob 
    act_ctc_pad[a3+1] = act_ctc_pad[a3+1] - trans_prob 
    act_ctc_3[a3+1] = act_ctc_3[a3+1] + trans_prob 
   
    ##interpolation
    CTC_X = np.arange(array_len) 
    spline1_ctc = make_interp_spline(CTC_X, act_ctc_1)
    CTC_y1 = spline1_ctc(CTC_X)
    spline2_ctc = make_interp_spline(CTC_X, act_ctc_2)
    CTC_y2 = spline2_ctc(CTC_X)
    spline3_ctc = make_interp_spline(CTC_X, act_ctc_3)
    CTC_y3 = spline3_ctc(CTC_X)
    spline_pad = make_interp_spline(CTC_X, act_ctc_pad)
    CTC_y_pad = spline_pad(CTC_X)

    #plot activations

    axes[0][0].plot(CE_X, CE_y1, color="red", label="SH",linewidth=5)
    axes[0][0].plot(CE_X, CE_y2, color="orange", label="AE",linewidth=5)
    axes[0][0].plot(CE_X, CE_y3, color="green", label="M", linewidth=5)
    axes[0][1].plot(CTC_X, CTC_y1, color="red", label="SH", linewidth=5)
    axes[0][1].plot(CTC_X, CTC_y2, color="orange", label="AE", linewidth=5)
    axes[0][1].plot(CTC_X, CTC_y3, color="green", label="M", linewidth=5)
    axes[0][1].plot(CTC_X, CTC_y_pad, color="gray", label="blank")
        
    axes[0][0].set_title("CE-trained model" )
    axes[0][0].legend()
    axes[0][1].set_title("CTC-trained model" )
    axes[0][1].legend()

    ##plot groundtruth 
    pos_g = 2
    axes[1][0].plot((p1_l, p1_r), (pos_g,pos_g), '|r', markersize=100)
    axes[1][0].plot((p1_l, p1_r), (pos_g,pos_g), '-r', linewidth=5.0)
    axes[1][0].plot((p2_l, p2_r), (pos_g,pos_g), '|', color="orange", markersize=100)
    axes[1][0].plot((p2_l, p2_r), (pos_g,pos_g), '-', color="orange", linewidth=5.0)
    axes[1][0].plot((p3_l, p3_r), (pos_g,pos_g), '|g', markersize=100)
    axes[1][0].plot((p3_l, p3_r), (pos_g,pos_g), '-g', linewidth=5.0)

    axes[1][1].plot((p1_l, p1_r), (pos_g,pos_g), '|r', markersize=100)
    axes[1][1].plot((p1_l, p1_r), (pos_g,pos_g), '-r', linewidth=5.0)
    axes[1][1].plot((p2_l, p2_r), (pos_g,pos_g), '|', color="orange", markersize=100)
    axes[1][1].plot((p2_l, p2_r), (pos_g,pos_g), '-', color="orange", linewidth=5.0)
    axes[1][1].plot((p3_l, p3_r), (pos_g,pos_g), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r), (pos_g,pos_g), '-g', linewidth=5.0)


   
    fig.supxlabel('speech frames')
    fig.supylabel('activations')
    mpl.rcParams['lines.linewidth'] = 5

    plt.savefig(out_file)
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 2:
        sys.exit("this script takes 1 argument <out.png> ") 
    
    plot_it(sys.argv[1])
    
    

   







    
  
