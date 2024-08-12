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
    fig, axes = plt.subplots(2,3,figsize=(90, 30), sharex="col", sharey="row", layout="constrained", width_ratios=[1,4,4], height_ratios=[1,2])
    axes[0][0].set_frame_on(False) 
    axes[0][0].set_xlim(-2, 0) 
    #axes[0][0].set_ylim(-4, 6) 
    axes[0][0].axis("off") 
    axes[1][0].set_frame_on(False) 
    axes[1][0].set_xlim(-2, 0) 
    axes[1][0].set_ylim(-4, 6) 
    axes[1][0].axis("off") 
    ##plot the acitvations
    ##CE
    act_ce_1 = np.full(array_len, 0)
    act_ce_2 = np.full(array_len, 0)
    act_ce_3 = np.full(array_len, 0)
    act_ce_sil = np.full(array_len, 0)
    
    act_ce_1[p1_l:p1_r+1] = 1
    act_ce_2[p2_l:p2_r+1] = 1
    act_ce_3[p3_l:p3_r+1] = 1
    act_ce_sil[:p1_l] = 1
    act_ce_sil[p3_r+1:] = 1

    act_ce_1[p1_l] = act_ce_1[p1_l] - trans_prob 
    act_ce_sil[p1_l] = act_ce_sil[p1_l] + trans_prob 
    act_ce_1[p1_r] = act_ce_1[p1_r] - trans_prob 
    act_ce_2[p1_r] = act_ce_2[p1_r] + trans_prob 
    act_ce_2[p2_l] = act_ce_2[p2_l] - trans_prob 
    act_ce_1[p2_l] = act_ce_1[p2_l] + trans_prob 
    act_ce_2[p2_r] = act_ce_2[p2_r] - trans_prob 
    act_ce_3[p2_r] = act_ce_3[p2_r] + trans_prob 
    act_ce_3[p3_l] = act_ce_3[p3_l] - trans_prob 
    act_ce_2[p3_l] = act_ce_2[p3_l] + trans_prob 
    act_ce_3[p3_r] = act_ce_3[p3_r] - trans_prob 
    act_ce_sil[p3_r] = act_ce_sil[p3_r] + trans_prob 

    ##interpolation
    CE_X = np.arange(array_len) 
    spline1 = make_interp_spline(CE_X, act_ce_1)
    CE_y1 = spline1(CE_X)
    spline2 = make_interp_spline(CE_X, act_ce_2)
    CE_y2 = spline2(CE_X)
    spline3 = make_interp_spline(CE_X, act_ce_3)
    CE_y3 = spline3(CE_X)
    spline_sil = make_interp_spline(CE_X, act_ce_sil)
    CE_sil = spline_sil(CE_X)

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

    axes[0][1].plot(CE_X, CE_y1, color="red", label="SH",linewidth=8)
    axes[0][1].plot(CE_X, CE_y2, color="orange", label="AE",linewidth=8)
    axes[0][1].plot(CE_X, CE_y3, color="green", label="M", linewidth=8)
    axes[0][1].plot(CE_X, CE_sil, color="gray", label="SIL", linewidth=5)
    axes[0][2].plot(CTC_X, CTC_y1, color="red", label="SH", linewidth=8)
    axes[0][2].plot(CTC_X, CTC_y2, color="orange", label="AE", linewidth=8)
    axes[0][2].plot(CTC_X, CTC_y3, color="green", label="M", linewidth=8)
    axes[0][2].plot(CTC_X, CTC_y_pad, color="gray", label="B", linewidth=5)
        
    axes[0][1].set_title("CE-trained model" )
    axes[0][1].legend()
    axes[0][2].set_title("CTC-trained model" )
    axes[0][2].legend()

    axes[0][0].text(0,0.5, 'model\'s activations', style='italic', color="tab:blue",
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, horizontalalignment="right", verticalalignment="center")

    pos_label = -1
    ##plot groundtruth 
    pos_g = 4
    axes[1][1].plot((p1_l, p1_r+1), (pos_g,pos_g), '|r', markersize=100)
    axes[1][1].plot((p1_l, p1_r+1), (pos_g,pos_g), '-r', linewidth=8.0)
    axes[1][1].plot((p2_l, p2_r+1), (pos_g,pos_g), '|', color="orange", markersize=100)
    axes[1][1].plot((p2_l, p2_r+1), (pos_g,pos_g), '-', color="orange", linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_g,pos_g), '-g', linewidth=8.0)
    axes[1][1].plot((0, p1_l), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][1].plot((0, p1_l), (pos_g,pos_g), '-', color="gray",  linewidth=8.0)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_g,pos_g), '-', color="gray", linewidth=8.0)

    pos_g_text = pos_g + 1
    axes[1][1].text(p1_l/2-3, pos_g_text, 'SIL', color='gray')
    axes[1][1].text((p3_r+array_len-1)/2-3, pos_g_text, 'SIL', color='gray')
    axes[1][1].text((p1_l+p1_r)/2-2, pos_g_text, 'SH', color='red')
    axes[1][1].text((p2_l+p2_r)/2-2, pos_g_text, 'AE', color='orange')
    axes[1][1].text((p3_l+p3_r)/2-1, pos_g_text, 'M', color='green')

    axes[1][2].plot((a1-1, a1+1), (pos_g,pos_g), '|r', markersize=100)
    axes[1][2].plot((a1-1, a1+1), (pos_g,pos_g), '-r', linewidth=8.0)
    axes[1][2].plot((a2-1, a2+1), (pos_g,pos_g), '|', color="orange", markersize=100)
    axes[1][2].plot((a2-1, a2+1), (pos_g,pos_g), '-', color="orange", linewidth=8.0)
    axes[1][2].plot((a3-1, a3+1), (pos_g,pos_g), '|g', markersize=100)
    axes[1][2].plot((a3-1, a3+1), (pos_g,pos_g), '-g', linewidth=8.0)

    axes[1][2].plot((0, a1-1), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][2].plot((a1+1, a2-1), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][2].plot((a2+1, a3-1), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][2].plot((a3+1, array_len-1), (pos_g,pos_g), '|', color="gray", markersize=100)
    axes[1][2].plot((0, a1-1), (pos_g,pos_g), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a1+1, a2-1), (pos_g,pos_g), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a2+1, a3-1), (pos_g,pos_g), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a3+1, array_len-1), (pos_g,pos_g), '-', color="gray", linewidth=5.0)


    axes[1][2].text(a1/2-1, pos_g_text, 'B', color='gray')
    axes[1][2].text((a1+a2)/2-1, pos_g_text, 'B', color='gray')
    axes[1][2].text((a2+a3)/2-1, pos_g_text, 'B', color='gray')
    axes[1][2].text((a3+array_len-1)/2-1, pos_g_text, 'B', color='gray')
    axes[1][2].text(a1-2, pos_g_text, 'SH', color='red')
    axes[1][2].text(a2-2, pos_g_text, 'AE', color='orange')
    axes[1][2].text(a3-1, pos_g_text, 'M', color='green')
   
    axes[1][0].text(0, pos_g, 'ground truth alignment', style='italic', color="tab:blue",
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, horizontalalignment="right", verticalalignment="center")
    #plot substituion
    pos_s = pos_g - 2
    axes[1][0].text(0, pos_s, 'simulated substitution', style='italic', color="tab:blue",
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, horizontalalignment="right", verticalalignment="center")

    p2_sub_r = p2_r + 4
    p3_sub_l = p3_l + 4 
    axes[1][1].plot((p1_l, p1_r+1), (pos_s,pos_s), '|r', markersize=100)
    axes[1][1].plot((p1_l, p1_r+1), (pos_s,pos_s), '-r', linewidth=8.0)
    axes[1][1].plot((p2_l, p2_sub_r+1), (pos_s,pos_s), '|', color="orange", markersize=100)
    axes[1][1].plot((p2_l, p2_sub_r+1), (pos_s,pos_s), '-', color="orange", linewidth=8.0)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '|g', markersize=100)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '-g', linewidth=8.0)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '|g', markersize=100)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '-g', linewidth=8.0)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '|g', markersize=100)
    axes[1][1].plot((p3_sub_l, p3_r+1), (pos_s,pos_s), '-g', linewidth=8.0)
    axes[1][1].plot((0, p1_l), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][1].plot((0, p1_l), (pos_s,pos_s), '-', color="gray",  linewidth=8.0)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_s,pos_s), '-', color="gray", linewidth=8.0)

    pos_s_text = pos_s + 1
    axes[1][1].text(p1_l/2-3, pos_s_text, 'SIL', color='gray')
    axes[1][1].text((p3_r+array_len-1)/2-3, pos_s_text, 'SIL', color='gray')
    axes[1][1].text((p1_l+p1_r)/2-2, pos_s_text, 'SH', color='red')
    axes[1][1].text((p2_l+p2_sub_r)/2-2, pos_s_text, 'AE', color='orange')
    axes[1][1].text((p3_sub_l+p3_r)/2-1, pos_s_text, 'L', color='green')

    axes[1][2].plot((a1-1, a1+1), (pos_s,pos_s), '|r', markersize=100)
    axes[1][2].plot((a1-1, a1+1), (pos_s,pos_s), '-r', linewidth=8.0)
    axes[1][2].plot((a2-1, a2+1), (pos_s,pos_s), '|', color="orange", markersize=100)
    axes[1][2].plot((a2-1, a2+1), (pos_s,pos_s), '-', color="orange", linewidth=8.0)
    axes[1][2].plot((a3-1, a3+1), (pos_s,pos_s), '|g', markersize=100)
    axes[1][2].plot((a3-1, a3+1), (pos_s,pos_s), '-g', linewidth=8.0)

    axes[1][2].plot((0, a1-1), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][2].plot((a1+1, a2-1), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][2].plot((a2+1, a3-1), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][2].plot((a3+1, array_len-1), (pos_s,pos_s), '|', color="gray", markersize=100)
    axes[1][2].plot((0, a1-1), (pos_s,pos_s), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a1+1, a2-1), (pos_s,pos_s), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a2+1, a3-1), (pos_s,pos_s), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a3+1, array_len-1), (pos_s,pos_s), '-', color="gray", linewidth=5.0)


    axes[1][2].text(a1/2-1, pos_s_text, 'B', color='gray')
    axes[1][2].text((a1+a2)/2-1, pos_s_text, 'B', color='gray')
    axes[1][2].text((a2+a3)/2-1, pos_s_text, 'B', color='gray')
    axes[1][2].text((a3+array_len-1)/2-1, pos_s_text, 'B', color='gray')
    axes[1][2].text(a1-2, pos_s_text, 'SH', color='red')
    axes[1][2].text(a2-2, pos_s_text, 'AE', color='orange')
    axes[1][2].text(a3-1, pos_s_text, 'L', color='green')

    #plot insertion
    pos_d = pos_g - 4
    axes[1][0].text(0, pos_d, 'simulated insertion', style='italic', color="tab:blue",
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, horizontalalignment="right", verticalalignment="center")

    p2_del_l = p1_l
    axes[1][1].plot((p2_del_l, p2_r+1), (pos_d,pos_d), '|', color="orange", markersize=100)
    axes[1][1].plot((p2_del_l, p2_r+1), (pos_d,pos_d), '-', color="orange", linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_d,pos_d), '-g', linewidth=8.0)
    axes[1][1].plot((0, p1_l), (pos_d,pos_d), '|', color="gray", markersize=100)
    axes[1][1].plot((0, p1_l), (pos_d,pos_d), '-', color="gray",  linewidth=8.0)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_d,pos_d), '|', color="gray", markersize=100)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_d,pos_d), '-', color="gray", linewidth=8.0)

    pos_d_text = pos_d + 1
    axes[1][1].text(p1_l/2-3, pos_d_text, 'SIL', color='gray')
    axes[1][1].text((p3_r+array_len-1)/2-3, pos_d_text, 'SIL', color='gray')
    axes[1][1].text((p2_del_l+p2_r)/2-2, pos_d_text, 'AE', color='orange')
    axes[1][1].text((p3_l+p3_r)/2-1, pos_d_text, 'M', color='green')

    axes[1][2].plot((a2-1, a2+1), (pos_d,pos_d), '|', color="orange", markersize=100)
    axes[1][2].plot((a2-1, a2+1), (pos_d,pos_d), '-', color="orange", linewidth=8.0)
    axes[1][2].plot((a3-1, a3+1), (pos_d,pos_d), '|g', markersize=100)
    axes[1][2].plot((a3-1, a3+1), (pos_d,pos_d), '-g', linewidth=8.0)

    axes[1][2].plot((0, a2-1), (pos_d,pos_d), '|', color="gray", markersize=100)
    axes[1][2].plot((a2+1, a3-1), (pos_d,pos_d), '|', color="gray", markersize=100)
    axes[1][2].plot((a3+1, array_len-1), (pos_d,pos_d), '|', color="gray", markersize=100)
    axes[1][2].plot((0, a2-1), (pos_d,pos_d), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a2+1, a3-1), (pos_d,pos_d), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a3+1, array_len-1), (pos_d,pos_d), '-', color="gray", linewidth=5.0)


    axes[1][2].text(a2/2-1, pos_d_text, 'B', color='gray')
    axes[1][2].text((a2+a3)/2-1, pos_d_text, 'B', color='gray')
    axes[1][2].text((a3+array_len-1)/2-1, pos_d_text, 'B', color='gray')
    axes[1][2].text(a2-2, pos_d_text, 'AE', color='orange')
    axes[1][2].text(a3-1, pos_d_text, 'M', color='green')
    #plot deletion
    pos_i = pos_g - 6
    axes[1][0].text(0, pos_i, 'simulated deletion', style='italic', color="tab:blue",
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}, horizontalalignment="right", verticalalignment="center")

    
    ins_p_l = p2_l - 2
    ins_p_r = p2_l 
    p1_ins_r = ins_p_l

    axes[1][1].plot((p1_l, p1_ins_r), (pos_i,pos_i), '|r', markersize=100)
    axes[1][1].plot((p1_l, p1_ins_r), (pos_i,pos_i), '-r', linewidth=8.0)
    axes[1][1].plot((ins_p_l, ins_p_r), (pos_i,pos_i), '|c', markersize=100)
    axes[1][1].plot((ins_p_l, ins_p_r), (pos_i,pos_i), '-c', linewidth=8.0)
    axes[1][1].plot((p2_l, p2_r+1), (pos_i,pos_i), '|', color="orange", markersize=100)
    axes[1][1].plot((p2_l, p2_r+1), (pos_i,pos_i), '-', color="orange", linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '-g', linewidth=8.0)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '|g', markersize=100)
    axes[1][1].plot((p3_l, p3_r+1), (pos_i,pos_i), '-g', linewidth=8.0)
    axes[1][1].plot((0, p1_l), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][1].plot((0, p1_l), (pos_i,pos_i), '-', color="gray",  linewidth=8.0)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][1].plot((p3_r+1, array_len-1), (pos_i,pos_i), '-', color="gray", linewidth=8.0)

    pos_i_text = pos_i + 1
    axes[1][1].text(p1_l/2-3, pos_i_text, 'SIL', color='gray')
    axes[1][1].text((p3_r+array_len-1)/2-3, pos_i_text, 'SIL', color='gray')
    axes[1][1].text((p1_l+p1_r)/2-2, pos_i_text, 'SH', color='red')
    axes[1][1].text((ins_p_l+ins_p_r)/2-1, pos_i_text, 'S', color='cyan')
    axes[1][1].text((p2_l+p2_r)/2-2, pos_i_text, 'AE', color='orange')
    axes[1][1].text((p3_l+p3_r)/2-1, pos_i_text, 'M', color='green')

    axes[1][2].plot((a1-1, a1+1), (pos_i,pos_i), '|r', markersize=100)
    axes[1][2].plot((a1-1, a1+1), (pos_i,pos_i), '-r', linewidth=8.0)
    axes[1][2].plot((a1+1, a1+3), (pos_i,pos_i), '|c', markersize=100)
    axes[1][2].plot((a1+1, a1+3), (pos_i,pos_i), '-c', linewidth=8.0)
    axes[1][2].plot((a2-1, a2+1), (pos_i,pos_i), '|', color="orange", markersize=100)
    axes[1][2].plot((a2-1, a2+1), (pos_i,pos_i), '-', color="orange", linewidth=8.0)
    axes[1][2].plot((a3-1, a3+1), (pos_i,pos_i), '|g', markersize=100)
    axes[1][2].plot((a3-1, a3+1), (pos_i,pos_i), '-g', linewidth=8.0)

    axes[1][2].plot((0, a1-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][2].plot((a1+3, a2-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][2].plot((a1+3, a2-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][2].plot((a2+1, a3-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][2].plot((a3+1, array_len-1), (pos_i,pos_i), '|', color="gray", markersize=100)
    axes[1][2].plot((0, a1-1), (pos_i,pos_i), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a1+3, a2-1), (pos_i,pos_i), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a2+1, a3-1), (pos_i,pos_i), '-', color="gray", linewidth=8.0)
    axes[1][2].plot((a3+1, array_len-1), (pos_i,pos_i), '-', color="gray", linewidth=5.0)


    axes[1][2].text(a1/2-1, pos_i_text, 'B', color='gray')
    axes[1][2].text((a1+3+a2)/2-1, pos_i_text, 'B', color='gray')
    axes[1][2].text((a2+a3)/2-1, pos_i_text, 'B', color='gray')
    axes[1][2].text((a3+array_len-1)/2-1, pos_i_text, 'B', color='gray')
    axes[1][2].text(a1-2, pos_i_text, 'SH', color='red')
    axes[1][2].text(a1+1, pos_i_text, 'S', color='cyan')
    axes[1][2].text(a2-2, pos_i_text, 'AE', color='orange')
    axes[1][2].text(a3-1, pos_i_text, 'M', color='green')
    fig.supxlabel('speech frames')
    #fig.supylabel('activations')

    plt.savefig(out_file)
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 2:
        sys.exit("this script takes 1 argument <out.png> ") 
    
    plot_it(sys.argv[1])
    
    

   







    
  
