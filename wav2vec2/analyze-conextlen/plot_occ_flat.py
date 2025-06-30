import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForCTC
from my_w2v2_package.custom_processor import My_Wav2Vec2Processor
import torch
from pathlib import Path
import pdb
import json
import matplotlib.pyplot as plt

def plot_it(out_file,rep,occ_vec_sd, occ_vec_sdi):
   
    fig, ax = plt.subplots()
    x_ticks = np.arange(0,rep+1)
    #plt.rcParams['font.size'] = 50
    plt.xlabel('Repetiton')
    plt.xticks(x_ticks)
    plt.yticks(np.arange(0,rep+4))
    plt.ylabel('Estimated occupancy')
    #plt.ylim(-80,20)


    ax.plot(x_ticks, occ_vec_sd, 'o-',  label="GOP-CTC-SS-SD")
    ax.plot(x_ticks, occ_vec_sdi, 'o-',  label="GOP-CTC-SS-SDI")
    
    
    plt.legend()
    fig.savefig(out_file)
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 2:
        sys.exit("this script takes 1 arguments <out-png>.") 
    frame_rep = 8
    occ_vec_sd = [0.0007085017266677115, 1.0000116049925472, 1.9999844251664245, 2.9999586125227213, 3.999932880806475, 4.9999071539541236, 5.99988142739406, 6.999855700851562, 7.999829974310119]
    occ_vec_sdi = [0.015030799300759541, 1.6132596196990954, 2.6132429647150692, 3.6132187071984183, 4.6131944495111075, 5.613170191823792, 6.613145934136478, 7.613121676449163, 8.613097418761852]
    plot_it(sys.argv[1], frame_rep, occ_vec_sd, occ_vec_sdi)








    
  
