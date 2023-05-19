import sys
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pdb
import numpy as np

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)?')
def readGOPToDF(df, gop_file, label):
    with open(gop_file, 'r') as in_file:
        isNewUtt = True
        p_list=[]
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
                isNewUtt = True
                continue
            if len(fields) != 3:
                continue
            cur_match = re_phone.match(fields[1])
            if (cur_match):
                cur_phoneme = cur_match.group(1)
            else:
                sys.exit("non legal phoneme found in the gop file")
            cur_score = float(fields[2])
            p_list.append((cur_phoneme, round(cur_score, 3), label))
    return df.append(pd.DataFrame(p_list, columns=['phoneme','score','label']))

def plot(df, data_labels):
    label = df['label'].unique()
    all_phonemes = df['phoneme'].unique()
    fig, ax = plt.subplots(1, figsize=(8,6))
    data = [ df.loc[(df["phoneme"] != "SIL") & (df["phoneme"] != "SPN")  & (df["label"] == lb), 'score'].to_numpy() for lb in label]
    plot_labels=[]
    for row,lb,pos in zip(data,label,range(len(label))):
        add_label(ax.violinplot(row,vert=False, quantiles=[0.25,0.5,0.75], points=500, positions=[pos/1.5]), data_labels[lb-1], round(np.std(row),3), plot_labels)
    #ax.set_title('Gop distribution for different data sets using DNN')
    ax.get_yaxis().set_visible(False)
    ax.set_xlim([-20, 0.1])
    plot_labels.reverse()
    ax.legend(*zip(*plot_labels), loc=3, fontsize=12, framealpha=1.0)
    outFile = "./out-compare-dnn-vilion/all-tri-DNN.png"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outFile)
        

    print("done")

def add_label(violin, label, score, labels,):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), "{0}, std={1}".format(label, score)))

if __name__ == "__main__":
    if len(sys.argv) <= 1 :
        sys.exit("this script takes <GOP file 1> <GOP file 2> <GOP file 3>... as arguments. It plots the GOP distributions for each phoneme")
    
    df = pd.DataFrame(columns=('phoneme','score','label'))
    for i in range(1,len(sys.argv)):
        df = readGOPToDF(df, sys.argv[i], i)
        print("read one GOP")

    plot(df, ['Librispeech', 'Voxforge', 'Tedlium','CMU-kids-filtered', 'CMU-kids'])
