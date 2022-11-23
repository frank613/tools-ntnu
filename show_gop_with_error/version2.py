import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import re
import pandas as pd
sys.path.append('/home/stipendiater/xinweic/tools/edit-distance')
from edit import edit_dist
from edit import get_sub_pair_list 
from sklearn import metrics
import pdb

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)?')
#for each phone in the GOP file:
    #label "S" if it's a substitution error based on the other input files.
    #label "D" if it's a deletion error based on the other input files.
    #label "C" if it's the same in both files
    
def labelError(GOP_file, cano_file, tran_file):
    gop_df = readGOPToDF(GOP_file)
    cano_df = readTRANToDF(cano_file)
    tran_df = readTRANToDF(tran_file)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    confusion_map = {} #each entry is a cano->{p:count}
    #outer loop is the cano df because it is extracted from the gop file and they should have the same number of uttids
    for index, row in cano_df.iterrows(): 
        uttid = row['uttid']
        cano_seq = row['seq']
        if uttid not in tran_df['uttid'].unique():
            print("warning: uttid {0} can't be found in the transcription".format(uttid))
            continue
        tran_df_filtered = tran_df.loc[tran_df["uttid"] == uttid, "seq"]
        if len(tran_df_filtered) != 1:
            sys.exit("duplicate uttids detected in the transcription file, check the input")
        tran_seq = tran_df_filtered.tolist()[0] #[0] converts 2d list to 1d
        dist, labels = edit_dist(cano_seq, tran_seq)
        #if dist == 0:
        #    continue
        if uttid not in gop_df['uttid'].unique():
            print("warning: uttid {0} can't be found in the gop file".format(uttid))
            continue
        labels_resized = [ label for idx, label in enumerate(labels) if label is not 'I']
        if len(labels_resized) != len(cano_seq):
            sys.exit("length of edit distance not maching the gop")

        extended = [ pair + (labels_resized[idx], uttid) for idx, pair in enumerate(gop_df.loc[gop_df['uttid'] == uttid, 'seq-score'].tolist()[0]) ]
        df = df.append(pd.DataFrame(extended, columns=['phonemes','scores','labels', 'uttid']))
        ##v2 adding statistics for error prons
        sub_list = get_sub_pair_list(cano_seq, tran_seq, labels)
        for canoP, tranP in sub_list:
            if canoP not in confusion_map.keys():
                confusion_map[canoP] = {tranP:1}
            elif tranP not in  confusion_map[canoP].keys():
                confusion_map[canoP][tranP] = 1
            else:
                confusion_map[canoP][tranP] += 1

    return (df,confusion_map)


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
        seq_score.append((cur_phoneme, cur_score)) 
    return df

def readTRANToDF(tran_file):
    in_file=open(tran_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq'))
    for line in in_file:
        uttid,seq = line.split()
        df.loc[len(df.index)] = [uttid, seq.split(';')]
    return df


plt.rcParams["figure.autolayout"] = True
#allowed_phonemes = ["OY", "AA", "SH", "EH"]
#allowed_phonemes = ["AA", "AO"]
def plot(df_labels, con_map, phoneme='all'):
    allowed_phonemes = df_labels["phonemes"].unique()
    #plt.rcParams["figure.autolayout"] = True
    allLabels = [ df_labels.loc[df_labels["phonemes"] == i,["scores","labels"]].to_numpy() for i in allowed_phonemes]
    dAndS = [ df_labels.loc[(df_labels["phonemes"] == i) & (df_labels['labels'].isin(["S", "D"])), "scores"].to_numpy() for i in allowed_phonemes]
    allC = [ df_labels.loc[(df_labels["phonemes"] == i) & (df_labels['labels'].isin(["C"])), "scores"].to_numpy() for i in allowed_phonemes]

    auc_vector = []
    fig, axs = plt.subplots(len(allLabels), figsize=(12,4*len(allLabels)))
    #plt.rcParams["figure.autolayout"] = True
    for idx,ax in enumerate(axs):
        ax.hist(allC[idx], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='g', label='{0} canonical phonemes'.format(len(allC[idx])))
        if allowed_phonemes[idx] in con_map.keys():
            freq_dict = con_map[allowed_phonemes[idx]]
            pNum = min(3, len(freq_dict))
            p_list = ','.join(sorted(freq_dict, key=freq_dict.get, reverse=True)[:pNum])
        else: 
            p_list = 'EMPT#Y'
        ax.hist(dAndS[idx], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='r', label='{0} sub/dels, top subs: {1}'.format(len(dAndS[idx]), p_list))
        if len(allC[idx]) != 0:
            ax.axvline(allC[idx].mean(), color='g', linestyle='dashed', linewidth=2, label='all-mean')
        if len(dAndS[idx]) != 0:
            ax.axvline(dAndS[idx].mean(), color='r', linestyle='dashed', linewidth=1, label='total error-mean, diff: {0}'.format(allC[idx].mean() - dAndS[idx].mean()))
        ax.legend(loc ="upper left")
        auc_value = auc_cal(allLabels[idx])
        ax.set_title('phoneme {0}, total AUC = {1}'.format(allowed_phonemes[idx], auc_value))
        if auc_value != 'NoDef':  ##exclude the phonemes that have only one lable
            auc_vector.append(auc_value)
    if len(auc_vector) != 0:
        print("average auc = {0}".format(sum(auc_vector)/len(auc_vector)))
    else:
        print("auc not available for current phonemes")

    outFile = "./out-error-cmu/all.png"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)    

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == "C" else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        return metrics.roc_auc_score(labels, -array[:, 0]) #negative because GOP is negatively correlated to the probablity of making an error


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <GOP file> <cano phone-seq-file> <transcribed phoneme-seq file> . It labels the phonemes and plot the points with the labels")

    df_labels, con_map = labelError(sys.argv[1], sys.argv[2], sys.argv[3])
    plot(df_labels, con_map)
    #showGOP(df_gop)

