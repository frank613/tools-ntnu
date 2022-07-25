import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
import sys
import re
import pandas as pd
sys.path.append('/home/stipendiater/xinweic/tools/edit-distance')
from edit import edit_dist
from sklearn import metrics

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*')
#for each phone in the GOP file:
    #label "S" if it's a substitution error based on the other input files.
    #label "D" if it's a deletion error based on the other input files.
    #label "C" if it's the same in both files
    
def labelError(GOP_file, cano_file, tran_file):
    gop_df = readGOPToDF(GOP_file)
    cano_df = readTRANToDF(cano_file)
    tran_df = readTRANToDF(tran_file)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
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
        if dist == 0:
            continue
        if uttid not in gop_df['uttid'].unique():
            print("warning: uttid {0} can't be found in the gop file".format(uttid))
            continue
        labels_resized = [ label for idx, label in enumerate(labels) if label is not 'I']
        if len(labels_resized) != len(cano_seq):
            sys.exit("length of edit distance not maching the gop")

        extended = [ pair + (labels_resized[idx], uttid) for idx, pair in enumerate(gop_df.loc[gop_df['uttid'] == uttid, 'seq-score'].tolist()[0]) ]
        df = df.append(pd.DataFrame(extended, columns=['phonemes','scores','labels', 'uttid']))

    return df


    


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
allowed_phonemes = ["AA"]
def plot(df_labels, phoneme='all'):
    allowed_phonemes = df_labels["phonemes"].unique()
    #plt.rcParams["figure.autolayout"] = True
    allLabels = [ df_labels.loc[df_labels["phonemes"] == i,["scores","labels"]].to_numpy() for i in allowed_phonemes]
    dAndS = [ df_labels.loc[(df_labels["phonemes"] == i) & (df_labels['labels'].isin(["S", "D"])), "scores"].to_numpy() for i in allowed_phonemes]
    assert(len(allLabels) == len(dAndS))
    #plt.legend(loc='upper right')
    if len(allLabels) == 1:
        fig, ax = plt.subplots(1, 1)
        ax.hist(allLabels[0][:, 0], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='all')
        ax.hist(dAndS[0], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='error')
        if len(allLabels[0]) != 0:
            ax.axvline(allLabels[0][:, 0].mean(), color='k', linestyle='dashed', linewidth=2, label='all-mean')
        if len(dAndS[0]) != 0:
            ax.axvline(dAndS[0].mean(), color='k', linestyle='dashed', linewidth=1, label='error-mean')
        ax.legend(loc ="upper left")
        ax.set_title('phoneme {0}, AUC = {1}'.format(allowed_phonemes[0], auc_cal(allLabels[0])))

    else:
        fig, axs = plt.subplots(len(allLabels), figsize=(10,3*len(allLabels)))
        plt.rcParams["figure.autolayout"] = True
        for idx,ax in enumerate(axs):
            ax.hist(allLabels[idx][:, 0], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='all')
            ax.hist(dAndS[idx], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='error')
            if len(allLabels[idx]) != 0:
                ax.axvline(allLabels[idx][:, 0].mean(), color='k', linestyle='dashed', linewidth=2, label='all-mean')
            if len(dAndS[idx]) != 0:
                ax.axvline(dAndS[idx].mean(), color='k', linestyle='dashed', linewidth=1, label='error-mean')
            ax.legend(loc ="upper left")
            ax.set_title('phoneme {0}, AUC = {1}'.format(allowed_phonemes[idx], auc_cal(allLabels[idx])))

    html_str = mpld3.fig_to_html(fig)
    Html_file= open("{0}.html".format(phoneme),"w")
    Html_file.write(html_str)
    Html_file.close()
            
def auc_cal(array): #input is a nX2 array, the with the columns "score", "label"
    labels = [ 0 if i == "C" else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDefine"
    else:
        return metrics.roc_auc_score(labels, -array[:, 0]) #negative because GOP is negative correlateing to the probablity of making an error


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <GOP file> <cano phone-seq-file> <transcribed phoneme-seq file> . It labels the phonemes and plot the points with the labels")

    df_labels = labelError(sys.argv[1], sys.argv[2], sys.argv[3])
    plot(df_labels)
    #showGOP(df_gop)

