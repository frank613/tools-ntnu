import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
import sys
import re
import pandas as pd
from sklearn import metrics

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*')

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        return metrics.roc_auc_score(labels, -array[:, 0]) #negative because GOP is negatively correlated to the probablity of making an error

def labelError(GOP_file, cano_GOP, from_phone):
    gop_df = readGOPToDF(GOP_file)
    cano_df = readGOPToDF(cano_GOP)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    entries = []
    if len(gop_df.index) != len(cano_df.index):
        print("number of the uttids of the two GOPs is not matching")
    
    for index, row in cano_df.iterrows():
        #if not any(gop_df["uttid"] == row["uttid"]):
        #    continue
        newSeq = gop_df.loc[gop_df["uttid"] == row["uttid"], "seq-score"]
        if len(newSeq) != 1:
            #print("same uttid found in the same file , ignored the utterance")
            print("uttid in the new gop not found, ignored the utterance")
            continue
        #skip SIL at the start/end and optional SIL
        newSeq = [ pair for pair in newSeq.tolist()[0] if pair[0] != "SIL" ]
        oldSeq = [ pair for pair in row["seq-score"] if pair[0] != "SIL" ]
        #if row['uttid'] != gop_df.loc[index, "uttid"]:
        #    sys.exit("uttid not matching, ignored the utterance")
        if len(newSeq) != len(oldSeq):
            #sys.exit("length of phonemes not matching for uttid: {0} in file {3}: ignored the utterence: {1} vs {2}".format(row['uttid'], len(row["seq-score"]), len(newSeq), GOP_file))
            print("length of phonemes not matching for uttid: {0} due to realignment with an alternative pron,  skipped the utterence".format(row['uttid'], len(row["seq-score"]), len(newSeq), GOP_file))
            continue

        labels = [1 if pair[0] == from_phone else 0 for pair in oldSeq]
        extended = [ pair + (labels[idx], row['uttid']) for idx, pair in enumerate(newSeq)]
        #df = df.append(pd.DataFrame(extended, columns=['phonemes','scores','labels', 'uttid']))
        entries = entries + extended
    return df.append(pd.DataFrame(entries, columns=['phonemes','scores','labels', 'uttid']))


##for the second phonme-replacement method, no need for the original GOP file
def labelError2(GOP_file, from_phone): 
    gop_df = readGOPToDF(GOP_file)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    entries = []
    
    for index, row in gop_df.iterrows():
        uttid = row["uttid"]
        newSeq = row["seq-score"]
        #if row['uttid'] != gop_df.loc[index, "uttid"]:
        #    sys.exit("uttid not matching, ignored the utterance")
        labels = [1 if pair[0] == from_phone else 0 for pair in newSeq]
        extended = [ pair + (labels[idx], uttid) for idx, pair in enumerate(newSeq)]
        entries = entries + extended
    return df.append(pd.DataFrame(entries, columns=['phonemes','scores','labels', 'uttid'])) 

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

plt.rcParams["figure.autolayout"] = True
def plot(df_labels, df_labels2, df_labels3, from_phoneme, phoneme):
    real = df_labels.loc[(df_labels["phonemes"] == phoneme) & (df_labels["labels"] == 0 ), "scores"].to_numpy()
    real2 = df_labels2.loc[df_labels2["phonemes"] == phoneme, "scores"].to_numpy()
    real3 = df_labels3.loc[df_labels3["phonemes"] == phoneme, "scores"].to_numpy()
    substituted = df_labels.loc[(df_labels["phonemes"] == phoneme ) & (df_labels["labels"] == 1),"scores"].to_numpy()
    substituted2 = df_labels2.loc[df_labels2["phonemes"] == from_phoneme, "scores"].to_numpy()
    substituted3 = df_labels3.loc[df_labels3["phonemes"] == from_phoneme, "scores"].to_numpy()
    auc_arr= df_labels.loc[(df_labels["phonemes"] == phoneme ), ["scores","labels"]].to_numpy()
    auc_arr2= df_labels2.loc[(df_labels2["phonemes"] == phoneme ) | (df_labels2["phonemes"] == from_phoneme) , ["scores","labels"]].to_numpy()
    auc_arr3= df_labels3.loc[(df_labels3["phonemes"] == phoneme ) | (df_labels3["phonemes"] == from_phoneme) , ["scores","labels"]].to_numpy()

    fig, ax = plt.subplots(2, 2, figsize=(12,4))
    if len(real) != 0:
        ax[0,0].hist(real, color='g', density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='real')
        #ax.hist(real.mean(), range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color="k", label='original-mean')
        real_mean = real.mean()
        sub_mean = substituted.mean()
        ax[0,0].axvline(real_mean, color='g', linestyle='dashed', linewidth=1, label='original-mean')
        #ax[0].annotate(str(round(real_mean,2)), (real_mean, 0), xytext=(0, -0.03))
    if len(substituted) != 0:
        ax[0,0].hist(substituted, color='r', density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.8, label='substituted from {0}'.format(from_phoneme))
        #ax.hist(substituted.mean(),  range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='k', label='wrong-mean')
        ax[0,0].axvline(sub_mean, color='r', linestyle='dashed', linewidth=1, label='error-mean')  
        #ax[0].annotate(str(round(sub_mean,2)), (sub_mean, 0), xytext=(0, -0.03))
        ax[0,0].legend(loc ="upper left")
        ax[0,0].set_title('Re-align: {0}<-{1}, AUC={2}, Diff_mean={3}'.format(phoneme, from_phoneme, round(auc_cal(auc_arr),3), round(real_mean - sub_mean, 3)))

    if len(real2) != 0:
        real_mean2 = real2.mean()
        sub_mean2 = substituted2.mean() 
        ax[0,1].hist(real2, color='g', density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='real')
        ax[0,1].axvline(real_mean2, color='g', linestyle='dashed', linewidth=1, label='original-mean')
        #ax[1].annotate(str(round(real_mean2)), (real_mean2, 0), xytext=(0, -0.03))
    if len(substituted2) != 0:
        ax[0,1].hist(substituted2, color='r',density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.8, label='substituted from {0}'.format(from_phoneme))
        #ax.hist(substituted.mean(),  range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='k', label='wrong-mean')
        ax[0,1].axvline(sub_mean2, color='r', linestyle='dashed', linewidth=1, label='error-mean')  
        #ax[1].annotate(str(round(sub_mean2,2)), (sub_mean2, 0), xytext=(0, -0.03))
    ax[0,1].legend(loc ="upper left")
    ax[0,1].set_title('Fix-align: {0}<-{1}, AUC={2}, Diff_mean={3}'.format(phoneme, from_phoneme, round(auc_cal(auc_arr2),3), round(real_mean2 - sub_mean2, 3)))

    if len(real3) != 0:
        real_mean3 = real3.mean()
        sub_mean3 = substituted3.mean() 
        ax[1,0].hist(real3, color='g', density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='real')
        ax[1,0].axvline(real_mean3, color='g', linestyle='dashed', linewidth=1, label='original-mean')
        #ax[1].annotate(str(round(real_mean2)), (real_mean2, 0), xytext=(0, -0.03))
    if len(substituted3) != 0:
        ax[1,0].hist(substituted3, color='r',density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.8, label='substituted from {0}'.format(from_phoneme))
        #ax.hist(substituted.mean(),  range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='k', label='wrong-mean')
        ax[1,0].axvline(sub_mean3, color='r', linestyle='dashed', linewidth=1, label='error-mean')  
        #ax[1].annotate(str(round(sub_mean2,2)), (sub_mean2, 0), xytext=(0, -0.03))
    ax[1,0].legend(loc ="upper left")
    ax[1,0].set_title('Fix-align-single-state: {0}<-{1}, AUC={2}, Diff_mean={3}'.format(phoneme, from_phoneme, round(auc_cal(auc_arr3),3), round(real_mean3 - sub_mean3, 3)))
    html_str = mpld3.fig_to_html(fig)
    Html_file= open("./output3/{0}/{1}.html".format(phoneme, from_phoneme),"w")
    Html_file.write(html_str)
    Html_file.close()     

    plt.savefig("./output3/{0}/{1}.png".format(phoneme, from_phoneme))

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 5 arguments <GOP file1> <GOP file2> <GOP file3> <cano GOP file> <phoneme-orig> <phoneme-to> . \
        It labels the substituted phonemes and plot the points with the labels")

    df_labels = labelError(sys.argv[1], sys.argv[4], sys.argv[5])
    print("label error done for the first gop")
    df_labels2 = labelError2(sys.argv[2], sys.argv[5])
    print("label error done for the second gop")
    df_labels3 = labelError2(sys.argv[3], sys.argv[5])
    print("label error done for the thrid  gop")
    plot(df_labels, df_labels2, df_labels3, sys.argv[5], sys.argv[6])
    #showGOP(df_gop)
