import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
import sys
import re
import pandas as pd

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*')

def labelError(GOP_file, cano_GOP, from_phone):
    gop_df = readGOPToDF(GOP_file)
    cano_df = readGOPToDF(cano_GOP)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    if len(gop_df.index) != len(cano_df.index):
        print("number of the uttids of the two GOPs is not matching")
    
    for index, row in cano_df.iterrows():
        if not any(gop_df["uttid"] == row["uttid"]):
            continue
        newSeq = gop_df.loc[gop_df["uttid"] == row["uttid"], "seq-score"]
        if len(newSeq) != 1:
            print("same uttid found in the same file , ignored the utterance")
            continue
        newSeq = newSeq.tolist()[0]
        #if row['uttid'] != gop_df.loc[index, "uttid"]:
        #    sys.exit("uttid not matching, ignored the utterance")
        if len(row["seq-score"]) != len(newSeq):
            print("length of phonemes not matching for uttid: {}, ignored the utterence".format(row['uttid']))
            continue
        labels = [1 if pair[0] == from_phone else 0 for pair in row["seq-score"]]
        extended = [ pair + (labels[idx], row['uttid']) for idx, pair in enumerate(newSeq)]
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

plt.rcParams["figure.autolayout"] = True
def plot(df_labels, from_phoneme, phoneme):
    real = df_labels.loc[(df_labels["phonemes"] == phoneme) & (df_labels["labels"] == 0 ), "scores"].to_numpy()
    substituted = df_labels.loc[(df_labels["phonemes"] == phoneme ) & (df_labels["labels"] == 1),"scores"].to_numpy()
    fig, ax = plt.subplots(1, 1)
    if len(real) != 0:
        ax.hist(real, density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='real')
        #ax.hist(real.mean(), range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color="k", label='original-mean')
        ax.axvline(real.mean(), color='k', linestyle='dashed', linewidth=1, label='original-mean')
    if len(substituted) != 0:
        ax.hist(substituted, density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.8, label='substituted from {0}'.format(from_phoneme))
        #ax.hist(substituted.mean(),  range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, color='k', label='wrong-mean')
        ax.axvline(substituted.mean(), color='k', linestyle='dashed', linewidth=1, label='error-mean')  
    ax.legend(loc ="upper left")
    ax.set_title('phoneme {0}, substitue from {1}'.format(phoneme, from_phoneme))

    html_str = mpld3.fig_to_html(fig)
    Html_file= open("./output_v2/{0}/{1}.html".format(phoneme, from_phoneme),"w")
    Html_file.write(html_str)
    Html_file.close()     

    plt.savefig("./output_v2/{0}/{1}.png".format(phoneme, from_phoneme))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP file> <cano GOP file> <phoneme-orig> <phoneme-to> . \
        It labels the substituted phonemes and plot the points with the labels")

    df_labels = labelError(sys.argv[1], sys.argv[2], sys.argv[3])
    plot(df_labels, sys.argv[3], sys.argv[4])
    #showGOP(df_gop)
