import sys
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb

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
    fig, axs = plt.subplots(len(all_phonemes), figsize=(12,4*len(all_phonemes)))
    for phoneme,ax in zip(all_phonemes, axs):
        data = [ df.loc[(df["phoneme"] == phoneme) & (df["label"] == lb), 'score'].to_numpy() for lb in label]
        #ax.boxplot(data, 0, 'rs', 0, labels = ['CMU', 'LIB'])
        ax.boxplot(data, 0, 'rs', 0, labels = data_labels)
        ax.set_title('Gop distribution for phoneme: ' + phoneme)
    outFile = "./out-compare-dnn/all_vox.png"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)
        

    print("done")

if __name__ == "__main__":
    if len(sys.argv) <= 1 :
        sys.exit("this script takes <GOP file 1> <GOP file 2> <GOP file 3>... as arguments. It plots the GOP distributions for each phoneme")
    
    df = pd.DataFrame(columns=('phoneme','score','label'))
    for i in range(1,len(sys.argv)):
        df = readGOPToDF(df, sys.argv[i], i)
        print("read one GOP")

    plot(df, ['LIBRI', 'CMU','TED'])
