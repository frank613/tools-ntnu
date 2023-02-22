import sys
import re
import pandas as pd
sys.path.append('/home/stipendiater/xinweic/tools/edit-distance')
from edit import edit_dist
import pdb

re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*')
#for each phone in the GOP file:
    #label "S" if it's a substitution error based on the other input files.
    #label "D" if it's a deletion error based on the other input files.
    #label "C" if it's the same in both files
    
def labelError(GOP_file, error_list, cano_file, tran_file):
    gop_df = readGOPToDF(GOP_file)
    cano_df = readTRANToDF(cano_file)
    tran_df = readTRANToDF(tran_file)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    tran_list = tran_df['uttid'].unique()
    gop_list = gop_df['uttid'].unique()
    extended = []
    #outer loop is the cano df because it is extracted from the gop file and they should have the same number of uttids
    for index, row in cano_df.iterrows(): 
        uttid = row['uttid']
        print("processing {}".format(uttid))
        cano_seq = row['seq']
        if uttid not in tran_list:
            print("warning: uttid {0} can't be found in the transcription".format(uttid))
            continue
        tran_df_filtered = tran_df.loc[tran_df["uttid"] == uttid, "seq"]
        if len(tran_df_filtered) != 1:
            sys.exit("duplicate uttids detected in the transcription file, check the input")
        tran_seq = tran_df_filtered.tolist()[0] #[0] converts 2d list to 1d
        dist, labels = edit_dist(cano_seq, tran_seq)
        #print(cano_seq)
        #print(tran_seq)
        if dist == 0:
            continue
        if uttid not in gop_list:
            print("warning: uttid {0} can't be found in the gop file".format(uttid))
            continue
        if uttid in error_list:
            labels_resized = [ label for idx, label in enumerate(labels) if label is not 'I']
            if len(labels_resized) != len(cano_seq):
                sys.exit("length of edit distance not maching the gop")
        else:
            labels_resized = ['C'] * len(labels)
        #print(len(labels), len(gop_df.loc[gop_df['uttid'] == uttid, 'seq-score'].tolist()[0]))
        #print(labels)
        #pdb.set_trace()
        extended += [ pair + (labels_resized[idx], uttid) for idx, pair in enumerate(gop_df.loc[gop_df['uttid'] == uttid, 'seq-score'].tolist()[0]) ]
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



if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <GOP file> <error-uttid-list> <cano phone-seq-file> <transcribed phoneme-seq file> <outFile>. It labels the phonemes in the GOP file with substitution or deletetion error based on edit distance")

    utt_list = []
    with open(sys.argv[2]) as ifile:
            for line in ifile:
                line = line.strip()
                fields = line.split()
                if len(fields) != 1:
                    sys.exit("wrong input line")
                utt_list.append(fields[0])
    df_labels = labelError(sys.argv[1], utt_list, sys.argv[3], sys.argv[4])
    df_labels.to_csv(sys.argv[5], header=None, index=True, sep=' ', mode='a')
