import sys
import pdb
import pandas as pd
import re
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?') # can be different for different models
opt_SIL = 'sil' ##can be different for different models
def readGOP(gop_file, p_table):
    in_file = open(gop_file, 'r')
    isNewUtt = True
    skip = False
    seq_score = []
    df_temp = []
    label_phoneme = []
    for line in in_file:
        line = line.strip()
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 1:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            skip = False
            if uttid not in p_table:
                skip = True
            else:
                label_phoneme = p_table[uttid][1]
            isNewUtt = False
            continue
        if line == '':
            if not skip:
                ## length in the gop file must the same as 2(sil) + len(anno)
                #assert( len(label_phoneme)+2 == len(seq_score))
                if len(label_phoneme) != len(seq_score):
                    pdb.set_trace()
                    sys.exit()
                df_temp.append((uttid, [(p,g,l) for (p,g),l in zip(seq_score[1:-1], label_phoneme)]))
            seq_score = []
            label_phoneme = []
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
        ####optional silence 
        if cur_phoneme != opt_SIL:
            seq_score.append((cur_phoneme, cur_score))
    return df_temp

def read_anno_p(anno_file):
    p_table = {}
    with open(anno_file) as ifile:
        for line in ifile:
            fields = line.strip().split()
            if len(fields) != 4:
                sys.exit("illegal line found in the anno file")
            if fields[0] not in p_table.keys():
                p_table[fields[0]] = (fields[1].split(','), [False if n_str == '1' else True for n_str in fields[2].split(',')])
            else: ## do the & on each value for disagreed anno
                p_table[fields[0]] = (p_table[fields[0]][0], [ x & bool(y) for x,y in zip(p_table[fields[0]][1], fields[2].split(','))])
    return p_table

def normalize_minmax(df):
    scaler = MinMaxScaler((-1,1))
    for ph in df['phoneme'].unique():
        scaled = scaler.fit_transform(df.loc[df['phoneme'] == ph, "score"].to_numpy().reshape(-1,1))
        df.loc[df['phoneme'] == ph, "score"] = scaled[:,0]

percentile=0.15
def normalize_minmax_remove_outliers(df):
    scaler = MinMaxScaler((-1,1))
    for ph in df['phoneme'].unique():
        temp_arr = df.loc[df['phoneme'] == ph, "score"].to_numpy()
        cut_off = np.percentile(temp_arr, percentile)
        df.loc[(df['phoneme'] == ph ) & (df['score'] < cut_off) , "score"] = cut_off 
        scaled = scaler.fit_transform(df.loc[df['phoneme'] == ph, "score"].to_numpy().reshape(-1,1))
        df.loc[df['phoneme'] == ph, "score"] = scaled[:,0]

def standarize_minmax_remove_outliers(df):
    scaler = StandardScaler()
    for ph in df['phoneme'].unique():
        temp_arr = df.loc[df['phoneme'] == ph, "score"].to_numpy()
        cut_off = np.percentile(temp_arr, percentile)
        df.loc[(df['phoneme'] == ph ) & (df['score'] < cut_off) , "score"] = cut_off
        scaled = scaler.fit_transform(df.loc[df['phoneme'] == ph, "score"].to_numpy().reshape(-1,1))
        df.loc[df['phoneme'] == ph, "score"] = scaled[:,0]

train_ratio=0.9
def normalize_logistic(df):
    for ph in df['phoneme'].unique():
        temp_arr = df.loc[df['phoneme'] == ph, ["score","label"]].to_numpy()
        X_train, _, y_train, _ = train_test_split(temp_arr[:, [0]], temp_arr[:,1].astype(bool), train_size=train_ratio, random_state=1, stratify=temp_arr[:,1].astype(bool))
        clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train, y_train)
        scores = clf.predict_proba(temp_arr[:, [0]])
        df.loc[df['phoneme'] == ph, "score"] = scores[:,0]


def read_anno(score_file):
    score_table = {}
    with open(score_file) as ifile:
        for line in ifile:
            fields = line.strip().split()
            if len(fields) != 3:
                pdb.set_trace()
                sys.exit("illegal line found in the anno file")
            if fields[0] not in score_table.keys():
                score_table[fields[0]] = int(fields[1])
            else: ## take the minimum for disagreed anno
                score_table[fields[0]] = min(score_table[fields[0]], int(fields[1]))
    return score_table

def get_data_list(df, score_table):
    df_temp = []
    for uttid in score_table.keys():
        if uttid not in df["uttid"].unique():
            continue
        avg_gop = df.loc[df["uttid"] == uttid, "score"].to_numpy().mean()
        df_temp.append((uttid, avg_gop, score_table[uttid]))
    return df_temp

N_SPLIT=10
def get_corr_linear_cv(df):
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    X = df["gop-avg"].to_numpy()
    Y = df['label'].to_numpy()
    corr_arr = np.zeros(N_SPLIT)
    n = 0
    for train_ix, test_ix in kfold.split(X, Y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_Y, test_Y = Y[train_ix], Y[test_ix]
        #pdb.set_trace()
        reg = LinearRegression().fit(train_X.reshape(-1,1), train_Y)
        #predict_Y = reg.predict(test_X.reshape(-1,1))
        predict_Y = np.around(reg.predict(test_X.reshape(-1,1)))
        corr = round(np.corrcoef(test_Y, predict_Y)[0,1],3)
        corr_arr[n] = corr
        print("the corr for fold {0} is {1}".format(n,corr))
        n += 1

    print("the stratified-KFold-CV correaltion is {}".format(corr_arr.mean()))

    #plt.scatter(X,Y)
    data = [ df.loc[df['label'] == i, "gop-avg" ].to_numpy() for i in [1,2,3,4,5]]
    plt.boxplot(data, 0, 'rs', 0, labels = ["1","2","3","4","5"])
    plt.savefig("./data_nomalized.png")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP file> <phoneme anno file> <score file> <outJson>. It labels the phonemes in the GOP file based on the phoneme anno file and outputs a summary in json format")

    p_table = read_anno_p(sys.argv[2])

    data_list = readGOP(sys.argv[1], p_table)

    gop_table=[]
    for item in data_list:
        for itm in item[1]:
            gop_table.append(list(itm) + [item[0]])
    df = pd.DataFrame(gop_table, columns=('phoneme','score','label','uttid'))
    #normalize_minmax(df)
    #normalize_minmax_remove_outliers(df)
    #standarize_minmax_remove_outliers(df)
    normalize_logistic(df)

    score_table = read_anno(sys.argv[3])
    data_list = get_data_list(df, score_table)
    df2 = pd.DataFrame(data_list, columns=('uttid','gop-avg','label'))
    get_corr_linear_cv(df2)

