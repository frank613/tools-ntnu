import sys
import pdb
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?') # can be different for different models
opt_SIL = 'sil' ##can be different for different models
def readGOP(gop_file, score_table):
    in_file = open(gop_file, 'r')
    isNewUtt = True
    skip = False
    seq_score = []
    df_temp = []
    for line in in_file:
        line = line.strip()
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 1:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            skip = False
            if uttid not in score_table:
                skip = True
            else:
                score = score_table[uttid]
            isNewUtt = False
            continue
        if line == '':
            if not skip:
                df_temp.append((uttid, np.array(seq_score).mean(), score))
            seq_score = []
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
            seq_score.append(cur_score)
    return df_temp

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
    plt.savefig("./data.png")

def get_corr_dt_cv(df):
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    X = df["gop-avg"].to_numpy()
    Y = df['label'].to_numpy()
    corr_arr = np.zeros(N_SPLIT)
    n = 0
    for train_ix, test_ix in kfold.split(X, Y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_Y, test_Y = Y[train_ix], Y[test_ix]
        #pdb.set_trace()
        model = DecisionTreeClassifier(class_weight="balanced")
        ##train_Y is now ordinal, thus regression tree is applied?
        model.fit(train_X.reshape(-1,1), train_Y)
        predict_Y = model.predict(test_X.reshape(-1,1))
        corr = round(np.corrcoef(test_Y, predict_Y)[0,1],3)
        corr_arr[n] = corr
        print("the corr for fold {0} is {1}".format(n,corr))
        evaluate_model(test_Y, predict_Y)
        n += 1
        
    print("the stratified-KFold-CV correaltion is {}".format(corr_arr.mean()))

    #plt.scatter(X,Y)
    #data = [ df.loc[df['label'] == i, "gop-avg" ].to_numpy() for i in [1,2,3,4,5]]
    #plt.boxplot(data, 0, 'rs', 0, labels = ["1","2","3","4","5"])
    #plt.savefig("./data.png")


def get_corr_dt_cat_cv(df):
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    X = df["gop-avg"].to_numpy()
    Y = df['label'].apply(str).to_numpy()
    corr_arr = np.zeros(N_SPLIT)
    n = 0
    for train_ix, test_ix in kfold.split(X, Y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_Y, test_Y = Y[train_ix], Y[test_ix]
        #pdb.set_trace()
        #model = DecisionTreeClassifier(class_weight="balanced")
        model = DecisionTreeClassifier(class_weight="balanced", max_depth=5)
        ##train_Y is now ordinal, thus regression tree is applied?
        model.fit(train_X.reshape(-1,1), train_Y)
        predict_Y = model.predict(test_X.reshape(-1,1)).astype(int)
        corr = round(np.corrcoef(test_Y.astype(int), predict_Y)[0,1],3)
        corr_arr[n] = corr
        print("the corr for fold {0} is {1}".format(n,corr))
        evaluate_model(test_Y.astype(int), predict_Y)
        n += 1
        
    print("the stratified-KFold-CV correaltion is {}".format(corr_arr.mean()))

    #plt.scatter(X,Y)
    #data = [ df.loc[df['label'] == i, "gop-avg" ].to_numpy() for i in [1,2,3,4,5]]
    #plt.boxplot(data, 0, 'rs', 0, labels = ["1","2","3","4","5"])
    #plt.savefig("./data.png")


def evaluate_model(data_y, y_pred):
    print("Accuracy: "+str(metrics.accuracy_score(data_y, y_pred))+", Recall(UAR): "+str(metrics.recall_score(data_y, y_pred, average='macro')))
    c = metrics.confusion_matrix(data_y, y_pred)
    print(c)
    return c
    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <GOP file> <score file>  <outJson>. It labels the phonemes in the GOP file based on the phoneme anno file and outputs a summary in json format")

    score_table = read_anno(sys.argv[2])
    data_list = readGOP(sys.argv[1], score_table)

    df = pd.DataFrame(data_list, columns=('uttid','gop-avg','label'))
    pdb.set_trace()
    #get_corr_linear_cv(df)
    #get_corr_dt_cv(df)
    get_corr_dt_cat_cv(df)
