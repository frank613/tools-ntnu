import sys
import pdb
import pandas as pd
import re
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from statistics import mean
import matplotlib.pyplot as plt

def read_anno(score_file, p_table):
    score_table = {}
    with open(score_file) as ifile:
        for line in ifile:
            fields = line.strip().split()
            if len(fields) != 3:
                pdb.set_trace()
                sys.exit("illegal line found in the anno file")
            if fields[0] not in p_table.keys():
                print("warning: {} does not exsist in the phoneme anno".format(fields[0]))
                continue
            if fields[0] not in score_table.keys():
                score_table[fields[0]] = (int(fields[1]), p_table[fields[0]])
            else: ## take the minimum for disagreed anno
                score_table[fields[0]] = (min(score_table[fields[0]][0], int(fields[1])), score_table[fields[0]][1])
    return score_table


N_SPLIT=10
##wrong phone = -1, correct phone = 1
def phone_error_score(df):
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    X = df["p-score"].to_numpy()
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
    data = [ df.loc[df['label'] == i, "p-score" ].to_numpy() for i in [1,2,3,4,5]]
    plt.boxplot(data, 0, 'rs', 0, labels = ["1","2","3","4","5"])
    plt.savefig("./data_cheating.png")
    



    
def read_anno_p(anno_file):
    p_table = {}
    with open(anno_file) as ifile:
        for line in ifile:
            fields = line.strip().split()
            if len(fields) != 4:
                sys.exit("illegal line found in the anno file")
            if fields[0] not in p_table.keys():
                p_table[fields[0]] = round(mean([1 if n_str == '1' else -1 for n_str in fields[2].split(',')]),3)
            else: ## do the & on each value for disagreed anno
                new_mean = round(mean([1 if n_str == '1' else -1 for n_str in fields[2].split(',')]),3)
                p_table[fields[0]] = round(mean([p_table[fields[0]], new_mean]), 3)
    return p_table


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <phone-anno-file> <score-file> <outJson>. It labels the phonemes in the GOP file based on the phoneme anno file and outputs a summary in json format")

    p_table = read_anno_p(sys.argv[1])
    score_table = read_anno(sys.argv[2], p_table)

    df = pd.DataFrame([(k,l,ps) for k,(l,ps) in score_table.items()], columns=('uttid','label','p-score'))
    pdb.set_trace()
    phone_error_score(df)
