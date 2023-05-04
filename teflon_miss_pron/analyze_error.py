import sys
import pdb
import pandas as pd
import re
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from statistics import mean

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

def read_anno(anno_file):
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

def get_acc_cheat(df):
    percentile_dict={}
    average_arr = np.zeros((len(df['phoneme'].unique()),3))
    for ph in df['phoneme'].unique():
        lb_arr = df.loc[df['phoneme'] == ph, "label"].to_numpy()
        percentile = lb_arr.sum() / len(lb_arr) * 100
        score_arr = df.loc[df['phoneme'] == ph, "score"].to_numpy()
        percentile_dict[ph] = np.percentile(score_arr, percentile)

    df['prediction'] = df.apply(lambda x: True if x['score'] <= percentile_dict[x['phoneme']] else False, axis=1)
    for i,ph in enumerate(df['phoneme'].unique()):
        acc = metrics.accuracy_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        precision = metrics.precision_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        recall = metrics.recall_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        #print("the accuracy of phoneme {} is {}".format(ph, acc))
        average_arr[i,:] = np.around([acc, precision, recall],3)
    average_vec = np.around(average_arr.mean(axis=0),3)
    print("the overall acc, precision, reall is {0}, {1}, {2}".format(average_vec[0], average_vec[1], average_vec[2]))

def get_acc_rand_guess(df):
    prop_dict={}
    average_arr = np.zeros((len(df['phoneme'].unique()),3))
    for ph in df['phoneme'].unique():
        lb_arr = df.loc[df['phoneme'] == ph, "label"].to_numpy()
        prop = lb_arr.sum() / len(lb_arr) * 100
        prop_dict[ph] = prop

    df['prediction'] = df.apply(lambda x: True if np.random.randint(1,101) <= prop_dict[x['phoneme']] else False, axis=1)
    for i,ph in enumerate(df['phoneme'].unique()):
        acc = metrics.accuracy_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        precision = metrics.precision_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        recall = metrics.recall_score(df.loc[df['phoneme'] == ph, "label"], df.loc[df['phoneme'] == ph, "prediction"])
        #print("the accuracy of phoneme {} is {}".format(ph, acc))
        average_arr[i,:] = np.around([acc, precision, recall],3)
    average_vec = np.around(average_arr.mean(axis=0),3)
    print("random guess: the overall acc, precision, reall is {0}, {1}, {2}".format(average_vec[0], average_vec[1], average_vec[2]))

def get_acc_noskill(df):
    average_arr = np.zeros((len(df['phoneme'].unique()),3))
    for i,ph in enumerate(df['phoneme'].unique()):
        lb_arr = df.loc[df['phoneme'] == ph, "label"].to_numpy()
        preidct_ns = np.array([False]*len(lb_arr))
        acc = metrics.accuracy_score(lb_arr, preidct_ns)
        precision = metrics.precision_score(lb_arr, preidct_ns)
        recall = metrics.recall_score(lb_arr, preidct_ns)
        average_arr[i,:] = np.around([acc, precision, recall],3)
    average_vec = np.around(average_arr.mean(axis=0),3)
    print("the noskill overall acc, precision, reall is {0}, {1}, {2}".format(average_vec[0], average_vec[1], average_vec[2]))


N_SPLIT=5
def get_acc_cv_stratify(df):
    kfold = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    for ph in df['phoneme'].unique():
        X = df.loc[df['phoneme'] == ph, "score"].to_numpy()
        Y = df.loc[df['phoneme'] == ph, "label"].to_numpy()
        acc_arr = np.zeros(N_SPLIT)
        n = 0
        for train_ix, test_ix in kfold.split(X, Y):
            train_X, test_X = X[train_ix], X[test_ix]
            train_Y, test_Y = Y[train_ix], Y[test_ix]
            percentile = train_Y.sum() / len(train_Y) * 100
            threshold = np.percentile(train_X, percentile)
            predict_Y = [ True if score <= threshold else False for score in test_X]
            acc = metrics.accuracy_score(test_Y, predict_Y)
            acc_arr[n] = acc
            n += 1
        print("the stratified-KFold-CV accuracy of phoneme {} is {}".format(ph,acc_arr.mean()))


def get_acc_cv(df):
    kfold = KFold(n_splits=N_SPLIT, shuffle=True, random_state=1)
    average_arr = np.zeros((len(df['phoneme'].unique()),3))
    for i,ph in enumerate(df['phoneme'].unique()):
        X = df.loc[df['phoneme'] == ph, "score"].to_numpy()
        Y = df.loc[df['phoneme'] == ph, "label"].to_numpy()
        acc_arr = np.zeros((N_SPLIT,3))
        n = 0
        for train_ix, test_ix in kfold.split(X, Y):
            train_X, test_X = X[train_ix], X[test_ix]
            train_Y, test_Y = Y[train_ix], Y[test_ix]
            percentile = train_Y.sum() / len(train_Y) * 100
            threshold = np.percentile(train_X, percentile)
            predict_Y = [ True if score <= threshold else False for score in test_X]
            acc = metrics.accuracy_score(test_Y, predict_Y)
            pr = metrics.precision_score(test_Y, predict_Y)
            rec = metrics.recall_score(test_Y, predict_Y)
            acc_arr[n,:] = [acc,pr,rec]
            n += 1
        #print("the Kfold-CV accuracy of phoneme {} is {}".format(ph,acc_arr.mean()))
        average_arr[i, :] = acc_arr.mean(axis=0)
    average_vec = np.around(average_arr.mean(axis=0), 3)
    print("the overall CV accuracy, precision and recall is {0}, {1}, {2}".format(average_vec[0], average_vec[1], average_vec[2]))




    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <GOP file> <phoneme anno file>  <outJson>. It labels the phonemes in the GOP file based on the phoneme anno file and outputs a summary in json format")

    p_table = read_anno(sys.argv[2])
    data_list = readGOP(sys.argv[1], p_table)

    gop_table=[]
    for item in data_list:
        for itm in item[1]:
            gop_table.append(list(itm) + [item[0]])
    df = pd.DataFrame(gop_table, columns=('phoneme','score','label','uttid'))
    pdb.set_trace()
    #get_acc_cheat(df)
    get_acc_cv(df)
    #get_acc_noskill(df)
    #get_acc_rand_guess(df)

    #don't use the stratified because it's cheating again for our classfifier!
    #get_acc_cv_stratify(df)
