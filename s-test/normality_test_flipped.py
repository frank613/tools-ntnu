import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import re
import os
from scipy.stats import shapiro
from scipy.stats import normaltest
import numpy as np


re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)?')
def read_gop(gopFile):
    in_file = open(gopFile, 'r')
    #df = pd.DataFrame(columns=('uttid', 'GOP-score', 'symbol'))
    isNewUtt = True
    gopList = []
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
        #consider all _I/B/E/S
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            sys.exit("non legal phoneme found in the gop file")
        cur_score = float(fields[2])
        gopList.append((cur_phoneme, cur_score, uttid))
    return pd.DataFrame(gopList, columns=('symbol', 'GOP-score', 'uttid'))

def shapiro_test(gop_df, symbol):
    length = len(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])
    mask = np.array(get_mask(length))
    flipped = sorted(np.array(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])) * mask
    stat, p = shapiro(flipped)
    print('Shapiro statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def skew_kurtosis_test(gop_df, symbol):
    length = len(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])
    mask = np.array(get_mask(length))
    flipped = sorted(np.array(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])) * mask
    stat, p = normaltest(flipped)
    print('D’Agostino’s K2 statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def get_mask(length):
    value = [ 1 if ((2 & i) >> 1) == 0 else -1 for i in range(1, 1+length)]
    return value
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <gop-file> <symbol>. \n \
        it reads the gop file and plot the qqplot ")

    gop_df = read_gop(sys.argv[1])
    shapiro_test(gop_df, sys.argv[2])
    skew_kurtosis_test(gop_df, sys.argv[2])
