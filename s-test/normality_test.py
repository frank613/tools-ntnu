import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import re
import os
from scipy.stats import shapiro
from scipy.stats import normaltest


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
    stat, p = shapiro(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])
    print('Shapiro statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def skew_kurtosis_test(gop_df, symbol):
    stat, p = normaltest(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])
    print('D’Agostino’s K2 statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <gop-file> <symbol>. \n \
        it reads the gop file and plot the qqplot ")

    gop_df = read_gop(sys.argv[1])
    shapiro_test(gop_df, sys.argv[2])
    skew_kurtosis_test(gop_df, sys.argv[2])
