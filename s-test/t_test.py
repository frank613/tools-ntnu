import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import re
import os
from scipy.stats import ttest_ind

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

def t_test(gop_df1, gop_df2,  symbol):
    stat, p = ttest_ind(gop_df1.loc[gop_df1["symbol"]== symbol, 'GOP-score'], gop_df2.loc[gop_df2["symbol"]== symbol, 'GOP-score'])
    print('default T statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    #stat, p = ttest_ind(gop_df1.loc[gop_df1["symbol"]== symbol, 'GOP-score'], gop_df2.loc[gop_df2["symbol"]== symbol, 'GOP-score'], trim=0.05)
    print('Trimed T statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    stat, p = ttest_ind(gop_df1.loc[gop_df1["symbol"]== symbol, 'GOP-score'], gop_df2.loc[gop_df2["symbol"]== symbol, 'GOP-score'], equal_var=False)
    print("Welch's T statistics=%.3f, p=%.3f" % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    stat, p = ttest_ind(gop_df1.loc[gop_df1["symbol"]== symbol, 'GOP-score'], gop_df2.loc[gop_df2["symbol"]== symbol, 'GOP-score'], permutations=min(len(gop_df1.loc[gop_df1["symbol"]== symbol, 'GOP-score']), len(gop_df2.loc[gop_df2["symbol"]== symbol, 'GOP-score']))*0.5)
    print('Permutation statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')








if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <gop-file1> <gop-files2> <symbol>. \n \
        it reads the gop files and does the mean test ")

    gop_df1 = read_gop(sys.argv[1])
    gop_df2 = read_gop(sys.argv[2])
    t_test(gop_df1, gop_df2, sys.argv[3])
