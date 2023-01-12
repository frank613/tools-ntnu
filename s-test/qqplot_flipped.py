import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import re
from statsmodels.graphics.gofplots import qqplot
import os
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

def get_mask(length):
    value = [ 1 if ((2 & i) >> 1) == 0 else -1 for i in range(1, 1+length)]
    return value

def qq_plot(gopFile, symbol, outFile):
    gop_df = read_gop(gopFile)
    length = len(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])
    mask = np.array(get_mask(length))
    flipped = sorted(np.array(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'])) * mask
    qqplot(flipped, line='s')
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <gop-file> <symbol> <out-file>.\n \
        it reads the gop file and plot the qqplot ")


    qq_plot(sys.argv[1], sys.argv[2], sys.argv[3])
