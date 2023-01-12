import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pdb
import re
from statsmodels.graphics.gofplots import qqplot
import os
from scipy.stats import expon 

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

def trim(in_df, symbol, prob):
        score_serie= in_df[in_df.symbol == symbol].score
        sorted_s = score_serie.sort_values()
        #trimmed = score_serie[(score_serie < score_serie.quantile(1-prob)) & ( score_serie > score_serie.quantile(prob))]
        length = math.floor(len(sorted_s)*prob)
        mask = [ False if i < length or i > len(sorted_s)-1-length else True for i in range(len(sorted_s))]
        trimmed = sorted_s[mask]
        return trimmed

def qq_plot(gopFile, symbol, outFile):
    gop_df = read_gop(gopFile)
    #qqplot(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score'], line='s')
    qqplot(-1*(gop_df.loc[gop_df["symbol"]== symbol, 'GOP-score']), dist=expon ,line='s')
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <gop-file> <symbol> <out-file>.\n \
        it reads the gop file and plot the qqplot ")


    qq_plot(sys.argv[1], sys.argv[2], sys.argv[3])
