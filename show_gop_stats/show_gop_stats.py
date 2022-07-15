import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pandas as pd
import re
import mpld3
import plotly.io as pio
import plotly.express as px
import numpy as np

allowed_phonemes = ["OY", "AA", "SH", "EH"]
re_phone = re.compile(r'([a-zA-Z]+)[0-9]*(_\w)*')
def get_stat(in_file_path, in_phoneme=''):
    in_file = open(in_file_path, 'r')
    df = pd.DataFrame(columns=('phonemes','scores','uttid'))
    isNewUtt = True
    for line in in_file:
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 2:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            isNewUtt = False
        if line == '':
            isNewUtt = True
            continue
        if len(fields) != 4:
            continue
        cur_match = re_phone.match(fields[1])
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            continue
        if cur_phoneme not in allowed_phonemes:
            print("phoneme {0} is not supported".format(cur_phoneme))
            continue
        df.loc[len(df.index)] = [cur_phoneme, float(fields[2]) ,uttid]

    if in_phoneme == '': #return all allowed phones
        return [ df.loc[df["phonemes"] == i,"scores"].to_numpy() for i in allowed_phonemes]
    else:
        return [df.loc[df["phonemes"] == in_phoneme,"scores"].to_numpy()]

def plots(points,phoneme='all'):
    #pd.DataFrame(points).plot(kind='density')
    plt.rcParams["figure.autolayout"] = True
    if len(points) == 1: 
        fig, ax = plt.subplots(1, 1)
        ax.hist(points, density=True, aisttype='stepfilled', alpha=0.8)
        #ax.legend(loc='best', frameon=False)
        ax.set_title('phoneme {0}'.format(phoneme))
        #print("plot is ready")
        #plt.show(block=False)
        #plt.savefig('out.png')
    elif len(points) > 1:
        print(len(points))
        fig, axs = plt.subplots(len(points))
        count = 0
        for ax in axs:
            ax.hist(points[count], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.8)
            ax.set_title('phoneme {0}'.format(allowed_phonemes[count]))
            count += 1
    else:
        sys.exit("points is empty")

    html_str = mpld3.fig_to_html(fig)
    Html_file= open("{0}.html".format(phoneme),"w")
    Html_file.write(html_str)
    Html_file.close()

    #counts, bins = np.histogram(points, bins=range(-100, 0, 10))
    #fig = px.histogram(y=counts, x=bins, labels={'x':'scores', 'y':'P'}, histnorm='probability density') 
    #pio.write_html(fig, file="Forecast_HTML.html")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("this program takes 1 argument for input GOP file and/or a phone lable")

    if len(sys.argv)  == 2:
        points = get_stat(sys.argv[1])
        plots(points)
    else:
        if sys.argv[2] not in allowed_phonemes:
            sys.exit("phoneme label not supported")
        points = get_stat(sys.argv[1], sys.argv[2])
        plots(points, sys.argv[2])
