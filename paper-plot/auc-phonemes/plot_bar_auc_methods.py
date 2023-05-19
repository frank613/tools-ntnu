import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import pdb
import numpy as np

def read_json(path):
    with open(path,"r") as injson:
        return json.load(injson)

barWidth = 0.25
colors = ['r', 'purple', 'yellow', 'orange', 'blue']
def plot(json_dict, out_file):
    # Set position of bar on X axis
    #del target_dict["phonemes"]["SIL"]
    #del target_dict["phonemes"]["SPN"]

    data_names = list(json_dict.keys())
    method_names = list(json_dict[data_names[0]].keys())
    for data_name in  data_names:
        assert(len(json_dict[data_name].keys()) == len(method_names))
    data_pos = np.arange(len(data_names))*(1+barWidth*len(method_names))
    method_pos_list = [data_pos+i*barWidth for i in range(len(method_names))]
     
    # Make the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods): 
        auc_list = [ json_dict[data_name][method]["summary"]["average-AUC"] for data_name in data_names]
        ax.bar(method_pos_list[i], auc_list, width = barWidth, color = colors[i % len(colors)], 
                edgecolor ='grey', label = method_names[i])
     
    ax.set_ylabel('AUC-value', fontweight ='bold', fontsize = 15)
    ax.set_xlabel('data-set', fontweight ='bold', fontsize = 15)
    plt.xticks(data_pos + barWidth, data_names)
    #plt.gca().get_yaxis().set_visible(False)
    ax.set_ylim([0.5,1])
    ax.set_xlim(left=-0.3)
    plt.legend()
    plt.tight_layout()
    plt.title('AUC compare for different methods on aritificial errors over dataset')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)


if __name__ == "__main__":
    if len(sys.argv) <= 1 :
        sys.exit("this script takes <inJson-mtd1-data1>,<inJson-mtd2-data1>,<inJson-mtd3-data1>....<inJson-mtd1-data2>.. <output>. It plots the AUC for these jsons over each phoneme using bar plot")

    methods = ['gmm-mono-align', 'gmm-mono-frame', 'dnn-mono', 'dnn-tri']
    dnames = ['cmu', 'libri']

    assert(len(methods) * len(dnames) == len(sys.argv) - 2)
    
    json_dict = {dname:{} for dname in dnames} 

    for i,dname in enumerate(dnames):
        for j, method in enumerate(methods):
            json_dict[dname][method] = read_json(sys.argv[(i+1)*len(dnames) +j+1])

    plot(json_dict, sys.argv[-1])



