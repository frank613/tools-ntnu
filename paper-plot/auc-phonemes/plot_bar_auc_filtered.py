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
colors = ['r', 'purple', 'yellow']
def plot(target_dict, ref_dict_list, target_dname, ref_dname, out_file):
    # Set position of bar on X axis
    to_delete = []
    for p in target_dict["phonemes"]:
        #total number < 300, error number < 30
        if target_dict["phonemes"][p][4] + target_dict["phonemes"][p][5] < 30 or target_dict["phonemes"][p][6] < 300 :
            to_delete.append(p)
    ##remvoe SIL and SPN
    to_delete.append("SIL")
    to_delete.append("SPN")
    for p in to_delete:
        del target_dict["phonemes"][p]
    #assert(len(target_dict["phonemes"]) == len(ref_dict_list[0]["phonemes"]))
    phoneme_list = target_dict["phonemes"].keys()
    target_pos = np.arange(len(phoneme_list))*1.5
    ref_pos_list = [target_pos+(i+1)*barWidth for i in range(len(ref_dict_list))]
     
    # Make the plot
    fig, ax = plt.subplots(figsize=(20, 5))
    auc_target = [ target_dict["phonemes"][p][0] for p in phoneme_list]
    ax.bar(target_pos, auc_target, width = barWidth, color = colors[0],
            edgecolor ='grey', label = target_dname)
    for i, r_dict in enumerate(ref_dict_list): 
        auc_ref = [ r_dict["phonemes"][p][2] for p in phoneme_list]
        ax.bar(ref_pos_list[i], auc_ref, width = barWidth, color = colors[i+1 % len(colors)], 
                edgecolor ='grey', label = ref_dname[i])
     
    ax.set_ylabel('AUC-value', fontweight ='bold', fontsize = 12)
    ax.set_xlabel('phonemes', fontweight ='bold', fontsize = 12)
    plt.xticks(target_pos + barWidth, phoneme_list, rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.gca().get_yaxis().set_visible(False)
    ax.set_ylim([0.5,1])
    ax.set_xlim(left=-0.3, right=ref_pos_list[-1][-1]+0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    #plt.title('AUC compare over different phoenmes')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)


if __name__ == "__main__":
    if len(sys.argv) <= 4 :
        sys.exit("this script takes 4 arguments <in-json> <ref-json1> <ref-json2> <output>. It plots the AUC for these jsons over each phoneme using bar plot")

    target_dname = 'cmu-kids-real'
    ref_dname = ['cmu-kids-filtered-artificial','libri-artificial']

    target_dict = read_json(sys.argv[1])
    ref_dict_list = [read_json(sys.argv[2]), read_json(sys.argv[3])]
    assert(len(ref_dict_list) == len(ref_dname))

    plot(target_dict, ref_dict_list, target_dname, ref_dname, sys.argv[4])



