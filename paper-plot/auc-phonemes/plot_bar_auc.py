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
    ##remvoe SIL and SPN
    del target_dict["phonemes"]["SIL"]
    del target_dict["phonemes"]["SPN"]
    assert(len(target_dict["phonemes"]) == len(ref_dict_list[0]["phonemes"]))
    targets = list(target_dict["phonemes"].items())
    pdb.set_trace()
    targets = sorted(targets, key=lambda x:x[1][0])
    phoneme_list = [ k for k,v in targets ]

    #sort to AUC, increasing
    target_pos = np.arange(len(phoneme_list))*1.5
    ref_pos_list = [target_pos+(i+1)*barWidth for i in range(len(ref_dict_list))]
     
    # Make the plot
    fig, ax = plt.subplots(figsize=(16, 4))
    auc_target = [ target_dict["phonemes"][p][0] for p in phoneme_list] 
    ax.bar(target_pos, auc_target, width = barWidth, color = colors[0],
            edgecolor ='grey', label = target_dname)
    for i, r_dict in enumerate(ref_dict_list): 
        auc_ref = [ r_dict["phonemes"][p][2] for p in phoneme_list]
        ax.bar(ref_pos_list[i], auc_ref, width = barWidth, color = colors[i+1 % len(colors)], 
                edgecolor ='grey', label = ref_dname[i])
     
    ax.set_ylabel('AUC-value', fontweight ='bold', fontsize = 13)
    ax.set_xlabel('phonemes', fontweight ='bold', fontsize = 13)
    plt.xticks(target_pos + barWidth,list(phoneme_list), rotation='vertical')
    ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.gca().get_yaxis().set_visible(False)
    ax.set_ylim([0.5,1])
    ax.set_xlim(left=-0.3, right=ref_pos_list[-1][-1]+1)
    plt.legend(fontsize=12, loc=4, framealpha=1.0)
    plt.tight_layout()
    #plt.title('AUC compare over different phoenmes')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)


if __name__ == "__main__":
    if len(sys.argv) <= 4 :
        sys.exit("this script takes 4 arguments <in-json> <ref-json1> <ref-json2> <output>. It plots the AUC for these jsons over each phoneme using bar plot")

    target_dname = 'CMU-kids-real'
    ref_dname = ['CMU-kids-artificial','Librispeech-artificial']

    target_dict = read_json(sys.argv[1])
    ref_dict_list = [read_json(sys.argv[2]), read_json(sys.argv[3])]
    assert(len(ref_dict_list) == len(ref_dname))

    plot(target_dict, ref_dict_list, target_dname, ref_dname, sys.argv[4])



