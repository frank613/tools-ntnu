import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
import pdb
import numpy as np

def read_json(path):
    with open(path,"r") as injson:
        return json.load(injson)

colors = ['r', 'purple', 'yellow']
def plot(real_dict, art_dict, out_file):
    # Set position of bar on X axis
    to_delete = []
    ##remvoe SIL and SPN
    #to_delete.append("SIL")
    #to_delete.append("SPN")
    #for p in to_delete:
    #    del art_dict["phonemes"][p]
    #assert(len(target_dict["phonemes"]) == len(ref_dict_list[0]["phonemes"]))
    phoneme_list = art_dict["phonemes"].keys()
    target_pos = np.arange(len(phoneme_list))
     
    # Make the plot
    #fig, ax = plt.subplots()
    data_all = np.array([[ p, art_dict["phonemes"][p][2], art_dict["phonemes"][p][0], real_dict["phonemes"][p][0], real_dict["phonemes"][p][1]] for p in phoneme_list])
    #sorted_data = np.array(sorted(data_all, key = lambda x:x[1]))
    #pdb.set_trace()
    plt.scatter(data_all[:,1].astype("float"), data_all[:,3].astype("float"))
    #ax.scatter(target_pos, sorted_data[:,1], label="artificial")
    #ax.scatter(target_pos, sorted_data[:,3], label="real")

    pdb.set_trace()

    for i in range(len(data_all[:,0])):
        plt.annotate(data_all[i][2], (float(data_all[i][1]), float(data_all[i][3])))
        #ax.annotate(sorted_data[i][2], (target_pos[i], sorted_data[i][1]))
        #ax.annotate(sorted_data[i][4], (target_pos[i], sorted_data[i][3]))
     
    plt.xlabel('AUC (artificial)', fontweight ='bold', fontsize = 12)
    plt.ylabel('AUC (real)', fontweight ='bold', fontsize = 12)
    #plt.xticks(target_pos, sorted_data[:,0], rotation='vertical')
    #ax.tick_params(axis='both', which='major', labelsize=12)
    #plt.gca().get_yaxis().set_visible(False)
    plt.xlim([0.5,1])
    plt.ylim([0.5,1])
    #ax.set_ylim(left=0.5, right=1)
    #plt.legend(fontsize=12)
    #plt.tight_layout()
    #plt.title('AUC compare over different phoenmes')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    print()


if __name__ == "__main__":
    if len(sys.argv) <= 3 :
        sys.exit("this script takes 4 arguments <in-json-real> <in-json-artificial> <output>. It plots the scatter plot for the input jsons over each phoneme")


    real_dict = read_json(sys.argv[1])
    art_dict = read_json(sys.argv[2])

    plot(real_dict, art_dict, sys.argv[3])



