import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForPreTraining, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from external.new_modules import Wav2Vec2ForPhoneCE
import torch
from pathlib import Path
import pdb



datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
vowel_set = set(['AA', 'AH', 'AE', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
cons_set = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'W', 'V', 'W', 'Y', 'Z', 'ZH'])
p_set = vowel_set | cons_set

xstr = lambda s: s or ""

#RE for CMU files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

def seg_to_token_seq(pid_seq):
    segmented = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    temp,idx_start = '', 0
    for i,pid in enumerate(pid_seq):
        if temp != pid:
            segmented.append([temp, idx_start, i])
            temp = pid
            idx_start = i
    segmented.append([temp, idx_start, i+1])
    segmented = segmented[1:]
    return segmented

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 
    

def write(gops_map, tkn, outFile):

    #p:(closest_phoneme, mean_diff, auc_value, entropy, count_of_real, count_of_error)
    out_form = { \
                'phonemes':{},  
                'summary': {"average-mean-diff": None, "average-AUC": None, "total_real": None, "total_error":None}} 
    #count of phonemes}
    
    total_real = 0
    total_error = 0
    total_auc = 0
    total_mean_diff = 0
    total_entry = 0
    for (k,v) in gops_map.items():
        real_arr = np.array(v[k])
        if len(real_arr) == 0:
            continue
        total_entry += 1
        real_label = np.stack((real_arr, np.full(len(real_arr), 0)), 1)
        scores = []
        total_real += len(v[k]) 
        for p in set(gops_map.keys()) - set([k]):
            sub_arr = np.array(gops_map[p][k]) #for all the p phonemes that are substituted to k
            if len(sub_arr) == 0:
                continue
            sub_label = np.stack((sub_arr, np.full(len(sub_arr), 1)), 1)
            auc_value = auc_cal(np.concatenate((real_label, sub_label)))
            if auc_value != "NoDef":
                auc_value = round(auc_value, 3)
            scores.append((p, sub_arr.mean(), len(sub_arr), auc_value))
            total_error += len(sub_arr)
        
        confused_pid, p_mean, num_error, auc = sorted(scores, key = lambda x: x[3])[0]
        mean_diff = round(real_arr.mean() - p_mean, 3)
        out_form["phonemes"][tkn._convert_id_to_token(k)] = (tkn._convert_id_to_token(confused_pid), mean_diff, auc, len(real_arr), num_error) 
        total_auc += auc
        total_mean_diff += mean_diff
    out_form["summary"]["average-mean-diff"]=total_mean_diff/total_entry
    out_form["summary"]["average-AUC"]=total_auc/total_entry
    out_form["summary"]["total_real"]=total_real
    out_form["summary"]["total_error"]=total_error
    
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, "w") as f:
        json.dump(str(out_form), f)

def read_align(align_path):
    utt_list =[]
    with open(align_path, "r") as ifile:
        for line in ifile:
            line = line.strip()
            uttid, phonemes = line.split(' ')[0], line.split(' ')[1:]
            uttid = uttid.strip("lbi-")
            p_list = list(map(lambda x: re_phone.match(x).group(1), phonemes))
            utt_list.append((uttid, p_list))
    return pd.DataFrame(utt_list, columns=('uttid','phonemes')) 

def collect_stats_avg_posterior(post_mat, pid_seq, pid_set, gops_map):
    for order, (pid,start_idx, end_idx) in enumerate(pid_seq):
        length_seg = end_idx - start_idx
        if pid in pid_set:
            for pid_inner in pid_set:
                post_avg = post_mat[start_idx:end_idx][:,pid_inner].mean()
                gops_map[pid][pid_inner].append(post_avg)   

#can't be used for cmu, .sph file can't be detected? 
def load_dataset_local_audiofolder(folder_path):
    dataset = datasets.load_dataset("audiofolder", data_dir=folder_path, split="train") 
    pdb.set_trace()
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(ali_df.loc[ali_df.uttid == uid, "phonemes"].to_list()[0])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered

def load_dataset_local_from_dict(folder_path):
    datadict = {"audio": []}  
    with open(folder_path + '/metadata.csv') as csvfile:
        next(csvfile)
        for row in csvfile:
            datadict["audio"].append(folder_path + '/' + row.split(',')[0])
    ds = datasets.Dataset.from_dict(datadict) 
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        batch["id"] = [re_uttid.match(item["path"])[2] for item in batch["audio"]]
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(ali_df.loc[ali_df.uttid == uid, "phonemes"].to_list()[0])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered

    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <out-file>.\n \
        , it analyzes the AUC using replacement error, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    # load prior and the pretrained model
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    #model_path = ""
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-en/processor")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-en/processor")
    model = Wav2Vec2ForPhoneCE.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path)
    pdb.set_trace()
    #cuda = torch.device('cuda:1')
    
    #count = 0
    with torch.no_grad():
        pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_map = { p1:{ p2: [] for p2 in pid_set } for p1 in pid_set }  # map(p:map(p:average)
        for row in ds:
            #count += 1
            #if count > 10:
            #    break
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no alignment can be found")
                continue
            print("processing {0}".format(row['id']))
            #step 1, segmentation (pid_seq = list of (pid, start_idx, end_idx)
            p_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            pid_seq = seg_to_token_seq(p_tokenizer.convert_tokens_to_ids(p_seq))
            #step 2 get the posterior matrix:
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values)["logits"].squeeze(0)
            post_mat = logits.softmax(dim=-1)
            #step 3 compute and analyze the GOP
            #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
            collect_stats_avg_posterior(post_mat, pid_seq, pid_set, gops_map)      

    print("done with GOP computation")
    write(gops_map, p_tokenizer, sys.argv[4])






    
  
