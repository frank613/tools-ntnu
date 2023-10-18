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
sys.path.append('/home/stipendiater/xinweic/tools/edit-distance')
import edit
import pandas as pd



datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
#vowel_set = set(['AA', 'AH', 'AE', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
#cons_set = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'W', 'V', 'W', 'Y', 'Z', 'ZH'])
#p_set = vowel_set | cons_set
sil_tokens = set(["SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))


#RE for CMU files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#return both segmented phoneme seq and pid seq
def seg_seq(p_seq, p_tokenizer):
    pid_seq = p_tokenizer.convert_tokens_to_ids(p_seq)
    segmented_p = [] #list of pair (phoneme, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    segmented_pid = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    temp,temp_p,idx_start = '', '', 0
    for i,(pid,p) in enumerate(zip(pid_seq, p_seq)):
        if temp != pid:
            segmented_pid.append([temp, idx_start, i])
            segmented_p.append([temp_p, idx_start, i])
            temp = pid
            temp_p = p
            idx_start = i
    segmented_pid.append([temp, idx_start, i+1])
    segmented_p.append([temp_p, idx_start, i+1])
    segmented_pid = segmented_pid[1:]
    segmented_p = segmented_p[1:]
    return (segmented_pid,segmented_p)

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == "C" else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 
    

def write(df, sub_list, tkn, outFile):

    #p:(closest_phoneme, mean_diff, auc_value, entropy, count_of_real, count_of_error)
    out_form = { \
                'phonemes':{},  
                'summary': {"average-mean-diff": None, "average-AUC": None, "total_real": None, "total_error":None}} 
    #count of phonemes}
    
    ##substitution stats
    p_replace_set = df['phonemes'].unique()
    pair_dict = {phoneme_out: { phoneme_in: 0 for phoneme_in in p_replace_set} for phoneme_out in p_replace_set}
    for pair in sub_list:
        l,r = pair.split(' -> ')
        lid = tkn._convert_token_to_id(l)
        rid = tkn._convert_token_to_id(r)
        if lid not in p_replace_set or rid not in p_replace_set:
            continue
        pair_dict[lid][rid] += 1

    total_correct = 0
    total_error = 0
    total_auc = 0
    total_mean_diff = 0
    total_entry = 0
    for pid in df['phonemes'].unique():
        data_false = df.loc[(df["phonemes"] == pid) & (df["labels"] == 'C'), ['scores', 'labels']].to_numpy()
        total_correct += len(data_false)
        data_true = df.loc[(df["phonemes"] == pid) & (df["labels"] != 'C'), ['scores','labels']].to_numpy()
        if len(data_true) == 0:
            #no error
            out_form["phonemes"][tkn._convert_id_to_token(pid)] ="noDef"
            continue
        total_entry += 1
        total_error += len(data_true)
        auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
        total_auc += auc_value
        sorted_items = sorted(pair_dict[pid].items(), key=lambda kv: kv[1], reverse=True)
        freq_sub =  sorted_items[0][0]
        #mean diff between correct and all the incorrect realisations
        mean_diff = round(-data_false[:, 0].mean() + data_true[:, 0].mean(), 3)
        total_mean_diff += mean_diff
        out_form["phonemes"][tkn._convert_id_to_token(pid)] = (tkn._convert_id_to_token(freq_sub), mean_diff, auc_value, len(data_false), len(data_true)) 
    out_form["summary"]["average-mean-diff"]=total_mean_diff/total_entry
    out_form["summary"]["average-AUC"]=total_auc/total_entry
    out_form["summary"]["total_real"]=total_correct
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

def collect_stats_error(post_mat, p_seg, tran_seq, pid_seg, pid_set, results_list, sub_list):
    #compare and get the lable
    p_sequence = [ item[0] for item in p_seg ]
    dist, labels = edit.edit_dist(p_sequence, tran_seq)
    sub_list += edit.get_sub_pair_list(p_sequence, tran_seq, labels)
    labels_resized = [ label for idx, label in enumerate(labels) if label != 'I']
    if len(labels_resized) != len(p_sequence):
            sys.exit("length of edit distance not maching the p_sequence")
    if len(pid_seg) != len(p_sequence):
            sys.exit("length of phoneme sequence does not match between alignment and pid_sequence")
    for order, (pid,start_idx, end_idx) in enumerate(pid_seg):
        if pid in pid_set:
            post_avg = post_mat[start_idx:end_idx][:,pid].mean().item()
            results_list.append((pid, post_avg, labels_resized[order]))  

def collect_stats_correct(post_mat, pid_seq, pid_set, results_list):
    for order, (pid,start_idx, end_idx) in enumerate(pid_seq):
        #length_seg = end_idx - start_idx
        if pid in pid_set:
            post_avg = post_mat[start_idx:end_idx][:,pid].mean().item()
            results_list.append((pid, post_avg, "C"))   


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

def readTRANToDF(tran_file):
    in_file=open(tran_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq'))
    for line in in_file:
        uttid,seq = line.split()
        df.loc[len(df.index)] = [uttid, seq.split(';')]
    return df
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <transcribed phoneme-seq> <error-uttid-list> <out-file>.\n \
        , it analyzes the AUC using real errors derived from the transcription file. The error-uttid-list is annotated, we use it here to avoid treating multiple valid phonemes as errors when computing the edit-distance. The csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    ##read transcription
    tran_df = readTRANToDF(sys.argv[4])
    tran_list = tran_df['uttid'].unique()
    ##read the error list
    error_list = []
    with open(sys.argv[5]) as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split()
            if len(fields) != 1:
                sys.exit("wrong input line")
            error_list.append(fields[0])
    # load the pretrained model
    model_path = sys.argv[2]
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-en/processor")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-en/processor")
    model = Wav2Vec2ForPhoneCE.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    csv_path = sys.argv[3]
    ds = load_dataset_local_from_dict(csv_path)
    #cuda = torch.device('cuda:1')
    
    #count = 0
    with torch.no_grad():
        p_set = set(p_tokenizer.get_vocab().keys())
        p_set = p_set - sil_tokens - spec_tokens
        pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        #list of tuple('phonemes','scores','labels')
        labled_phonemes = []  
        sub_list = []
        for row in ds:
            #count += 1
            #if count > 100:
            #    break
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no alignment can be found")
                continue
            if row['id'] not in tran_list:
                print("ignore uttid: " + row['id'] + ", no transcription can be found")
                continue
            print("processing {0}".format(row['id']))
            #step 1, segmentation (pid_seq = list of (pid, start_idx, end_idx)
            p_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            pid_seg, p_seg = seg_seq(p_seq, p_tokenizer)
            #step 2 get the posterior matrix:
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values)["logits"].squeeze(0)
            post_mat = logits.softmax(dim=-1)
            #step 3 compute and analyze the GOP
            #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
            
            if row['id'] not in error_list:
                collect_stats_correct(post_mat, pid_seg, pid_set, labled_phonemes)
            else:
                tran_seq = (tran_df.loc[tran_df["uttid"] == row["id"], "seq"]).tolist()[0]
                collect_stats_error(post_mat, p_seg, tran_seq, pid_seg, pid_set, labled_phonemes, sub_list)      

    df = pd.DataFrame(labled_phonemes, columns=['phonemes','scores','labels'])
    print("done with GOP computation")
    write(df,sub_list, p_tokenizer,sys.argv[6])






    
  

