import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from external.new_modules import Wav2Vec2ForPhoneCE
import torch
from pathlib import Path
import pdb
import json



datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
noisy_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "SPN"))
sil_token = "SIL"

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

##segmentation, new phonemes are annotated as "_#", also return raw phoneme seq without SPN and SIL
def seg_to_token_seq(p_seq):
    segmented = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    temp,idx_start = '', 0
    raw_seq = []
    for i,p in enumerate(p_seq):
        if p.endswith('_#'):
            #pid = p_tokenizer._convert_token_to_id(p.strip("_#"))
            pid = p.strip("_#")
            segmented.append([temp, idx_start, i])
            temp = pid
            idx_start = i
            if pid != sil_token and pid not in noisy_tokens:
                raw_seq.append(pid)
    segmented.append([temp, idx_start, i+1])
    segmented = segmented[1:]
    return segmented,raw_seq


def read_align(align_path):
    utt_list =[]
    with open(align_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            uttid, phonemes = line.split(' ')[0], line.split(' ')[1:]
            uttid = uttid.strip("lbi-")
            p_list = []
            prev_tag = ''
            prev_p = ''
            prev_tone = '' 
            for p in phonemes:
                pure_phoneme = re_phone.match(p).group(1)
                tone = re_phone.match(p).group(2)
                tag = re_phone.match(p).group(3)
                if prev_p != pure_phoneme:
                    p_list.append(pure_phoneme + "_#") 
                elif prev_tone != tone:
                    p_list.append(pure_phoneme + "_#") 
                elif prev_tag != tag:
                    p_list.append(pure_phoneme + "_#") 
                #elif tag == '_I' and prev_tone != tone:
                #    p_list.append(pure_phoneme + "_#") 
                else:
                    p_list.append(pure_phoneme) 
                prev_p = pure_phoneme
                prev_tone = tone
                prev_tag = tag
                #p_list = list(map(lambda x: re_phone.match(x).group(1), phonemes))
            utt_list.append((uttid, p_list))
    return pd.DataFrame(utt_list, columns=('uttid','phonemes')) 


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
    #ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)
    ds_filtered = ds_map

    return ds_filtered


def process_logits(post_mat, human_align):
    ##reduce the logits to a matrix of Tx3 , for each t, record the activations of the current phoneme and the two neigbouring phonemes, according to human alignment: list of triple (phoneme, start_idx, end_idx)
    pass

    
def get_ali_pointers(post_mat, p_seq):
    seq_len = len(p_seq)
    numphones = post_mat.shape[0] # Number of labels, including SIL  
    L = seq_len ## added optional sil to each phoneme
    T = post_mat.shape[1]

    # alphas stores best posterior for the current s at t
    alphas= torch.zeros((L,T)).double()
    pointers = torch.zeros((L,T+1))

    # Initialize, note that the first SIL and last SIL is not optional in CE
    alphas[0,0] = post_mat[p_seq[0],0] 
    pointers[0,0] = -1
    for t in range(1,T):
        start = max(0, L-2*(T-t-1)-1)
        for s in range(start,L):
            #SIL
            s_index = p_seq[s]
            if s%2 == 0:
                if s == 0:
                    alphas[s,t] = alphas[s,t-1] * post_mat[s_index, t]
                    pointers[s,t] = s
                else:
                    s0 = alphas[s,t-1] 
                    s1 = alphas[s-1,t-1]
                    winner = max(s0,s1)
                    alphas[s,t] = winner * post_mat[s_index,t]
                    if winner == s0:
                        pointers[s,t] = s
                    else:
                        pointers[s,t] = s-1
            #Non-SIL
            else:
                if s == 1:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    winner = max(s0, s1)
                    alphas[s,t] = winner * post_mat[s_index,t]
                    if winner == s0:
                        pointers[s,t] = s
                    else:
                        pointers[s,t] = s-1
                else:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = alphas[s-2,t-1]
                    winner = max(s0,s1,s2)
                    alphas[s,t] = winner * post_mat[s_index,t]
                    if winner == s0:
                        pointers[s,t] = s
                    elif winner == s1:
                        pointers[s,t] = s-1
                    else:
                        pointers[s,t] = s-2
       
    ##last time-step for backtrace, stored at state 0 always
    pointers[0,T] = L-1 
    return pointers

    
# return the backtrace path for the current pointer table, to find the biggest contribution to denom liklihood, and also the sub path inside the arbitrary state
def get_backtrace_path(pointers):
    T = pointers.shape[1]
    S = pointers.shape[0]
    full_path = []
    full_path_int = []
    next_state = 0 
    sub_seq = [] ## label's id for the current token
    for t in list(range(T-1,-1,-1)):
        next_state = int(pointers[int(next_state), t])
        full_path_int.append(next_state)
    return full_path_int




if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-json>.\n \
        , it generates the logtis and viterbi path for the CE trained model") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForPhoneCE.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path)
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens
    count = 0
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        json_dict = {}  
        for row in ds:
                count += 1
                if count > 0:
                    break
                if row['id'] not in uttid_list:
                    print("ignore uttid: " + row['id'] + ", no alignment can be found")
                    continue
        print("processing {0}".format(row['id']))
        json_dict.update([("uid",row['id'])])
        #step 1, authentic segmentation based on human annotation/ alignments from GMM-mono (pid_seq = list of (pid, start_idx, end_idx)
        ali_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
        segmented, raw_seq = seg_to_token_seq(ali_seq)
        segmented = [(p,p_tokenizer._convert_token_to_id(p),s,e) for p,s,e in segmented]
        json_dict.update([("align-seq", segmented)])
        #step 2 get the posterior matrix:
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values)["logits"].squeeze(0)
        post_mat = logits.softmax(dim=-1)
        ##merge noisy tokens to SIL:
        sil_index = p_tokenizer._convert_token_to_id(sil_token)
        noisy_labels = p_tokenizer.convert_tokens_to_ids(list(noisy_tokens))
        post_mat[:, sil_index] = post_mat[:,sil_index] + torch.sum(post_mat[:,noisy_labels], axis=-1)
        json_dict.update([("post_mat", post_mat.tolist())])
        ##add optional SIL to p_seq
        p_seq = ["SIL"]
        ##simulated insertion
        #raw_seq.pop(2)
        ##simulated deletion
        #raw_seq.insert(3, "P")
        for p in raw_seq:
            p_seq.append(p)
            p_seq.append("SIL")
        pid_seq = p_tokenizer.convert_tokens_to_ids(p_seq)
        ##run viterbi
        pointers = get_ali_pointers(post_mat.transpose(0,1), pid_seq)
        path_int = get_backtrace_path(pointers)
        path_int.reverse()
        path_int = path_int[1:]
        path_str = [ p_seq[i] for i in path_int]
        path_pid = p_tokenizer.convert_tokens_to_ids(path_str)
        json_dict.update([("path_str", path_str)])
        json_dict.update([("path_pid", path_pid)])
        with open(sys.argv[5], 'w') as f:
            json.dump(json_dict,f)
       







    
  
