import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForCTC
from my_w2v2_package.custom_processor import My_Wav2Vec2Processor
import torch
from pathlib import Path
import pdb
import json

ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')

sil_token = "SIL"
pad_token = "<pad>"  ## the blank token is defined to be <pad> in wav2vec2 

noisy_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "SPN"))

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')
re_uttid_raw = re.compile(r'(.*)\.(.*$)')

##segmentation, new phonemes are annotated as "_#", also return raw phoneme seq without SPN and SIL
def seg_to_token_seq(p_seq):
    segmented = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    temp,idx_start = '', 0
    raw_seq = []
    for i,p in enumerate(p_seq):
        if p.endswith('_#'):
            #pid = p_tokenizer._convert_token_to_id(p.strip("_#"))
            pid = p.strip("_#")
            if temp == sil_token or temp in noisy_tokens:
                to_append = pad_token
            else:
                to_append = temp
            segmented.append([to_append, idx_start, i])
            temp = pid
            idx_start = i
            if pid != sil_token and pid not in noisy_tokens:
                raw_seq.append(pid)
    if temp == sil_token or temp in noisy_tokens:
        to_append = pad_token
    else:
        to_append = temp
    segmented.append([to_append, idx_start, i])
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


def load_dataset_local_from_dict(csv_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": []}  
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
                datadict["audio"].append(row.split(',')[0])
        ds = datasets.Dataset.from_dict(datadict) 
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
        ds.save_to_disk(cache_full_path)
    ds = datasets.Dataset.load_from_disk(cache_full_path) 
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        batch["id"] = [re_uttid_raw.match(item["path"])[1] for item in batch["audio"]]
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

    
def get_ali_pointers(post_mat, p_seq, blank=0):
    seq_len = len(p_seq)
    numphones = post_mat.shape[0] # Number of labels, including SIL  
    L = 2*seq_len + 1
    T = post_mat.shape[1]
    P = post_mat.shape[0]


    # alphas stores best posterior for the current s at t
    alphas= torch.zeros((L,T)).type(torch.float64)
    pointers = torch.zeros((L,T+1)).type(torch.float64)

    # Initialize, not that the first SIL and last SIL is not optional in CE
    alphas[0,0] = post_mat[blank,0] 
    alphas[1,0] = post_mat[p_seq[0],0] 
    pointers[0,0] = -1
    pointers[1,0] = -1
    for t in range(1,T):
        start = max(0, L-2*(T-t))
        for s in range(start,L):
            l = int((s-1)/2)
            #blank
            if s%2 == 0:
                if s == 0:
                    alphas[s,t] = alphas[s,t-1] * post_mat[blank, t]
                    pointers[s,t] = s
                else:
                    s0 = alphas[s,t-1] 
                    s1 = alphas[s-1,t-1]
                    winner = max(s0,s1)
                    alphas[s,t] = winner * post_mat[blank,t]
                    if winner == s0:
                        pointers[s,t] = s
                    else:
                        pointers[s,t] = s-1
            #Non-blank
            else:
                if s == 1:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    winner = max(s0, s1)
                    alphas[s,t] = winner * post_mat[p_seq[l], t]
                    if winner == s0:
                        pointers[s,t] = s
                    else:
                        pointers[s,t] = s-1
                else:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = alphas[s-2,t-1]
                    winner = max(s0,s1,s2)
                    alphas[s,t] = winner * post_mat[p_seq[l], t]
                    if winner == s0:
                        pointers[s,t] = s
                    elif winner == s1:
                        pointers[s,t] = s-1
                    else:
                        pointers[s,t] = s-2
    empty_p = alphas[L-1, T-1]
    final_p = alphas[L-2, T-1]
    winner = max(final_p,empty_p)
    if winner == final_p:
        pointers[0,T] = L-2
    else:
        pointers[0,T] = L-1
       
    return pointers

    
# return the backtrace path for the current pointer table, to find the biggest contribution to denom liklihood, and also the sub path inside the arbitrary state
def get_backtrace_path(pointers):
    T = pointers.shape[1]
    S = pointers.shape[0]
    full_path = []
    full_path_int = []
    next_state = 0 ## last timestep has only one state valid
    sub_seq = [] ## label's id for the current token
    for t in list(range(T-1,-1,-1)):
        next_state = int(pointers[int(next_state), t])
        full_path_int.append(next_state)
    return full_path_int




if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-json>.\n \
        , it generates the logtis and viterbi path for the CTC trained model") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
 
    processor = My_Wav2Vec2Processor.from_pretrained(prep_path) 
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens
    count = 0
    #target = 0
    target = "fabm2aa1"
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        json_dict = {}  
        for row in ds:
            #if count != target:
            #    count += 1
            #    continue
            if row['id'] != target:
                continue
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no alignment can be found")
                continue
            print("processing {0}".format(row['id']))
            json_dict.update([("uid",row['id'])])
            #step 1, authentic segmentation based on human annotation/ alignments from GMM-mono (pid_seq = list of (pid, start_idx, end_idx)
            ali_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            segmented, raw_seq = seg_to_token_seq(ali_seq)
            segmented = [(p, processor.tokenizer._convert_token_to_id(p),s,e) for p,s,e in segmented]
            json_dict.update([("align-seq", segmented)])
            #step 2 get the posterior matrix:
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values)["logits"].squeeze(0)
            post_mat = logits.softmax(dim=-1).type(torch.float64)
            json_dict.update([("post_mat", post_mat.tolist())])
            ##simulated insertion
            #raw_seq.pop(2)
            ##simulated deletion
            #raw_seq.insert(3, "P")
            ##add SIL
            raw_seq = [sil_token] + raw_seq + [sil_token] 
            pid_seq = processor.tokenizer.convert_tokens_to_ids(raw_seq)
            ##run viterbi
            pointers = get_ali_pointers(post_mat.transpose(0,1), pid_seq)
            path_int = get_backtrace_path(pointers)
            path_int.reverse()
            path_int = path_int[1:]
            path_str = [ raw_seq[int((s-1)/2)] if s%2!=0 else '<pad>' for s in path_int]
            path_pid = processor.tokenizer.convert_tokens_to_ids(path_str)
            json_dict.update([("path_str", path_str)])
            json_dict.update([("path_pid", path_pid)])
            with open(sys.argv[5], 'w') as f:
                json.dump(json_dict,f)
            break
       







    
  
