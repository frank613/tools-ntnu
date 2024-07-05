import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Processor,Wav2Vec2ForCTC
import torch
from pathlib import Path
import pdb
import matplotlib.pyplot as plt

ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_token = "SIL"
noisy_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "SPN"))


#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')


def writes(gops_list, outFile):
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, gop_list in gops_list:
            fw.write(key+'\n')
            for cnt, (p,score) in enumerate(gop_list):
                fw.write("%d %s %.3f\n"%(cnt, p, score))
            fw.write("\n")
    
def read_trans(trans_path):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                cur_uttid = items[0]
                trans_map[cur_uttid] = []
            phoneme = re_phone.match(items[4]).group(1)
            if phoneme not in (set([sil_token]) | noisy_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 


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
                batch["p_text"].append(tran_map[uid])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=3)

    return ds_filtered

##return the segement of the arbitrary state from the best path(with the largest conrtibution to the denominator), compare it to the forward algrotihm, we don't need to "remove" anything because we take the maximum for viterbi
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
            s_index = p_seq[s]
            #SIL
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
                    ## do we allow token1 -> token2  when they are identical? yes! since it's not forward(sum) but viterbi(max), so we are fine
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


# return the backtrace path for the current pointer table
def get_backtrace_path(pointers):
    
    T = pointers.shape[1]
    full_path_int = []
    sub_seq = [] ## label's id for the current token
    next_state = 0 #only one state defined for the additional time step
    for t in list(range(T-1,-1,-1)):
        next_state = int(pointers[int(next_state),t])
        full_path_int.append(next_state)
                 
    return (full_path_int)
    


   

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-file>.\n \
            , it analyzes the GOP using the ctc-align methods, the csv path must be a folder containing audios files and the metadata.csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1]) 
    uttid_list = tran_map.keys()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
   
    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, 'cmu-kids')
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens - sil_tokens
    count = 0
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_list = []  # (uttid, (phoneme, scores))
        for row in ds:
            #count += 1
            #if count > 10:
                #break
            #if row['id'] != 'fabm2ao2':
            #    continue
            print("processing {0}".format(row['id']))
            #get the total likelihood of the lable
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            #the default loss here in the config file is "ctc_loss_reduction": "sum" 
            raw_seq = row["p_text"]
            ##add optional SIL to p_seq
            p_seq = [sil_token]
            for p in raw_seq:
                p_seq.append(p)
                p_seq.append(sil_token)
            pid_seq = p_tokenizer.convert_tokens_to_ids(p_seq)
            #pid_seq = pid_seq.type(torch.int32)
            logits = model(input_values)["logits"].squeeze(0)
            post_mat = logits.softmax(dim=-1).type(torch.float64)
            ##merge noisy tokens to SIL:
            sil_index = p_tokenizer._convert_token_to_id(sil_token)
            noisy_labels = p_tokenizer.convert_tokens_to_ids(list(noisy_tokens))
            post_mat[:, sil_index] = post_mat[:,sil_index] + torch.sum(post_mat[:,noisy_labels], axis=-1)
            post_mat = post_mat.transpose(0,1)

            #step 2, compute the GOP
            pointers = get_ali_pointers(post_mat, pid_seq)
            full_path_int = get_backtrace_path(pointers)[:-1]
            full_path_int.reverse()
            gop_list = []
            last_state = 0
            post_count = 0
            post_total = 0
            for i,state in enumerate(full_path_int):
                l = int((last_state - 1)/2) 
                l_new = int((state - 1)/2) 
                if state != last_state:
                    if post_count != 0: ##previous state is not blank, token->blank or token1->token2
                        app_token = raw_seq[l]
                        gop_list.append((app_token, torch.log(post_total/post_count)))
                        post_count = 0
                        post_total = 0
                    #else: # blank->token
                if state%2 != 0:
                    post_count += 1
                    post_total += post_mat[pid_seq[state],i]
                last_state = state
            if post_count != 0:
                l = int((last_state - 1)/2)
                gop_list.append((raw_seq[l], torch.log(post_total/post_count)))
            ## the state is able to tell when to separate two idnetical tokens, for example "AH AH"
            assert len(gop_list) == len(raw_seq) 
            gops_list.append((row['id'], gop_list))
 
       

    print("done with GOP computation")
    writes(gops_list, sys.argv[5])
            
    

   







    
  
