import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
from pathlib import Path
import pdb



datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
sil_tokens = set(["SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

xstr = lambda s: s or ""

#RE for CMU files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 

##return only likeli
def ctc_loss(params, seq, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)

    alphas = torch.zeros((L,T)).double()

    # Initialize alphas and forward pass 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
            # same label twice
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                    * params[seq[l],t]
	    
    llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])
	
    return -llForward

    
##return only likeli, given the postion for allowing arbitrary tokens
def ctc_loss_denom(params, seq, pos, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)
    
    #skipped_vector = [False] * seqLen 

    alphas = torch.zeros((L,T)).double()

    # Initialize alphas and forward pass 
    # remain blanks blank
    alphas[0,0] = params[blank,0]  
    if pos == 0:
        alphas[1,0] = 1 - params[blank,0] # all non-blank output are allowed, because it is the first time entering the wildcard state
    else:
        alphas[1,0] = params[seq[0],0]

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        ##end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
           
            if s%2 == 0:  # blank
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                elif pos == l: ##leave the wildcard state with an emtpy token. 
                    #remove the path duplicated with the "skip" path from alphas[s,t-1]
                    removed1 =  alphas[s,t-1] - alphas[s-2,t-2] * params[blank,t-1]
                    #the wild card state must remove the "blank" path from the previous time(t-1), because it's overlapped with the first term:  alphas[s,t-1]
                    removed2 =  alphas[s-1,t-1] - alphas[s-1,t-2] * params[blank,t-1]
                    #we also allow "skip" action: alphas[s-2,t-1] for gop to model token deletion. (This skip is allowed only once for each blank state, otherwise duplicated computation)
                    alphas[s,t] = (removed1 + removed2 + alphas[s-2,t-1]) * params[blank,t]

                else: #normal update of blank
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                        * params[seq[l],t]
            elif pos == l-1: #previous token is the arbitrary token
                #removed = alphas[s-2,t-1]*(1 - params[blank,t-1] - params[seq[l],t-1] ) ## remove the blank token (already considered in the blank state) and the duplicated label of t-1
                #compare to the last equation
                if l == 1:  #l can't be 0
                    removed = alphas[s-2,t-1] - alphas[s-2,t-2]*(params[blank,t-1] + params[seq[l],t-1]) - alphas[s-3,t-2]*params[seq[l],t-1]
                elif seq[l] == seq[l-2]: ##already removed the path to s-2(same as s) in t-1 from s-4
                    removed = alphas[s-2,t-1] - alphas[s-2,t-2]*(params[blank,t-1] + params[seq[l],t-1]) - alphas[s-3,t-2]*params[seq[l],t-1]
                else:
                    removed = alphas[s-2,t-1] - alphas[s-2,t-2]*(params[blank,t-1] + params[seq[l],t-1]) - alphas[s-3,t-2]*params[seq[l],t-1] - alphas[s-4,t-2]*params[seq[l],t-1]
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + removed) * params[seq[l],t]
            else: #current pos can be arbitrary tokens, including the blanks
                if l == 0:
                    #alphas[s,t] = alphas[s,t-1] + alphas[s-1,t-1]*(1 - params[blank,t]) + alphas[s-2,t-1]*(1 - params[blank,t])
                    alphas[s,t] = alphas[s,t-1] + alphas[s-1,t-1]*(1 - params[blank,t]) 
                else:
                    alphas[s,t] = alphas[s,t-1] + alphas[s-1,t-1]*(1 - params[blank,t]) + alphas[s-2,t-1]*(1 - params[blank,t] - params[seq[l-1],t]) 
       
    if pos == seqLen-1:  ## this already contains the probability of being blank tokens as end of the sequence, L-3 and L-4 are also possible end states(same as what we do in alfree-v2, allow deletion )
        #pdb.set_trace()
        #llForward = torch.log(alphas[L-2, T-1] + alphas[L-3, T-1])
        llForward = torch.log(alphas[L-2, T-1] + alphas[L-3, T-2]* params[blank,T-1] + alphas[L-4, T-2]*(params[blank,T-1] +  params[seq[-2],T-1])) 
    else:  
        llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])
	
    return -llForward

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
        
        if len(scores) == 0:
            continue
        total_entry += 1
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
            if phoneme  not in (sil_tokens |spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 

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
                batch["p_text"].append(tran_map[uid])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    #ds_filtered = ds_map
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered

    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <out-file>.\n \
        , it analyzes the AUC using replacement error, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1])
    uttid_list = tran_map.keys()
    # load prior and the pretrained model
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    #model_path = ""
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-en/processor-en-ctc")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-en/processor-en-ctc")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path)
    #cuda = torch.device('cuda:1')
    
    # count = 0
    with torch.no_grad():
        p_set = set(p_tokenizer.get_vocab().keys())
        p_set = p_set - sil_tokens - spec_tokens
        pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_map = { p1:{ p2: [] for p2 in pid_set } for p1 in pid_set }  # map(p:map(p:average)
        for row in ds:
            # count += 1
            # if count > 5:
            #     break}
            #if row['id'] != "fabm2bp2":
            #    continue
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no alignment can be found")
                continue
            print("processing {0}".format(row['id']))
            #step 1, get the labels (pid_seq)
            labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
            labels = labels.type(torch.int32)
            
            ##step 2 run the model, return the post_mat and check the correctness of our loss function
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            return_dict = model(input_values, labels = labels)
            log_like_total = return_dict["loss"].squeeze(0)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1)
            
            #print("number of tokens:{}".format(labels.shape))
            #step 3  run phoneme replacement and compute GOP
            for order, pid in enumerate(labels):
                ll_denom = ctc_loss_denom(post_mat.transpose(0,1), labels, order, blank=0)
                if pid in pid_set:
                    for pid_inner in pid_set:
                        labels_edited = labels.clone()
                        labels_edited[order] = pid_inner
                        ll_num = ctc_loss(post_mat.transpose(0,1), labels_edited, blank=0)
                        #step 3.1, run phoneme replacement
                        gop_score = -ll_num + ll_denom
                        gops_map[int(pid)][pid_inner].append(gop_score)   

            json.dumps("./output_temp_gv2/{}.json".format(row["id"]))
                   
                
    print("done with GOP computation")
    write(gops_map, p_tokenizer, sys.argv[4])






    
  
