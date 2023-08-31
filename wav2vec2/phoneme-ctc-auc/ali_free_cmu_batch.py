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

##modified from Stanford-CTC, return the alphas and betas
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
    betas = torch.zeros((L,T)).double()

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

    # Initialize betas and backwards pass
    betas[-1,-1] = params[blank,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = betas[:,-1].sum()
    betas[:,-1] = betas[:,-1]
    for t in range(T-2,-1,-1):
        start = max(0,L-2*(T-t)) 
        end = min(2*t+2,L)
        for s in range(end-1,-1,-1):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s == L-1:
                    betas[s,t] = betas[s,t+1] * params[blank,t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
            # same label twice
            elif s == L-2 or seq[l] == seq[l+1]:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                    * params[seq[l],t]

    llBackward = torch.log(betas[0, 0] + betas[1, 0])
    # Check for underflow or zeros in denominator of gradient
    llDiff = np.abs(llForward-llBackward)
    if llDiff > 1e-5 :
        print("Diff in forward/backward LL : %f"%llDiff)
	
    return -llForward,alphas,betas   

def ctc_loss_batch(params, batch_label, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    batch_label - batch(N*L) of sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = batch_label.shape[1] # Length of label sequence (# phones)
    B = batch_label.shape[0]  #batch size
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)

    alphas = torch.zeros((B,L,T)).double()
    betas = torch.zeros((B,L,T)).double()

    # Initialize alphas and forward pass 
    alphas[:,0,0] = params[blank,0]
    index_init_alphas = torch.stack((batch_label[:,0],torch.ones_like(batch_label[:,0])*0),0).tolist()
    alphas[:,1,0] = params[index_init_alphas]
    
    for t in range(1,T):
        start = max(0,L-2*(T-t)) ##earliest start non-zero state for current t, T-t = how many t left to complete the sequence, for each t, it's able to cover at most 2 labels(directly skip from one non-black state to the blank state)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[:,s,t] = alphas[:,s,t-1] * params[blank,t]
                else:
                    alphas[:,s,t] = (alphas[:,s,t-1] + alphas[:,s-1,t-1]) * params[blank,t]
            
            else:
                index_list = torch.stack((batch_label[:,l],torch.ones_like(batch_label[:,l])*t),0).tolist()
                condition = torch.logical_or(torch.ones(batch_label.shape[0])==s, batch_label[:,l] == batch_label[:, l-1] )
                ## s-2 will be -1 when s == 1, still valid 
                alphas[:,s,t] = torch.where(condition, (alphas[:,s,t-1] + alphas[:,s-1,t-1]) * params[index_list], \
                                           (alphas[:,s,t-1] + alphas[:,s-1,t-1] + alphas[:,s-2,t-1]) * params[index_list])
	    
    llForward = torch.log(alphas[:,L-1, T-1] + alphas[:,L-2, T-1])

    # Initialize betas and backwards pass
    betas[:,-1,-1] = params[blank,-1]
    index_init_betas = torch.stack((batch_label[:,-1],torch.ones(batch_label.shape[0])*-1),0).tolist()
    betas[:,-2,-1] = params[index_init_betas]
    
    for t in range(T-2,-1,-1):
        end = min(2*t+2,L)  ## latest end non-zero state for current t, otherwise, with t steps, at most cover 2t+2(two initial state) previous states in l' 
        ##why (end-1)? end-1 is the index for alphas/betas, T,L,B are the lengths
        for s in range(end-1,-1,-1):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s == L-1:
                    betas[:,s,t] = betas[:,s,t+1] * params[blank,t]
                else:
                    betas[:,s,t] = (betas[:,s,t+1] + betas[:,s+1,t+1]) * params[blank,t]
            
            else:
                index_list = torch.stack((batch_label[:,l],torch.ones_like(batch_label[:,l])*t),0).tolist()
                #condition = torch.ones(batch_label.shape[0])*(L-2)==s
                #if all (condition):
                if s == L-2:
                    betas[:,s,t] = (betas[:,s,t+1] + betas[:,s+1,t+1]) * params[index_list]
                else:
                    condition = (batch_label[:,l] == batch_label[:, l+1])
                    betas[:,s,t] = torch.where(condition, (betas[:,s,t+1] + betas[:,s+1,t+1]) * params[index_list], (betas[:,s,t+1] + betas[:,s+1,t+1] + betas[:,s+2,t+1]) * params[index_list])
                #logical_or and where must assure both statement runable
                #condition = torch.logical_or(torch.ones(batch_label.shape[0])*(L-2)==s, batch_label[:,l] == batch_label[:, l+1] )

    llBackward = torch.log(betas[:, 0, 0] + betas[:, 1, 0])
    # Check for underflow or zeros in denominator of gradient
    llDiff = np.abs(llForward-llBackward)
    if any(llDiff > 1e-5) :
        print("Diff in forward/backward LL")
	
    return -llForward,alphas,betas   

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
            if phoneme not in (sil_tokens |spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 

#compute the gop of the sequece given the current phoneme number
def compute_gop_af(post_mat, labels, ll_self, alphas, betas, order):
    assert order < len(labels)
    phoneme_number = order + 1 #start from 1
    T = alphas.shape[1]
    pids = labels.tolist()
    pids_shiftdown = [None] + pids[:-1]
    pids_shiftup = pids[1:] + [None]  
    gop_list = []
    #[alpha_t1(s1) * (1-y(t1+1,P1)] * [(1 - y(t2-1,P2))*beta_t2(s2)]
    #t2 >= t1 + 3
    p_l,p_m,p_r = list(zip(pids_shiftdown, pids, pids_shiftup))[order]   
    denom_sum = 0
    start_s, end_s = 2*(phoneme_number)-1-1, 2*(phoneme_number)-1+1  #2i-1 is the middle phoneme index
    if phoneme_number == 1:
        s2 = end_s + 1
        for t2 in range(1,T): 
            denom_sum += (1-post_mat[t2-1, p_r]) * betas[s2, t2]
    elif phoneme_number == len(labels):
        s1 = start_s - 1
        for t1 in range(T-1):
            denom_sum += alphas[s1,t1]*(1-post_mat[t1+1,p_l]) 
    else:
        s1 = start_s - 1 
        s2 = end_s + 1
        ## for all the possible alignments of the current phoneme constraint by other canonical phonemes in the sequence
        for t1 in range(T-3): 
            for t2 in range(t1+3,T):
                denom_sum += alphas[s1,t1]*(1-post_mat[t1+1,p_l]) * (1-post_mat[t2-1, p_r]) * betas[s2,t2]
    
    gop = -ll_self + np.log(denom_sum)
    return gop

def compute_gop_af_batch(post_mat, ll_self, batch_label, alphas, betas, order):
    assert order < batch_label.shape[1]
    phoneme_number = order + 1 #start from 1
    T = alphas.shape[2]
    p_m = batch_label[:,order]
   
    #[alpha_t1(s1) * (1-y(t1+1,P1)] * [(1 - y(t2-1,P2))*beta_t2(s2)]
    #t2 >= t1 + 3
    #p_l,p_m,p_r = list(zip(pids_shiftdown, pids, pids_shiftup))[order]   
    denom_sum = torch.zeros_like(ll_self)
    start_s, end_s = 2*(phoneme_number)-1-1, 2*(phoneme_number)-1+1  #2i-1 is the middle phoneme index
    if phoneme_number == 1:
        p_r = batch_label[:,order+1]
        s2 = end_s + 1
        for t2 in range(1,T): 
            index_list = torch.stack((torch.ones(batch_label.shape[0])*(t2-1), p_r),0).tolist()
            denom_sum += (1-post_mat[index_list]) * betas[:, s2, t2]
    elif phoneme_number == len(labels):
        p_l = batch_label[:,order-1]
        s1 = start_s - 1
        for t1 in range(T-1):
            index_list = torch.stack((torch.ones(batch_label.shape[0])*(t1+1), p_l),0).tolist()
            denom_sum += betas[:,s1,t1]*(1-post_mat[index_list]) 
    else:
        p_r = batch_label[:,order+1]
        p_l = batch_label[:,order-1]
        s1 = start_s - 1 
        s2 = end_s + 1
        ## for all the possible alignments of the current phoneme constraint by other canonical phonemes in the sequence
        for t1 in range(T-3): 
            for t2 in range(t1+3,T):
                index_list_r = torch.stack((torch.ones(batch_label.shape[0])*(t2-1), p_r),0).tolist()
                index_list_l = torch.stack((torch.ones(batch_label.shape[0])*(t1+1), p_l),0).tolist()
                denom_sum += alphas[:,s1,t1]*(1-post_mat[index_list_l]) * (1-post_mat[index_list_r]) * betas[:,s2,t2]
    
    gop = -ll_self + torch.log(denom_sum)
    return gop

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
    
    #count = 0
    with torch.no_grad():
        p_set = set(p_tokenizer.get_vocab().keys())
        p_set = p_set - sil_tokens - spec_tokens
        pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_map = { p1:{ p2: [] for p2 in pid_set } for p1 in pid_set }  # map(p:map(p:average)
        for row in ds:
            #count += 1
            #if count > 1:
            #    break
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
            
            #step 3 compute and analyze the GOP
            for order, pid in enumerate(labels):
                if pid in pid_set:
                    batch_label = labels.repeat(len(pid_set),1)
                    batch_label[:,order] = torch.Tensor(pid_set)
                    ll_self, alphas, betas = ctc_loss_batch(post_mat.transpose(0,1), batch_label, blank=0)  
                    #step 3.1
                    gop_scores = compute_gop_af_batch(post_mat, ll_self, batch_label, alphas, betas, order)
                    for gop,pid_inner in zip(gop_scores,pid_set):
                        gops_map[int(pid)][pid_inner].append(gop)   
                   
                
    print("done with GOP computation")
    write(gops_map, p_tokenizer, sys.argv[4])






    
  
