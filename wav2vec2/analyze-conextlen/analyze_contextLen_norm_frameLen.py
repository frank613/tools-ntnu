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
import random

random.seed(0)
ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])
PAD_SIL_TOKEN = "SIL"

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
 
#pad_sil_token on begin and end of the sequence if not None   
def read_trans(trans_path, pad_sil_token=None):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                if pad_sil_token: ##add SIL at the begining and end of the sequence 
                    if cur_uttid != "":
                        trans_map[cur_uttid].append(pad_sil_token)
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = [pad_sil_token]
                else:
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = []
                cur_uttid = items[0]
            phoneme = re_phone.match(items[4]).group(1)                
            if phoneme not in (sil_tokens | spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    if pad_sil_token and trans_map[cur_uttid][-1] != pad_sil_token:
        trans_map[cur_uttid].append(pad_sil_token)
    return trans_map 


def load_dataset_local_from_dict(csv_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": []}  
        #with open(folder_path + '/metadata.csv') as csvfile:
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
                #datadict["audio"].append(folder_path + '/' + row.split(',')[0])
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
            if uid not in uttid_list or uid in error_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(tran_map[uid])
        return batch
    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=3)
    #ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered

##return the segement of the arbitrary state from the best path(with the largest conrtibution to the denominator), compare it to the forward algrotihm, we don't need to "remove" anything because we take the maximum for viterbi
def viterbi_ctc(params, seq, blank=0):
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
    P = params.shape[0] # number of non-blank tokens    

    ##the alphas[s,t] stors the best posterior for the current s at t
    alphas = torch.zeros((L,T)).double()
    
    ##For backtrace, the pointer[s,t] stores the source state of the alphas[s,t]. For t = 0 store -1. 
    #At T+1, the last time step store the winner of final state 0 (only one state (0) valid at T+1)
    pointers = torch.zeros((L,T+1)).double()
    
    # Initialize alphas for viterbi
 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    pointers[0,0] = -1
    pointers[1,0] = -1

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    pointers[s,t] = s #stays at s=0
                else:         
                    winner = max(alphas[s,t-1], alphas[s-1,t-1])
                    alphas[s,t] = winner * params[blank,t]
                    pointers[s,t] = s if alphas[s,t] == alphas[s,t-1]* params[blank,t] else s-1
            else:
                if l == 0 or seq[l] == seq[l-1]:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = -1
                    
                else:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = alphas[s-2,t-1]
                winner = max(s0,s1,s2)
                alphas[s,t] = winner * params[seq[l],t]
                if winner == s0: ## stays at s=0
                    pointers[s,t] = s
                elif winner == s1: ## leaving the arbitrary state at t, keep 
                    pointers[s,t] = s-1
                else:
                    pointers[s,t] = s-2

    empty_p = alphas[L-1, T-1]
    final_p = alphas[L-2, T-1]
    winner = max(final_p, empty_p)
    if winner == final_p: 
        pointers[0,T] = L-2
    else:
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

##return only likeli
def ctc_loss(params, seq, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)

    alphas = torch.zeros((L,T)).double()
    alpha_bar = torch.zeros(T).double()

    # Initialize alphas and forward pass 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    alpha_bar[0] = torch.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0] /  alpha_bar[0]

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
        alpha_bar[t] = torch.sum(alphas[:,t])
        alphas[:,t] = alphas[:,t] / alpha_bar[t]
    
    llForward = torch.log(alpha_bar).sum()   
	
    return -llForward

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
##zero pos here starst from 0def check_arbitrary(in_alphas, s, t, zero_pos=[]):
def check_arbitrary(in_alphas, s, t, zero_pos=[]):
    if in_alphas[s,t].sum() > 0:
        if len(zero_pos) != 0:
            mask = torch.ones_like(in_alphas[s,t])
            for i in zero_pos:
                mask[i] = 0
            return sum(in_alphas[s,t][mask.bool()])
        else:
            return sum(in_alphas[s,t][:])
    else:
        return False

# def get_alpha_bar(alphas, t, blank, next_label_idx, pos):
#     ## for comupting the alpha bar, we need to remove the blank state and next_label state in the arbitrary state  
#     ###exclude the same state in the "ärbitrary" state when computing the alpha_bar
#     arbitrary_state = 2*pos + 1 
#     alpha_mask = torch.ones(alphas.shape[2], dtype=torch.bool)
#     alpha_mask[blank] = False
#     if next_label_idx is not None:
#         alpha_mask[next_label_idx] = False
#     return alphas[:arbitrary_state,t,0].sum() + alphas[arbitrary_state+1:,t,0].sum() + alphas[arbitrary_state,t,alpha_mask].sum()


##we need to remove the duplication for the "arbitrary" state
def get_alpha_bar(alphas, t, blank, pos, next_label, leakage):
    arbitrary_state = 2*pos + 1 
    alpha_mask = torch.ones(alphas.shape[2], dtype=torch.bool)
    alpha_mask[blank] = False  ## the same as the next blank state, so we remove
    if next_label is not None:
        alpha_mask[next_label] = False  ## the same as the next non-blank state, so we remove
    ret = alphas[:arbitrary_state,t,0].sum() + alphas[arbitrary_state,t,alpha_mask].sum() + alphas[arbitrary_state+1:,t,0].sum() - leakage
    return ret

##This version composes of deletion subsitution using normalized alphas, so we need to concern skip paths
## it also tracts the leakage probability and substract it
def ctc_loss_denom(params, seq, pos, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)
    P = params.shape[0] # number of tokens    

    ## constraint mask for disabling insertion, and in this version we don't allow phoneme->blank but remains in the arbitrary state 
    mask_ins = torch.eye(P)
    #mask_ins[blank,:] = torch.ones(P)
    
    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()
    alpha_bar = torch.zeros(T).double()
    
    if pos == seqLen - 1:
        next_label_idx = None
    else:
        next_label_idx = seq[pos+1]
    
    ##### extra duplication at empty state after arbitrary state
    leakage = 0
    extra = 0
    # Initialize alphas 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]
        # can totally skip the pos, in this version we remove the initialization of state 2
        alphas[2,0,0] = 0
        if len(seq) > 1:
            alphas[3,0,0] = params[seq[1],0]
        alphas[1,0] = params[0:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0] 
        alpha_bar[0] = get_alpha_bar(alphas, 0, blank, pos, next_label_idx, leakage)

    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]
        alpha_bar[0] =  alphas[0,0,0] + alphas[1,0,0]
        
    alphas[:,0,:] = alphas[:,0,:] /  alpha_bar[0]
    
    for t in range(1,T):
        lowest_state = L-2*(T-t)
        if (lowest_state-1) / 2 == pos: ### -2 for possible skip paths at the arnitrary state
            lowest_state = lowest_state - 2
        start = max(0,lowest_state) 
        if start > 2*pos + 2:
            leakage = 0
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, [blank]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
                        ##in this version no need to remove for t=1, because state 2 is not initialized anymore
                        alphas[s,t,0] = (alphas[s,t-1,0] + sum) * params[blank,t]
                        ## extra duplication at empty state after arbitrary state
                        if next_label_idx != None:
                            extra = alphas[s-1, t-1, next_label_idx] * params[blank, t]
                            leakage = leakage*params[blank, t] + extra 
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                else:
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                        * params[seq[l],t]
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension, and also consider the skip paths
                sum = check_arbitrary(alphas, s-2, t-1, [blank,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                if l-2 < 0 or seq[l-2] == seq[l]: ###dont allow token skip
                    skip_token = 0
                else:
                    skip_token = alphas[s-4,t-1,0] * params[seq[l],t]
                ##we allow skip empty in this version, following the graph in the paper. No need to remove because the state[s-1] is blocked for skip.
                ##also we allow skip empty because we removed the state 2 probability in intialization
                skip_empty = alphas[s-3,t-1,0] * params[seq[l],t] 
                alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum ) * params[seq[l],t] + skip_empty + skip_token
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths    
                if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[blank] = 0
                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + empty_prob
                    
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. alph
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[blank] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[blank] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + skip_prob + empty_prob 

        alpha_bar[t] = get_alpha_bar(alphas, t, blank, pos, next_label_idx, leakage)
        alphas[:,t,:] = alphas[:,t,:] / alpha_bar[t]
        leakage = leakage / alpha_bar[t]
        
    occ = alphas[2*pos+1,:,:].sum()
    llForward = torch.log(alpha_bar).sum()
    return (-llForward, occ, alpha_bar)

def single_process(example, p_tokenizer, processor, model, max_context_len, out_path):
    row = example
    proc_id = str(os.getpid())
    #if row["id"] != "fahj1by1":
    #    return    
    print("processing {0}".format(row['id']))
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:   
        ##step1  get logits     
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        #the default loss here in the config file is "ctc_loss_reduction": "sum" 
        pid_seq = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
        pid_seq = pid_seq.type(torch.int32)
        ##return the log_like to check the correctness of our function
        return_dict = model(input_values, labels = pid_seq)
        logits = return_dict["logits"].squeeze(0) 
        post_mat = logits.softmax(dim=-1).type(torch.float64)
        ## run viterbi
        token_range = get_token_range(post_mat, pid_seq)        
        ## run context length test on the middle phoneme
        context_range =  int(max_context_len/2)
        p_index = int(len(pid_seq)/2)
        pid_from = pid_seq[p_index]
        p_from =  p_tokenizer._convert_id_to_token(int(pid_from))
        p_set = set(p_tokenizer.get_vocab().keys())
        ## p_set for select replacements
        p_list = list(p_set - spec_tokens - {p_from})
        ## select wrong phonemes
        p_to_pair = []
        for i in range(5):
            p_to = p_list[random.randrange(0, len(p_list))]
            pid_to = p_tokenizer._convert_token_to_id(p_to)
            p_to_pair.append((p_to,pid_to))
        ##compute full context gop for correct 
        ll_self = ctc_loss(post_mat.transpose(0,1), pid_seq, blank=0)
        ll_denom,occ,_ = ctc_loss_denom(post_mat.transpose(0,1), pid_seq, p_index, blank=0)
        occ = max(1, occ)
        gop = (-ll_self + ll_denom)/occ
        f.write("%s,%s,%s->%s,%s\n"%(row['id']+"-"+str(p_index)+"-"+p_from, "full", p_from, p_from, gop.item()))  
        #wrong phonemes               
        for p_to, pid_to in p_to_pair:
            labels_replaced = pid_seq.clone()
            labels_replaced[p_index] = pid_to
            ll_self_replaced = ctc_loss(post_mat.transpose(0,1), labels_replaced, blank=0)
            gop_replaced = (-ll_self_replaced + ll_denom)/occ
            f.write("%s,%s,%s->%s,%s\n"%(row['id']+"-"+str(p_index)+"-"+p_to, "full", p_from, p_to, gop_replaced.item()))
        if p_index - context_range < 0 or p_index + context_range > len(pid_seq) - 1:
            print("not enough left or right context")  
            return
        
        ### try with other content-len
        # substitutions       
        for p_to,pid_to in p_to_pair:
            labels_wrong = pid_seq.clone()
            labels_wrong[p_index] = pid_to
            for i in range(0, int(context_len/2)+1):
                frame_range_l = int(sum(token_range[(2*(p_index-i) + 1) -1])/2)
                frame_range_r = int(sum(token_range[(2*(p_index+i) + 1) + 1])/2)
                #correct
                labels = torch.Tensor(pid_seq[p_index-i:p_index + i + 1]).type(torch.int32) ## start from no context
                ll_self = ctc_loss(post_mat[frame_range_l:frame_range_r+1,:].transpose(0,1), labels, blank=0)
                ll_denom,occ,_ = ctc_loss_denom(post_mat[frame_range_l:frame_range_r+1,:].transpose(0,1), labels, i, blank=0)
                occ = max(1, occ)
                gop = (-ll_self + ll_denom)/occ
                f.write("%s,%s,%s->%s,%s\n"%(row['id']+"-"+str(p_index)+"-"+p_from, 2*i, p_from, p_from, gop.item()))            
                ##wrong
                labels_replaced = torch.Tensor(labels_wrong[p_index-i:p_index + i + 1]).type(torch.int32) ## start from no context
                ll_self_replaced = ctc_loss(post_mat[frame_range_l:frame_range_r+1,:].transpose(0,1), labels_replaced, blank=0)
                gop_replaced = (-ll_self_replaced + ll_denom)/occ
                f.write("%s,%s,%s->%s,%s\n"%(row['id']+"-"+str(p_index)+"-"+p_to, 2*i, p_from, p_to, gop_replaced.item()))
        

#run viterbi and get the dict of range of each token, there might be blanks being skipped, we need to add dummy segment for those
def get_token_range(post_mat, pid_seq):
    pointers = viterbi_ctc(post_mat.transpose(0,1), pid_seq)
    path_int = get_backtrace_path(pointers)
    path_int.reverse()
    path_int = path_int[1:]
    token_range = []
    last = -1
    left = 0
    right = -1
    for c,current in enumerate(path_int):
        if last == -1 and current == 1: ##skipped the first blank
            token_range.append((0,0))
        if current != last and last != -1:
            right = c-1
            token_range.append((left, right))
            if current - last > 1: ## skipped blank, so we will cut at the "right" frame later
                token_range.append((right, right))
            left = c
        last = current
    if last%2 == 0: ##end with blank 
        token_range.append((left, len(path_int)-1))
    else:
        token_range.append((left, len(path_int)-1))
        token_range.append((len(path_int)-1, len(path_int)-1))
    assert(len(token_range) == 2*len(pid_seq) + 1)
    return token_range
        

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <error-uttid_list> <out-file>.\n \
        , it runs the context length test for GOP-AF-SD on correct phonemes and simulated error phonemes")  
    # load the pretrained model and data
    tran_path = sys.argv[1]
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
     
    #step 0, read the files
    tran_map = read_trans(tran_path) 
    uttid_list = tran_map.keys()
    error_list = []
    with open(sys.argv[5]) as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split()
            if len(fields) != 1:
                sys.exit("wrong input line")
            error_list.append(fields[0])
    
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    ### the length of the surrounding tokens
    context_len = 2*8
    
    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model":model, "max_context_len":context_len, "out_path":sys.argv[6]}, num_proc=10) 
    print("done")
    
    
       







    
  
