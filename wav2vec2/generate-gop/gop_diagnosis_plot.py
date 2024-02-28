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


datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')
    
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
            if phoneme not in (sil_tokens | spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 


def load_dataset_local_from_dict(folder_path, uttid):
    datadict = {"audio": []}  
    with open(folder_path + '/metadata.csv') as csvfile:
        next(csvfile)
        for row in csvfile:
            uid = row.split(',')[0].split('.')[0]     
            if uttid == uid:
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


    return ds_map

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

##return only likeli
def ctc_loss_scaled(params, seq, blank=0):
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
    c = alphas[:,0].sum()
    alphas[:,0] = alphas[:,0] / c
    llForward = torch.log(c)

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
            
        # normalize at current time (prevent underflow)
        c = alphas[start:end,t].sum()
        alphas[start:end,t] = alphas[start:end,t] / c
        llForward += torch.log(c)
        
    return -llForward

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
##zero pos here starst from 0
def check_arbitrary(in_alphas, s, t, zero_pos=[]):
    mask = torch.ones_like(in_alphas[s,t])
    if torch.count_nonzero(in_alphas[s,t]) > 1:
        if len(zero_pos) != 0:
            for i in zero_pos:
                mask[i] = 0
        return torch.sum(in_alphas[s,t][mask.bool()])
    else:
        return False
    

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum, return the vector for back trace
# (different from alphas by assignubg 0 to zero_pos)
def check_arbitrary_back(in_alphas, s, t, zero_pos=[]):
    tensor_t = in_alphas[s,t]
    mask = torch.ones_like(in_alphas[s,t])
    if torch.count_nonzero(tensor_t) > 1:
        if len(zero_pos) != 0:
            for i in zero_pos:
                tensor_t[i] = 0
        return (torch.sum(tensor_t), tensor_t)
      
    else:
        return (False,False)
    
# return False if it's not arbitrary state, return the max label index otherwise
def check_pointer_back(pointers, s, t):
    if torch.count_nonzero(pointers[s,t,1:]) > 0:
        return torch.argmax(pointers[s,t,1:])   
    else:
        return -1


# reduce the last dimension and divide it by the NN output
def check_and_divide(in_alphas, post_mat, label_seq):
    for s in range(in_alphas.shape[0]):
        for t in range(in_alphas.shape[1]):
            if s%2 == 0:
                
                continue
            l = int((s-1)/2)
            if torch.count_nonzero(in_alphas[s,t]) > 1:
                in_alphas[s,t] = in_alphas[s,t] / post_mat[:,t]
            else:
                in_alphas[s,t,0] = in_alphas[s,t,0] / post_mat[label_seq[l],t] 
 
    return None

##return likeli, given the postion for arbitrary state
def ctc_loss_denom_all(params, seq, pos, blank=0):
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

    ##alphas
    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()

    # Initialize alphas and forward pass 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]
        # can totally skip the pos
        alphas[2,0,0] = params[blank,0]
        alphas[3,0,0] = params[seq[1],0]
        
        alphas[1,0] = params[0:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0]
    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]

    for t in range(1,T):
        ###different from the gv3+, we need to strictly limit the max and end, because the alphas and betas are needed not only for computing the total likelihood 
        start = max(0,L-2*(T-t)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, [0]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension       
                        if t == 1:
                            ## should be = 0, totally remove the path because it's the same as the skip path
                            # which means the only two ways to start from s=2 at t=0 is 2->3 or 2->4
                            removed = alphas[s,t-1,0] - alphas[s-2,t-1,0]
                            alphas[s,t,0] = (removed + sum + alphas[s-2,t-1,0]) * params[blank,t]
                        else:
                            removed =  alphas[s,t-1,0] - alphas[s-2,t-2,0] * params[blank,t-1]  ## allow for jump, but only once, same as in v2
                            alphas[s,t,0] = (removed + sum + alphas[s-2,t-1,0]) * params[blank,t]  
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                else:
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                        * params[seq[l],t]
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension
                sum = check_arbitrary(alphas, s-2, t-1, [0,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum) * params[seq[l],t]
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths       
                if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + empty_prob
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[0] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + skip_prob + empty_prob
         
    sum = check_arbitrary(alphas, L-2, T-1)    
    if sum: # last label is arbitrary, inlcludes empty as well so we don't need the last term alphas[L-1,T-1,0], but we need the skip path
        #need explictly the alphas of T-2 for skip path because at T-1 only two states have values(L-1,L-2)
        llForward = torch.log(sum + alphas[L-3, T-2,0]* params[blank,T-1] + alphas[L-4, T-2, 0]*(params[blank,T-1] +  params[seq[-2],T-1]))
    else:
        llForward = torch.log(alphas[L-1, T-1, 0] + alphas[L-2, T-1, 0])

    ####betas
    betas = torch.zeros((L,T,P)).double()
    
    # Initialize betas and backward pass
    if pos == seqLen - 1:
        betas[-1,-1,0] = params[blank,-1]
        # can totally skip the pos
        betas[-3,-1,0] = params[blank,-1]
        betas[-4,-1,0] = params[seq[-2],-1]
        
        betas[-2,-1] = params[-1:,-1]  #an list all tokens
        betas[-2,-1,0] = 0  #can't be blank
    else:
        betas[-1,-1,0] = params[blank,-1]
        betas[-2,-1,0] = params[seq[-1],-1]

    ###for loop from T-1 -> 0
    for t in range(T-2,-1, -1):
        end = min(2*t+2,L) 
        for s in range(end-1,-1,-1):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==L-1:
                    betas[s,t,0] = betas[s,t+1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(betas, s+1, t+1, [0]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension       
                        if t == T-2:
                            removed = betas[s,t+1,0] - betas[s+2,t+1,0] ## should be = 0, totally remove the path because it's the same as the skip path
                            betas[s,t,0] = (removed + sum + betas[s-2,t-1,0]) * params[blank,t]
                        else:
                            removed =  betas[s,t+1,0] - betas[s+2,t+2,0] * params[blank,t+1]  ## allow for jump, but only once, same as in v2
                            betas[s,t,0] = (removed + sum + betas[s+2,t+1,0]) * params[blank,t]  
                    else:
                        betas[s,t,0] = (betas[s,t+1,0] + betas[s+1,t+1,0]) * params[blank,t]
            elif pos != l and pos != l+1:
                if s == L-2 or seq[l] == seq[l+1]:   # the first label or same label twice
                    betas[s,t,0] = (betas[s,t+1,0] + betas[s+1,t+1,0]) * params[seq[l],t]
                else:
                    betas[s,t,0] = (betas[s,t+1,0] + betas[s+1,t+1,0] + betas[s+2,t+1,0]) \
                        * params[seq[l],t]
            elif pos == l+1: #last token is the arbitrary token, need to collect the probability from the additional dimension
                sum = check_arbitrary(betas, s+2, t+1, [0,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                betas[s,t,0] = (betas[s,t+1,0] + betas[s+1,t+1,0] + sum) * params[seq[l],t]
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths       
                if s == L-2: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = betas[s+1,t+1,0] * params[:,t]
                    empty_prob[0] = 0

                    betas[s,t,:] = (betas[s,t+1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + empty_prob
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                    skip_prob = betas[s+2,t+1,0] * params[:,t]  
                    skip_prob[seq[l+1]] = 0    
                    skip_prob[0] = 0    

                    empty_prob = betas[s+1,t+1,0] * params[:,t]
                    empty_prob[0] = 0

                    betas[s,t,:] = (betas[s,t+1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + skip_prob + empty_prob
    
    sum = check_arbitrary(betas, 1, 0)    
    if sum: 
        #need explictly the betas of t=1 for skip path because at 0 only two states have values(0,1)
        llBackward = torch.log(sum + betas[2,1,0]* params[blank,0] + betas[3,1,0]*(params[blank,0] +  params[seq[1],0]))
    else:
        llBackward = torch.log(betas[0, 0, 0] + betas[1, 0, 0])
    
    if np.abs(llForward-llBackward) > 1e-2 :
        pdb.set_trace()
        sys.exit("llDiff too big, check the results")
        
    return (-llForward, alphas, betas)

##return the segement of the arbitrary state from the best path(with the largest conrtibution to the denominator), compare it to the forward algrotihm, we don't need to "remove" anything because we take the maximum for viterbi
def viterbi_denom(params, seq, pos, blank=0):
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

    ##the alphas[s,t] stors the best posterior for the current s at t,  also extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()
    
    ##For backtrace, the pointer[s,t,0] stores the source state of the alphas[s,t] at the first element. For t = 0 store -1. 
    ## Store the source alphas pointers[s,t,1:] after blocking illegal paths for arbitrary token
    #T+1, the last time step store the winner of final state 0 (only one state (0) valid at T+1)
    pointers = torch.zeros((L,T+1,P+1)).double()
    
    # Initialize alphas for viterbi
    if pos == 0:
        alphas[1,0] = params[:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0]
        alphas[0,0,0] = params[blank,0]
        pointers[0,0,0] = -1
        pointers[1,0,0] = -1  
        # can totally skip the pos
        alphas[2,0,0] = params[blank,0]
        alphas[3,0,0] = params[seq[1],0]
        pointers[2,0,0] = -1
        pointers[3,0,0] = -1  
    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]
        pointers[0,0,0] = -1
        pointers[1,0,0] = -1

    for t in range(1,T):
        start = max(0,L-2*(T-t+1)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                    pointers[s,t,0] = s #stays at s=0
                else:
                    sum, back_states = check_arbitrary_back(alphas, s-1, t-1, [0]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state      
                        if t == 1:
                            removed = alphas[s,t-1,0] - alphas[s-2,t-1,0] ## should be = 0, totally remove the path because it's the same as the skip path
                        else:
                            removed = alphas[s,t-1,0] - alphas[s-2,t-2,0] * params[blank,t-1]  ## allow for jump, but only once, same as in v2
                        s0 = removed
                        s1 = sum 
                        s2 = alphas[s-2,t-1,0]
                        winner = max(s0,s1,s2)
                        alphas[s,t,0] = winner * params[blank,t]
                        if winner == s0: ## stays at s=0
                            pointers[s,t,0] = s
                        elif winner == s1: ## leaving the arbitrary state at t
                            pointers[s,t,0] = s-1
                            pointers[s,t,1:] = back_states
                        else:
                            pointers[s,t,0] = s-2
                    else:
                        winner = max(alphas[s,t-1,0], alphas[s-1,t-1,0])
                        alphas[s,t,0] = winner * params[blank,t]
                        pointers[s,t,0] = s if alphas[s,t,0] == alphas[s,t-1,0]* params[blank,t] else s-1
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    winner = max(alphas[s,t-1,0], alphas[s-1,t-1,0])
                    alphas[s,t,0] = winner * params[seq[l],t]
                    pointers[s,t,0] = s if winner == alphas[s,t-1,0] else s-1
                else:
                    s0 = alphas[s,t-1,0]
                    s1 = alphas[s-1,t-1,0]
                    s2 = alphas[s-2,t-1,0]
                    winner = max(s0,s1,s2)
                    alphas[s,t,0] = winner * params[seq[l],t]
                    if winner == s0: ## stays at s=0
                        pointers[s,t,0] = s
                    elif winner == s1: ## leaving the arbitrary state at t, keep 
                        pointers[s,t,0] = s-1
                    else:
                        pointers[s,t,0] = s-2
                        
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension
                sum, back_states = check_arbitrary_back(alphas, s-2, t-1, [0,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                s0 = alphas[s,t-1,0]
                s1 = alphas[s-1,t-1,0]
                s2 = sum
                winner = max(s0,s1,s2)
                alphas[s,t,0] = winner * params[seq[l],t]
                if winner == s0: ## stays at s=0
                    pointers[s,t,0] = s
                elif winner == s1: ## leaving the arbitrary state at t, keep 
                    pointers[s,t,0] = s-1
                else:
                    pointers[s,t,0] = s-2
                    pointers[s,t,1:] = back_states
               
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow for all the paths       
                if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0
                    empty_sum = empty_prob.sum()

                    alphas_prob = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1)
                    alphas_sum = alphas_prob.sum()
                    winner = max(empty_sum, alphas_sum)
                   
                    if winner == empty_sum: 
                        alphas[s,t,:] = empty_prob
                        pointers[s,t,0] = s - 1
                    else:
                        alphas[s,t,:] = alphas_prob
                        pointers[s,t,0] = s
                        pointers[s,t,1:] = alphas[s,t-1,:]  # we always store the t-1 alphas in pointers 
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[0] = 0 
                    skip_sum = skip_prob.sum()  

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0
                    empty_sum = empty_prob.sum()

                    alphas_prob = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1)
                    alphas_sum = alphas_prob.sum()
                    
                    winner = max(alphas_sum, empty_sum, skip_sum)
                    if winner == skip_sum: 
                        alphas[s,t,:] = skip_prob
                        pointers[s,t,0] = s-2
                    elif winner == empty_sum: 
                        alphas[s,t,:] = empty_prob
                        pointers[s,t,0] = s-1
                    else:
                        alphas[s,t,:] = alphas_prob
                        pointers[s,t,0] = s
                        pointers[s,t,1:] = alphas[s,t-1,:]  # we always store the t-1 alphas in pointers 
                    
    sum, back_states = check_arbitrary_back(alphas, L-2, T-1)    
    if sum: # last label is arbitrary, we still need to compare the skip and empty path
        #need explictly the alphas of T-2 for skip path because at T-1 only two states have values(L-1,L-2)
        empty_p = alphas[L-3, T-1, 0]
        skip_p = alphas[L-4, T-1, 0]
        winner = max(sum, empty_p, skip_p)
        if winner == empty_p: 
            pointers[0,T,0] = L-3
        elif winner == skip_p: 
            pointers[0,T,0] = L-4
        else:
            pointers[0,T,0] = L-2
            pointers[0,T,1:] = alphas[L-2,T-1,:]  # we always store the t-1 alphas in pointers 
                    
    else:
        empty_p = alphas[L-1, T-1, 0]
        final_p = alphas[L-2, T-1, 0]
        winner = max(final_p, empty_p)
        if winner == final_p: 
            pointers[0,T,0] = L-2
        else:
            pointers[0,T,0] = L-1             
    return pointers

def check_exit(istring):
    if istring == "exit":
        sys.exit(1)

# return the backtrace path for the current pointer table, to find the biggest contribution to denom liklihood, and also the sub path inside the arbitrary state
def get_backtrace_path(pointers):
    
    # always return the path for continous states with "extended pointer"
    # if no "extended pointer" found in the full path, meaning a deletion happens!
    T = pointers.shape[1]
    full_path = []
    full_path_int = []
    sub_seq = [] ## label's id for the current token
    next_state = 0 #only one state defined for the additional time step
    for t in list(range(T-1,-1,-1)):
        check_return = check_pointer_back(pointers,int(next_state),t)
        next_state = int(pointers[int(next_state),t,0])
        if check_return != -1: #the arbitrary token
            full_path.append(str(next_state) + '***')
            sub_seq.append(int(check_return))
            full_path_int.append(next_state)
        else:
            full_path.append(str(next_state))
            full_path_int.append(next_state)
                 
    return (full_path, sub_seq, full_path_int)
    
def plot_posterior_gragh(label_ids, post_mat, full_path, sub_seq, alphas, betas, out_fname, p_tokenizer,point_index,wav):
    ## plot the poseterior heat for each time step and state, also draw the full_path line
    #reduced_a = alphas.sum(-1)
    #reduced_b = betas.sum(-1)
    # p_matrix = torch.zeros_like(reduced_a)
    # p_matrix[::2] = post_mat[0]
    # for i,l in enumerate(label_ids):
    #     p_matrix[2*i+1] = post_mat[l]      
    # posterior_matrix = reduced_a * reduced_b / p_matrix
    # ##normalize each time step
    # #posterior_matrix = (posterior_matrix/(posterior_matrix.sum(0))).flip(0)
    # posterior_matrix = posterior_matrix/(posterior_matrix.sum(0))
    # posterior_matrix = posterior_matrix.numpy()

    ##fixed posterior_matrix
    reduced_a = alphas.sum(-1)
    reduced_b = betas.sum(-1)
    check_and_divide(alphas, post_mat, label_ids)
    posterior_matrix = (alphas * betas).sum(-1)
    posterior_matrix = posterior_matrix/(posterior_matrix.sum(0))
    posterior_matrix = posterior_matrix.numpy()
    
    ##plot
    #fig, axes = plt.subplots(3,2,figsize=(30, 20), sharex="col", constrained_layout=True, gridspec_kw={'width_ratios': [30, 1], 'height_ratios': [1, 3, 3]})
    fig, axes = plt.subplots(4,2,figsize=(30, 30), sharex="col", gridspec_kw={'width_ratios': [30, 1], 'height_ratios': [1, 4, 4, 4]})
    axes[0, 1].axis('off')
    #axes[1,0].sharex(axes[0,0])
    #axes[0,0].sharex(axes[1,0])

    im = axes[1,0].imshow(posterior_matrix, origin="lower")
    im2 = axes[2,0].imshow(reduced_a, origin="lower")
    im3 = axes[3,0].imshow(reduced_b, origin="lower")

    labels = p_tokenizer.convert_ids_to_tokens(label_ids)
    labels = [["#",item] for item in labels]
    labels = [ col for row in labels for col in row] + ["#"]

    wav_min = min(wav)
    wav = [p - wav_min for p in wav]
    axes[0,0].plot(point_index, wav)
    
    axes[1,0].autoscale(False)
    axes[2,0].autoscale(False) 
    axes[3,0].autoscale(False)
    axes[0,0].autoscale(False)

    # Create colorbar
    cbar = fig.colorbar(im, cax=axes[1,1])
    cbar.ax.set_ylabel("normalized posteior of pathes that went through (s,t)", rotation=-90, va="bottom")
    
    cbar = fig.colorbar(im2, cax=axes[2,1])
    cbar.ax.set_ylabel("unnormlized alphas for each (s,t)", rotation=-90, va="bottom")
    
    cbar = fig.colorbar(im3, cax=axes[3,1])
    cbar.ax.set_ylabel("unnormlized betas for each (s,t)", rotation=-90, va="bottom")
    # Show all ticks a09.9nd label them with the respective list entries
    #axes[1,0].set_xticks(np.arange(posterior_matrix.shape[1]), labels=np.arange(posterior_matrix.shape[1]))
    axes[1,0].set_yticks(np.arange(posterior_matrix.shape[0]), labels=labels)
    axes[2,0].set_yticks(np.arange(posterior_matrix.shape[0]), labels=labels)
    axes[3,0].set_yticks(np.arange(posterior_matrix.shape[0]), labels=labels)

    # full_path
    #full_path_height = len(full_path) - 1  - np.array(full_path[1:])
    full_path_height = np.array(full_path[1:])
    axes[1,0].step(np.arange(posterior_matrix.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
    axes[1,0].grid()
    axes[1,0].legend(loc=2)
    
    axes[2,0].step(np.arange(reduced_a.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
    axes[2,0].grid()
    axes[2,0].legend(loc=2)
    
    axes[3,0].step(np.arange(reduced_b.shape[1]),full_path_height, where="post", color="r", label="Viterbi best path: {}".format(sub_seq)) 
    axes[3,0].grid()
    axes[3,0].legend(loc=2)
    fig.tight_layout()
    #plt.show()
    out_file = "./out-plot" + "/" + out_fname + ".png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    axes[0,0].set_aspect("auto")
    axes[1,0].set_aspect("auto")
    axes[2,0].set_aspect("auto") 
    axes[3,0].set_aspect("auto") 
    
    plt.savefig(out_file)
    
    

def decode_post(post_mat, p_tokenizer): 
    ##plot decode path using "maximum t" strategy
    ids = torch.argmax(post_mat, 0)
    return p_tokenizer.convert_ids_to_tokens(ids)
    

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir>.\n \
            , it analyzes the GOP using the AF-gv2 methods, the csv path must be a folder containing audios files and the metadata.csv") 
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
   
    uttid = None
    pos = None
    labels = None
    while True:
        if uttid is None: 
            uttid = input("type in the uttid or exit:\n")
            check_exit(uttid)
            if uttid not in tran_map:
                print("uttid not found in the transcription file")
                uttid = None
                continue  
            # load dataset and read soundfiles for the given uttid
            ds= load_dataset_local_from_dict(csv_path, uttid)
            if len(ds) == 0:
                print("uttid not found in the audio folder")
                uttid = None
                continue
            print("The dictionary phoneme sequence: ")
            dict_seq = tran_map[uttid]
            #tran_pairs = [(i,phoneme) for i,phoneme in zip(range(len(dict_seq)), dict_seq)]
            tran_pairs = [(i,phoneme) for i,phoneme in enumerate(dict_seq)]
            continue
        
        if pos is None:
            print(tran_pairs)
            pos = input("type in the index of phoneme to be replaced or exit:\n")
            check_exit(pos)
            pos = int(pos)
            if pos >= len(dict_seq):
                print("pos is larger than the limit of the current utterence")
                pos = None
                continue
            
        if labels is None:
            cano_seq = input("type in the canonical phoneme sequence or exit, can be empty for synthesizing phoeneme insertion:\n")
            check_exit(cano_seq)
            cano_list = cano_seq.split()
            
            labels_text = dict_seq[:pos] + cano_list + dict_seq[pos+1:]
            labels = p_tokenizer.convert_tokens_to_ids(labels_text)
            if None in labels:
                print("invalid phoneme in the canonical phonemes, please check and type again!")
                labels = None
                continue
        print("the current canonical seqence is:")
        print(labels_text)
        with torch.no_grad():
            row = ds[0]
            print("processing {0}".format(row['id']))
            #get the total likelihood of the lable
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            labels = torch.Tensor(labels)
            labels = labels.type(torch.int32)
            ##return the log_like to check the correctness of our function
            return_dict = model(input_values, labels = labels)
            log_like_total = return_dict["loss"].squeeze(0)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1).type(torch.float64)
            ll_self = ctc_loss(post_mat.transpose(0,1), labels, blank=0)
            #ll_self = ctc_loss_scaled(post_mat.transpose(0,1), labels, blank=0)
            llDiff = np.abs(log_like_total - ll_self)
            ##frame index for plotting the wav
            nframe_points = 16000*0.02
            #frame_index = [ p/nframe_points for p in range(len(row["speech"]))]
            #truncated by wav2vec2?
            frame_index = [ p/nframe_points for p in range(0,int(logits.shape[0]*nframe_points))]
            wav_truncated = row["speech"][:len(frame_index)]

            print("the best decode path:")
            best_decode_path = decode_post(post_mat.transpose(0,1), p_tokenizer)
            print(best_decode_path)
            if llDiff > 1 :
                print(f"model ll: {log_like_total}, function ll: {ll_self}")
    
            for i in range(labels.shape[0]):
                ll_denom, alphas, betas = ctc_loss_denom_all(post_mat.transpose(0,1), labels, i, blank=0)
                gop = -ll_self + ll_denom
                pointers = viterbi_denom(post_mat.transpose(0,1), labels, i, blank=0)
                full_path, sub_seq_ids, full_path_int = get_backtrace_path(pointers)
                full_path.reverse()
                full_path_int.reverse()
                sub_seq_ids.reverse()
                print("{0}: {1}, GOP = {2}".format(i, labels_text[i], gop))
                print("the best path for this token:")
                print(full_path[1:])
                print("the best sub_seq for this token:")
                sub_seq = p_tokenizer.convert_ids_to_tokens(sub_seq_ids)
                print(sub_seq)
                print("generating plots")
                plot_posterior_gragh(labels, post_mat.transpose(0,1), full_path_int, sub_seq, alphas, betas, labels_text[i]+str(i),p_tokenizer,frame_index, wav_truncated)
            pos = None
            labels = None
            
    

   







    
  
