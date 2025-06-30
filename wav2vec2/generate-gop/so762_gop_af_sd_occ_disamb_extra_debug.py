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


ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])
PAD_SIL_TOKEN = "SIL"

#RE for SO762
re_uttid = re.compile(r'(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
    

def load_dataset_local_from_dict(folder_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": [], "p_text":[]}
        with open(folder_path + '/metadata.csv') as csvfile:
            next(csvfile)
            for row in csvfile:
                filename,trans,scores = row.split(',')
                datadict["audio"].append(folder_path + '/' + filename)
                datadict["p_text"].append(trans.split(' '))
        ds = datasets.Dataset.from_dict(datadict) 
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
        ds.save_to_disk(cache_full_path)
    ds = datasets.Dataset.load_from_disk(cache_full_path)
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["id"] = [re_uttid.match(item["path"])[1] for item in batch["audio"]]
        return batch
    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100, num_proc=5)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=5)
    
    return ds_filtered

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
    #occ_vec = alphas.sum(dim=1)
    #genius
    occ_vec = (alphas*alpha_bar[None,:]).sum(-1)  	
    return (-llForward,occ_vec)	

# ##return also alphas for the target position
# def ctc_loss_plus(params, seq, pos, blank=0):
#     """
#     CTC loss function.
#     params - n x m matrix of n-D probability distributions(softmax output) over m frames.
#     seq - sequence of phone id's for given example.
#     Returns objective, alphas and betas.
#     """
#     seqLen = seq.shape[0] # Length of label sequence (# phones)
#     L = 2*seqLen + 1 # Length of label sequence with blanks
#     T = params.shape[1] # Length of utterance (time)

#     alphas = torch.zeros((L,T)).double()
#     alpha_bar = torch.zeros(T).double()

#     # Initialize alphas and forward pass 
#     alphas[0,0] = params[blank,0]
#     alphas[1,0] = params[seq[0],0]
#     alpha_bar[0] = torch.sum(alphas[:,0])
#     alphas[:,0] = alphas[:,0] /  alpha_bar[0]

#     for t in range(1,T):
#         start = max(0,L-2*(T-t)) 
#         end = min(2*t+2,L)
#         for s in range(start,L):
#             l = int((s-1)/2)
#             # blank
#             if s%2 == 0:
#                 if s==0:
#                     alphas[s,t] = alphas[s,t-1] * params[blank,t]
#                 else:
#                     alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
#             # same label twice
#             elif s == 1 or seq[l] == seq[l-1]:
#                 alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
#             else:
#                 alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
#                     * params[seq[l],t]
#         alpha_bar[t] = torch.sum(alphas[:,t])
#         alphas[:,t] = alphas[:,t] / alpha_bar[t]
    
#     #occ = alphas[2*pos+1,:].sum()
#     #genius
#     occ = (alphas[2*pos+1,:]*alpha_bar).sum()  	
#     llForward = torch.log(alpha_bar).sum()  	
#     return (-llForward,occ)

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
#zero_pos is used for blocking the path
def check_arbitrary(in_alphas, s, t, pos, zero_pos=[]):
    if (s-1)/2 == pos: ## is arbitrary
        if len(zero_pos) != 0:
            mask = torch.ones_like(in_alphas[s,t])
            for i in zero_pos:
                mask[i] = 0
            return (in_alphas[s,t] * mask ).sum()
        else:
            return (in_alphas[s,t]).sum()
    else:
        return None
    
def get_alpha_bar(alphas, t):
    return alphas[:,t,:].sum()

##This version composes of deletion subsitution using normalized alphas, so we need to concern skip paths
## it does not tract leakage because we removed the ambiguate connection , the cost is no substitution to the next phoenme can be detected, only substituion errors
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
  
    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()
    alpha_bar = torch.zeros(T).double()
    
    if pos == seqLen - 1:
        next_label_idx = None
    else:
        next_label_idx = seq[pos+1]
    
    ## constraint mask for disabling insertion, and in this version we don't allow phoneme->blank but remains in the arbitrary state 
    ## in this version we also block the path to next label for disambiguation
    mask_ins = torch.eye(P)
    if next_label_idx:
        mask_ins[next_label_idx, next_label_idx] = 0
    #mask_ins[blank,:] = torch.ones(P)
    
    # Initialize alphas 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]
        # can totally skip the pos, in this version we remove the initialization of state 2
        alphas[2,0,0] = 0
        if len(seq) > 1:
            alphas[3,0,0] = params[seq[1],0]
        alphas[1,0] = params[0:,0]  #an list all tokens
        alphas[1,0,blank] = 0  #can't stay at blank, same as the alphas[0,0,0]
        if next_label_idx:
            alphas[1,0,next_label_idx] = 0 ## in this version we block this for disambuation
        alpha_bar[0] = get_alpha_bar(alphas, 0)
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
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, pos, [blank]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum is not None: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
                        ##in this version no need to remove for t=1, because state 2 is not initialized anymore
                        alphas[s,t,0] = (alphas[s,t-1,0] + sum) * params[blank,t]
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                else:
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                        * params[seq[l],t]
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension, and also consider the skip paths
                sum = check_arbitrary(alphas, s-2, t-1, pos, [blank,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
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
                    ## in this version we also block the path to next label for disambiguation
                    if next_label_idx:
                        empty_prob[next_label_idx] = 0
                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + empty_prob
                    
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. alph
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[blank] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[blank] = 0

                    ## in this version we also block the path to next label for disambiguation
                    if next_label_idx:
                        empty_prob[next_label_idx] = 0
                        skip_prob[next_label_idx] = 0
                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + skip_prob + empty_prob 

        alpha_bar[t] = get_alpha_bar(alphas, t)
        alphas[:,t,:] = alphas[:,t,:] / alpha_bar[t]
        
    #occ = alphas[2*pos+1,:,:].sum()
    #gernius
    occ = (alphas[2*pos+1,:,:].sum(-1)*alpha_bar).sum() 
    llForward = torch.log(alpha_bar).sum()
    return (-llForward,occ)

def single_process(batch, p_tokenizer, processor, model_path, out_path):
    #row = example
    proc_id = str(os.getpid())
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:
        for i,uid in enumerate(batch["id"]): 
            print("processing {0}".format(uid))
            f.write(uid + '\n')
            #get the total likelihood of the lable
            input_values = processor(batch["speech"][i], return_tensors="pt", sampling_rate=16000).input_values
            #the default loss here in the config file is "ctc_loss_reduction": "sum" 
            labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(batch["p_text"][i]))
            labels = labels.type(torch.int32)
            ##return the log_like to check the correctness of our function
            return_dict = model(input_values, labels = labels)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1).type(torch.float64)
            ll_self, occ_vec_orig = ctc_loss(post_mat.transpose(0,1), labels, blank=0)
            #step 2, compute the GOP
            pids = labels.tolist()
            for i,pid in enumerate(pids):
                if i == 5:
                    pdb.set_trace()
                occ_num = occ_vec_orig[2*i+1]
                ll_denom,occ = ctc_loss_denom(post_mat.transpose(0,1), labels, i, blank=0)
                ## The missing extra path (the subsitution to the next pid) in the denominator
                occ_extra = torch.zeros(1)
                if i!= len(pids)-1:
                    labels_extra = labels.clone()
                    labels_extra[i] = pids[i+1]
                    ll_extra,occ_vec_extra = ctc_loss(post_mat.transpose(0,1), labels_extra, blank=0)
                    ll_denom = -((-ll_denom).exp() + (-ll_extra).exp()).log()
                    if occ_vec_extra[2*(i+1)+1] < 0.1:
                        #We need to use i+1 position for extra path because of deletion
                        occ_extra = occ_vec_extra[2*(i+1)+1]
                    else:
                        ##not a deletion, we need to compute the occ for the target position
                        occ_extra = occ_vec_extra[2*i+1]
                gop = -ll_self + ll_denom
                f.write("%d %s %s %s %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid)), gop.item(), occ.item(), occ_num.item(), occ_extra.item()))
            f.write("\n")

        

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments  <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <SIL-token> <out-file>.\n \
        , it generates the GOP using a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv. SIL indicates the token used for pad the SIL at the BOS/EOS")  
    # load the pretrained model and data
    model_path = sys.argv[1]
    csv_folder = sys.argv[2]
    prep_path = sys.argv[3]
    sil_token = sys.argv[4]

    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_folder, "speechocean762")
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model_path":model_path, "out_path":sys.argv[5]}, batched=True, batch_size=50, num_proc=1) 
    
    print("done")
    
    
       







    
  
