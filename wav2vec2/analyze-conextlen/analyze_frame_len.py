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

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')


###the index of the target phoneme, start from 0
p_index = 30
### the length of the surrounding tokens
context_len = 2*0
### frame repetation 
frame_rep = 40

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


#run viterbi and get the dict of range of each token, there might be blanks being skipped, we need to add dummy segment for those
def get_token_range(post_mat, pid_seq):
    pointers = get_ali_pointers(post_mat.transpose(0,1), pid_seq)
    path_int = get_backtrace_path(pointers)
    path_int.reverse()
    path_int = path_int[1:]
    token_range = []
    last = -1
    left = 0
    right = -1
    for c,current in enumerate(path_int):
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
    #pdb.set_trace()
    assert(len(token_range) == 2*len(pid_seq) + 1)
    return token_range

###return the tensor of the requried range and repetition
def get_rep_tensor(in_tensor, target_range_l, target_range_r, empty_range_l, empty_range_r, rep):
    left = in_tensor[:, empty_range_l:target_range_l]
    right = in_tensor[:, target_range_r+1:empty_range_r+1]
    repeated = in_tensor[:, target_range_l:target_range_r+1].repeat(1,rep)
    ret = torch.concat((left, repeated, right), 1) 
    return ret
    

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
    
##return only likeli, given the postion for arbitrary state, 
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
    P = params.shape[0] # number of tokens    

    ## constraint mask for disabling insertion, and in this version we don't allow phoneme->blank but remains in the arbitrary state 
    mask_ins = torch.eye(P)
    #mask_ins[blank,:] = torch.ones(P)
    
    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()

    # Initialize alphas 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]
        # in this version we don't allow initial with the second blank,  it will simplifiy the later computation especially for the “normalized” version
        alphas[2,0,0] = 0
        if len(seq) > 1:
            alphas[3,0,0] = params[seq[1],0]
        alphas[1,0] = params[0:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0]
    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]
        
    for t in range(1,T):
        if pos == seqLen-1: ###different from non-composed one, +1 below for possible skip paths at the final states
            lowest_state = L-2*(T-t+1)
        else:
            lowest_state = L-2*(T-t)
        start = max(0,lowest_state) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, [blank]) 
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
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
                    empty_prob[0] = 0
                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + empty_prob
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[0] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + skip_prob + empty_prob
         
    sum = check_arbitrary(alphas, L-2, T-1, [blank])    
    if sum: # last label is arbitrary,  we need the skip path alphas[L-3, T-1, 0] + alphas[L-4, T-1, 0]
        if len(seq) == 1:
            llForward = torch.log(alphas[L-1, T-1, 0] + sum + alphas[L-3, T-1, 0])
        else:
            llForward = torch.log(alphas[L-1, T-1, 0] + sum + alphas[L-3, T-1, 0] + alphas[L-4, T-1, 0])
    else:
        llForward = torch.log(alphas[L-1, T-1, 0] + alphas[L-2, T-1, 0])

    return -llForward

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-json>.\n \
        , it analyzes the GOP with respect to context length") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
    ### the length of the surrounding tokens
 
    processor = My_Wav2Vec2Processor.from_pretrained(prep_path) 
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens
    count = 0
    #target = 0
    #target = "fabm2ab2"
    target = "fabm2ay2"
    #target = "fclm2ah1"
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
                break
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
            pid_seq = processor.tokenizer.convert_tokens_to_ids(raw_seq)
            ###check length
            if context_len >= len(pid_seq) + 1:
                print("not enough context in the utterance")
                break
            print(f"testing GOP of conext length: {context_len}")
            ##run viterbi and get the dict of range of each token
            token_range = get_token_range(post_mat, pid_seq) 
                
            ## for specific phoneme
            cur_p = pid_seq[p_index]
            context_range =  int(context_len/2)
            if p_index - context_range < 0 or p_index + context_range > len(pid_seq) - 1:
                print("not enough left or right context")  
                break
            print(f"testing the erronous phoneme: {processor.tokenizer._convert_id_to_token(cur_p)}")
            for i in range(1, frame_rep+1):
                labels = torch.Tensor(pid_seq[p_index - context_range:p_index + context_range + 1]).type(torch.int32) ## start from no context
                empty_range_l = int(sum(token_range[(2*(p_index - context_range) + 1) - 1])/2)
                empty_range_r = int(sum(token_range[(2*(p_index + context_range) + 1) + 1])/2)
                target_range_l, target_range_r = token_range[(2*(p_index + context_range) + 1)]
                repeated = get_rep_tensor(post_mat.transpose(0,1),target_range_l, target_range_r, empty_range_l, empty_range_r, i)
                #pdb.set_trace()
                ll_self = ctc_loss(repeated, labels, blank=0)
                ll_denom = ctc_loss_denom(repeated, labels, context_range, blank=0)
                gop = -ll_self + ll_denom
                print(f"GOP with frame repetition {i}: {gop} num: {ll_self} denom:{ll_denom}") 
                #print(f"GOP with frame repetition {i} after normalization: {gop/i}") 
            break
       







    
  