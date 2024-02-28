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

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')


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
            if phoneme not in (sil_tokens | spec_tokens):
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
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)
    #ds_filtered = ds_map

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
                    #remove the path duplicated with the "skip" path from alphas[s,t-1], t-2 is a bug when t=1?
                    #removed1 =  alphas[s,t-1] - alphas[s-2,t-2] * params[blank,t-1]
                    #fixing the bug?
                    if t == 1:
                        removed1 =  alphas[s,t-1] - alphas[s-2,t-2] * params[blank,t-1]
                        #the wild card state must remove the "blank" path from the previous time(t-1), because it's overlapped with the first term:  alphas[s,t-1]
                        removed2 =  alphas[s-1,t-1] - alphas[s-1,t-2] * params[blank,t-1]
                    else: # t=1
                        removed1 = 0
                        removed2 = alphas[s-1,t-1] ## already removed blank at initial step
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
                #removed = alphas[s-2,t-1]*(1 - params[blank,t-1] - params[seq[l],t-1] ) ## remove the blank token (already considered in the blank state) and the duplicated label of t-1, compare to the last equation
                if t == 1 and s == 3: # the only possible path is from s=1(pos) -> s=3
                    removed = alphas[s-2,t-1] - params[seq[l],t-1]
                    alphas[s,t] =  removed * params[seq[l],t]
                else:    
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
       
    if pos == seqLen-1:  ## this already contains the probability of being blank tokens as end of the sequence, L-3 and L-4 are also possible end states(same as what we do in alfree-v2, allow deletion/skip )
        #llForward = torch.log(alphas[L-2, T-1] + alphas[L-3, T-1])
        ##we need to compute the last two terms explicitly for T-2, because they are pruned at T-1 due to the original algorthm
        llForward = torch.log(alphas[L-2, T-1] + alphas[L-3, T-2]* params[blank,T-1] + alphas[L-4, T-2]*(params[blank,T-1] +  params[seq[-2],T-1]))
        ##since we already allow skip, there we can skip from L-4 to L-2, so we can use the the same equation as below?
        ##llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])
    else:  
        llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])

    return -llForward
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-file>.\n \
        , it generates the GOP using a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv") 
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
    ds= load_dataset_local_from_dict(csv_path)
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens - sil_tokens
    #count = 0
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_list = []  # (uttid, (phoneme, scores))
        for row in ds:
            #count += 1
            #if count > 10:
            #    break
            #if row['id'] != 'fabm2cy2':
            #    continue
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no transcription can be found")
                continue
            print("processing {0}".format(row['id']))
            #get the total likelihood of the lable
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            #the default loss here in the config file is "ctc_loss_reduction": "sum" 
            labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
            labels = labels.type(torch.int32)
            ##return the log_like to check the correctness of our function
            return_dict = model(input_values, labels = labels)
            log_like_total = return_dict["loss"].squeeze(0)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1).type(torch.float64)
            ll_self = ctc_loss(post_mat.transpose(0,1), labels, blank=0)
            #ll_self = ctc_loss_scaled(post_mat.transpose(0,1), labels, blank=0)
            llDiff = np.abs(log_like_total - ll_self)
            if llDiff > 1 :
                print(f"model ll: {log_like_total}, function ll: {ll_self}")

            #step 2, compute the GOP
            pids = labels.tolist()
            gop_list = []
            for i,pid in enumerate(pids):
                ll_denom = ctc_loss_denom(post_mat.transpose(0,1), labels, i, blank=0)
                gop = -ll_self + ll_denom
                gop_list.append((p_tokenizer._convert_id_to_token(pid), gop))
            gops_list.append((row['id'], gop_list))
 
       

    print("done with GOP computation")
    writes(gops_list, sys.argv[5])






    
  
