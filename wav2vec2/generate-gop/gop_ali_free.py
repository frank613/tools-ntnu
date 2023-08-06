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
sil_tokens = set(["sil"])

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
                
            if items[4] not in sil_tokens:
                trans_map[cur_uttid].append(re_phone.match(items[4]).group(1))
    return trans_map 


def load_dataset_local_from_dict(folder_path):
    datadict = {"audio": []}  
    with open(folder_path + '/metadata.csv') as csvfile:
        next(csvfile)
        for row in csvfile:
            datadict["audio"].append(folder_path + '/train/' + row.split(',')[0])
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
    #ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)
    ds_filtered = ds_map

    return ds_filtered

 ##modified from Stanford-CTC, return the rescaled alphas and betas
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

    # Initialize betas and backwards pass
    betas[-1,-1] = params[blank,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = betas[:,-1].sum()
    betas[:,-1] = betas[:,-1] / c
    llBackward = torch.log(c)
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

        c = betas[start:end,t].sum()
        betas[start:end,t] = betas[start:end,t] / c
        llBackward += np.log(c)

    # Check for underflow or zeros in denominator of gradient
    llDiff = np.abs(llForward-llBackward)
    if llDiff > 1e-5 :
        print("Diff in forward/backward LL : %f"%llDiff)

    return -llForward,alphas,betas

 
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <out-file>.\n \
        , it generates the GOP using a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1]) 
    uttid_list = tran_map.keys()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
 
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-nor/processor-ctc")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-nor/processor-ctc")
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
            post_mat = logits.softmax(dim=-1)
            ll_self, alphas, betas = ctc_loss(post_mat.transpose(0,1), labels, blank=0)
            llDiff = np.abs(log_like_total - ll_self)
            if llDiff > 1 :
                print(f"model ll: {log_like_total}, function ll: {ll_self}")

            
            T = alphas.shape[1]
            pids = labels.tolist()
            pids_shiftdown = [None] + pids[:-1]
            pids_shiftup = pids[1:] + [None]  
            gop_list = []
            #[alpha_t1(s1) * (1-y(t1+1,P1)] * [(1 - y(t2-1,P2))*beta_t2(s2)]
            #t2 >= t1 + 3
            for i, (p_l,p_m,p_r) in enumerate(zip(pids_shiftdown, pids, pids_shiftup)):
                denom_sum = 0
                start_s, end_s = 2*i-1-1, 2*i-1+1  #2i-1 is the middle phoneme index
                if i == 0:
                    s2 = end_s + 1
                    for t2 in range(1,T): 
                        denom_sum += (1-post_mat[t2-1, p_r]) * betas[s2, t2]
                elif i == len(labels) - 1:
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
                gop_list.append((p_tokenizer._convert_id_to_token(p_m), gop))
            gops_list.append((row['id'], gop_list))
 
       

    print("done with GOP computation")
    writes(gops_list, sys.argv[4])






    
  
