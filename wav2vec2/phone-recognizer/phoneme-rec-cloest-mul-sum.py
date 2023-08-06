import sys
import re
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForPreTraining, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import os
import pdb
import torch
import pickle
import numpy as np
import math
import json
from pathlib import Path

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
vowel_set = set(['AA', 'AH', 'AE', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
cons_set = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'W', 'V', 'W', 'Y', 'Z', 'ZH'])
p_set = vowel_set | cons_set

def read_align(align_path):
    utt_list =[]
    with open(align_path, "r") as ifile:
        for line in ifile:
            line = line.strip()
            uttid, phonemes = line.split(' ')[0], line.split(' ')[1:]
            uttid = uttid.strip("lbi-")
            p_list = list(map(lambda x: re_phone.match(x).group(1), phonemes))
            utt_list.append((uttid, p_list))
    return pd.DataFrame(utt_list, columns=('uttid','phonemes')) 

# get the p(phoneme|code) from count file, the format is a normalized torch.tensor with shape=(320,320,N_phoneme)
def read_prior_phoneme(count_path) -> torch.tensor:
    count = torch.load(count_path)  #shape = (N_phoneme, 320, 320)
    permuted = count.permute(1,2,0)
    mask = permuted.sum(-1) > 0
    permuted[mask] = permuted[mask] / permuted[mask].sum(-1, keepdim=True)  
    return permuted

   
def compute_cos_compact(X, Y, dim=-1): 
    #X.shape = (seq_len,256) Y.shape = (320,320,256) return shape = (seq_len,320,320)
    split_n = 10
    res = torch.Tensor()
    t_list = []
    for batch in X.split(split_n): #shape(batch) = (split_n, 256)
        t_list.append(torch.cosine_similarity(batch.view(-1,1,1,256), Y, -1)) #(320,320)
    res = torch.cat(t_list, 0)
    return res        
 
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <train/test/dev> <num-utts> <phoneme-alignment file> <code-count-file> <wav2vec2-dir>. It uses the code-prior-file for phoneme recognition, it outputs a file with transcribed phoneme sequence and computes the frame-wise PER based on alignment file")

    ali_df = read_align(sys.argv[3]) 
    # load prior and the pretrained model
    prior_phoneme = read_prior_phoneme(sys.argv[4]) #prior: #shape = (N_phoneme, 320, 320)
    model_path = sys.argv[5]
    #model_path = ""
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data/phoneme-tokenizer")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    if sys.argv[1] != "train.100" and sys.argv[1] != "validation":
        sys.exit("The first argument must be validation or train.100") 
    ds = datasets.load_dataset("librispeech_asr", "clean", split=sys.argv[1])
    #ds = datasets.load_dataset("librispeech_asr", "clean", split="train")
    #ds = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    #get the array for single row
    def map_to_array(batch):
        print("process once")    
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        return batch

    dataset = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    #cuda = torch.device('cuda:1')
    with torch.no_grad():
        #step 0 project the codewords to the space for comparison
        codewords = model.quantizer.codevectors #torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        num_groups = model.quantizer.num_groups
        if num_groups != 2:
            sys.exit("the codebook must have exactly two groups of codes, otherwise not implemented")
        num_vars = model.quantizer.num_vars
        uttid_list = ali_df['uttid'].unique()

        #compute the projected centroids
        codewords = model.quantizer.codevectors #torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        idx_pairs = torch.cartesian_prod(torch.arange(num_vars), torch.arange(num_vars))
        c_view = codewords.view(1,num_groups, num_vars, -1) #shape = (1,2,320,128)
        concat_list = [] #shape =(1,320^2,256)
        for idx_pair in idx_pairs:
            concat_code = torch.Tensor()
            for i,idx in zip(range(num_groups), idx_pair):
                concat_code = torch.cat((concat_code, c_view[0][i][idx]), 0) #shape = (256)
            concat_list.append(concat_code)
        concat_tensor = torch.stack(concat_list,0) #shape =(320^2,256)
        projected_cws = model.project_q(concat_tensor).view(num_vars, num_vars, -1) #shape=(320,320,256)
        #starting processing
        #cnt=0
        total_frames=0
        total_error=0
        total_frames_nonzero=0
        total_error_nonzero=0
        for row in dataset:
            #if cnt >=2:
            #    break
            #cnt+=1
            print("processing {}".format(row["id"]))
            if row["id"] not in uttid_list:
                print("not found in the alignment, skipped")
                continue

            p_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values   
            results = model(input_values)
            projected_vector = results.projected_states.squeeze(0) #shape = (seq_length, 256)
            sim_mat = compute_cos_compact(projected_vector.float(), projected_cws.float(), dim=-1) #shape = (seq_len,320,320)
            weighted_prior = sim_mat.view(-1,320,320,1)*prior_phoneme #shape = (seq_len,320,320,N_phonemes)
            prior_seq = weighted_prior.sum(dim=(1,2)) ### shape = (N_frames, N_phonemes
            id_seq = prior_seq.max(-1)[1]
            assert len(p_seq) == len(id_seq)
            ref_id_seq = p_tokenizer.convert_tokens_to_ids(p_seq)
            assert None not in ref_id_seq
            ref_id_seq = torch.Tensor(ref_id_seq)
            total_frames += len(p_seq)
            total_error += (ref_id_seq != id_seq ).sum()
            non_zero_idx = id_seq.nonzero().squeeze(-1)
            total_frames_nonzero += len(non_zero_idx)
            total_error_nonzero += (ref_id_seq[non_zero_idx] != id_seq[non_zero_idx] ).sum()
            
        print("Total: {0}, Error: {1}, The PER of {2} ({3} utterances) is {4}".format(total_frames, total_error, sys.argv[1], sys.argv[2], torch.round(total_error/total_frames, decimals=3)))
        print("Total-non-zero: {0}, Error: {1}, The PER of {2} ({3} utterances) is {4}".format(total_frames_nonzero, total_error_nonzero, sys.argv[1], sys.argv[2], torch.round(total_error_nonzero/total_frames_nonzero, decimals=3)))
           

