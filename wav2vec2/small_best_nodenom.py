import sys
import re
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForPreTraining, Wav2Vec2Processor
import os
import pdb
import torch
import pickle
import numpy as np
import json
from sklearn import metrics
import math

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

def read_prior(prior_path) -> Dict[str, np.ndarray]:
    with open(prior_path, "rb") as file:
        Phone_Matrix = pickle.load(file)
    phonemes = dir(Phone_Matrix)[:41]
    new_dict = {}
    for phoneme in phonemes:
        Phone_M = Phone_Matrix.current(phoneme)
        count = Phone_M.sum()
        new_dict.update({phoneme:Phone_M/count})
    return new_dict

def compute_entropy(in_list, minV=-20, maxV=0, nBin=50):

    copied = in_list.copy()
    for n,value in enumerate(copied):
        if value < minV:
            copied[n] = minV
        elif value > maxV:
            copied[n] = maxV
        else:
            pass
    hist1 = np.histogram(copied, bins=nBin, range=(minV,maxV), density=True)
    stats = hist1[0]
    stats = stats[stats!=0]
    ncat = len(stats)
    stats = stats/stats.sum()
    ent = round(-(stats*np.log(np.abs(stats))).sum(), 3)
    ##scale it back to 0-1
    if ncat == 1:
        pass
    else:
        ent = round(ent/np.log(ncat),3)
    return (np.float64(ent),ncat)

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3)
 
def writes(gops_maps, outFile):
    for key,gops_map in gops_maps.items():
        out_form = { \
                    'phonemes':{},
                    'summary': {"average-mean-diff": None, "average-AUC": None, "total_real": None, "total_error":None}}
        total_real = 0
        total_error = 0
        total_auc = 0
        total_mean_diff = 0
        for (k,v) in gops_map.items():
            real_arr = np.array(gops_map[k][k])
            if len(real_arr) == 0:
                continue
            ent,nbin = compute_entropy(real_arr)
            real_label = np.stack((real_arr, np.full(len(real_arr), 0)), 1)
            scores = []
            total_real += len(real_arr)
            for p in set(gops_map[k].keys()) - set([k]):
                sub_arr = np.array(gops_map[p][k]) #for all the p phonemes that are substituted to k
                if len(sub_arr) == 0:
                    continue
                sub_label = np.stack((sub_arr, np.full(len(sub_arr), 1)), 1)
                auc_value = auc_cal(np.concatenate((real_label, sub_label)))
                scores.append((p, sub_arr.mean(), len(sub_arr), round(auc_value,3)))
                total_error += len(sub_arr)

            if len(scores) == 0:
                continue
            confused_p, p_mean, num_error, auc = sorted(scores, key = lambda x: x[3])[0]
            mean_diff = round(real_arr.mean() - p_mean, 3)
            out_form["phonemes"][k] = (confused_p, np.float64(mean_diff), auc, ent, len(real_arr), num_error)
            total_auc += auc
            total_mean_diff += mean_diff
        out_form["summary"]["average-mean-diff"]=total_mean_diff/len(gops_map.items())
        out_form["summary"]["average-AUC"]=total_auc/len(gops_map.items())
        out_form["summary"]["total_real"]=total_real
        out_form["summary"]["total_error"]=total_error

        os.makedirs(os.path.dirname(outFile), exist_ok=True)
        with open(outFile+'.'+str(key), "w") as f:
            json.dump(out_form, f)

   
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
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <phoneme-alignment file> <code-prior-file> <wav2vec2-dir> <out-json>. It computes the gop using wav2vec2 and analyzes the performance with synthesized data")

    ali_df = read_align(sys.argv[1]) 
    # load prior and the pretrained model
    prior = read_prior(sys.argv[2]) #prior: dictionary of 2D array for each phoneme
    model_path = sys.argv[3]
    #model_path = ""
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds = datasets.load_dataset("librispeech_asr", "clean", split="validation")
    #ds = datasets.load_dataset("librispeech_asr", "clean", split="train")
    #ds = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    #get the array for single row
    def map_to_array(batch):
        print("process once")    
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        return batch

    dataset = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=10)
    #cuda = torch.device('cuda:1')
    with torch.no_grad():
        #step 0 project the codewords to the space for comparison
        codewords = model.quantizer.codevectors #torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        num_groups = model.quantizer.num_groups
        if num_groups != 2:
            sys.exit("the codebook must have exactly two groups of codes, otherwise not implemented")
        num_vars = model.quantizer.num_vars
        idx_pairs = torch.cartesian_prod(torch.arange(num_vars), torch.arange(num_vars))
        c_view = codewords.view(1,num_groups, num_vars, -1) #shape = (1,2,360,128)
        concat_list = [] #shape =(1,360^2,256)
        for idx_pair in idx_pairs:
            concat_code = torch.Tensor()
            for i,idx in zip(range(num_groups), idx_pair):
                concat_code = torch.cat((concat_code, c_view[0][i][idx]), 0) #shape = (256)
            concat_list.append(concat_code)
        concat_tensor = torch.stack(concat_list,0) #shape =(360^2,256)
        projected_cws = model.project_q(concat_tensor).view(num_vars, num_vars, -1) #shape=(320,320,256)
        uttid_list = ali_df['uttid'].unique()

        #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
        #Plot the mean for all the other model and compute the overall AUC for that phoneme
        threshold = [0,0.0001,0.001,0.01,0.05,0.1]
        gops_map = { th:{ p1:{ p2: [] for p2 in p_set } for p1 in p_set } for th in threshold }  # map(p:map(p:average)
        #starting processing
        cnt=0
        for row in dataset:
            #if cnt >=1:
            #    break
            cnt+=1
            print("processing {}".format(row["id"]))
            if row["id"] not in uttid_list:
                print("not found in the alignment, skipped")
                continue
            #step 1 segmentation of phonemes
            p_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            segmented = []
            temp,idx_start = '', 0
            for i,ph in enumerate(p_seq):
                if temp != ph:
                    segmented.append([temp, idx_start, i])
                    temp = ph
                    idx_start = i
            segmented.append([temp, idx_start, i+1])
            segmented = segmented[1:]
            input_values = processor(row["speech"], return_tensors="pt").input_values   
            #step 2 get the transoformed and closest vector of the output for each frame, disable time mask using model.eval(), no padding
            results = model(input_values) 
            projected_vector = results.projected_states.squeeze(0) #shape = (seq_length, 256)
            #pv_view = projected_vector.view(projected_vector.shape[0], 1, 1, -1) #shape = (seq_len, 1, 1, 256) for boradcasting
            closest_vector = results.projected_quantized_states.squeeze(0)    #shape = (seq_len,256)  
            #step 3 (denominator) faiss(PQ with/without slice?) neareast search
            #sim_denom = torch.cosine_similarity(projected_vector, closest_vector) #shape= (seq_len)
            #step 4 (numerator)calculate the similarity compared to the transformed codebook centriods for each frame given the prior
            similarity_num = compute_cos_compact(projected_vector.float(), projected_cws.float(), dim=-1) #shape = (seq_len,320,320)
            for th in threshold:
                for p in p_set: 
                    weight_tensor = torch.Tensor(prior[p]) #shape = (320,320)
                    weight_tensor = torch.where(weight_tensor >= th, 1, 0)
                    best, _ = sim_num = (weight_tensor * similarity_num).max(-1)
                    best, _ = best.max(-1)
                    #step 5 gop calculation
                    sim_diff = best #shape= (seq_len)
                    for phoneme,start_idx,end_idx in segmented:
                        if phoneme == "SIL" or phoneme == "SPN":
                            continue
                        gops_map[th][phoneme][p].append(sim_diff[start_idx:end_idx].mean())
    #dump the distribution
    writes(gops_map, sys.argv[4])
