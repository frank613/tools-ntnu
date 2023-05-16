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

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')

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

 
def writes(gop_list, key_list, outFile):
    assert(len(gop_list) == len(key_list))
    #outFile = "./output-gop-nodecode/all_cmu.gop"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, gop in zip(key_list, gop_list):
            fw.write(key+'\n')
            for cnt, (p,score, n, d) in enumerate(gop):
                fw.write("%d %s %.3f\n"%(cnt, p, score))
            fw.write("\n")
   
def compute_cos_compact(X, Y, dim=-1): 
    #X.shape = (seq_len,256) Y.shape = (320,320,256) return shape = (seq_len,320,320)
    res = torch.Tensor()
    t_list = []
    for i in range(X.shape[0]):
        t_list.append(torch.cosine_similarity(X[i], Y, -1)) #(320,320)
    res = torch.stack(t_list, 0)
    return res        
 
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <phoneme-alignment file> <code-prior-file> <wav2vec2-dir> <out-file>.\n \
        , it computes the gop based on the segmentation from the alignment and the distance metric from the wav2vec2 hidden space")

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
    gop_list = []
    key_list = []
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
        #starting processing
        for row in dataset:
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
                    if len(segmented) == 0: 
                        temp = ph
                        idx_start = i
                        continue
                    segmented.append((temp, idx_start, i))
                    temp = ph
                    idx_start = i
            segmented.append((temp, idx_start, i+1))
            input_values = processor(row["speech"], return_tensors="pt").input_values   
            #step 2 get the transoformed and closest vector of the output for each frame, disable time mask using model.eval(), no padding
            results = model(input_values) 
            projected_vector = results.projected_states.squeeze(0) #shape = (seq_length, 256)
            #pv_view = projected_vector.view(projected_vector.shape[0], 1, 1, -1) #shape = (seq_len, 1, 1, 256) for boradcasting
            closest_vector = results.projected_quantized_states.squeeze(0)    #shape = (seq_len,256)  
            #step 3 (numerator)calculate the similarity compared to the transformed codebook centriods for each frame given the prior
            #similarity_num = torch.cosine_similarity(pv_view.float(), projected_cws.float(), dim=-1)  #shape = (seq_len,320,320)
            similarity_num = compute_cos_compact(projected_vector.float(), projected_cws.float(), dim=-1) #shape = (seq_len,320,320)
            weights = [torch.Tensor(prior[p]) for p in p_seq] # list(seq_len) of numpy 2D tensor
            weights_tensor = torch.stack(weights, 0) #shape = (seq_len, 320, 320)
            assert(weights_tensor.shape[0] == similarity_num.shape[0])
            sim_num = (weights_tensor * similarity_num).sum(dim=(1,2)) #shape = (seq_len)
            #step 4 (denominator) faiss(PQ with/without slice?) neareast search
            sim_denom = torch.cosine_similarity(projected_vector, closest_vector) #shape= (seq_len)
            #step 5 gop calculation
            #sim_diff = torch.log(sim_num) - torch.log(sim_denom) #shape= (seq_len)
            sim_diff = sim_num - sim_denom #shape= (seq_len)
            key_list.append(row["id"])
            gop_scores = []
            for phoneme,start_idx,end_idx in segmented:
                gop_scores.append((phoneme, sim_diff[start_idx:end_idx].mean()))
            gop_list.append(gop_scores)
            
    
    #dump the distribution
    writes(gop_list, key_list, sys.argv[4])
