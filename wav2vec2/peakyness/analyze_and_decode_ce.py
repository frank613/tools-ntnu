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
from my_w2v2_package.entropy_loss import ctc_entropy_cost
import my_w2v2_package.entropy_loss as entropy_loss
import pdb

entropy_loss.cuda = False    

ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
sil_tokens = set(["sil", "SIL", "SPN"])
noisy_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "SPN"))
sil_vocab = "SIL"

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
    

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
            if phoneme not in (sil_tokens | noisy_tokens):
                trans_map[cur_uttid].append(phoneme)
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

def remove_duplicate(hyp):
    last_ph = ""
    hyp_new = []
    for ph in hyp.split(' '):
        if ph not in sil_tokens and ph != last_ph:
            hyp_new.append(ph)
        ## if identical phoneme is separted by SIL/SPN, then we don't remove
        last_ph = ph
    return " ".join(hyp_new)



def single_process(example, p_tokenizer, processor, model, sil_token_id, out_path):
    row = example
    proc_id = str(os.getpid())
    # if row["id"] != "fabm2bt2":
    #     return
    print("processing {0}".format(row['id']))
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:
        #f.write(row['id']+'\n')
        #get the total likelihood of the lable
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        #the default loss here in the config file is "ctc_loss_reduction": "sum" 
        labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
        labels = labels.type(torch.int32)
        ##return the log_like to check the correctness of our function
        return_dict = model(input_values, labels = labels)
        logits = return_dict["logits"].squeeze(0) 
        prob = logits.softmax(dim=-1).type(torch.float32)
        ##merge noisy tokens to SIL
        noisy_labels = p_tokenizer.convert_tokens_to_ids(list(noisy_tokens))
        prob[:, sil_token_id] = prob[:,sil_token_id] + torch.sum(prob[:,noisy_labels], axis=-1)
        #step 2, compute the conditoned entropy, here we only use batch_size = 1 
        log_prob = prob.log().type(torch.float32)
        len_labels = torch.Tensor([labels.shape[0]]).type(torch.int)
        len_T = torch.Tensor([log_prob.shape[0]]).type(torch.int)
        ## we don't need conditioned on the label for calculate the vocab-entropy, as it is for training. Here as a measure, unconditioned ensures the number of values are same over all the utterences
        ## For CE we can just specify SIL as blank tokens, optional SIL
        entropy, logP = ctc_entropy_cost(log_prob[:,None], labels, len_T, len_labels, sumed=True, blank=sil_token_id) 
        ##step 3, blank coverage
        bc = prob[:,sil_token_id].mean()
        ##step 4, decoding
        hyp_ids = torch.argmax(logits, dim=-1)
        tokens = processor.tokenizer.batch_decode(hyp_ids)
        hyp = processor.tokenizer.convert_tokens_to_string(tokens,spaces_between_special_tokens=True)
        hyp = remove_duplicate(hyp['text'])
        f.write("%s,%d,%d,%s,%s,%s\n"%(row['id'], len_labels, len_T, entropy.item(), bc.item(), hyp))

        

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <transcription file, kaldi-CTM format> <error-list> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <out-file>.\n \
        , it calculates ConEN, BC and decode for the whole dataset") 
    
    #step 0, read the files
    tran_map = read_trans(sys.argv[1]) 
    uttid_list = tran_map.keys()
    
    error_list = []
    with open(sys.argv[2]) as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split()
            if len(fields) != 1:
                sys.exit("wrong input line")
            error_list.append(fields[0])
    
    # load the pretrained model and data
    model_path = sys.argv[3]
    csv_path = sys.argv[4]
    prep_path = sys.argv[5]
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    sil_id = p_tokenizer._convert_token_to_id(sil_vocab)
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model":model, "sil_token_id":sil_id, "out_path":sys.argv[6]}, num_proc=1) 
    
    print("done")
    
    
       







    
  
