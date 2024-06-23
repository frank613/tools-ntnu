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
from my_w2v2_package.entropy_loss import ctc_prior_entropy_cost
import my_w2v2_package.entropy_loss as entropy_loss
import pdb

entropy_loss.cuda = False    

ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])

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
            if phoneme not in (sil_tokens | spec_tokens):
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
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(tran_map[uid])
        return batch
    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=3)
    #ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered





def single_process(example, p_tokenizer, processor, model, out_path):
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
        ## we use the log version of the code from en-ctc
        log_prob = logits.log_softmax(dim=-1).type(torch.float32)
        #step 2, compute the conditoned entropy, here we only use batch_size = 1 
        pdb.set_trace()
        len_labels = torch.Tensor([labels.shape[0]]).type(torch.int)
        len_T = torch.Tensor([log_prob.shape[0]]).type(torch.int)
        ## we don't need conditioned on the label for calculate the vocab-entropy, as it is for training. Here as a measure, unditioned ensures the number of values are same over all the utterences
        entropy, label_entropy, logP = ctc_prior_entropy_cost(log_prob[:,None], labels, len_T, len_labels, sumed=True, conditioned=False,  blank=p_tokenizer.pad_token_id) 
        f.write("%s %d %d %s %s %s\n"%(row['id'], len_labels, len_T, entropy.item(), label_entropy.item(), logP.item()))

        

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <out-file>.\n \
        , it calculates the conditioned entropy and also vocabulary entropy for the whole dataset") 
    
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
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model":model, "out_path":sys.argv[5]}, num_proc=1) 
    
    print("done")
    
    
       







    
  
