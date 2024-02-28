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
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set("sil")

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')


def writes(gop_list, outFile):
    #outFile = "./output-gop-nodecode/all_cmu.gop"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, score in gop_list:
            fw.write("%s %.3f\n"%(key, score))
  
    
def read_trans(trans_path):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                trans_map[cur_uttid] = []
                cur_uttid = items[0]
            if items[4] not in sil_tokens:
                trans_map[cur_uttid].append(re_phone.match(items[4]).group(1))
    return trans_map 

###simplifed, because of frame-wise CE training, no "-" symbols will be outputed
def collect_stats_avg_posterior(post_mat, pid_seq, pid_set, gops_map):
    for order, (pid,start_idx, end_idx) in enumerate(pid_seq):
        length_seg = end_idx - start_idx
        if pid in pid_set:
            for pid_inner in pid_set:
                post_avg = post_mat[start_idx:end_idx][:,pid_inner].mean()
                gops_map[pid][pid_inner].append(post_avg)   


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
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

    return ds_filtered

    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <out-file>.\n \
        , it generates the GOP for utterence level evaluation, using the log of total likelihood of a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1]) 
    uttid_list = tran_map.keys()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
 
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-nor/processor")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-nor/processor")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path)
    pdb.set_trace()
    #cuda = torch.device('cuda:1')
    
    p_set = set(p_tokenizer.encoder.keys()) - spec_tokens - sil_tokens
    #count = 0
    with torch.no_grad():
        pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_list = []  # (uttid, scores)
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
            lables = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
           
            log_like_total = model(input_values, lables = lables)["loss"].squeeze(0)  
            gops_list.append((row['id'], log_like_total))
 
       

    print("done with GOP computation")
    writes(gops_list, sys.argv[4])






    
  
