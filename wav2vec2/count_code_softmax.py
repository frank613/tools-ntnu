import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForPreTraining, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import pdb
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

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

def read_plist(ptable_path):
    p_list=[]
    with open(ptable_path, "r") as ifile:
        for line in ifile:
            phonemes = line.split('')
            if len(phonemes) != 1:
                sys.exit("bad line in phoneme table")
            p_list.append(phonemes[0]) 
    return p_list
        


if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <phoneme-alignment file> <wav2vec2-dir> <out-file>.\n \
        , it outpus the code count numpy matrix of the train-100h, the matrix's shape = N_phone * 320 * 320")

    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    #phoneme_list = read_plist(sys.argv[2])
    # load pretrained model
    model_path = sys.argv[2]
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data/phoneme-tokenizer")
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    model.eval()
    #shape = (phoneme,320,320)
    results = torch.zeros(len(p_tokenizer.encoder.keys()),320,320)
    ##step 1 load the dataset and change the format
    ds = datasets.load_dataset("librispeech_asr", "clean", split="train.100")
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(ali_df.loc[ali_df.uttid == uid, "phonemes"].to_list()[0])
        return batch

    print("pre-processing")
    dataset = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = dataset.filter(lambda example: example['p_text'] is not None)

    #step 2 process the data in batch
    def process_batch(batch):
        with torch.no_grad():
            print("process one batch")
            input_values = processor(batch["speech"], return_tensors="pt", padding=False, sampling_rate=16000).input_values
            ##get the CNN output
            #cnn_hiddens =  model.wav2vec2.feature_extractor(input_values)
            #cnn_hiddens = cnn_hiddens.transpose(1,2)
            model_results = model.wav2vec2(input_values)
            cnn_hiddens = model.dropout_features(model_results[1])
            ##get the mapped logits
            logits = model.quantizer.weight_proj(cnn_hiddens)
            softmax_mat = logits.view(-1, 2, 320).softmax(dim=-1) #(N,2,320)
            soft1 = softmax_mat[:, 0, :].unsqueeze(2).expand(-1, -1, 320)
            soft2 = softmax_mat[:, 1, :].unsqueeze(1).expand(-1, 320, -1)
            softmax_mat = soft1 * soft2 #shape = (N, 320, 320)
            ##not woking by directly call the tokenizer, word delimeter will be inserted
            #list_tokens = [ ' '.join(seq) for seq in batch["p_text"]] 
            #padded_idx = torch.Tensor(p_tokenizer(list_tokens, padding="longest", is_split_into_words=True)["input_ids"]) ##the code of <pad> is zero, see <vocab.json>, shape = (N,)
            ##padding must operate on dict and the field "input_ids" --- hack
            dict_of_idxlist = {"input_ids":[]}
            for item in batch["p_text"]:
                id_list = p_tokenizer.convert_tokens_to_ids(item)
                assert len(id_list) == len(item) and None not in id_list
                dict_of_idxlist["input_ids"].append(id_list)
            padded_idx = torch.Tensor(p_tokenizer.pad(dict_of_idxlist)["input_ids"])   #shape = (N, 860)
            ##phoneme lables
            ## put the phoneme identity in the one_hot_idx for scatter add (all other code points as well as padded points will be added to the phoenme "zero" group)
            results.scatter_add_(0, padded_idx.view(-1,1,1).expand(-1,320,320).type(torch.int64), softmax_mat)

    

    ds_filtered = ds_filtered.map(process_batch, batched=True, batch_size=1)
    
    #dump the distribution
    pdb.set_trace()
    torch.save(results, sys.argv[3])


