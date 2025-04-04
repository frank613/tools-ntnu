import sys
from pathlib import Path
import torch
import logging
import numpy as np
import pandas as pd
import re
import math
import os
import datasets
from vall_e.utils import to_device, set_seed, clamp, wrapper as ml
from vall_e.config import cfg
from vall_e.engines import load_engines
from vall_e.data import get_phone_symmap, get_lang_symmap, tokenize, text_tokenize, sentence_split
from vall_e.emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
from vall_e.emb import g2p, qnt
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer
from tqdm import trange, tqdm

import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

ds_data_path = '/home/xinweic/cached-data/vall-e-mdd/data'
ds_cache_path = "/home/xinweic/cached-data/vall-e-mdd/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)/(.*)\..*')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)

##copied from inference.py
def encode_text(text, phn_symmap, language="auto", precheck=True, phonemize=True ):
    # already a tensor, return it
    if isinstance( text, torch.Tensor ):
        return text
    # check if tokenizes without any unks (for example, if already phonemized text is passes)
    if precheck and "<unk>" in phn_symmap:
        tokens = tokenize( text )
        if phn_symmap["<unk>"] not in tokens:
            return torch.tensor( tokens )

    if not phonemize:
        return torch.tensor( text_tokenize( text ) )

    return torch.tensor( tokenize( g2p.encode(text, language=language) ) )

##copied from inference.py
def encode_lang(language):
    symmap = get_lang_symmap()
    id = 0
    if language in symmap:
        id = symmap[language]
    return torch.tensor([ id ])

##copied from inference.py
def encode_audio(path, trim_length=0.0 ):
    # already a tensor, return it
    if isinstance( path, torch.Tensor ):
        return path

    # split string into paths
    if isinstance( path, str ):
        path = Path(path)

    prom = qnt.encode_from_file(path)
    if hasattr( prom, "codes" ):
        prom = prom.codes
    prom = prom[0][:, :].t().to(torch.int16)
    res = prom
    if trim_length:
        res = repeat_extend_audio( res, int( cfg.dataset.frames_per_second * trim_length ) )
    return res

#copied from ar_nar.py
def load_artifact(path):
    artifact = np.load(path, allow_pickle=True)[()]
    audio = torch.from_numpy(artifact["codes"].astype(np.int16))[0, :, :].t().to(dtype=torch.int16, device=cfg.device)
    return audio

#def get_text(text_in, device, phn_symmap):
def get_text(text_in, phn_symmap, lang_code):
    lines = sentence_split(text_in)
    assert len(lines) == 1
    text = lines[0]
    #lang = g2p.detect_language(text)
    phns = encode_text( text, phn_symmap, language=lang_code)
    #phns = phns.to(device=device, dtype=torch.uint8 if len(phn_symmap) < 256 else torch.int16)
    phns = phns.to(dtype=torch.uint8 if len(phn_symmap) < 256 else torch.int16)
    lang = encode_lang(lang_code)
    #lang = lang.to(device=device, dtype=torch.uint8)
    lang = lang.to(dtype=torch.uint8)
    return phns,lang

##return audio prompt(or noisy prompt) and full sequence as resp
#def get_emb(audio_in_path, device, trim_length=0, remove_last_n=0, noise=False):
def get_emb(audio_in_path, trim_length=0, noise=False):
    full = encode_audio(audio_in_path) 
    if not noise:
        res = encode_audio(audio_in_path, trim_length=trim_length)
    else:
        noisy_code = load_artifact(Path("/home/xinweic/git-repos/vall-e/data/noise.enc"))
        if trim_length:
            res = repeat_extend_audio( noisy_code, int( cfg.dataset.frames_per_second * trim_length ) )
        else:
            res = noisy_code
    #return res.to(device=device, dtype=torch.int16), full.to(device=device, dtype=torch.int16)
    return res.to(dtype=torch.int16), full.to(dtype=torch.int16)
 
def read_ctm(ctm_path, tokenizer):
    ret_dict = {}
    phone_list = []
    cur_uttid = None
    with open(ctm_path, "r") as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split(' ')
            assert len(fields) == 5
            uttid, start, end, phoneme = fields[0], float(fields[2]), float(fields[2]) + float(fields[3]), fields[4]
            #uttid = uttid.strip("lbi-")
            phoneme = re_phone.match(phoneme).group(1)
            pid = tokenizer._convert_token_to_id(phoneme)
            if uttid != cur_uttid and cur_uttid is not None:
                ret_dict.update({cur_uttid: phone_list})
                phone_list = []
            cur_uttid = uttid           
            phone_list.append([pid, start, end])             
        if not cur_uttid in ret_dict:
            ret_dict.update({cur_uttid: phone_list})
    return ret_dict

##segmentation, new phonemes are annotated as "_N"
# def seg_to_token_seq(p_tokenizer,p_seq):
#     segmented = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
#     temp,idx_start = '', 0
#     for i,p in enumerate(p_seq):
#         if p.endswith('_#'):
#             pid = p_tokenizer._convert_token_to_id(p.strip("_#"))
#             segmented.append([temp, idx_start, i])
#             temp = pid
#             idx_start = i
#     segmented.append([temp, idx_start, i+1])
#     segmented = segmented[1:]
#     return segmented

def read_trans(tran_path):
    tran_map = {}
    with open(tran_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            uttid, sent = line.split(' ')[0], line.split(' ')[1:]
            tran_map.update({uttid: (' '.join(sent)).lower()})
    return tran_map

def read_dur(dur_path):
    dur_list = []
    with open(dur_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            fields = line.split(' ')
            if len(fields) != 2:
                sys.exit("Kaldi format utt2dur format")
            uttid, dur = fields[0], float(fields[1])
            dur_list.append((uttid, dur))
    return dur_list

## convert resolution from ctm alignment to target code rate(in frame-shift/unit)
def resol_conversion(pid_seq, rate_target):
    #pdb.set_trace()
    pid_seq_ret = []
    ms_frame = round(1000/rate_target, 5)
    ## *2 because ctm file uses default frame_shift = 10
    total_len = math.floor(float(pid_seq[-1][2]*2*1000 / ms_frame))
    time_end = [ e for id,s,e in pid_seq ] 
    total_time = time_end[-1]
    frame_end = [ math.ceil((time/total_time)*total_len) for time in time_end ]  ### ceiling for covering a wider range of code in case of phoneme state transition 
    start = 0
    for (pid,_,_),frame_e in zip(pid_seq, frame_end):
        pid_seq_ret.append([pid, start, frame_e])
        start = frame_e - 1  ### -1 for covering a wider range of code in case of phoneme state transition 
    return pid_seq_ret
 
### non-batch version
def get_avg_posterior(model, text_in, prop_in, resp_in, lang, pid_seq, cmp_len=False, is_ar_level_0 = False, masking_nar_level_0 = True, total_levels=0):
    pid_seq = resol_conversion(pid_seq, rate_target=cfg.dataset.frames_per_second) ## 75 for current config
    ##kaldi will randomly cut 10ms/20ms(1 or 2 frames) in the number of MFCC features, so we extend the SIL(or other phonemes in the last) to match the number of codes  
    frame_diff = resp_in.shape[0] - pid_seq[-1][-1]
    if frame_diff > 5 or frame_diff < 0:
        sys.exit("problem with length of CTM and encodec output")
    #elif frame_diff > 0 and pid_seq[-1][0] == sil_token_id:
    elif frame_diff > 0:
        pid_seq[-1][-1] = pid_seq[-1][-1] + frame_diff   
    assert pid_seq[-1][-1] == resp_in.shape[0] 
    b_size = len(pid_seq)
    if cmp_len:
        pass
        #lenth_in = resp_in.shape[0]
        #len_list = [ lenth_in ] * b_size
    else:
        len_list = None
    if masking_nar_level_0:
        ## to batch for MDD
        input_kwargs = dict(
                    text_list=[text_in] * b_size, 
                    task_list=["tts"] * b_size,
                    raw_text_list=None,
                    proms_list=[prop_in] * b_size,
                    resps_list = [resp_in] * b_size,
                    lang_list=[lang] * b_size,
                    disable_tqdm=False,
                    use_lora=True,
                    is_mdd=True,
                    is_masking_nar_level_0=True,
                    pid_seq = pid_seq,
                    total_levels = total_levels
                )
    else: ## single processing
        sys.exit("for now, only support masking")
    if total_levels < 0 or total_levels > model.config.resp_levels:
        sys.exit("specify a correct level range, usually from [0,7]")
    ##first level
    if not is_ar_level_0: ## len+NAR
        if not 'nar' in model.config.capabilities:
            sys.exit("the model does not support NAR")
        if cmp_len:
            ## predict len
            #kwargs = {"temperature": 2}
            #len_list = model( **input_kwargs, task_list=["len"], **{"max_duration": 10, "temperature": 2} )
            sys.exit("not yet support to compare length")
               
        ## NAR+len, return a list of avg-posterior, the length is based on total_levels
        ret_value = model( **input_kwargs)
    else:
        sys.exit("not supporting AR+NAR in this version")
    # pdb.set_trace()
    # _logger.info(f"MDD done")
    return ret_value
    
def load_dataset_local_from_dict(csv_path, cache_additional, trans_map, uttid_list, lang_code, subset=None, last=None):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    lang_code = lang_code
    if not os.path.exists(cache_full_path):
        datadict = {"audio": []}  
        #with open(folder_path + '/metadata.csv') as csvfile:
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
                #datadict["audio"].append(folder_path + '/' + row.split(',')[0])
                datadict["audio"].append(row.split(',')[0])
        ds = datasets.Dataset.from_dict(datadict) 
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=24000))
        ## do further transformation 
        def map_to_array(batch):   
            batch["speech"] = [ item["array"] for item in batch["audio"] ]
            batch["id"] = [re_uttid_raw.match(item["path"])[2] for item in batch["audio"]]
            batch["phns"] = []
            batch["lang"] = []
            batch["prompt"] = []
            batch["resp"] = []
            phn_symmap = get_phone_symmap()
            for i, uid in enumerate(batch["id"]):
                if uid not in uttid_list:
                    batch["phns"].append(None)
                    batch["lang"].append(None)
                    batch["prompt"].append(None)
                    batch["resp"].append(None)
                else:
                    text_in = trans_map[uid]
                    phns, lang = get_text(text_in, phn_symmap, lang_code)
                    batch["phns"].append(phns)
                    batch["lang"].append(lang)
                    ##prompt
                    audio_in_path = batch["audio"][i]["path"]
                    prompt, resp = get_emb(audio_in_path, trim_length=3, noise=False)
                    batch["prompt"].append(prompt)
                    batch["resp"].append(resp)
            return batch
        ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
        ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['phns']], batched=True, batch_size=100, num_proc=10)
        ds_filtered.save_to_disk(cache_full_path)
        
    ds_filtered = datasets.Dataset.load_from_disk(cache_full_path)
    if subset is not None:
        ds_filtered = ds_filtered.filter(lambda batch: [ item in subset for item in batch['id']], batched=True, batch_size=100, num_proc=10)
    if last is not None:
        last_index = -1 
        for i, uid in enumerate(ds_filtered["id"]):
            if uid == last:
                last_index = i
                break
        if last_index == -1:
            sys.exit("last not found, check input")
        ds_filtered = ds_filtered.select(range(last_index+1, len(ds_filtered)))
    return ds_filtered
   
# def single_process(example, p_tokenizer, device, out_path):
#     #pdb.set_trace()
#     model = model_globle
#     row = example
#     proc_id = str(os.getpid())
#     uid = row['id']
#     print("processing {0}".format(uid))
#     #return None
#     #if uid == "fabm2bz1":
#     #    pdb.set_trace()
#     with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:    
#         pid_seq = ctm_dict[uid]
#         phns = torch.tensor(row["phns"], device=device, dtype=torch.int16)
#         prompt = torch.tensor(row["prompt"], device=device, dtype=torch.int16)
#         resp = torch.tensor(row["resp"], device=device, dtype=torch.int16)
#         lang = torch.tensor(row["lang"], device=device, dtype=torch.uint8)
#         avg_post_list = get_avg_posterior(model, phns, prompt, resp, lang, pid_seq)
#         ## convert L*T list to T*L
#         avg_post_list = [list(x) for x in zip(*avg_post_list)]
#         assert len(pid_seq) == len(avg_post_list)
#         ### write files       
#         f.write(row['id']+'\n')
#         for i, avg_post in enumerate(avg_post_list):
#             gop = ",".join([ str(item) for item in avg_post])
#             f.write("%d %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid_seq[i][0])), gop))
#         f.write("\n")

def batch_process(batch, p_tokenizer, device, out_path=None):
    engines = load_engines(training=False, is_mdd=True)
    assert len(engines) == 1
    models = []
    for name, engine in engines.items():
        if type != torch.int8:
            models.append(engine.module.to(device, dtype=dtype if not amp else torch.float32))           
    model = models[0]
    model.eval()
    proc_id = str(os.getpid())
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:
        for i,uid in enumerate(batch["id"]):
            print("processing {0}".format(uid))    
            pid_seq = ctm_dict[uid]
            phns = torch.tensor(batch["phns"][i], device=device, dtype=torch.int16)
            prompt = torch.tensor(batch["prompt"][i], device=device, dtype=torch.int16)
            resp = torch.tensor(batch["resp"][i], device=device, dtype=torch.int16)
            lang = torch.tensor(batch["lang"][i], device=device, dtype=torch.uint8)
            avg_post_list = get_avg_posterior(model, phns, prompt, resp, lang, pid_seq, total_levels=8)
            ## convert L*T list to T*L
            avg_post_list = [list(x) for x in zip(*avg_post_list)]
            assert len(pid_seq) == len(avg_post_list)
            ### write files      
            f.write(uid+'\n')
            for i, avg_post in enumerate(avg_post_list):
                gop = ",".join([ str(item) for item in avg_post])
                f.write("%d %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid_seq[i][0])), gop))
            f.write("\n")
    load_engines.cache_clear()
    unload_model()
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 8:
        sys.exit("this script takes 7 arguments <model-ckpt-sft> <in-audio-csv-folder> <in-raw-text-file> <CTM-alignment-file> <gop-tokenizer-path> <utt2dur> <out-gop-path> \n \
        , it loads the TTS model and compute the GOP")
           
    ## default cfg and then update cfg from model, similar to inferece.py
    cfg.load_model(Path(sys.argv[1]))
    cfg.format( training=False )
    #cfg.device = "cpu"
    ## cfg related attributes
    dtype = cfg.inference.dtype
    amp = cfg.inference.amp
    device = cfg.device
    
    ## load the model and engine(engine helps to create model and load from stat_dict)
    #cfg.ckpt_dir = Path(sys.argv[1])
    # engines = load_engines(training=False, is_mdd=True)
    # assert len(engines) == 1
    # models = []
    # for name, engine in engines.items():
    #     if type != torch.int8:
    #         models.append(engine.module.to(device, dtype=dtype if not amp else torch.float32))
            
    # model_globle = models[0]
    # model_globle.eval()
    #_logger.info(f"model loaded")
    model = None

    ##load alignment and transcription
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(sys.argv[5])
    trans_map = read_trans(sys.argv[3])
    ctm_dict = read_ctm(sys.argv[4], p_tokenizer)
    uttid_list = list(ctm_dict.keys())
    dur_list = read_dur(sys.argv[6])
    subset_list = [ uttid for uttid, dur in dur_list if dur < 12 ]
    csv_path = Path(sys.argv[2])
    
    out_path = sys.argv[7]
    last_utt = "mjsd3ac2"
    
    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids", trans_map, uttid_list, lang_code="en-us", subset=subset_list, last=last_utt)
    # ds could be loaded from disk, need to move the tensors to device 
    #ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "model":model, "device":device, "out_path":out_path}, num_proc=2) 
    ds.map(batch_process, fn_kwargs={"p_tokenizer":p_tokenizer, "device":device, "out_path":out_path}, batched=True, batch_size=100, num_proc=1)
    
    #pdb.set_trace()
    # iter=tqdm(ds.iter(batch_size=1), total = len(ds))
    # for i in iter:
    #     row = i
    #     proc_id = str(os.getpid())
    #     uid = row['id'][0]
    #     print("processing {0}".format(uid))
    #     #if uid == "fabm2bz1":
    #     #    pdb.set_trace()
    #     with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:    
    #         pid_seq = ctm_dict[uid]
    #         phns = torch.tensor(row["phns"][0], device=device, dtype=torch.int16)
    #         prompt = torch.tensor(row["prompt"][0], device=device, dtype=torch.int16)
    #         resp = torch.tensor(row["resp"][0], device=device, dtype=torch.int16)
    #         lang = torch.tensor(row["lang"][0], device=device, dtype=torch.uint8)
    #         avg_post_list = get_avg_posterior(model, phns, prompt, resp, lang, pid_seq)
    #         ## convert L*T list to T*L
    #         avg_post_list = [list(x) for x in zip(*avg_post_list)]
    #         assert len(pid_seq) == len(avg_post_list)
    #         ### write files       
    #         f.write(uid+'\n')
    #         for i, avg_post in enumerate(avg_post_list):
    #             gop = ",".join([ str(item) for item in avg_post])
    #             f.write("%d %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid_seq[i][0])), gop))
    #         f.write("\n")
    #         torch.cuda.empty_cache()
               
    ##unload qnt models
    # load_engines.cache_clear()
    # unload_model()
    