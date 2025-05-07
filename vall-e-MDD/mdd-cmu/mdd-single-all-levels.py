import sys
from pathlib import Path
import torch
import logging
import numpy as np
import pandas as pd
import re
import math
from vall_e.utils import to_device, set_seed, clamp, wrapper as ml
from vall_e.config import cfg
from vall_e.engines import load_engines
from vall_e.data import get_phone_symmap, get_lang_symmap, tokenize, text_tokenize, sentence_split
from vall_e.emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
from vall_e.emb import g2p, qnt
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer

import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
#sil_tokens = set(["sil","SIL","SPN"])
sil_token = "SIL"

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

def get_text(text_in, device, phn_symmap, lang_code):
    lines = sentence_split(text_in)
    assert len(lines) == 1
    text = lines[0]
    #lang = g2p.detect_language(text)
    phns = encode_text( text, phn_symmap, language=lang_code)
    phns = phns.to(device=device, dtype=torch.uint8 if len(phn_symmap) < 256 else torch.int16)
    lang = encode_lang(lang_code)
    lang = lang.to(device=device, dtype=torch.uint8)
    return phns,lang

##return audio prompt(or noisy prompt) and full sequence as resp
def get_emb(audio_in_path, device):
    full = encode_audio(audio_in_path)
    return full.to(device=device, dtype=torch.int16)


##return audio prompt(or noisy prompt) and full sequence as resp
def get_prompt(audio_in_path, device, trim_length=0, noise=False):
    #full = encode_audio(audio_in_path)
    ## remove last n codes, to match the KALDI MFCC feature numbers (dropped last frame) 
    #if remove_last_n:
    #    full = full[:-remove_last_n,:] 
    if not noise:
        res = encode_audio(audio_in_path, trim_length=trim_length)
    else:
        noisy_code = load_artifact(Path("/home/xinweic/git-repos/vall-e/data/noise.enc"))
        if trim_length:
            res = repeat_extend_audio( noisy_code, int( cfg.dataset.frames_per_second * trim_length ) )
        else:
            res = noisy_code
    return res.to(device=device, dtype=torch.int16)
 
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
            tran_map.update({uttid: ' '.join(sent)})
    return tran_map

## convert resolution from ctm alignment to target code rate(in frame-shift/unit)
def resol_conversion(pid_seq, rate_target):
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
def get_avg_posterior(model, text_in, prop_in, resp_in, lang, pid_seq, cmp_len=False, is_ar_level_0 = False, masking_nar_level_0 = True, total_levels=1):
    pid_seq = resol_conversion(pid_seq, rate_target=cfg.dataset.frames_per_second) ## 75 for current config
    ##kaldi will randomly cut 10ms/20ms(1 or 2 frames) in the number of MFCC features, so we extend the SIL to match the number of codes  
    frame_diff = resp_in.shape[0] - pid_seq[-1][-1]
    if frame_diff > 5 or frame_diff < 0:
        sys.exit("problem with length of CTM and encodec output")
    #elif frame_diff > 0 and pid_seq[-1][0] == sil_token_id:
    elif frame_diff > 0:
        pid_seq[-1][-1] = pid_seq[-1][-1] + frame_diff         
    assert pid_seq[-1][-1] == resp_in.shape[0] 
    b_size = len(pid_seq)
    if cmp_len:
        lenth_in = resp_in.shape[0]
        len_list = [ lenth_in ] * b_size
    else:
        len_list = None
    if masking_nar_level_0:
        ## to batch
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
    if total_levels < 1 or total_levels > model.config.resp_levels:
        sys.exit("specify a correct level range, usually from [1,8]")
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
        #pdb.set_trace()
 
    else:
        sys.exit("not supporting AR+NAR in this version")
    print(ret_value)
    pdb.set_trace()
    _logger.info(f"MDD done")
    
    return ret_value
    
    

   

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 8:
        sys.exit("this script takes 7 arguments <model-ckpt-sft> <in-audio-wav-file> <in-prompt-wav-file> <in-raw-text-file> <CTM-alignment-file> <gop-tokenizer-path> <out-gop-path> \n \
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
    engines = load_engines(training=False, is_mdd=True)
    assert len(engines) == 1
    models = []
    for name, engine in engines.items():
        if type != torch.int8:
            models.append(engine.module.to(device, dtype=dtype if not amp else torch.float32))
            
    model = models[0]
    model.eval()
    _logger.info(f"model loaded")
    pdb.set_trace()
    ##load alignment and transcription
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(sys.argv[6])
    trans_map = read_trans(sys.argv[4])
    ctm_dict = read_ctm(sys.argv[5], p_tokenizer) 
    uttid_list = list(ctm_dict.keys())
    
    ## prepare data    
    ## target
    uid_align = "mbak2az2"
    uid_text = "mbak2az2"
    #uid = "fjdk1bm2"
    ##text
    phn_symmap = get_phone_symmap()
    text_in = trans_map[uid_text].lower()
    #text_in = trans_map[uid]
    lang = "en-us"
    phns,lang = get_text(text_in, device, phn_symmap, lang_code=lang)
    ##reps
    audio_in_path = Path(sys.argv[2])
    resp = get_emb(audio_in_path, device)
    ##prompt
    prompt_in_path = Path(sys.argv[3])
    prompt = get_prompt(prompt_in_path, device, trim_length=3, noise=False)
    #prompt = get_prompt(prompt_in_path, device, trim_length=3, noise=False)
    ##step 1, get segmentation (pid_seq = list of (pid, start_idx, end_idx)
    pid_seq = ctm_dict[uid_align]
    ##step 2, get avg_posterior for each phoneme
    #set_seed()
    with torch.no_grad():
        #avg_post_list = get_avg_posterior(model, phns, prompt, resp, lang, pid_seq, sil_token_id=p_tokenizer._convert_token_to_id(sil_token), total_levels=8)
        avg_post_list = get_avg_posterior(model, phns, prompt, resp, lang, pid_seq, total_levels=8)
           
    ##unload qnt models
    load_engines.cache_clear()
    unload_model()
    