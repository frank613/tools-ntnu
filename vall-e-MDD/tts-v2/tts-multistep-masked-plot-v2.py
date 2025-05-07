import sys
from pathlib import Path
import torch
import logging
import numpy as np
from vall_e.utils import to_device, set_seed, clamp
from vall_e.config import cfg
from vall_e.engines import load_engines
from vall_e.data import get_phone_symmap, get_lang_symmap, tokenize, text_tokenize, sentence_split
from vall_e.models.ar_nar import AR_NAR
from vall_e.emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
from vall_e.emb import g2p, qnt
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer
import matplotlib.pyplot as plt
import re
import math
import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)/(.*)\..*')

##copied from inference.py
def encode_text(text, language="auto", precheck=True, phonemize=True ):
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

def get_text(text_in, device, phn_symmap, lang):
    lines = sentence_split(text_in)
    assert len(lines) == 1
    text = lines[0]
    #lang = g2p.detect_language(text)
    phns = encode_text( text, language=lang)
    phns = phns.to(device=device, dtype=torch.uint8 if len(phn_symmap) < 256 else torch.int16)
    lang = encode_lang(lang)
    lang = lang.to(device=device, dtype=torch.uint8)
    return phns,lang

##return audio prompt or noisy prompt
def get_promp_emb(audio_in_path, device, trim_length=0, noise=0):
    if not noise:
        res = encode_audio(audio_in_path, trim_length=trim_length)
    ##noisy
    elif noise == 1:
        noisy_code = load_artifact(Path("/home/xinweic/git-repos/vall-e/data/noise.enc"))
        if trim_length:
            res = repeat_extend_audio( noisy_code, int( cfg.dataset.frames_per_second * trim_length ) )
        else:
            res = noisy_code
    ##zero input
    else:
        res = torch.zeros((cfg.dataset.frames_per_second * trim_length, resp_level), dtype=torch.int16)
    return res.to(device=device, dtype=torch.int16)

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

def read_trans(tran_path):
    tran_map = {}
    with open(tran_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            uttid, sent = line.split(' ')[0], line.split(' ')[1:]
            tran_map.update({uttid: (' '.join(sent)).lower()})
    return tran_map

def mdd_mask( pid_seq, index, length, mask_ratio, device):
    ##return the original index, not the extended
    mask = torch.full((length,), False, dtype=torch.bool, device=device )
    pid, l_orig, r_orig = pid_seq[index]
    if mask_ratio < 1:
        sys.exit("mask_ratio must greater than 1") 
    extend = math.floor((r_orig-l_orig) * (mask_ratio - 1) / 2)
    l = l_orig - extend if l_orig - extend >= 0 else 0
    r = r_orig + extend if r_orig + extend <= length-1 else length-1
    mask[l:r] = True ## 1 is masked! same as above, different from below, because later will we use "where" operation 
    return mask,l_orig,r_orig

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

def compute_gop(out_prob, left, right):
    ## do a small sanity check here
    assert out_prob.shape[0] >= right-left and out_prob[:left].sum() + out_prob[right:] == 0
    return out_prob[left:right].log().mean()
    
    
##plot the graph
def plot_activations(logits, out_prob, resps_in, target_phoneme, left, right, out_path):
    #pdb.set_trace()
    assert logits.shape[-1] == resps_in.shape[-1] 
    ##get the last "len" logits
    assert logits.shape[0] >= resps_in.shape[0]
    seq_len = resps_in.shape[0]
    logits= logits[-seq_len:, :, :].softmax(dim=1).cpu().numpy() 
    
    plt.rcParams['font.size'] = 50
    plt.tight_layout()
    fig, axes = plt.subplots(logits.shape[-1],1,figsize=(25*logits.shape[-1], 300), sharex="col",layout="constrained")
    for i in range(logits.shape[-1]):
        ##compute gop
        gop = compute_gop(out_prob[:,i], left, right)
        ##heat map
        #pdb.set_trace()
        im = axes[i].imshow(np.transpose(logits[:,:,i]), norm="linear", origin="lower") #cmap="YlGn")
        cbar =  axes[i].figure.colorbar(im, ax=axes[i])
        cbar.ax.set_ylabel("normalized logits", rotation=-90, va="bottom")
        ##input codes
        #axes[i].step(np.arange(seq_len),resps_in[:,i].cpu().numpy(), "w-*", where="post", label="codes to be evaluated" )
        axes[i].plot(np.arange(seq_len),resps_in[:,i].cpu().numpy(), "o--", color="white", alpha=0.5, label="codes to be evaluated" )
        ##output codes
        #axes[i].step(np.arange(seq_len),resps_out[:,i].cpu().numpy(), "r-*", where="post", label="sampled codes" )
        #axes[i].plot(np.arange(seq_len),resps_out[:,i].cpu().numpy(), "o--", color="red", alpha=0.5, label="sampled codes" )
        ##mask
        #axes[i].axvspan(left,right, color='0.5')
        axes[i].axvline(left, color='g')
        axes[i].axvline(right-1, color='g')
        
        axes[i].text(left+right/2, 0, f"GOP for {target_phoneme}:{gop}", size="large", color="white")
        
        axes[i].set_aspect("auto")
        axes[i].set_title(f"Masking phoneme {target_phoneme}, code level {i}")
        axes[i].legend()
        
        
    fig.supxlabel("Encodec frames")
    fig.supylabel("Code numbers 0-9999")
    plt.savefig(out_path)

def get_tts_logtis_and_plot(model, text_in, prop_in, lang, is_ar, device, reps_in, predict_level_0=True, pid_seq=None, mask_index=None, target_phoneme=None, mask_ratio=1, out_path=None): 

    ##do single mask
    if pid_seq is None or mask_index is None:
        sys.exit("must provide pid_seq and mask_index for masked generation")
    phoneme_mask,l,r = mdd_mask(pid_seq, mask_index, len(reps_in), mask_ratio, device)
    len_list = [len(reps_in)]
    input_kwargs = dict(
                phns_list=[text_in], 
                proms_list=[prop_in],
                lang_list=[lang],
                task_list=["mdd-plot"],
                len_list=len_list,
                disable_tqdm=False,
                use_lora=True,
                resps_list=[reps_in],
                phoneme_mask=phoneme_mask,
                n_step_level_0 = 10,
            )

    for i in range(1):
        ## predict len
        #len_list = model( **input_kwargs, task_list=["len"], **{"max_duration": 10, "temperature": 2} )
        len_list = [len(reps_in)]
        ## NAR
        kwargs = {"temperature": 0, "cfg_strength":2.5}
        out_probs, logit_list = model( **input_kwargs, **(kwargs))
        ## decode
        out_prob = out_probs[0]
        plot_activations(logit_list[0].transpose(0,1).transpose(1,2), out_prob, reps_in, target_phoneme, left=l, right=r, out_path=out_path+f"-plot-{mask_index}_{target_phoneme}-{i}.png")
        #wav, sr = qnt.decode_to_file(resps, out_path+f"MNP-{mask_index}_{target_phoneme}-{i}.wav", device=device) 
  
    _logger.info(f"decoding done")
    

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <model-ckpt-sft> <in-audio-wav-file> <in-raw-text-file> <CTM-alignment-file> <gop-tokenizer-path> <out-wave-path> \n \
        , it maskes level-0 code and plot the activations")
          
    ## default cfg and then update cfg from model, similar to inferece.py
    cfg.load_model(Path(sys.argv[1]))
    cfg.format( training=False )
    ## cfg related attributes
    dtype = cfg.inference.dtype
    amp = cfg.inference.amp
    device = cfg.device
    resp_level = cfg.model.resp_levels
    
    ##load meta data
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(sys.argv[5])
    trans_map = read_trans(sys.argv[3])
    ctm_dict = read_ctm(sys.argv[4], p_tokenizer)
    uttid_list = list(ctm_dict.keys())
    
    ## load the model and engine(engine helps to create model and load from stat_dict)
    #cfg.ckpt_dir = Path(sys.argv[1])
    engines = load_engines(training=False, is_mdd=True)
    assert len(engines) == 1
    models = []
    for name, engine in engines.items():
        if type != torch.int8:
            models.append(engine.module.to(device, dtype=dtype if not amp else torch.float32))
            
    models[0].eval()
    _logger.info(f"model loaded")

    phn_symmap = get_phone_symmap()
    
    #prepare input
    uid = "fabm2aa1"
    #uid = "fabm2cb2"
    if uid not in uttid_list:
        sys.exit("can't find the uid in data")
    text_in = trans_map[uid]
    lang_code = "en-us"
    phns,lang = get_text(text_in, device, phn_symmap, lang=lang_code)
    audio_in_path = Path(sys.argv[2])
    prompt,resps = get_emb(audio_in_path, trim_length=3, noise=0)
    pid_seq = ctm_dict[uid]
    pid_seq = resol_conversion(pid_seq, rate_target=cfg.dataset.frames_per_second)
    
    ###disable prompt
    prompt=None
    
    #phns = None
    #phns = torch.tensor([1, 2], device=device)
    #phns[3] = 4
    #phns[4] = phns[4]
    #phns[4] = 4
    #phns[1] = 101
    #phns = torch.cat((phns[:4], phns[4+1:]))
    #pdb.set_trace()
    #phns = torch.tensor([1, 2], device=device)
    
    #phns[:] = 4
    #phns[0] = 1
    #phns[-1] = 2
    
    ipa_dict = cfg.tokenizer.get_vocab()
    ipa_dict_inv = {v:k for k,v in ipa_dict.items()}
    sym_list = [ f"{i}-{p}:{ipa_dict_inv[p]}" for i, p in enumerate(phns.tolist())]
    print(sym_list)
    ## TTS
    mask_index = 3
    #number_steps_level_0 = 1
    mask_ratio = 1
    target_phoneme=p_tokenizer._convert_id_to_token(int(pid_seq[mask_index][0]))
    set_seed()
    with torch.no_grad():
        get_tts_logtis_and_plot(models[0], phns, prompt, lang, False, device, resps, pid_seq=pid_seq, mask_index=mask_index, target_phoneme=target_phoneme, mask_ratio=mask_ratio, out_path=sys.argv[6])
    
    ##unload qnt models
    load_engines.cache_clear()
    unload_model()
    