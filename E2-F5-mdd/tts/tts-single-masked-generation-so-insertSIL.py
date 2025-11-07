import sys
import os
from pathlib import Path
import torch
import logging
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import get_class
#sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/F5-link/f5_tts/infer")
#from infer.utils_infer import (
from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    infer_process,
    load_model,
    load_model_mdd,
    load_vocoder,
    save_spectrogram,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

from f5_tts.model.utils import (
    convert_char_to_pinyin,
)
import torchaudio
import re
import math
import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

#hf_cache_path_model = "/home/xinweic/cached-model/vocoder-cache/F5"
hf_cache_path_model = "/home/xinweic/cached-model/vocoder-cache/F5/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)/(.*)\..*')


def get_audio(audio_in_path, target_rms, target_sample_rate, device):
    audio, sr = torchaudio.load(audio_in_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True) 
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        
    #return res.to(device=device, dtype=torch.int16), full.to(device=device, dtype=torch.int16)
    return audio.to(device=device)

def read_ctm(ctm_path, tokenizer=None):
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
            #pid = tokenizer._convert_token_to_id(phoneme)
            if uttid != cur_uttid and cur_uttid is not None:
                ret_dict.update({cur_uttid: phone_list})
                phone_list = []
            cur_uttid = uttid           
            #phone_list.append([pid, start, end])  
            phone_list.append([phoneme, start, end])           
        if not cur_uttid in ret_dict:
            ret_dict.update({cur_uttid: phone_list})
    return ret_dict

def read_trans(tran_path):
    tran_map = {}
    with open(tran_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            uttid, sent = line.split('\t')[0], line.split('\t')[1:]
            tran_map.update({uttid: (' '.join(sent)).lower()})
    return tran_map


## convert resolution from ctm alignment to target code rate(in frame-shift/unit)
def resol_conversion(pid_seq, rate_target):
    #pdb.set_trace()
    pid_seq_ret = []
    ## *2 because ctm file uses default frame_shift = 10, but actually is 20
    total_len = math.floor(float(pid_seq[-1][2]*2 * rate_target))
    time_end = [ e for id,s,e in pid_seq ] 
    total_time = time_end[-1]
    frame_end = [ math.ceil((time/total_time)*total_len) for time in time_end ]  ### ceiling for covering a wider range of code in case of phoneme state transition 
    start = 0
    for (pid,_,_),frame_e in zip(pid_seq, frame_end):
        pid_seq_ret.append([pid, start, frame_e])
        start = frame_e - 1  ### -1 for covering a wider range of code in case of phoneme state transition 
    return pid_seq_ret

## convert resolution from ctm alignment to target dur length
def resol_conversion_duration(pid_seq, dur_target):
    #pdb.set_trace()
    pid_seq_ret = []
    total_len = dur_target
    time_end = [ e for id,s,e in pid_seq ] 
    total_time = time_end[-1]
    frame_end = [ math.ceil((time/total_time)*total_len) for time in time_end ]  ### ceiling for covering a wider range of code in case of phoneme state transition 
    start = 0
    for (pid,_,_),frame_e in zip(pid_seq, frame_end):
        pid_seq_ret.append([pid, start, frame_e])
        start = frame_e - 1  ### -1 for covering a wider range of code in case of phoneme state transition 
    return pid_seq_ret
    

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <model-path> <model-config-yaml-path> <in-audio-wav-file> <in-raw-text-file> <CTM-alignment-file> <out-wave-path> \n \
        , it loads the model and run TTS's speech infilling task, similar to training")    
   
    ## load model config
    yaml_path = sys.argv[2]
    model_cfg = OmegaConf.load(yaml_path)
    vocoder_name = model_cfg.model.mel_spec.mel_spec_type 
    ## paramters
    target_rms = 0.1
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    frames_per_second = target_sample_rate // hop_length
    mask_ratio_min = 1.1
    mask_vs_average_ratio = 2.0
    fixed_SIL_frame = 1
    SIL_ratio = 1.5
    seed = None
    filling = False
    null_cond = False
    
    #load vocab and tokenizer
    tokenizer = model_cfg.model.tokenizer

    if model_cfg.model.tokenizer_path is not None or tokenizer != "pinyin":
        sys.exit("check the tokenizer and vocab path")
    vocab_path = f"{sys.argv[1]}/vocab.txt"
    #tokenizer_file = f"{sys.argv[1]}/vocab.txt"
    # with open(vocab_path, "r", encoding="utf-8") as f:
    #     vocab_char_map = {}
    #     for i, char in enumerate(f):
    #         vocab_char_map[char[:-1]] = i
    # #vocab_size = len(vocab_char_map)
    # assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"
    
    ## load model
    model_path = f"{sys.argv[1]}/model_1250000.safetensors"
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch    
    model_name = model_cfg.model.name
    print(f"Using {model_name}...")
    ema_model = load_model_mdd( model_cls, model_arc, model_path, mel_spec_type=vocoder_name, vocab_file=vocab_path, device=device, use_ema=True)
    #load vocoder
    vocoder_local_path = hf_cache_path_model
    if os.path.exists(vocoder_local_path):
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=vocoder_local_path, device=device)
    else:
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, hf_cache_dir=vocoder_local_path, device=device)
         
    ##load meta data
    #p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(sys.argv[7])
    trans_map = read_trans(sys.argv[4])
    ctm_dict = read_ctm(sys.argv[5])
    uttid_list = list(ctm_dict.keys())
    
    #prepare input
    #If English punctuation marks the end of a sentence, 
    #make sure there is a space " " after it. Otherwise not regarded as when chunk.
    uid = "000050118"
    #uid = "fabm2bn2"
    if uid not in uttid_list:
        sys.exit("can't find the uid in data")

    target_text = trans_map[uid]
    text_list = [target_text]
    ## mask and gen
    audio_in_path = Path(sys.argv[3])
    cond = get_audio(audio_in_path, target_rms, target_sample_rate, device)
    ###Mel generator
    mel_spec_kwargs=dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    mel_spec = MelSpec(**mel_spec_kwargs)
    assert cond.ndim == 2
    cond = mel_spec(cond)
    cond = cond.permute(0, 2, 1)
    duration_mel = cond.shape[1]
    pid_seq_sec = ctm_dict[uid]
    pid_seq = resol_conversion_duration(pid_seq_sec, dur_target=duration_mel)
    n_p = len(pid_seq)
   
    ###Null Condition
    ###Warning, single " " == <SPACE><PAD>....!= all <SPACE>, check class TextEmbedding
    if null_cond:
        #text_list = [ " " for text in text_list ]
        #text_list = ["a a a a a a a a a a a a a a a a a a" for text in text_list]
        #text_list = ["e e e e e e e e e e e e e e e e e e" for text in text_list]
        text_list = ["只 只 只 只 只 只 只 只 只 只 只 只 只 只" for text in text_list]
        #text_list = ["a pianist walked through a field"]
    
    #tokenizer
    if tokenizer == "pinyin":
        final_text_list = convert_char_to_pinyin(text_list)
    else:
        final_text_list = text_list
    
    ##append space into the EOS 
    for i,text in enumerate(final_text_list): 
        if text[-1] != " ":
            final_text_list[i].append(" ")
        
    print(f"text  : {text_list}")
    print(f"pinyin: {final_text_list}")

    ## TTS filling the next half
    repeats = 3
    for i in range(repeats):
        l = int(duration_mel/2)
        cond_mask = torch.ones(duration_mel,  dtype=torch.bool, device=device)
        cond_mask[l:] = False
        
        # Inference
        with torch.inference_mode():
            generated, trajectory, cond_mel = ema_model.sample(
                cond=cond,
                text=final_text_list,
                duration=int(duration_mel*1.1), ##scale it
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                edit_mask=cond_mask,
            )
        print(f"Generated mel_{i}: {generated.shape}")
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)  ##batch, lenth, channel
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
        ##save
        os.makedirs(sys.argv[6], exist_ok=True)
        #save_spectrogram(gen_mel_spec[0].cpu().numpy(), f"{sys.argv[6]}/speech_edit_{pid}.png")
        torchaudio.save(f"{sys.argv[6]}/filling_half_{i}.wav", generated_wave, target_sample_rate)
        print(f"Generated wav_half{i}: {generated_wave.shape}")

    if pid_seq[0][0] != "SIL":
        sys.exit("this version must have an SIL at the start")
    SIL_to_append = cond[:, 0:pid_seq[0][-1]]
    len_SIL = SIL_to_append.shape[1]
    if mask_ratio_min < 1:
        sys.exit("mask_ratio must greater than 1")  
    ## TTS, "cond" is in sampling_rate while mask is in frame_rate   
    repeats = 1
    avg_len = math.ceil(pid_seq[-1][-1]/n_p)
    for i, (pid, l_orig, r_orig) in enumerate(pid_seq):
        mask_ratio = max(mask_vs_average_ratio*avg_len/(r_orig-l_orig), mask_ratio_min)
        extend = math.ceil((r_orig-l_orig) * (mask_ratio - 1) / 2)
        if mask_ratio != 1:
            l_orig = math.floor(l_orig - extend) if l_orig - extend >= 0 else 0
            r_orig = math.ceil(r_orig + extend) if r_orig + extend <= duration_mel else duration_mel
        
        if fixed_SIL_frame is not None:
            extend_SIL = fixed_SIL_frame
        else:
            extend_SIL = math.ceil((r_orig-l_orig) * (SIL_ratio - 1) / 2)
        l = l_orig
        r = r_orig
        ##pad the sil
        if pid != "SIL":
            if extend_SIL > len_SIL:
                sys.exit("no enough SIL to append")
            if i == 0 or pid_seq[i-1][0] == "SIL": ## only pad right hand side
                if i+1 >= n_p or pid_seq[i+1][0] == "SIL":
                    cond_new = cond
                    continue
                cond_new= torch.cat((cond[:, :r_orig], SIL_to_append[:, :extend_SIL, :], cond[:, r_orig:]), dim=1)
            elif i == n_p-1 or pid_seq[i+1][0] == "SIL": ## only pad left
                if i-1 < 0 or pid_seq[i-1][0] == "SIL":
                    cond_new = cond
                    continue
                cond_new = torch.cat((cond[:, :l_orig], SIL_to_append[:,:extend_SIL], cond[:,l_orig:]), dim=1)
                l = l_orig + extend_SIL
                r = r_orig + extend_SIL
            else: ## pad both sides
                cond_new = torch.cat((cond[:, :l_orig], SIL_to_append[:,:extend_SIL], cond[:,l_orig:r_orig],  SIL_to_append[:,:extend_SIL], cond[:,r_orig:]), dim=1)
                l = l_orig + extend_SIL
                r = r_orig + extend_SIL
        else:
            cond_new = cond
        ##cond mask
        cond_mask = torch.ones(cond_new.shape[1],  dtype=torch.bool, device=device)
        cond_mask[l:r] = False
        # Inference
        with torch.inference_mode():
            generated, trajectory, cond_mel = ema_model.sample(
                cond=cond_new,
                text=final_text_list,
                duration=duration_mel,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                edit_mask=cond_mask,
            )
        print(f"Generated mel_{pid}: {generated.shape}. mask-ratio:{mask_ratio}")
        generated = generated.to(torch.float32)
        ## start from all so we do speech edit
        if not filling:
            start_frame = 0 
            generated = generated[:, start_frame:, :]
        else: ## for masked GEN/MDD we hope only the target segment is generated
            generated = torch.where(cond_mask[None, :, None], cond_mel, generated)
        gen_mel_spec = generated.permute(0, 2, 1)  ##batch, lenth, channel
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
        ##save
        os.makedirs(sys.argv[6], exist_ok=True)
        #save_spectrogram(gen_mel_spec[0].cpu().numpy(), f"{sys.argv[6]}/speech_edit_{pid}.png")
        torchaudio.save(f"{sys.argv[6]}/speech_edit_{i}_{pid}.wav", generated_wave, target_sample_rate)
        print(f"Generated wav_{pid}: {generated_wave.shape}")