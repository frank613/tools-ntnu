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
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <model-path> <model-config-yaml-path> <in-audio-wav-file> <out-wave-path> \n \
        , it loads the model and run TTS")    
   
    ## load model config
    yaml_path = sys.argv[2]
    model_cfg = OmegaConf.load(yaml_path)
    vocoder_name = model_cfg.model.mel_spec.mel_spec_type 
    ## paramters
    target_rms = 0.1
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    hop_length = model_cfg.model.mel_spec.hop_length
    frames_per_second = target_sample_rate // hop_length
    seed = None
    
    null_cond = False
    no_audio = False
    no_text = False
    
    
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
    ema_model = load_model( model_cls, model_arc, model_path, mel_spec_type=vocoder_name, vocab_file=vocab_path, device=device, use_ema=True)
    #load vocoder
    vocoder_local_path = hf_cache_path_model
    if os.path.exists(vocoder_local_path):
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=vocoder_local_path, device=device)
    else:
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, hf_cache_dir=vocoder_local_path, device=device)
         
    #prepare input
    #If English punctuation marks the end of a sentence, 
    #make sure there is a space " " after it. Otherwise not regarded as when chunk.
    #ref_text = "I am Xinwei."
    ref_text = "公司要倒闭了. 我要跑路了."
    #target_text = "我还可以说中文呢."
    #target_text = "你今天什么时候来上班"
    target_text = "I love to eat dicks. 我想吃大鸡巴."
    text_list = [ref_text+target_text]
    
    ## mask and gen
    audio_in_path = Path(sys.argv[3])
    cond = get_audio(audio_in_path, target_rms, target_sample_rate, device)
    ref_mel_len = math.floor(cond.shape[-1] / hop_length)
    target_mel_len =  math.ceil((cond.shape[-1]/hop_length)*(len(target_text)/len(ref_text)))
    final_mel_len = ref_mel_len + target_mel_len

   
    ###Null Condition, change the input
    if null_cond:
        ##[SPACE] is 0 and according to TextEmbedding.forward(), all text is set to 0 and padded to seq_len
        if no_audio: ## with text only
            text_list = [target_text]
            ref_mel_len = 0
            final_mel_len = target_mel_len
        elif no_text: ## with audio, no extra text = nothing to generate? will be padded zeros until seq_len
            ##fully no text
            text_list = [ " " for text in text_list ]
            #only text corresponding to prompt
            #text_list = [ref_text]
        else: ## no cond and no text
            text_list = [ " " for text in text_list ] 
            #copied from InputEmbedding.forward()
            cond = torch.zeros_like(cond)           
    
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
    
    ## TTS, "cond" is in sampling_rate while mask is in frame_rate   
    repeats = 5
    for i in range(repeats):
        # Inference
        with torch.inference_mode():
            generated, trajectory = ema_model.sample(
                cond=cond,
                text=final_text_list,
                duration= final_mel_len,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        print(f"Generated mel_{i}: {generated.shape}")
        generated = generated[:, ref_mel_len:, ].to(torch.float32)
        assert target_mel_len == generated.shape[1]
        gen_mel_spec = generated.permute(0, 2, 1)  ##to batch, lenth, channel
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
        ##save
        os.makedirs(sys.argv[4], exist_ok=True)
        #save_spectrogram(gen_mel_spec[0].cpu().numpy(), f"{sys.argv[4]}/{i}.png")
        torchaudio.save(f"{sys.argv[4]}/{i}.wav", generated_wave, target_sample_rate)
        print(f"Generated wav_{i}: {generated_wave.shape}")