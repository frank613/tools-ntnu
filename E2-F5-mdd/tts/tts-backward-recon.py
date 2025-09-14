import sys
from pathlib import Path
import torch
import logging
import numpy as np
import re
import math
import os
import datasets
from omegaconf import OmegaConf
from hydra.utils import get_class
import torchaudio
from tqdm import trange, tqdm
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
    load_model_mdd,
    load_vocoder,
    save_spectrogram,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    convert_char_to_pinyin,
    default
)
import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

ds_data_path = '/home/xinweic/cached-data/f5-mdd/data'
ds_cache_path = "/home/xinweic/cached-data/f5-mdd/ds-cache"
vocoder_cache_path_model = "/home/xinweic/cached-model/vocoder-cache/F5/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

hf_cache_path_model = "/home/xinweic/cached-model/vocoder-cache/F5/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)/(.*)\..*')
max_batch = 16



##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)

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
            uttid, sent = line.split(' ')[0], line.split(' ')[1:]
            tran_map.update({uttid: (' '.join(sent)).lower()})
    return tran_map

##return a list of phoneme masks for each phoneme, before and after scaled by mask_ratio
def mdd_mask( pid_seq, mask_ratio, device):
    length = pid_seq[-1][-1]
    mask_list = []
    mask_list_orig = []
    for index in range(len(pid_seq)):
        mask = torch.full((length,), True, dtype=torch.bool, device=device )
        mask_orig = torch.full((length,), True, dtype=torch.bool, device=device )
        pid, l_orig, r_orig = pid_seq[index]
        if mask_ratio < 1:
            sys.exit("mask_ratio must greater than 1") 
        extend = math.floor((r_orig-l_orig) * (mask_ratio - 1) / 2)
        l = math.floor(l_orig - extend) if l_orig - extend >= 0 else 0
        r = math.ceil(r_orig + extend) if r_orig + extend <= length else length
        mask[l:r] = False ## False == masked!
        mask_orig[l_orig:r_orig] = False
        mask_list.append(mask)
        mask_list_orig.append(mask_orig)
    return mask_list, mask_list_orig

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
    assert pid_seq_ret[-1][-1] == dur_target
    return pid_seq_ret
    
 
### non-batch version
def get_avg_posterior(model, text_in, mel, pid_seq, cfg_strength_gop=0, diff_symbol=None, masking_ratio=1, sway_sampling_coef=-1, steps=32, n_samples=5):
    assert mel.shape[-1] == model.num_channels
    duration_mel = mel.shape[-2]
    pid_seq = resol_conversion_duration(pid_seq, dur_target=duration_mel)
    assert masking_ratio >= 1  ###in this version (ODE-solver) only the segment within the phoneme_mask_list are valid 
    #phoneme_mask_list, phoneme_mask_list_orig = mdd_mask(pid_seq, masking_ratio, device)
    #num_phonemes = len(pid_seq)

    ##mask all speech prompt for test
    #l = int(1*duration_mel/2)
    #l = duration_mel -1
    l=0
    cond_mask = torch.ones(duration_mel,  dtype=torch.bool, device=device)
    cond_mask[l:] = False
    input_kwargs = dict(
                mel_target=[mel],
                text=[text_in], 
                duration = duration_mel,
                steps = steps,
                cfg_strength = cfg_strength_gop,
                phoneme_mask_list_orig = [cond_mask],
                sway_sampling_coef = sway_sampling_coef,
                n_samples = n_samples,           
        )
        ## Hut approximation, directly return aggreagated probability for each phoneme
    #gop, y0 = model.compute_prob_hut_start(**input_kwargs)
    gop, recon, recon2, recon3 = model.compute_prob_hut_recon(**input_kwargs)
    return gop, recon, recon2, recon3
    
def load_dataset_local_from_dict(csv_path, cache_additional, trans_map, uttid_list, subset=None, last=None):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
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
        datadict = {"audio": []}  
        #with open(folder_path + '/metadata.csv') as csvfile:
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
                #datadict["audio"].append(folder_path + '/' + row.split(',')[0])
                datadict["audio"].append(row.split(',')[0])
        ds = datasets.Dataset.from_dict(datadict) 
        #ds = ds.cast_column("audio", datasets.Audio(sampling_rate=target_sample_rate))
        ## do further transformation 
        def map_to_array(batch):   
            batch["id"] = [re_uttid_raw.match(item)[2] for item in batch["audio"]]
            batch["tokens"] = []
            batch["mel"] = []
            for i, uid in enumerate(batch["id"]):
                if uid not in uttid_list:
                    batch["tokens"].append(None)
                    batch["mel"].append(None)
                else:
                    text_in = trans_map[uid]
                    #tokenizer
                    if tokenizer == "pinyin":
                        final_text_list = convert_char_to_pinyin([text_in])
                    else:
                        final_text_list = [text_in]
                    batch["tokens"].append(final_text_list[0])
                    audio_in_path = batch["audio"][i]
                    cond = get_audio(audio_in_path, target_rms, target_sample_rate, device)
                    assert cond.ndim == 2
                    cond = mel_spec(cond)
                    cond = cond.permute(0, 2, 1).squeeze()
                    #duration_mel = math.ceil(cond.shape[-1] / hop_length)
                    batch["mel"].append(cond)
            return batch
        ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
        ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['tokens']], batched=True, batch_size=100, num_proc=10)
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
        ds_filtered = ds_filtered.select(range(last_index, len(ds_filtered)))
    return ds_filtered
   

def batch_process(uid, text, mel, device, out_path):
    model = load_model_mdd( model_cls, model_arc, model_path, mel_spec_type=mel_spec_type, vocab_file=vocab_path, device=device, use_ema=True)
    dtype = next(model.parameters()).dtype
    ###to save memory for MDD
    for param in model.parameters():
        param.requires_grad = False   
    ##mdd parameters:
    cfg_strength_gop=0
    #diff_symbol="p"
    diff_symbol=None
    masking_ratio=1
    steps=64
    n_samples=10
    #sway_sampling_coef = None
    sway_sampling_coef = -1
    
    print(f"Using cfg={cfg_strength_gop}, mr={masking_ratio}, steps={steps}, sway={sway_sampling_coef}, diff={diff_symbol}")
    #We need training mode because ODE?
    #model.eval()
    with torch.no_grad():
        print("processing {0}".format(uid))    
        pid_seq = ctm_dict[uid]
        #gop, y0, cond = get_avg_posterior(model, text, mel, pid_seq, cfg_strength_gop=cfg_strength_gop, diff_symbol=diff_symbol, masking_ratio=masking_ratio, sway_sampling_coef=sway_sampling_coef, steps=steps, n_samples=n_samples) 
        gop, generated, generated2, generated3 = get_avg_posterior(model, text, mel, pid_seq, cfg_strength_gop=cfg_strength_gop, diff_symbol=diff_symbol, masking_ratio=masking_ratio, sway_sampling_coef=sway_sampling_coef, steps=steps, n_samples=n_samples)             
        #print(f"Overall GOP score: {gop.item()}")
        #tts
        #load vocoder
        vocoder_local_path = hf_cache_path_model
        if os.path.exists(vocoder_local_path):
            vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=True, local_path=vocoder_local_path, device=device)
        else:
            vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, hf_cache_dir=vocoder_local_path, device=device)

        # with torch.inference_mode():
        #     generated, trajectory = model.sample_from_start(
        #         cond=cond[None],
        #         text=[text],
        #         duration= y0.shape[1],
        #         steps=32,
        #         cfg_strength=2,
        #         sway_sampling_coef=-1,
        #         seed=None,
        #         edit_mask=None,
        #         start=y0,
        #     )
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)  ##batch, lenth, channel
        generated2 = generated2.to(torch.float32)
        gen_mel_spec2 = generated2.permute(0, 2, 1)  ##batch, lenth, channel
        generated3 = generated3.to(torch.float32)
        gen_mel_spec3 = generated3.permute(0, 2, 1)  ##batch, lenth, channel
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
            generated_wave_2 = vocoder.decode(gen_mel_spec2).cpu()
            generated_wave_3 = vocoder.decode(gen_mel_spec2).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
            generated_wave_2 = vocoder(gen_mel_spec2).squeeze(0).cpu()
            generated_wave_3 = vocoder(gen_mel_spec2).squeeze(0).cpu()
        ##save
        os.makedirs(out_path, exist_ok=True)
        #save_spectrogram(gen_mel_spec[0].cpu().numpy(), f"{sys.argv[6]}/speech_edit_{pid}.png")
        torchaudio.save(f"{out_path}/inverse_flow_recon_1.wav", generated_wave, target_sample_rate)
        torchaudio.save(f"{out_path}/inverse_flow_recon_2.wav", generated_wave_2, target_sample_rate)
        torchaudio.save(f"{out_path}/inverse_flow_recon_3.wav", generated_wave_3, target_sample_rate)
        print(f"Generated: {generated_wave.shape}")
    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <model-path> <model-config-yaml-path> <in-audio-wav-file> <in-raw-text-file> <CTM-alignment-file> <out-wave-path>\n \
        , it loads the TTS model and compute the GOP")
    
    ## load model config       
    yaml_path = sys.argv[2]
    model_cfg = OmegaConf.load(yaml_path)
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type 
    ## paramters
    target_rms = 0.1
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    frames_per_second = target_sample_rate // hop_length
    #load vocab and tokenizer
    tokenizer = model_cfg.model.tokenizer
    vocoder_name = model_cfg.model.mel_spec.mel_spec_type 

    if model_cfg.model.tokenizer_path is not None or tokenizer != "pinyin":
        sys.exit("check the tokenizer and vocab path")
    vocab_path = f"{sys.argv[1]}/vocab.txt"
    
    ## load model
    model_path = f"{sys.argv[1]}/model_1250000.safetensors"
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch    
    model_name = model_cfg.model.name
    print(f"Using {model_name}...")
    #load vocoder
    # vocoder_local_path = vocoder_cache_path_model
    # if os.path.exists(vocoder_local_path):
    #     vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=True, local_path=vocoder_local_path, device=device)
    # else:
    #     vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, hf_cache_dir=vocoder_local_path, device=device)
             
    ##load alignment and transcription
    trans_map = read_trans(sys.argv[4])
    ctm_dict = read_ctm(sys.argv[5])
    uttid_list = list(ctm_dict.keys())
    
    ## mask and gen
    audio_in_path = Path(sys.argv[3])
    uid = "fabm2aa1"
    cond = get_audio(audio_in_path, target_rms, target_sample_rate, device)
    duration_mel = math.ceil(cond.shape[-1] / hop_length)
    
    mel_spec_kwargs=dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    mel_spec = MelSpec(**mel_spec_kwargs)
    mel = mel_spec(cond)
    mel = mel.permute(0, 2, 1).squeeze()
    #pid_seq_sec = ctm_dict[uid]
    #pid_seq = resol_conversion_duration(pid_seq_sec, dur_target=duration_mel)
    
    batch_process(uid, trans_map[uid], mel, device, sys.argv[6])
    # new_folder = os.path.dirname(out_path)
    # if not os.path.exists(new_folder):
    #     os.makedirs(new_folder)
    # load dataset and read soundfiles
    #ds= load_dataset_local_from_dict(csv_path, "cmu-kids", trans_map, uttid_list, subset=subset_list, last=last_utt)
    # ds could be loaded from disk, need to move the tensors to device 
    #ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "model":model, "device":device, "out_path":out_path}, num_proc=2) 
    #ds.map(batch_process, fn_kwargs={"device":device}, batched=True, batch_size=1, num_proc=1)
    