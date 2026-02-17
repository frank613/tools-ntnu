import argparse
import os
import shutil
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM_MDD, UNetT, DiT, Trainer_MDD_word
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset
from hydra.utils import get_class
import pdb
from torch.utils.data import Subset

import sys
from omegaconf import OmegaConf
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

def read_word_ctm(ctm_path, tokenizer=None):
    ret_dict = {}
    word_list = []
    cur_uttid = None
    with open(ctm_path, "r") as ifile:
        for line in ifile:
            line = line.strip()
            fields = line.split(' ')
            assert len(fields) == 5
            uttid, start, end, word = fields[0], float(fields[2]), float(fields[2]) + float(fields[3]), fields[4]
            uttid = uttid.strip("lbi-")
            #pid = tokenizer._convert_token_to_id(phoneme)
            if uttid != cur_uttid and cur_uttid is not None:
                ret_dict.update({cur_uttid: word_list})
                word_list = []
            cur_uttid = uttid           
            #phone_list.append([pid, start, end])  
            word_list.append([word.lower(), start, end])           
        if not cur_uttid in ret_dict:
            ret_dict.update({cur_uttid: word_list})
    return ret_dict

def main():
    ##load pretrained model
    #model = load_model_mdd(model_cls, model_arc, model_path, mel_spec_type=mel_spec_type, vocab_file=vocab_path, device=device, use_ema=True)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    model = CFM_MDD(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
    )
    
    trainer = Trainer_MDD_word(
        model,
        epochs=15,
        learning_rate=2e-5,
        num_warmup_updates=3000,
        save_per_updates=500,
        keep_last_n_checkpoints=-1,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=3500,
        batch_size_type="frame",
        max_samples=64,
        grad_accumulation_steps=5,
        max_grad_norm=1.0,
        logger="wandb",
        wandb_project="fine-tune-f5",
        wandb_run_name=None,
        wandb_resume_id=None,
        log_samples=True,
        last_per_updates=5000,
        bnb_optimizer=bnb_optimizer,
        text_loss_weight=0.1,  
        ctm_dict=ctm_dict,         
    )
    ##load libripssech datasets? validation set how to add?
    train_dataset = load_dataset(train_path, dataset_type="CustomDatasetPath", mel_spec_kwargs=mel_spec_kwargs, filter_list=ctm_dict.keys())
    #indices_to_keep = [i for i, sample in enumerate(train_dataset) if sample['uid'] in ctm_dict]
    #train_dataset = Subset(train_dataset, indices_to_keep)        
    #train_dataset = train_dataset.filter(lambda example: example["uid"] in ctm_dict)
    val_dataset = load_dataset(dev_path, dataset_type="CustomDatasetPath", mel_spec_kwargs=mel_spec_kwargs)

    trainer.train(
        train_dataset,
        val_dataset,
        resumable_with_seed=1,  # seed for shuffling dataset
    )


if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <model-path> <model-config-yaml-path> <word-ctm-file> <training-dataset-path> <dev-dataset-path> \n \
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
    bnb_optimizer=model_cfg.optim.bnb_optimizer
    #load vocab and tokenizer
    tokenizer = model_cfg.model.tokenizer
    if model_cfg.model.tokenizer_path is not None or tokenizer != "pinyin":
        sys.exit("check the tokenizer and vocab path")  
    vocab_path = f"{sys.argv[1]}/vocab.txt"
    ## load model
    checkpoint_path = f"{sys.argv[1]}"
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch    
    model_name = model_cfg.model.name
    print(f"Using {model_name}...")
    
    ctm_dict = read_word_ctm(sys.argv[3])
    train_path = sys.argv[4]
    dev_path = sys.argv[5]    
    # import json
    # with open(f"{train_path}/duration.json", "r", encoding="utf-8") as f:
    #         data_dict = json.load(f)
    # durations = data_dict["duration"]
    # pdb.set_trace()
    main()
