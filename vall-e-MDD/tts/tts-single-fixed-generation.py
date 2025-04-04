import sys
from pathlib import Path
import torch
import logging
import numpy as np
from vall_e.utils import to_device, set_seed, clamp, wrapper as ml
from vall_e.config import cfg
from vall_e.engines import load_engines
from vall_e.data import get_phone_symmap, get_lang_symmap, tokenize, text_tokenize, sentence_split
from vall_e.models.ar_nar import AR_NAR
from vall_e.emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
from vall_e.emb import g2p, qnt

import pdb

_logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.INFO)

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

def get_tts_results(model, text_in, prop_in, lang, is_ar, device, reps_in, fix_level=0, out_path=None): 
    #duration_padding = 1.05
    #if len(reps_in) != fix_level+1:
    #    sys.exit("in input reps should be of length quant_level+1")
    input_kwargs = dict(
                text_list=[text_in], 
                raw_text_list=None,
                proms_list=[prop_in],
                lang_list=[lang],
                disable_tqdm=False,
                use_lora=True,
                resps_list=[reps_in[:, :fix_level+1]],
                fix_level = fix_level,
            )

    if not is_ar: ## len+NAR
        for i in range(5):
            ## predict len
            #len_list = model( **input_kwargs, task_list=["len"], **{"max_duration": 10, "temperature": 2} )
            len_list = [len(reps_in)]
            ## NAR
            kwargs = {"temperature": 2}
            resps_list = model( **input_kwargs, len_list=len_list, task_list=["tts"], **(kwargs))
            ## decode
            resps = resps_list[0]
            wav, sr = qnt.decode_to_file(resps, out_path+f"-fix-level{fix_level}-{i}.wav", device=device)
    else:
        sys.exit("not supporting AR+NAR in this version")
    _logger.info(f"decoding done")
    

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <model-ckpt-sft> <in-audio-wav-file> <out-wave-path> \n \
        , it loads the model and run TTS based on input prompt fixing the level-0 code")
        
    
    ## default cfg and then update cfg from model, similar to inferece.py
    cfg.load_model(Path(sys.argv[1]))
    cfg.format( training=False )
    ## cfg related attributes
    dtype = cfg.inference.dtype
    amp = cfg.inference.amp
    device = cfg.device
    resp_level = cfg.model.resp_levels
    
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

    ## prepare data
    ##text
    phn_symmap = get_phone_symmap()
    #text_in = "A scientist walked through a field"
    #text_in = "The day is friday, I am happy"
    text_in = "I am Xinwei"
    #text_in = "I want to try how I beat Elon Mask in the future"
    #text_in = "A SCIENTIST WALKED THROUGH A FIELD"
    #text_in = "a large size in stockings is hard to sell"
    lang_code = "en-us"
    phns,lang = get_text(text_in, device, phn_symmap, lang=lang_code)
    #pdb.set_trace()
    ##prompt
    audio_in_path = Path(sys.argv[2])
    prompt,resps = get_emb(audio_in_path, trim_length=3, noise=0)
    ###disable prompt
    #prompt=None
    #text_in = None
    ## TTS
    set_seed()
    with torch.no_grad():
        get_tts_results(models[0], phns, prompt, lang, False, device, resps, fix_level=0, out_path=sys.argv[3])
    
    ##unload qnt models
    load_engines.cache_clear()
    unload_model()
    