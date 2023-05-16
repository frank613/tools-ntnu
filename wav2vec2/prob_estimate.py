import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForPreTraining, Wav2Vec2Processor
import torch
import pdb
import sys
from pathlib import Path
import re
import panda as pd

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')

def read_align(align_path):
    utt_list =[]
    with open(align_path, r) as if:
        for line in if:
            line = line.strip()
            uttid, phonemes = line.split(' ')[0], line.split(' ')[1:]
            utt_list.append((uttid, map(lambda x: re_phone.match(x).group(1), phonemes)))
    return pd.DataFrame(utt_list, columns=('uttid','phonemes')) 


#return an N(phonemes)*N(codes) numpy array
def get_dist(ali_df, processor, model, ds):

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <phoneme-alignment file> <wav2vec2-dir> <out-file>.\n \
        , it estimates the p(code|phoneme) given the dataset and its alignment for the current wav2vec2 model")

    ali_df = read_align(sys.argv[1]) 
    # load pretrained model
    model_path = sys.argv[2]
    model_path = "model_path = sys.argv[2]"
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

    # load dataset and read soundfiles
    ds = datasets.load_dataset("librispeech_asr", "clean", split="validation")

    # estimate the p(c|p)
    code_distritbuion = get_dist(ali_df, processor, model, ds)

    #dump the distribution
    pdb.set_trace()


