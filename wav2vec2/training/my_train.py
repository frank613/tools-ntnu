import os,sys,pdb,re,json
import torch
import datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, AutoConfig, AutoTokenizer, AutoFeatureExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from pathlib import Path
import pandas as pd


datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])
#RE for filenames
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def read_trans(trans_path):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                cur_uttid = items[0]
                trans_map[cur_uttid] = []
            phoneme = re_phone.match(items[4]).group(1)                
            if phoneme not in (sil_tokens | spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 

def extract_all_phonemes(batch):
    all_phonemes = " ".join(batch["p_text"])
    vocab = list(set(all_phonemes.split(" ")))
    return {"vocab": [vocab], "all_text": [all_phonemes]}

def load_dataset_local_from_dict(folder_path,tran_map, uttid_list):
    audio_df = pd.read_csv(folder_path)
    ds = datasets.Dataset.from_dict({"audio": audio_df['file_name'].values}).cast_column("audio", datasets.Audio(sampling_rate=16000))
    #get the array for single row
    def map_to_array(batch):
        batch["speech"] = [ item["array"] for item in batch["audio"]] 
        batch["p_text"] = []
        batch["id"] = ["lbi-"+re_uttid.match(item["path"])[2] for item in batch["audio"]]
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(" ".join(tran_map[uid]))
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100)
    #ds_filtered = ds_map
    return ds_filtered

##we don't pad, so process one entry at a time
def prepare_dataset(batch):
    batch["input_values"] = processor(audio=batch["speech"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])  
    ##manully split into words(phonemes) so that we don't need word delimeter
    #text_batch = batch["p_text"].split(" ")
    pdb.set_trace()
    #processor.tokenizer.tokenize(text=batch["p_text"])
    #processor.tokenizer.tokenize(text=text_batch, is_split_into_words= True)
    processor.tokenizer.tokenize(text=batch["p_text"], is_split_into_words= True)
    batch["labels"] = processor(text = text_batch, is_split_into_words= True).input_ids
    return batch

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <train transcription, kaldi-CTM format> <dev transcription, kaldi-CTM format> <w2v2-pretrained-model-dir> <local-data-csv-folder-train> <local-data-csv-folder-dev> <out-model-dir>.\n \
        , it fine-tuens model with standard CTC for the given training data") 
    
    tran_path_train = sys.argv[1]
    tran_path_dev = sys.argv[2]
    model_path = sys.argv[3]
    csv_path_train = sys.argv[4]
    csv_path_dev = sys.argv[5]
    #step 1, load the dataset
    tran_map_train = read_trans(tran_path_train)
    tran_map_dev = read_trans(tran_path_dev)
    uttid_list_train = tran_map_train.keys()
    uttid_list_dev = tran_map_dev.keys()  
    
    #train_ds = load_dataset_local_from_dict(csv_path_train,tran_map_train, uttid_list_train)
    dev_ds = load_dataset_local_from_dict(csv_path_dev,tran_map_dev, uttid_list_dev)
    
    #step 2, define the processor, the feature extractor can be directly loaded 
    #Howeverm the ctc-tokenizer needs to be created
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    #vocabs = train_ds.map(extract_all_phonemes, batched=True, atch_size=-1, keep_in_memory=True,  
    #        remove_columns=train_ds.column_names["train"])
    #vocabs_train = train_ds.map(extract_all_phonemes, batched=True, atch_size=-1, keep_in_memory=True)
    vocabs_dev = dev_ds.map(extract_all_phonemes, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dev_ds.column_names)
    #vocab_list = list(set(vocabs_train["vocab"][0]) | set(vocabs_dev["vocab"][0]))
    vocab_list = list(set(vocabs_dev["vocab"][0]))
    vocab_dict = {v : k+1 for k, v in enumerate(vocab_list)}
    vocab_dict["<pad>"] = 0
    with open('./vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    #tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    #config = AutoConfig.from_pretrained(model_path)
    #tokenizer_type = config.model_type if config.tokenizer_class is None else None
    #config = config if config.tokenizer_class is not None else None
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./",
        unk_token=None,
        pad_token="<pad>",
        word_delimiter_token=None
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #step 3, prepare the trainig data, we don't pad here, so disable batch
    #train_ds = train_ds.map(prepare_dataset, batched=True, batch_size=100)
    dev_ds = dev_ds.map(prepare_dataset, num_proc=1)
    #step 4, fine-tune the model, the datacollator does the padding.
    pdb.set_trace()

