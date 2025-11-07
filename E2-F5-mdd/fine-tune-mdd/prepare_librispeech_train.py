import os
import sys
import pdb

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from f5_tts.model.utils import (
    convert_char_to_pinyin,
)
import re


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.flac"))
    text_dict = {}

    for line in audio_lists:
        text_path = re.sub("(-.*)(-.*?)\.","\\1.",str(line.with_suffix(".trans.txt")))
        uid = line.stem
        if uid in text_dict:
            text = text_dict[uid]
        else:
            with open(text_path, "r") as fr:
                for t_line in fr:
                    t_line = t_line.strip()
                    match = re.match("(.*?) (.*)", t_line)
                    id,text = match[1], match[2].lower()
                    text = convert_char_to_pinyin([text])[0]
                    text_dict.update({id:text})
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 20:
            continue
        #sub_result.append({"audio_path": str(line), "text": text, "duration": duration, "uid":uid})
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    
    #dataset_path = Path(os.path.join(dataset_dir, sub_set))
    #deal_with_audio_dir(next(dataset_path.iterdir()))

    
    dataset_path = Path(os.path.join(dataset_dir, sub_set))
    [
        futures.append(executor.submit(deal_with_audio_dir, audio_dir))
        for audio_dir in dataset_path.iterdir()
        if audio_dir.is_dir()
    ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 1

    tokenizer = "char"  # "pinyin" | "char"

    sub_set = "train-clean-100"
    #sub_set = "dev-clean"
    dataset_dir = "/talebase/data/speech_raw/libri_speech/LibriSpeech"
    dataset_name = f"Librispeech_{sub_set}_{tokenizer}"
    save_dir = "/home/xinweic/data/data-for-F5/libri/" + sub_set
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
