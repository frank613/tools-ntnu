import os, sys
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from transformers import PreTrainedTokenizer
from transformers.utils import to_py_obj, logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from transformers.tokenization_utils_base import AddedToken
from transformers.tokenization_utils import _insert_one_token_to_ordered_list
from itertools import groupby
import torchaudio
from datasets import Dataset

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json",
    },
    "tokenizer_config_file": {
        "facebook/wav2vec2-base-960h": (
            "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json"
        ),
    },
}

# Wav2Vec2 has no max input length
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-base-960h": sys.maxsize}

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
        path_features = [{"path": feature["path"]} for feature in features]

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
        batch["path"] = path_features

        return batch


@dataclass
class DataCollatorPhoneWithPadding:
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



def get_list_of_filenames(dir, flags=None, format=".WAV"):
    filenames_list = []
    for root, directories, filenames in os.walk(dir):
        # for directory in directories:
        #    print(os.path.join(root, directory))
        #print(root)
        if flags is None:
            for filename in filenames:
                if not filename.endswith(format):
                    continue

                else:
                    filenames_list.append(os.path.join(root, filename))
                    # print(os.path.join(root,filename))

        else:
            for filename in filenames:
                if filename in flags:
                    # print(os.path.join(root,filename))
                    filenames_list.append(os.path.join(root, filename))

    return filenames_list


class PhoneMatrix:
    def __init__(self, phones):
        #self.matrix =
        #self.phones = phones
        for phone in phones:
            #self.phone = np.zeros((320,320),dtype=int)
            setattr(self,phone,np.zeros((320,320),dtype=int))

    def accumulate(self, phones, idx_info):
        for idx, phone in enumerate(phones):
            current_value = getattr(self, phone)
            idx1 = idx_info[idx, 0, 0]
            idx2 = idx_info[idx, 1, 0]
            current_value[idx1, idx2] = current_value[idx1, idx2] + 1
            setattr(self, phone, current_value)

    def current(self, phone):
        return getattr(self, phone)


class Wav2Vec2PhoneTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            **kwargs,
        )

        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.replace_word_delimiter_char = replace_word_delimiter_char

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder.keys():
            if len(token) > 1:
                self.unique_no_split_tokens.append(token)

        self._create_trie(self.unique_no_split_tokens)

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", self.word_delimiter_token))

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(
            self,
            tokens: List[str],
            group_tokens: bool = True,
            spaces_between_special_tokens: bool = False,
            output_char_offsets: bool = False,
            output_word_offsets: bool = False,
    ) -> Dict[str, Union[str, float]]:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        if len(tokens) == 0:
            return {"text": "", "char_offsets": [], "word_offsets": []}
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
        else:
            chars = tokens
            char_repetitions = len(tokens) * [1]

        # filter self.pad_token which is used as CTC-blank token
        processed_chars = list(filter(lambda char: char != self.pad_token, chars))

        # replace delimiter token
        processed_chars = [
            self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
        ]

        # retrieve offsets
        char_offsets = word_offsets = None
        if output_char_offsets or output_word_offsets:
            char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)

            if len(char_offsets) != len(processed_chars):
                raise ValueError(
                    f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                    " have to be of the same length, but are: "
                    f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                    f" {len(processed_chars)}"
                )

            # set tokens to correct processed token
            for i, char in enumerate(processed_chars):
                char_offsets[i]["char"] = char

            # retrieve word offsets from character offsets
            word_offsets = None
            if output_word_offsets:
                word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)

            # don't output chars if not set to True
            if not output_char_offsets:
                char_offsets = None

        # join to string
        join_char = " " if spaces_between_special_tokens else ""
        string = join_char.join(processed_chars).strip()

        if self.do_lower_case:
            string = string.lower()

        return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}

    @staticmethod
    def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
    ) -> List[Dict[str, Union[str, int]]]:
        end_indices = np.asarray(char_repetitions).cumsum()
        start_indices = np.concatenate(([0], end_indices[:-1]))

        offsets = [
            {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
        ]

        # filter out CTC token
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
        return offsets

    @staticmethod
    def _get_word_offsets(
            offsets: Dict[str, Union[str, float]], word_delimiter_char: str = " "
    ) -> Dict[str, Union[str, float]]:
        word_offsets = []

        last_state = "SPACE"
        word = ""
        start_offset = 0
        end_offset = 0
        for i, offset in enumerate(offsets):
            char = offset["char"]
            state = "SPACE" if char == word_delimiter_char else "WORD"

            if state == last_state:
                # If we are in the same state as before, we simply repeat what we've done before
                end_offset = offset["end_offset"]
                word += char
            else:
                # Switching state
                if state == "SPACE":
                    # Finishing a word
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                else:
                    # Starting a new word
                    start_offset = offset["start_offset"]
                    end_offset = offset["end_offset"]
                    word = char

            last_state = state
        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        return word_offsets

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if is_split_into_words:
            text = " " + text
        return (text, kwargs)

    def _decode(self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        ** kwargs
        ) -> str:
        """
        special _decode function is needed for Wav2Vec2PhoneTokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)
            # Each phone will end up with a word/phone determination token such that
            # The final output can produce a frame-wise phone label output
            result.append(self.word_delimiter_token)

        text = self.convert_tokens_to_string(result)

        #if clean_up_tokenization_spaces:
        #    clean_text = self.clean_up_tokenization(text)
        #    return clean_text
        #else:
        #    return text
        return text

    def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            output_char_offsets: bool = False,
            output_word_offsets: bool = False,
            **kwargs
    ) -> str:

        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            output_char_offsets=output_char_offsets,
            output_word_offsets=output_word_offsets,
            **kwargs,
        )

    def batch_decode(
            self,
            sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            output_char_offsets: bool = False,
            output_word_offsets: bool = False,
            **kwargs
    ) -> List[str]:
        batch_decoded = [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                output_char_offsets=output_char_offsets,
                output_word_offsets=output_word_offsets,
                **kwargs,
            )
            for seq in sequences
        ]
        if output_char_offsets or output_word_offsets:
            # transform list of dicts to dict of lists
            #return Wav2Vec2CTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})
            raise NotImplementedError

        return batch_decoded

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the `unk_token` to them).
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Example:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info(f"Adding {token} to the vocabulary")

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before)
        for token in tokens_to_add:
            if len(token) > 1:
                self._additional_special_tokens.append(AddedToken(token))
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, token)

        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)








