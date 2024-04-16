from transformers import Wav2Vec2CTCTokenizer

class My_Wav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):
 
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
            vocab_file,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            replace_word_delimiter_char=" ",
            do_lower_case=False,
            **kwargs
        )

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", ""))