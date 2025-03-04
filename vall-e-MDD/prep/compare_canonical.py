import sys
import re
import pdb

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])

#pad_sil_token on begin and end of the sequence if not None   
def read_trans(trans_path, pad_sil_token=None):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                if pad_sil_token: ##add SIL at the begining and end of the sequence 
                    if cur_uttid != "":
                        trans_map[cur_uttid].append(pad_sil_token)
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = [pad_sil_token]
                else:
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = []
                cur_uttid = items[0]
            phoneme = re_phone.match(items[4]).group(1)                
            if phoneme not in (sil_tokens | spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    if pad_sil_token and trans_map[cur_uttid][-1] != pad_sil_token:
        trans_map[cur_uttid].append(pad_sil_token)
    return trans_map 

def read_tts(tts_file):
    trans_map = {}
    vocab = set()
    with open(tts_file, "r") as ifile:
        for line in ifile:
            items = line.strip().split("\t")
            if len(items) != 2:
                sys.exit("input line should be separated with tabs")
            uttid = items[0]
            phone_w_list = items[1].split()
            all = map( lambda x: x.split("-"), phone_w_list)
            if uttid in trans_map:
                sys.exit("duplicated uttid")
            phonemes = []
            for p_list in all:
                phonemes = phonemes + p_list
            trans_map[uttid] = phonemes
            vocab = vocab.union(set(phonemes))
    return (trans_map, vocab)


if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <tts-phoneme> <kaldi-CTM> \n \
        , it compares the canonical phonemes")
    tts_file = sys.argv[1]
    ctm_file = sys.argv[2]
    #read tts-input
    tts_map, vocabs = read_tts(tts_file)
    #read ctm
    ctm_map = read_trans(ctm_file) 
    pdb.set_trace()    
    
    

