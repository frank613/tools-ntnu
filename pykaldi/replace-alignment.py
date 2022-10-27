import sys
import re
from kaldi.alignment import GmmAligner
from kaldi.util.table import SequentialIntVectorReader
from kaldi.util.table import SequentialMatrixReader
from kaldi.hmm import split_to_phones
from kaldi.util.io import xopen
from kaldi.tree import ContextDependency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
vowel_set = set(['AA', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
cons_set = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'W', 'V', 'W', 'Y', 'Z', 'ZH'])

# cur_match = re_phone.match(fields[1])
#         if (cur_match):
#             cur_phoneme = cur_match.group(1)

xstr = lambda s: s or ""
class PhonemesMap:
    
    def __init__(self, phoneme_dict) -> None:
        self._dict = phoneme_dict
        self._dict_invert = { v:k for k,v in phoneme_dict.items() }
        
    def get_phone_group(self, p_in): # p_in = AH or AH0_B
        return [ k for k,v in self._dict.items() if re_phone.match(v).group(1) == re_phone.match(p_in).group(1) ]

    def get_raw_symbol(self, p_id):
        return re_phone.match(self._dict[p_id]).group(1)

    def get_symbol(self, p_id):
        return self._dict[p_id]

    def get_phone_id(self, p_in): #p_in = AH0_B
        if not p_in in self._dict_invert.keys():
            sys.exit("can't get the id for the given phoneme, not in the phones.txt")
        return self._dict_invert[p_in] 

    def switch_symbol(self, fromP, toP): #keep the stress if the target is a vowel  fromP/toP = AH or AH0_B
        comps_frm = re_phone.match(fromP)
        comps_to = re_phone.match(toP)

        if toP in vowel_set:
            return xstr(comps_to.group(1)) + xstr(comps_frm.group(2)) + xstr(comps_frm.group(3))
        elif toP in cons_set:
            return xstr(comps_to.group(1)) + xstr(comps_frm.group(3))
        else:
            sys.exit("phoneme substitution not supported")


def seg_to_phonemelist(segmented, trMod):
    r_list = [] #list of pair (phone, offset-0-based)
    count = 0 
    for s_vector in segmented:
        if len(s_vector) == 0:
            sys.exit("wrong segmentation detected")
        r_list.append((trMod.transition_id_to_phone(s_vector[0]), count))
        count += len(s_vector)
    return r_list

def compute_like_sum(acMod, trMod, transSeq, featureSeq):
    if len(featureSeq) != len(transSeq):
        sys.exit("length of features and transition-ids are not matching")
    results = 0
    for t_id,features in zip(transSeq, featureSeq):
        results += acMod.log_likelihood(trMod.transition_id_to_pdf(t_id), features)
    return results

def compute_like_replaced(acMod, trMod, transSeq, tree, toP_id, featureSeq):
    if len(featureSeq) != len(transSeq):
        sys.exit("length of features and transition-ids are not matching")
    results = 0 
    for t_id, features in zip(transSeq, featureSeq): 
        ##monophone model here 
        pdf_id = tree.compute([toP_id], trMod.transition_id_to_pdf_class(t_id))
        results += acMod.log_likelihood(pdf_id, features)
    return results
    # pair_to_transition_id(trans_state:int, trans_index:int) → int
    # trMod.tuple_to_transition_state(toP_id, hmm_state:int, pdf:int, self_loop_pdf:int) → int
    # transition_id_to_hmm_state(trans_id:int) → int
    # transition_id_to_pdf(trans_id:int) → int
    # transition_id_to_phone(trans_id:int) → int

    # pair_to_transition_id(trans_state:int, trans_index:int) → int
    # transition_id_to_transition_index(trans_id:int) → int
    # transition_id_to_transition_state(trans_id:int) → int

def plot(results, fromP):
    SIZE = (12,4)
    fig, axs = plt.subplots(len(results.keys()), figsize=(12,4*len(results.keys())))
    for ax, (k,v) in zip(axs, results.items()):
        like_arr = np.array(v)
        ax.hist(like_arr, color='g', density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='real')
        ax.axvline(like_arr.mean(), color='g', linestyle='dashed', linewidth=1, label='error-mean')  
        ax.axvline(0, color='r', linestyle='dashed', linewidth=1, label='error-mean')  
        ax.legend(loc ="upper left")
        ax.set_title('{0}<-{1}, Diff_mean={2}'.format(k, fromP, round(0 - like_arr.mean(), 3)))
    plt.savefig("./output/{0}/all.png".format(fromP))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <model-dir> <phones.txt> <data-dir> <target-phoneme>.\n \
        it replace the original alignment and output per-frame likelihood")
    
    #step 0, read the files
    trans_m, acoustic_m = GmmAligner.read_model(sys.argv[2]+"/final.mdl")
    # #or using this style
    # trans_model = TransitionModel()
    # am_gmm = AmDiagGmm()
    # with xopen(model_rxfilename) as ki:
    # trans_model.read(ki.stream(), ki.binary)
    # am_gmm.read(ki.stream(), ki.binary)
    tree = ContextDependency()
    with xopen(sys.argv[2]+"/tree") as ki:
        tree.read(ki.stream(), ki.binary)

    targetP = sys.argv[5]
    if targetP not in vowel_set | cons_set:
        sys.exit("target phoneme not supported")
    p_set = vowel_set | cons_set - set([targetP])
    p_set = ['S', 'AH', 'AO', 'D', 'F']

    phoneme_dict = {}
    with open(sys.argv[3], 'r') as p_table:
        for line in p_table:
            pair_temp = line.split(' ')
            if len(pair_temp) != 2:
                sys.exit("wrong line in phones.txt")
            phoneme_dict.update({int(pair_temp[1]) : pair_temp[0]})
    p_map = PhonemesMap(phoneme_dict) 

    align_dict = {}
    with SequentialIntVectorReader(sys.argv[1]) as reader:
          for num_done, (key, vec) in enumerate(reader):
              align_dict.update({key:vec})
 
    # Define feature pipeline as a Kaldi rspecifier
    feats_rspecifier = ( 
            "ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:{0}/utt2spk scp:{0}/cmvn.scp scp:{0}/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
    ).format(sys.argv[4])

    results = { p:[] for p in p_set }  # map of likelihood ratio list for all phoneme pairs
    with SequentialMatrixReader(feats_rspecifier) as f:
        for fkey, feats in f:
            if fkey not in align_dict:
                #print("ignore uttid: " + fkey + ", no alignment can be found")
                continue
            #step one, segmentation (segmented: list[list[int]])
            done, segmented = split_to_phones(trans_m, align_dict[fkey])
            if not done:
                sys.exit("split to phones failed")
            pid_seq = seg_to_phonemelist(segmented, trans_m)
            #step two, search and compute the likelihood for denominator
            for order, (p_id,start_idx) in enumerate(pid_seq):
                p_sym_raw = p_map.get_raw_symbol(p_id)
                p_sym = p_map.get_symbol(p_id)
                if p_sym_raw == targetP:
                    denom = compute_like_sum(acoustic_m, trans_m, segmented[order], feats[start_idx: start_idx+len(segmented[order])])
                    #step three, replace the transition id and compute the ratio
                    for p in p_set:
                        toP = p_map.switch_symbol(p_sym, p)
                        toPid = p_map.get_phone_id(toP)
                        numer = compute_like_replaced(acoustic_m, trans_m, segmented[order], tree, toPid, feats[start_idx: start_idx+len(segmented[order])])
                        log_ratio_avg = (numer - denom)/len(segmented[order])
                        results[p].append(log_ratio_avg)                
    plot(results, targetP)
    






    
  
