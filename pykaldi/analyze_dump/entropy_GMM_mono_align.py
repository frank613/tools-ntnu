import sys
import re
import os
from kaldi.alignment import GmmAligner
from kaldi.util.table import SequentialIntVectorReader
from kaldi.util.table import SequentialMatrixReader
from kaldi.hmm import split_to_phones
from kaldi.util.io import xopen
from kaldi.tree import ContextDependency
from kaldi.util.table import SequentialLatticeReader
from kaldi.fstext import LatticeVectorFstMutableArcIterator
from kaldi.fstext import LatticeVectorFstStateIterator
from sklearn import metrics
import json


import numpy as np
import pdb


re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
#vowel_set = set(['AA', 'IH', 'UH', 'ER', 'IH', 'OW', 'UH'])
#cons_set = set(['B', 'CH', 'D', 'F', 'G', 'HH', 'L', 'M', 'R', 'S', 'T'])
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

def compute_entropy(in_list, minV=-20, maxV=0, nBin=50):

    copied = in_list.copy()
    for n,value in enumerate(copied):
        if value < minV:
            copied[n] = minV
        elif value > maxV:
            copied[n] = maxV
        else:
            pass
    hist1 = np.histogram(copied, bins=nBin, range=(minV,maxV), density=True)
    stats = hist1[0]
    stats = stats[stats!=0]
    ncat = len(stats)
    stats = stats/stats.sum()
    ent = round(-(stats*np.log(np.abs(stats))).sum(), 3)
    ##scale it back to 0-1
    ent = round(ent/np.log(ncat),3)
    return (ent,ncat)


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

def get_likeli_from_lat(acMod, trMod, lat, featureSeq):  #return a floatvector 
    #num_states = lat.NumStates()
    count = 0
    tran_seq = []
    #for s in LatticeVectorFstStateIterator(lat):
    for s, state in enumerate(lat.states()):
        #for arc in LatticeVectorFstMutableArcIterator(lat,s):
        for arc in lat.arcs(s):
            if arc.nextstate != s + 1:
                sys.exit("not sorted or not best path ")
            if arc.ilabel != 0:
                count +=1
                tran_seq.append(arc.ilabel)
    
    if len(featureSeq) != len(tran_seq):
        sys.exit("length of features and transition-ids are not matching")
    results = []
    for t_id,features in zip(tran_seq, featureSeq):
        results.append(acMod.log_likelihood(trMod.transition_id_to_pdf(t_id), features))

    return results

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 

def get_likeli_for_denom(aligner, featureSeq, text_list):  #return the best phoneme for the given feature segments and the total sum of likelihood

    ## single aligner that decodes many times for all the phoneme
    max_like = float('-inf')
    for text in text_list:
        results  = aligner.align_ignore(featureSeq, text)
        if results is None:
            continue
        if  -results['weight'].value2 > max_like:
            max_like = -results['weight'].value2   
    return (max_like)

def write(results, gops_real, outFile):
        #p:(closest_phoneme, mean_diff, auc_value, entropy, count_of_real, count_of_error)
    out_form = { \
                'phonemes':{},
                'summary': {"average-mean-diff": None, "average-AUC": None, "total_real": None, "total_error":None}}
    #count of phonemes}
    total_real = 0
    total_error = 0
    total_auc = 0
    total_mean_diff = 0
    for (k,v) in gops_real.items():
        real_arr = np.array(v)
        ent,nbin = compute_entropy(real_arr)
        real_label = np.stack((real_arr, np.full(len(real_arr), 0)), 1)
        scores = []
        total_real += len(v)
        for p in set(gops_real.keys()) - set([k]):
            sub_arr = np.array(results[p][k]) #for all the p phonemes that are substituted to k
            sub_label = np.stack((sub_arr, np.full(len(sub_arr), 1)), 1)
            auc_value = auc_cal(np.concatenate((real_label, sub_label)))
            scores.append((p, sub_arr.mean(), len(sub_arr), auc_value))
            total_error += len(sub_arr)
        confused_p, p_mean, num_error, auc = sorted(scores, key = lambda x: x[3])[0]
        mean_diff = round(real_arr.mean() - p_mean, 3)
        out_form["phonemes"][k] = (confused_p, mean_diff, auc, ent, len(v), num_error)
        total_auc += auc
        total_mean_diff += mean_diff
    out_form["summary"]["average-mean-diff"]=total_mean_diff/len(gops_real.items())
    out_form["summary"]["average-AUC"]=total_auc/len(gops_real.items())
    out_form["summary"]["total_real"]=total_real
    out_form["summary"]["total_error"]=total_error
    
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, "w") as f:
        json.dump(out_form, f)

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 8:
        sys.exit("this script takes 7 arguments <cano-alignment file> <model-dir> <phones.txt> <data-dir> <lexicon-file-for-denom> <words-symbol-for-denom> <out-file>>.\n \
        it replace the original alignment for different phonemes and analyze the GOPs according to the replacements")
    
    #step 0, read the files
    trans_m, acoustic_m = GmmAligner.read_model(sys.argv[2]+"/final.mdl")
  
    tree = ContextDependency()
    with xopen(sys.argv[2]+"/tree") as ki:
        tree.read(ki.stream(), ki.binary)

    p_set = vowel_set | cons_set 

    phoneme_dict = {}
    with open(sys.argv[3], 'r') as p_table:
        for line in p_table:
            pair_temp = line.split(' ')
            if len(pair_temp) != 2:
                sys.exit("wrong line in phones.txt")
            phoneme_dict.update({int(pair_temp[1]) : pair_temp[0]})
    p_map = PhonemesMap(phoneme_dict) 


    fake_lexicon = GmmAligner.read_lexicon(sys.argv[5])
    fake_words = GmmAligner.read_symbols(sys.argv[6])
    fake_text_list = []
    with open(sys.argv[6], 'r') as w_table:
        for line in w_table:
            pair_temp = line.split(' ')
            if len(pair_temp) != 2:
                sys.exit("wrong line in phones.txt")
            fake_text_list.append(pair_temp[0])

    align_dict = {}
    with SequentialIntVectorReader(sys.argv[1]) as reader:
          for num_done, (key, vec) in enumerate(reader):
              align_dict.update({key:vec})
 
    #Aligner denom
    aligner = GmmAligner(trans_m, acoustic_m, tree, fake_lexicon, fake_words, beam=128, disambig_symbols=[], acoustic_scale=1.0)
    
    # Define feature pipeline as a Kaldi rspecifier
    feats_rspecifier = ( 
            "ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:{0}/utt2spk scp:{0}/cmvn.scp scp:{0}/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
    ).format(sys.argv[4])


    #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
    #Plot the mean for all the other model and compute the overall AUC for that phoneme
    gops_map = { p1:{ p2: [] for p2 in p_set if p2 != p1 } for p1 in p_set }  # map(p:map(p:average)
    gops_real = { p:[] for p in p_set }  # gops  for each real(unreplaced) phoneme
    with SequentialMatrixReader(feats_rspecifier) as f:
        for fkey, feats in f:
            print("processing {0}".format(fkey))
            if fkey not in align_dict:
                #print("ignore uttid: " + fkey + ", no alignment can be found")
                continue
            #step one, segmentation (segmented: list[list[int]])
            done, segmented = split_to_phones(trans_m, align_dict[fkey])
            if not done:
                sys.exit("split to phones failed")
            pid_seq = seg_to_phonemelist(segmented, trans_m)
            #step two, search and compute the likelihood for denominator and numerator
            for order, (p_id,start_idx) in enumerate(pid_seq):
                length_seg = len(segmented[order])
                denom_sum = get_likeli_for_denom(aligner, feats[start_idx: start_idx + length_seg].clone(), fake_text_list)
                p_sym_raw = p_map.get_raw_symbol(p_id)
                p_sym = p_map.get_symbol(p_id)
                
                #step three, compute the GOPs for this segment
                if p_sym_raw in p_set:
                    real_lik = compute_like_sum(acoustic_m, trans_m, segmented[order], feats[start_idx: start_idx + length_seg])
                    gops_real[p_sym_raw].append((real_lik - denom_sum )/length_seg)     

                    for p in p_set - set([p_sym_raw]):
                        toP = p_map.switch_symbol(p_sym, p)
                        toPid = p_map.get_phone_id(toP)
                        numer = compute_like_replaced(acoustic_m, trans_m, segmented[order], tree, toPid, feats[start_idx: start_idx+length_seg])
                        log_ratio_avg = (numer - denom_sum)/length_seg
                        gops_map[p_sym_raw][p].append(log_ratio_avg)         

    print("done with GOP computation")
    write(gops_map, gops_real, sys.argv[7])
    






    
  
