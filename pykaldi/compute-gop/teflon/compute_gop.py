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
import pdb


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
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
        ##monophone model here, only one elment in the context window 
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

def writes(gop_list, key_list):
    assert(len(gop_list) == len(key_list))
    outFile = "./output-teflon/all.gop.lattice.full"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, gop in zip(key_list, gop_list):
            fw.write(key+'\n')
            for i, (p,score, n, d) in enumerate(gop):
                #fw.write("%s\t%.3f_%.3f_%.3f\n"%(p,score,n,d))
                fw.write("%d %s %.3f\n"%(i,p,score))
            fw.write("\n")



if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <model-dir> <phones.txt> <data-dir> <lattice-1best>.\n \
        , it writes out the gop score for each phoneme based on the segmentation of the cano-alignment") 
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
            "ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:{0}/utt2spk scp:{0}/cmvn.scp scp:{0}/feats.scp ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats {1}/final.mat ark:- ark:- |"
    ).format(sys.argv[4], sys.argv[2])

    # Read the lattice
    lat_dict = {}
    with SequentialLatticeReader(sys.argv[5]) as reader:
          for num_done, (key, vec) in enumerate(reader):
              lat_dict.update({key:vec})

    gops_list = []  # gops(|uttid|x|phoneme-seq|)
    key_list= []
    with SequentialMatrixReader(feats_rspecifier) as f:
        for fkey, feats in f:
            if fkey not in align_dict or fkey not in lat_dict:
                #print("ignore uttid: " + fkey + ", no alignment can be found")
                continue
            #step one, segmentation (segmented: list[list[int]])
            done, segmented = split_to_phones(trans_m, align_dict[fkey])
            if not done:
                sys.exit("split to phones failed")
            key_list.append(fkey)
            gops_list.append([])
            pid_seq = seg_to_phonemelist(segmented, trans_m)
            #step two, get the denom likelihood from lattice
            denom_lik = get_likeli_from_lat(acoustic_m, trans_m, lat_dict[fkey], feats)
            #step three compute the likelihood for numerator 
            for order, (p_id,start_idx) in enumerate(pid_seq):
                length_seg = len(segmented[order])
                denom_sum = sum(denom_lik[start_idx:start_idx + len(segmented[order])])
                p_sym_raw = p_map.get_raw_symbol(p_id)
                p_sym = p_map.get_symbol(p_id)
                num_lik = compute_like_sum(acoustic_m, trans_m, segmented[order], feats[start_idx: start_idx + length_seg])
                gops_list[len(key_list)-1].append((p_sym, (num_lik - denom_sum )/length_seg, num_lik, denom_sum))
    writes(gops_list, key_list)
    






    
  
