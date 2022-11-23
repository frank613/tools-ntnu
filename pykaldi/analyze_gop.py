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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pdb


re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
#vowel_set = set(['AA', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
#vowel_set = set(['AA', 'AH', 'AY', 'EH', 'ER', 'IH', 'OW', 'UH'])
vowel_set = set(['AA', 'IH', 'UH', 'ER', 'IH', 'OW', 'UH'])
#cons_set = set(['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'W', 'V', 'W', 'Y', 'Z', 'ZH'])
#cons_set = set(['B', 'CH', 'D', 'F', 'G', 'HH', 'K', 'L', 'N', 'R', 'S', 'SH', 'T'])
cons_set = set(['B', 'CH', 'D', 'F', 'G', 'HH', 'L', 'M', 'R', 'S', 'T'])

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

def plot(results, gops_real):
    #SIZE = (12,4)
    fig, axs = plt.subplots(len(gops_real.keys()), figsize=(12,4*len(gops_real.keys())))
    nop = 5
    color_set = cm.rainbow(np.linspace(0, 1, nop+1)) #only plot the top 5 confusion phonemes
    for ax, (k,v) in zip(axs, gops_real.items()):
        print("plotting " + k)
        real_arr = np.array(v)
        real_label = np.stack((real_arr, np.full(len(real_arr), 0)), 1)
        ax.hist(real_arr, color=color_set[0], density=True, range=[-100, 10], bins=100, histtype='stepfilled', alpha=0.5, label='substituted')
        ax.axvline(real_arr.mean(), color=color_set[0], linestyle='dashed', linewidth=1, label='real-' + k)
       
        scores = []
        for p in set(gops_real.keys()) - set([k]):
            sub_arr = np.array(results[p][k]) #for all the p phonemes that are substituted to k
            sub_label = np.stack((sub_arr, np.full(len(sub_arr), 1)), 1)
            auc_value = auc_cal(np.concatenate((real_label, sub_label)))
            scores.append((p, sub_arr.mean(), len(sub_arr), auc_value))
        
        for items,c in zip(sorted(scores, key = lambda x: x[1], reverse = True)[:nop], color_set[1:]):
            ax.axvline(items[1], color=c, linestyle='dashed', linewidth=1, label="{0} {1}s: diff:{2} auc:{3}".format(items[2], items[0], real_arr.mean()-items[1], items[3])) #phone: diff-AUC-number 
        num_sum = round(sum([ items[2] for items in scores ]), 5)
        mean_all = sum([ round(items[1]*items[2]/num_sum, 3) for items in scores ])
        ax.legend(loc ="upper left")
        ax.set_title('{0}, overall_mean_diff={1}'.format(k, round(real_arr.mean() - mean_all, 3)))
    
    outFile = "./analyze-gop-auc/all.png"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <model-dir> <phones.txt> <data-dir> <lattice-1best>.\n \
        it replace the original alignment for different phonemes and analyze the GOPs according to the replacements")
    
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

    p_set = vowel_set | cons_set 

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

    # Read the lattice
    lat_dict = {}
    with SequentialLatticeReader(sys.argv[5]) as reader:
          for num_done, (key, vec) in enumerate(reader):
              lat_dict.update({key:vec})

    #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
    #Plot the mean for all the other model and compute the overall AUC for that phoneme
    gops_map = { p1:{ p2: [] for p2 in p_set if p2 != p1 } for p1 in p_set }  # map(p:map(p:average)
    gops_real = { p:[] for p in p_set }  # gops  for each real(unreplaced) phoneme
    with SequentialMatrixReader(feats_rspecifier) as f:
        for fkey, feats in f:
            if fkey not in align_dict or fkey not in lat_dict:
                #print("ignore uttid: " + fkey + ", no alignment can be found")
                continue
            #step one, segmentation (segmented: list[list[int]])
            done, segmented = split_to_phones(trans_m, align_dict[fkey])
            if not done:
                sys.exit("split to phones failed")
            pid_seq = seg_to_phonemelist(segmented, trans_m)
            #step two, get the denom likelihood from lattice
            denom_lik = get_likeli_from_lat(acoustic_m, trans_m, lat_dict[fkey], feats)
            #step three, search and compute the likelihood for denominator
            for order, (p_id,start_idx) in enumerate(pid_seq):
                length_seg = len(segmented[order])
                denom_sum = sum(denom_lik[start_idx:start_idx + len(segmented[order])])
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
    plot(gops_map, gops_real)
    






    
  
