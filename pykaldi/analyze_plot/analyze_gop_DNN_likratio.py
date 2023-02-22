import sys
import re
import os
from kaldi.asr import NnetRecognizer
import kaldi.nnet3 as _nnet3
from kaldi.matrix._matrix import Vector
from kaldi.util.table import SequentialIntVectorReader
from kaldi.util.table import SequentialMatrixReader
from kaldi.hmm import split_to_phones
from kaldi.util.io import xopen
from kaldi.tree import ContextDependency
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pdb


re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
vowel_set = set(['AA', 'IH', 'UH', 'ER', 'IH', 'OW', 'UH'])
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

def compute_like_sum(decodable, trMod, transSeq, start_pos, length):
    log_sum = 0
    for i in range(start_pos, start_pos + length): 
        loglike_vec = get_vector_for_frame(decodable, i)
        log_sum += loglike_vec._getitem(trMod.transition_id_to_pdf(transSeq[i-start_pos]))
    return log_sum

def compute_like_replaced(decodable, trMod, transSeq, tree, context, start_pos, length):
    log_sum = 0
    # print(tree.context_width())
    for i in range(start_pos, start_pos + length):
        loglike_vec = get_vector_for_frame(decodable, i)
        log_sum += loglike_vec._getitem(tree.compute(context, trMod.transition_id_to_pdf_class(transSeq[i-start_pos])))
    return log_sum

def get_likeli_for_denom(decodable, start_pos, length):  #return the total sum of log likelihood for each segment
    log_sum = 0
    for i in range(start_pos, start_pos + length):
        loglike_vec = get_vector_for_frame(decodable, i)
        log_sum += loglike_vec.max()
    return log_sum


def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 
    
def get_vector_for_frame(decodable, frame_num):
    #loglike_vec = Vector.from_size(am_nnet.get_nnet().output_dim("output"))
    loglike_vec = Vector(global_out_dim)
    decodable.get_output_for_frame(frame_num, loglike_vec)
    loglike_vec.apply_log_()
    loglike_vec._add_vec_(-1, real_logprior)
    #return loglike_vec._add_vec_(-1, am_nnet.priors()) nnet3 output by default use the log priors
    return loglike_vec

def plot(results, gops_real, outFile):
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
    
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    plt.savefig(outFile)

#variable used all over the places 
global_out_dim = None
real_logprior = None

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <nnmodel-dir> <trimodel-dir> <data-dir> <out-file>.\n \
        , it writes out the gop score for each phoneme based on the segmentation of the cano-alignment") 
    #step 0, read the files
    trans_m, acoustic_m = NnetRecognizer.read_model(sys.argv[2]+"/final.mdl")
    global_out_dim = acoustic_m.get_nnet().output_dim("output")
    real_logprior = acoustic_m.priors()
    real_logprior.apply_log_()
    # #or using this style
    # trans_model = TransitionModel()
    # am_gmm = AmDiagGmm()
    # with xopen(model_rxfilename) as ki:
    # trans_model.read(ki.stream(), ki.binary)
    # am_gmm.read(ki.stream(), ki.binary)
    tree = ContextDependency()
    with xopen(sys.argv[3]+"/tree") as ki:
        tree.read(ki.stream(), ki.binary)

    p_set = vowel_set | cons_set 
    phoneme_dict = {}
    with open(sys.argv[3]+"/phones.txt", 'r') as p_table:
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
            "ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:{0}/utt2spk scp:{0}/cmvn.scp scp:{0}/feats.scp ark:- |"
    ).format(sys.argv[4])

    #Analyze for each target phoneme, investigate the GOP of all the other phonemes that are evaluated with the target phoneme model.
    #Plot the mean for all the other model and compute the overall AUC for that phoneme
    gops_map = { p1:{ p2: [] for p2 in p_set if p2 != p1 } for p1 in p_set }  # map(p:map(p:average)
    gops_real = { p:[] for p in p_set }  # gops  for each real(unreplaced) phoneme
    with SequentialMatrixReader(feats_rspecifier) as f:
        for frame_num, (fkey, feats) in enumerate(f):
            print("processing {0}".format(fkey))
            if fkey not in align_dict:
                #print("ignore uttid: " + fkey + ", no alignment can be found")
                continue
            #step 1, segmentation (segmented: list[list[int]])
            done, segmented = split_to_phones(trans_m, align_dict[fkey])
            if not done:
                sys.exit("split to phones failed")
            pid_seq = seg_to_phonemelist(segmented, trans_m)
            #step 2 get the likelihood matrix:
            decodable_opts = _nnet3.NnetSimpleComputationOptions()
            decodable_opts.acoustic_scale = 1 
            compiler = _nnet3.CachingOptimizingCompiler.new_with_optimize_opts(acoustic_m.get_nnet(), decodable_opts.optimize_config)
            #vector_prior = acoustic_m.priors(), prior can't be provided directly, because it assumes our output layer is a log-softmax
            empty_prior = Vector(0)
            decodable = _nnet3.DecodableNnetSimple(decodable_opts, acoustic_m.get_nnet(), empty_prior, feats, compiler)
            #print("Numframes {0}, OutputDim {1}".format(decodable.num_frames(), decodable.output_dim()))
            #step 3 compute the GOP 
            for order, (p_id,start_idx) in enumerate(pid_seq):
                length_seg = len(segmented[order])
                #step 3.1, get the denom likelihood 
                denom_sum = get_likeli_for_denom(decodable, start_idx, length_seg)
                #print(p_map.get_raw_symbol(pid_best))
                p_sym_raw = p_map.get_raw_symbol(p_id)
                p_sym = p_map.get_symbol(p_id)
                #get the context and padding if needed
                if order == 0 and len(pid_seq) > 1:
                    context = [p_id, p_id, pid_seq[order+1][0]]
                elif order == len(pid_seq) - 1 and len(pid_seq) > 1:
                    context = [pid_seq[order-1][0], p_id, p_id]
                elif len(pid_seq) > 2:
                    context = [pid_seq[order-1][0], p_id, pid_seq[order+1][0]]
                else: ## len = 1
                    context = [p_id, p_id, p_id]
                #step 3.2, compute the GOPs
                if p_sym_raw in p_set:
                    real_lik = compute_like_sum(decodable, trans_m, segmented[order], start_idx, length_seg)
                    gops_real[p_sym_raw].append((real_lik - denom_sum )/length_seg)     
                    for p in p_set - set([p_sym_raw]):
                        toP = p_map.switch_symbol(p_sym, p)
                        toPid = p_map.get_phone_id(toP)
                        context[1] = toPid
                        numer = compute_like_replaced(decodable, trans_m, segmented[order], tree, context, start_idx, length_seg)
                        log_ratio_avg = (numer - denom_sum)/length_seg
                        gops_map[p_sym_raw][p].append(log_ratio_avg)         

    print("done with GOP computation")
    plot(gops_map, gops_real, sys.argv[5])






    
  
