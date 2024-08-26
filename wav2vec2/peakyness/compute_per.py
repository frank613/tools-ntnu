import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from pathlib import Path
import jiwer
import pdb

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')


def read_res(res_path):
    con_list =[]
    bc_list = []
    hyp_map = {}
    with open(res_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split(',')
            if len(items) != 6:
                sys.exit("input res format error")
            con_list.append(float(items[3]))
            bc_list.append(float(items[4]))
            hyp_map.update({items[0]:items[5]})
    return con_list,bc_list,hyp_map 


def read_ref(ref_path):
    ref_map={}
    with open(ref_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 2:
                sys.exit("input trasncription file format error")
            ref_map.update({items[0]:items[1]})
    return ref_map

#return tuples of (hyp, ref)       
def filter_ref(hyp_map, ref_map):
    filtered_pair_list = []
    vocabs = list(set((" ".join(list(hyp_map.values())).split(' '))))
    for k,v in hyp_map.items():
        if k in ref_map.keys():
            ref_filtered = [ phone for phone in ref_map[k].split(';') if phone in vocabs]
            filtered_pair_list.append((v, " ".join(ref_filtered)))

    return filtered_pair_list

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <hyp-file> <ref-file-mapped> \n \
        , it prints out ConEN, BC and PER") 
    
    #step 0, read the files
    con_list, bc_list, hyp_map = read_res(sys.argv[1]) 
    ref_map = read_ref(sys.argv[2])
    filtered_pair_list = filter_ref(hyp_map, ref_map) 

    hyp_ref = list(zip(*filtered_pair_list))

    #compute and report
    output = jiwer.process_words(list(hyp_ref[1]), list(hyp_ref[0]))
    print("PER: {:.4f}".format(output.wer))
    print("average-conEN: {:.4f}".format(sum(con_list)/len(con_list)))
    print("average-BC: {:.4f}".format(sum(bc_list)/len(bc_list)))
    
    
    
    
    
       







    
  
