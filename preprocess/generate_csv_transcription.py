import sys
import os
import re
import json
import pdb
from functools import reduce

header=("file_name","transcription","p_scores")
records = []
re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')

if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <score.json> <target-dir>. generate a csv file for audios given the folder (without subfolders)")
    with open(sys.argv[1], 'r') as inf:        
        anno = json.load(inf)

    for f in os.listdir(sys.argv[2]):
        f = f.split('/')[-1]
        uid = f.split('.')[0]
        if re.match('.*\.WAV', f):
            if uid in anno:
                word_phones = [ item for item in [ word["phones"] for word in anno[uid]["words"]]]
                phonemes = reduce(lambda z, y:z+y, word_phones)
                phonemes = [re_phone.match(item).group(1) for item in phonemes]

                w_scores =  [item for item in [ word["phones-accuracy"] for word in anno[uid]["words"]]]
                p_scores = reduce(lambda z, y:z+y, w_scores)
                item = (f, " ".join(phonemes), " ".join([ str(score) for score in p_scores]))
                records.append(item)

    with open(sys.argv[2] + '/metadata.csv', 'w') as of:
        of.write(','.join(header)+'\n')
        for item in records:
            of.write(','.join(item)+'\n')



