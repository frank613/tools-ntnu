import os
import sys
import pdb

delimiter=' '
def merge(v1,v2):
    res = []
    for a,b in zip(v1,v2):
        res.append('{0} {1}'.format(a,b))
    return res

if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.exit("this script takes 3 arguments <ali-symbol> <ali-score> <uttid>.\n \
        , it merges symbol and score for the given uttid")

   
    with open(sys.argv[1], 'r') as inFile, open(sys.argv[2], 'r') as inFile2:
        for line in inFile:
            fields = line.strip().split(delimiter)
            if len(fields) >= 2 and fields[0] == sys.argv[3]: 
                symbols = fields[1:]
                for line2 in inFile2:
                    fields2 = line2.strip().split('[')
                    if len(fields2) >= 2 and fields2[0].strip() == sys.argv[3]:
                        scores = fields2[1].strip('[] \n').split(delimiter)

                        if len(scores) == len(symbols):
                            res = merge(symbols,scores)
                            for i,item in enumerate(res):
                                print(i,item)
                        else:
                            print('frame numbers not matching')



