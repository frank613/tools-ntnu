#!/bin/sh

#  splitAlignments.py
#  
#
#  Created by Eleanor Chodroff on 3/25/15.
#
#
#
import sys,csv

if len(sys.argv) != 3:
            sys.exit("this script takes 2 arguments <in ctm file> <outdir> .")

results=[]
name=""
infile=sys.argv[1]
outdir=sys.argv[2] + '/'

try:
    with open(infile) as f:
        #next(f) #skip header
        for line in f:
            columns=line.split(" ")
            name_prev = name
            name = columns[0]
            if (name_prev != name):
                try:
                    with open(outdir + (name_prev) +".txt",'w') as fwrite:
                        writer = csv.writer(fwrite)
                        fwrite.write("\n".join(results))
                        fwrite.close()
                #print name
                except Exception, e:
                    print "Failed to write file",e
                    sys.exit(2)
                del results[:]
                results.append(line[0:-1])
            else:
                results.append(line[0:-1])
except Exception, e:
    print "Failed to read file",e
    sys.exit(1)
# this prints out the last textfile (nothing following it to compare with)
try:
    with open(outdir + (name_prev) +".txt",'w') as fwrite:
        writer = csv.writer(fwrite)
        fwrite.write("\n".join(results))
        fwrite.close()
                #print name
except Exception, e:
    print "Failed to write file",e
    sys.exit(2)
