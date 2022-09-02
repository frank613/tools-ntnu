import sys
#from enum import Enum

#class pType(Enum):
#    H = 64
#    M = 48
#    L = 39

def main(fIn,fMap):

    map_file = open(fMap, 'r')
    pDict = {}
    for line in map_file:
        fields  = line.split()
        if len(fields) == 0 or len(fields) > 2:
            sys.exit("illegal line in the map file, must be one ortwo columns")
        elif len(fields) == 1:
            continue
        else:
            pDict.update({fields[0].upper():fields[1].upper()})

    in_file = open(fIn, 'r')
    for line in in_file:
        fields = line.split()
        if len(fields) != 2:
                sys.exit("illegal line in the input file")
        pSet = fields[1].split(';')
        pSet_mapped = [pDict.get(p, p) for p in pSet]
        print("{0} {1}".format(fields[0], (';').join(pSet_mapped)) )


if __name__ == "__main__":
    #if len(sys.argv) != 4 and sys.argv[3] not in [pType.H, pType.M, pTyle.L] :
    if len(sys.argv) != 3 :
        #sys.exit("this script takes  arguments for <input-file> <map-file> and a number in 64,49 or 39 as target phoneme set")
        sys.exit("this script takes  arguments for <input-file> <map-file> which maps phonemes in col1 to col2")

    main(sys.argv[1], sys.argv[2])


    
