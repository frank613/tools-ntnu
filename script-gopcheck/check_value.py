import os
import sys
import pdb

delimiter=' '
lam = lambda a: True if (a>1) else False

if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit("this script takes 2 arguments <in_txt_file> field-number.\n \
        , it interpretss the field as number and prints out the line number when certain conditioin is true")

   
    with open(sys.argv[1], 'r') as inFile:
        for index,line in enumerate(inFile, 1):
            field_idx = int(sys.argv[2]) - 1 
            fields = line.strip().split(delimiter)
            if len(fields) >= field_idx:
                number = float(fields[field_idx])
                if lam(number):
                    print("Line {0}: {1}".format(index, line))

