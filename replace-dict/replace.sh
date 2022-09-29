#!/bin/bash
# replace unique words in one sentence with 

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input-dict-file>" >&2
    exit 1
fi


inFile=$1

generate_dict() {
  frm=$1
  to=$2
  dict=$3
  awk -v frm=$frm -v to=$to '{ a=$1"\t"; for(i=2; i<=NF; i++) {($i == frm)?a=a to " ":a=a $i " "} {print substr(a, 1, length(a)-1)}}' $dict > ./out/$1_$2.lex
}

generate_dict 'AA0' 'OK' $inFile
generate_dict 'EH1' 'OK' $inFile
