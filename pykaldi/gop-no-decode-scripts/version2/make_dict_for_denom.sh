#!/bin/bash

if [ $# -ne 1 ]; then
  cat >&2 <<EOF
Make dictionary for the gop denominator
Usage: $0 phones.txt
EOF
   exit 1;
fi

p_list=$(cut -d' ' -f1 $1 | grep -v '[#<]' | tr '\n' ' ')
cnt=0

for i in $p_list;do
	echo "PRON_$i $i"
        echo "PRON_$i $cnt" >> words.txt
	cnt=$(($cnt+1))	
done > lexicon_for_denom.txt

perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < ./lexicon_for_denom.txt > ./lexiconp_for_denom.txt || exit 1;

/localhome/stipendiater/xinweic/kaldi/egs/librispeech/s5/utils/lang/make_lexicon_fst.py ./lexiconp_for_denom.txt | \
    fstcompile --isymbols=$1 --osymbols='./words.txt' \
      --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > ./L_for_denom.fst || exit 1;

echo "done"
