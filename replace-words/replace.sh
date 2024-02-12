#!/bin/bash
# replace unique words in one sentence with 

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input-text-file>" >&2
fi


inFile=$1
##get the top words
#cut -d' ' -f2- /localhome/stipendiater/xinweic/kaldi/egs/librispeech/s5/data/gop_combined/text | tr ' ' "\n" | sort | uniq -c | sort -n -k 1.1 -r | head -n5 | sed 's/^ *//' | cut -f2##the replacements

original_words=(THE AND OF TO A)
replace_words=(B LET BIG K O)

if [ "${#original_words[@]}" -ne "${#replace_words[@]}" ]; then
	echo "number of tokens in the arrays not match" && exit 1
fi

length=${#original_words[@]}
while read p;do
	for i in $(eval echo {0..$length})
	  do 
	  	echo $p
		p=$(echo $p | sed "s/${original_words[$i]}/${replace_words[$i]}/g")
	  done
	echo $p
done < $inFile
