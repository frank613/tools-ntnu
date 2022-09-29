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
  ##wether to keep streess in the target phoenme, Ture/False
  keepStress=$3
  dict=$4
  if [ $keepStress == "True" ]; then
	  gawk -v frm=$frm -v to=$to '{ a=$1"\t"; for(i=2; i<=NF; i++) { if ($i ~ frm) {match($i, /[A-Z]+([0-9])*/, ary); { a=a to ary[1] " "}} else {a=a $i " "}} {print substr(a, 1, length(a)-1)}}' $dict | sort -u > ./out/$1_$2.lex
  elif [ $keepStress == "False" ]; then
	  gawk -v frm=$frm -v to=$to '{ a=$1"\t"; for(i=2; i<=NF; i++) { if ($i ~ frm) {a=a to " "} else {a=a $i " "}} {print substr(a, 1, length(a)-1)}}' $dict | sort -u > ./out/$1_$2.lex
  else
	echo "the third argument must be False/True" 1>&2 && exit 1
  fi
}

#generate_dict 'AA' 'OK'  True  $inFile
#generate_dict 'AA' 'OK'  False  $inFile


#cat phones.txt | cut -d' ' -f1 | grep '[A-Z]\+[0-9]' | cut -d'_' -f1 | sed 's/\([A-Z]\+\).*/\1/g' | sort -u | tr '\n' ' '
vowel_set=(AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW)
cons_set=(B CH D DH F G HH JH K L M N NG P R S SH SIL SPN T TH W V W Y Z ZH)

for vowel in ${vowel_set[@]};do
	for vowel2 in ${vowel_set[@]};do
		generate_dict $vowel $vowel2 True $inFile
	done
	for cons in ${cons_set[@]};do
		generate_dict $vowel $cons False $inFile
	done	
done

for cons in ${cons_set[@]};do
	for all in ${vowel_set[@]} ${cons_set[@]};do
		generate_dict $cons $all False $inFile
	done
done
