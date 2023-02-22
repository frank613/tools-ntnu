#!/bin/bash
#convert gop to canonical phone sequence 

gopfile=$1 

while read p; do
	#echo $p
	arr=($p)
	if [ 1 -eq ${#arr[@]} ]
	then
		temp_arr=(${arr[0]}_BEGIN)
	fi
	if [ 3 -eq ${#arr[@]} ]
	then
		phoneme=$(echo ${arr[1]} | sed -r 's/([0-9]*_)[A-Z]+//')
		temp_arr+=(${phoneme})
	fi
	if [ 0 -eq ${#arr[@]} ]
	then
		#echo ${temp_arr[@]} | sed 's/ /;/g' | sed 's/_BEGIN;/ /' | sed 's/SIL;//' | sed 's/;SIL$//g'
		echo ${temp_arr[@]} | sed 's/ /;/g' | sed 's/_BEGIN;/ /'
	fi
done <$gopfile
