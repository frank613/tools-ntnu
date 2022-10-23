#!/bin/bash
#convert gop to canonical phone sequence 

gop_cano=./gop.score.all.symbol.orig
gop_subdir1=/localhome/stipendiater/xinweic/kaldi/egs/librispeech/s5/exp/gop_replace_all
gop_subdir2=/localhome/stipendiater/xinweic/kaldi/egs/librispeech/s5/exp/gop_replace_all_v2

dirlist=$(find $gop_subdir1 -mindepth 1 -maxdepth 1 -type d)
#dirlist=/localhome/stipendiater/xinweic/kaldi/egs/librispeech/s5/exp/gop_replace_all/AA_B

for dir in ${dirlist[*]};do
	fname=$(basename $dir)
	echo "prcessing ${fname}"
	fro_to=(${fname//_/ })
	if [ ! 2 -eq ${#fro_to[@]} ];then
		echo "bad-dir-name"
		continue
	fi
	if [ -d output2/${fro_to[1]} ];then
		continue
	fi
	if [ -d $gop_subdir2/$fname ];then
		mkdir -p output2/${fro_to[1]}
		python analyze_gop_compare.py $dir/gop.score.all.symbol $gop_subdir2/$fname/gop.score.all.symbol $gop_cano ${fro_to[0]} ${fro_to[1]}
	fi
done


