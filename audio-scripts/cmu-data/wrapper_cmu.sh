#!/bin/bash
#wrapper to create textgrid file for a given CTM/alignment dir
stage=1
aliDir=$1
outDir=$2
audioDir=$3

pctm=$1/all.phonemes.ctm
wctm=$1/all.words.ctm
headr=./header.txt

words_out=$outDir/ctms-words
phones_out=$outDir/ctms-phones
textgrid_out=$outDir/textgrid

if [ $stage -le 1 ]; then
	if [ ! -d $words_out ]; then
		mkdir -p $words_out
	fi
	if [ ! -d $phones_out ]; then
		mkdir -p $phones_out
	fi
	if [ ! -d $textgrid_out ]; then
		mkdir -p $textgrid_out
	fi
	python splitAlignments.py $pctm $words_out
	python splitAlignments.py $wctm $phones_out

	./add_header.sh $headr $words_out
	./add_header.sh $headr $phones_out
fi

#/Applications/Praat.app/Contents/MacOS/Praat --run "./combined_textgrid.praat.txt" $words_out $phones_out $audioDir 30
./praat --run "combined_textgrid_cmu.praat.txt" $words_out $phones_out $textgrid_out $audioDir 30

