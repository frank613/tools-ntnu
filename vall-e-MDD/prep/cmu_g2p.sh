#! /bin/bash
### using g2p for Vall-e for CMU-kids
set -eu
root_path=$1
out_path=$2
for kid in $root_path/kids/*; do
	if [ -d $kid ]; then
		spkID=$(basename $kid)
		text="$kid/trans"
	fi
	for utt in $text/*; do
		uttID=$(basename $utt)
		uttID=${uttID%".trn"}
		sentID=${uttID#$spkID}
		sentID=${sentID:0:3}
		sent=$(grep $sentID $root_path/tables/sentence.tbl | cut -f 3- | tr -d '[:cntrl:]')			
		echo $uttID >> $out_path/uttid
		python vall_link/g2p_sep.py --no-stress --no-punctuation "$sent" >> $out_path/phonemes
	done
done
paste $out_path/uttid $out_path/phonemes > $out_path/merged.txt
