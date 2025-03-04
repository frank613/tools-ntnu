#! /bin/bash
### using g2p for Vall-e for CMU-kids
set -eu
root_path=$1
out_path=$2
for kid in $root_path/kids/*; do
	if [ -d $kid ]; then
		spkID=$(basename $kid)
		signal="$kid/signal"
	fi
	for sph in $signal/*; do
		uttID=$(basename $sph)
		uttID=${uttID%".sph"}
		wav=${uttID}.wav
		mkdir -p $out_path/$spkID
		sox -v 0.99 -G -t sph "$sph" -r 24000 -t wav "$out_path/$spkID/${wav}"
	done
done
