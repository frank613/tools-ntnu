#!/bin/bash
##generate a csv file for audios given the folder (with subfolders)
main_dir=$1
output_file=$2
header="file_name,transcription"

echo $header > $output_file
files=$(find ${main_dir} -name "*.flac")
for file in ${files[*]}
     do
	     echo "$(readlink -e ${file}),Null" >> $output_file
     done
