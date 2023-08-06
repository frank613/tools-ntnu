#!/bin/bash
##generate a csv file for audios given the folder (without subfolders)
main_dir=$1
output_file=$2
header="file_name,transcription"

echo $header > $output_file
files=$(ls ${main_dir})
for file in ${files[*]}
     do
       echo "${file},Null" >> $output_file
     done
