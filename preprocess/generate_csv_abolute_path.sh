#!/bin/bash
##generate a csv file for audios given the folder (with subfolders), the input shoud be an absolute path
main_dir=$1
output_file=$2
dirlist=$(find $1 -mindepth 1 -maxdepth 1 -type l)
header="file_name,transcription"

echo $header > $output_file
for dir in ${dirlist[*]}
 do
   files=$(ls ${dir}/signal/*)
   for file in ${files[*]}
     do
       echo "${file},Null" >> $output_file
     done
 done
