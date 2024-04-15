#!/bin/bash
##generate a audio folder for links from another folder(with subfolders)
main_dir=$1
output_folder=$2
dirlist=$(find $1 -type d)
header="file_name,transcription"

mkdir -p $output_folder
for dir in ${dirlist[*]}
 do
   files=$(find ${dir} -name "*.WAV")
   for file in ${files[*]}
     do
     	ln -s $file $output_folder  
     done
 done
