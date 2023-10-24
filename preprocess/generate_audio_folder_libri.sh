#!/bin/bash
##generate a audio folder for links from another folder(with subfolders)
main_dir=$1
output_folder=$2
dirlist=$(find $1 -mindepth 2 -maxdepth 2 -type d)
header="file_name,transcription"

mkdir $output_folder
for dir in ${dirlist[*]}
 do
   files=$(find ${dir} -name "*.flac")
   for file in ${files[*]}
     do
     	ln -s $file $output_folder  
     done
 done
