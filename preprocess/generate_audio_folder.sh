#!/bin/bash
##generate a audio folder for links from another folder(with subfolders)
main_dir=$1
output_folder=$2
dirlist=$(find $1 -mindepth 1 -maxdepth 1 -type l)
header="file_name,transcription"

mkdir $output_folder
for dir in ${dirlist[*]}
 do
   files=$(ls ${dir}/signal/*)
   for file in ${files[*]}
     do
     	ln -s $file $output_folder  
     done
 done
