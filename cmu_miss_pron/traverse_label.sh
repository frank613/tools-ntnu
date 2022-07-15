#!/bin/bash
# not needed to use this script for cmu.
dirlist=$(find /talebase/data/speech_raw/cmu_kids_v2/kids -mindepth 1 -maxdepth 1 -type l)

for dir in ${dirlist[*]}
 do
   files=$(ls ${dir}/label)
   for file in ${files[*]}
     do
       echo -n "${file%.*} "
       cat ${dir}/label/$file | grep 'phone>' | tr -s ' ' | cut -d' ' -f2 | tr '\n' ';' | sed 's/;$/\n/' | sed 's/([^;]*)//g' | sed 's/SIL;//g' | sed 's/[a-z]//g'
     done
 done
