#!/bin/bash
# get the phoneme transcription from the label folder for each uttid 
dirlist=$(find /talebase/data/speech_raw/cmu_kids_v2/kids -mindepth 1 -maxdepth 1 -type l)

for dir in ${dirlist[*]}
 do
   files=$(ls ${dir}/label)
   for file in ${files[*]}
     do
       echo -n "${file%.*} "
       #cat ${dir}/label/$file | grep 'phone>' | tr -s ' ' | cut -d' ' -f2 | tr '\n' ';' | sed 's/;$/\n/' | sed 's/([^;]*)//g' | sed 's/[a-z]//g' | sed 's/SIL;//g' | sed 's/;SIL$//g'
       cat ${dir}/label/$file | grep 'phone>' | tr -s ' ' | cut -d' ' -f2 | tr '\n' ';' | sed 's/;$/\n/' | sed 's/([^;]*)//g' | sed 's/[a-z]//g' | sed 's/SIL;//g' | sed 's/;SIL$//g' | sed 's/\+INHALE\+//g' | sed 's/\+SMACK\+//g' | sed 's/\+NOISE\+//g' | sed 's/\+EXHALE\+//g' | sed 's/\+RUSTLE\+//g' | sed 's/\+SWALLOW\+//g'
       #cat ${dir}/label/$file | grep 'phone>' | tr -s ' ' | cut -d' ' -f2 | tr '\n' ';' | sed 's/;$/\n/' | sed 's/([^;]*)//g' | sed 's/[a-z]//g'
     done
 done
