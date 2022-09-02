####get the uttids with wrong phonemes
#cat /talebase/data/speech_raw/cmu_kids_v2/tables/point.tbl | grep -e "sub\|del" | cut -f 1 > uttid.temp
#cat /talebase/data/speech_raw/cmu_kids_v2/tables/point.tbl | grep -e "sub\|del" | cut -f 2 | cut -d' ' -f 3- > point.temp
#paste uttid.temp point.temp > misspron.txt

####get the label and cano files
#./traverse_label.sh > label.txt
#python phoneme_map.py label.txt 60-39.map  > label_mapped.txt
#./get_phone_from_gop.sh gop.score.all.symbol > cano.txt

##
python error_phone.py gop.score.all.symbol cano.txt label_mapped.txt


##check phoneme set
#cut -d' ' -f2 gop.score.all.symbol | sed 's/[0-9]*_[A-Z]//g' | sort -u  > libri.phone.txt

