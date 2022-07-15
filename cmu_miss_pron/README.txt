#cat /talebase/data/speech_raw/cmu_kids_v2/tables/point.tbl | grep -e "sub\|del" | cut -f 1 > uttid.temp
#cat /talebase/data/speech_raw/cmu_kids_v2/tables/point.tbl | grep -e "sub\|del" | cut -f 2 | cut -d' ' -f 3- > point.temp
#paste uttid.temp point.temp > misspron.txt

#./traverse_label.sh > label.txt

#cut -d' ' -f2 gop.score.all.symbol | sed 's/[0-9]*_[A-Z]//g' | sort -u  > libri.phone.txt

#python phoneme_map.py label.txt 60-39.map  > label_mapped.txt
