
#header="./header.txt"
header=$1
#ctm="./ctm"                
ctm=$2                

# direct the terminal to the directory with the newly split session files
# ensure that the RegEx below will capture only the session files
# otherwise change this or move the other .txt files to a different folder
                
mkdir -p tmp

for i in $ctm/*.txt;
do
    cat "$header" "$i" | sed 's/ *$//g' > ./tmp/xx.$$
    mv ./tmp/xx.$$ "$i"
done;
