sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/0/zero /g" -e "s/1/one /g" -e "s/2/two /g" -e "s/3/three /g" -e "s/4/four /g" -e "s/5/five /g" -e "s/6/six /g" -e "s/7/seven /g" -e "s/8/eight /g" -e "s/9/nine /g" < corpuses/dbArticles/englisharticles.txt | tr -c "A-Za-z'_ \n" " " > e1
tr A-Z a-z < e1 > e2
