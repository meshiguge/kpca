import csv
import sys


csv.field_size_limit(sys.maxsize)
a =0
b=0
f = open('englisharticles.txt', 'a')
#(sourcename, hashprefix, articlelink, language, author,content, feedlink, metadata, published, state, summary, title, updated, year)
with open('news_article2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        language = row[3]
        content = row[5]
        state = row[9]#state
        if(language=='EN'):
            if(state == "content" or state=="structinfo"):
                a =a+1
                print(a)
                f.write(content+"\n")
        elif(language=='DE'):
            b = b+1
            print("de>>>>>",b)
f.close()

