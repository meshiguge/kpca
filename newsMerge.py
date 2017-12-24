import csv
#(sourcename, hashprefix, articlelink, language, author,content, feedlink, metadata, published, state, summary, title, updated, year)
with open('news_article2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
    	language = row[3]
    	content = row[5]
    	state = row[9]#state
    	f = open('englisharticles.txt', 'w')
    	if(language =="EN"):
    		if(state == "content" or state=="structinfo"):
    			f.write(content+"\n")
    	f.close()
