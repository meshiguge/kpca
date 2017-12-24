#(sourcename, hashprefix, articlelink, language, author,content, feedlink, metadata, published, state, summary, title, updated, year)
f = open('news_article2.csv','r')
line = f.readline()
print(line)
wf = open('englisharticles.txt', 'w')
a=0
b=0
while line:
	row = line.split(",")
	language = row[3]
	content = row[5]
	state = row[9]#state
	print(language)
	if(language =="EN"):
		print(state)
		if(state == "content" or state=="structinfo"):
			print("here")
			wf.write(content+"\n")
			a=a+1
			print(a)
	elif(language == 'DE'):
		b = b+1
	line = f.readline()
print(">>",b)
wf.close()


