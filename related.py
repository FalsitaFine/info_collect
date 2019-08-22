import os
import re

file_dir = "./detail/"
keyword = "breast cancer"

related_list = []

for files in os.listdir(file_dir):
	flag = 0
	filename = files
	print(filename)
	if filename != ".DS_Store":
		word = str(filename).replace(".txt",'')
		filename = file_dir + str(filename)
		readf = open(filename,'r')
		line = readf.readline()
		while(line):
			if (re.search(keyword, line, re.IGNORECASE)):
				flag = 1
				break
			line = readf.readline()
		if flag == 1:
			related_list.append(word)


print(related_list)
out_dir = "./dictionary/" + keyword + ".txt" 
writef = open(out_dir,'a')

for word in related_list:
	writef.write(word+"\n")




