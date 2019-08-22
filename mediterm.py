import requests
from bs4 import BeautifulSoup
import re
import time

red_match = "<.*?>"

wordslist = []



#readpre = open("./raw_data/vocabs",'r')
#writef = open("./raw_data/web_train.txt",'a')

#line = readpre.readline()

'''
while line != "":
	wordslist.append(line)
	line = readpre.readline()

print(len(wordslist))
'''

charalist = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']

#test_length = 10
index = 0
for current_chara in charalist:
	#link = "https://www.dictionary.com/browse/" + currentword
	link = "https://www.medicinenet.com/script/main/alphaidx.asp?p="+current_chara+"_dict"
	


	print("Getting chara ", current_chara)
	res = requests.get(link)
	res.encoding = 'utf-8'
	#print(res.text)
	plain_text = res.text
	text_line = []
	abd_list = ["[","]"]
	reduce_list = ["<p>","<b>","</p>","</b>",'<span id="cntid"></span>','<div class="apPage">']




	soup = BeautifulSoup(plain_text)
	#print(soup.prettify())
	#print(soup.find_all("li"))
	soupx = soup.find(class_ = "AZ_results")
	#print(soupx)
	#print("-----")
	soupxx = soupx.find_all("li")
	#print(soupxx)

	
	for i in range(len(soupxx)):
		soup_ori = str(soupxx[i])
		soup_red = re.sub(red_match,"",str(soupxx[i]))
		#soup_red = str(soupxx[i])
		for abd in abd_list:
			soup_red = soup_red.replace(abd,"")


		text_line.append(soup_red)
		text_line[i] = text_line[i].replace("/"," or ")

		soup_ori = soup_ori.replace(soup_red,"")
		soup_ori = soup_ori.replace("<li><a href=","")
		soup_ori = soup_ori.replace('"',"")
		soup_ori = soup_ori.replace('></a></li>',"")

		print(text_line[i])
		print(soup_ori)

		link = soup_ori

		print("Getting term ", text_line[i])
		res = requests.get(link)
		res.encoding = 'utf-8'
		#print(res.text)
		plain_text = res.text
		termpage = BeautifulSoup(plain_text)
		#print(soup.prettify())
		#print(soup.find_all("li"))
		termsoupx = termpage.find(class_ = "apPage")

		term_result = str(termsoupx)
		term_result = term_result.split("</p>")[0]

		for reducer in reduce_list:
			term_result = term_result.replace(reducer,"")


		print(term_result)
		#print("-----")
		#print(soupxx)

		file_name = "./detail/" + text_line[i] + ".txt"
		writef = open(file_name,'a')
		writef.write(term_result)

	




	

	'''
	index += 1
	if index >= test_length:
		break
	'''
	file_name = "./list/"+current_chara + ".txt"
	writef = open(file_name,'a')
	for text_single in text_line:
		writef.write((text_single+"\n"))

	#soup_red = re.sub(red_match,"",soup)
	#print(soup_red)
