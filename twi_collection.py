import twint
import os

source_dir = "./dictionary/"
limit_info = 100
mode = "all"

for files in os.listdir(source_dir):
	filename = files
	print(filename)
	if filename != ".DS_Store":
		word = str(filename).replace(".txt",'')

		filename = source_dir + str(filename)
		readf = open(filename,'r')
		line = readf.readline()
		while(line):
			line = line.replace("\n","")
			if mode == "all":
				out_dir = "./twi_collection/" + word + "/" + line + ".csv" 
				conf = twint.Config()
				conf.Search = line
				conf.Output = out_dir
				conf.Limit = limit_info
				conf.Store_csv = True

				twint.run.Search(conf)
			line = readf.readline()


