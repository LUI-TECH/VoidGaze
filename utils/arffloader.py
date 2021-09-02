from scipy.io import arff
import pandas as pd
import re
import os 

import numpy as np
from numpy import save



from utils.coord_Transform import *









def loadArff(path , mode = "full"):
	data = arff.loadarff(path)
	dataarray = pd.DataFrame(data[0]).to_numpy()
	dataArray = np.delete(dataarray,8,1)
	delete = []

	for indice, i in enumerate(dataArray):
		if i[3] <= 0.9 :
			delete.append(indice)
			continue

		if i[7] == b'fixation':
			dataArray[indice,7] = 0
		elif i[7] == b'saccade':
			dataArray[indice,7] = 1
		elif i[7] == b'SP':
			dataArray[indice,7] = 1
		elif i[7] == b'noise':
			delete.append(indice)
	data = np.delete(dataArray,delete,0)

	f = open(path, "r")
	metadata = dict()
	while True:
		line = f.readline()
		if  "@ATTRIBUTE time INTEGER" in line:
			break
		if "%@METADATA" in line:
			word = re.findall(r'\S+', line)
			metadata[word[1]] = float(word[2])

	if mode == "full":
		return data, metadata
	elif mode == "saccade":
		return data[data[:,7]==1], metadata

	elif mode == "fixation":
		return data[data[:,7]==0], metadata





def Loader(path):
	return np.load(path)



#if __name__ == "__main__":



	
	#directory = '/Users/louitech_zero/Desktop/360_em_dataset/ground_truth/test/'
	#files = []
	#for r, d, f in os.walk(directory):
	#	for file in f:
	#	files.append(os.path.join(r, file))

	
