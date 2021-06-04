from scipy.io import arff
import pandas as pd
import re
import os 

import numpy as np
from numpy import save



from utils.coord_Transform import AngularDisp, PosTransform






def loadArff(path):
	data = arff.loadarff(path)
	dataarray = pd.DataFrame(data[0]).to_numpy()
	dataArray = np.delete(dataarray,8,1)
	delete = []

	for indice, i in enumerate(dataArray):
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

	return data, metadata

def createInputs(dataArray,metadata):

	Data = np.zeros((len(dataArray),9))
	for i in range(len(dataArray)):
		(CartHead,CartGaze,GazeVec_FOV,sphereHead,sphereGaze) = PosTransform(dataArray[i],metadata)



		if i == 0:
			headDisp = np.array([0.0,0.0])
			angDisp = 0.0
			angVelocity = 0.0
		if i> 0:
			headDisp = sphereHead - sphereHead_last

			angDisp = AngularDisp(CartHead,CartHead_last)
			angVelocity = 1000000*angDisp/(dataArray[i,0] - dataArray[i-1,0])

		Data[i] = np.array([sphereGaze[0],sphereGaze[1],
							sphereHead[0],sphereHead[1], 
							headDisp[0], headDisp[1],
							angDisp, angVelocity, dataArray[i,7]
									],dtype=float)
		sphereHead_last = sphereHead
		CartHead_last = CartHead

	return Data

def DataSaver(file_paths):
	data = None
	count = 0
	for path in file_paths:

		df , metadata = loadArff(path)
		proceeded = createInputs(df,metadata)

		if count == 0:
			data = proceeded
		else:
			data = np.append(data,proceeded,axis=0)
		count = 1

	save('/Users/louitech_zero/VoidGaze/data/TestSet.npy', data)
	print("size of data saved:",data.shape)


def Loader(path):
	return np.load(path)



if __name__ == '__main__':



	
	directory = '/Users/louitech_zero/Desktop/360_em_dataset/ground_truth/test/'
	files = []
	for r, d, f in os.walk(directory):
		for file in f:
			files.append(os.path.join(r, file))

	DataSaver(files)
	
