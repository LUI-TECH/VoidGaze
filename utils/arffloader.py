from scipy.io import arff
import pandas as pd
import numpy as np

import re
from coord_Transform import AngularDisp, PosTransform
from numpy import save
import os 







def loadArff(path):
	data = arff.loadarff(path)
	dataarray = pd.DataFrame(data[0]).to_numpy()
	dataArray = np.delete(dataarray,8,1)
	for i in dataArray:
		if i[7] == b'fixation':
			i[7] = 1
		elif i[7] == b'saccade':
			i[7] = 2
		elif i[7] == b'sp':
			i[7] = 3
		elif i[7] == b'noise':
			i[7] = 4
		else:
			i[7] = 0


	f = open(path, "r")
	metadata = dict()
	while True:
		line = f.readline()
		if  "@ATTRIBUTE time INTEGER" in line:
			break
		if "%@METADATA" in line:
			word = re.findall(r'\S+', line)
			metadata[word[1]] = float(word[2])

	return dataArray, metadata

def createInputs(dataArray,metadata):

	CartPosition = np.zeros((len(dataArray),12))
	for i in range(len(dataArray)):
		(CartHead,CartGaze,GazeVec_FOV) = PosTransform(dataArray[i],metadata)

		if i == 0:
			headDisp = np.array([[0.0],[0.0],[0.0]])
			angDisp = 0.0
			angVelocity = 0.0
		if i> 0:
			headDisp = CartHead - CartHead_last

			angDisp = AngularDisp(CartHead,CartHead_last)
			angVelocity = angDisp/(dataArray[i,0] - dataArray[i-1,0])

		CartPosition[i] = np.array([CartGaze[0,0],CartGaze[1,0],CartGaze[2,0],
			CartHead[0,0],CartHead[1,0],CartHead[2,0], 
			headDisp[0,0], headDisp[1,0], headDisp[2,0],
			angDisp, angVelocity, dataArray[i,7]
			],dtype=float)
		CartHead_last = CartHead
	return CartPosition

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
	
	save('TrainSet.npy', data)
	print("size of data saved:",data.shape)




if __name__ == '__main__':

	np.load('TrainSet.npy')

	'''
	directory = '/Users/louitech_zero/Desktop/360_em_dataset/ground_truth/train/'
	files = []
	for r, d, f in os.walk(directory):
		for file in f:
			files.append(os.path.join(r, file))

	DataSaver(files)
	'''
