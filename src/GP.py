from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import numpy as np
import pickle

class GaussianProcess:
	"""
		Class to wrap Gaussian Process
	"""

	def __init__(self,kernel = None):

		self.kernel = kernel
		if kernel == None:
			self.kernel = DotProduct() + WhiteKernel()
		self.gpr = GaussianProcessRegressor(kernel=kernel,random_state=0)


	def fit(self,trainFeature, trainTarget):
	'''
		fir model with training data
		inputs :
			trainFeature : array of floats (n,5)
			trainTarget : array of floats (n,2)
	'''
		print("This can take quite a while if the dataset is large")
		self.gpr = self.gpr.fit(trainFeature, trainTarget)


	def infer(self,feature):
	'''
		predict with valid data
		inputs :
			feature : array of floats (n,5)
		returns:
			mean : array of floats (n,2)
			std : array of floats (n,1)
	'''
		result = self.gpr.predict(feature,return_std=True)
		mean = result[0][0]
		std = result[0][1]
		return mean, std

	def save_model(self,path = 'gpr.pkl'):
	'''
		save model to file path
		inputs :
			path : string
	'''
		with open(path, 'wb') as f:
			pickle.dump(self.gpr,f)
		print('Model saved')

	def load_model(self,path = 'gpr.pkl'):
	'''
		load model from file path
		inputs :
			path : string
	'''
		with open(path, 'rb') as f:
			self.gpr = pickle.load(f)
		print('Model loaded')