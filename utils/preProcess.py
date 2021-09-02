from utils.coord_Transform import *
from utils.arffloader import *
import numpy as np
import numpy  
import pylab
import os
class GazeHeadMovement(object):
	def __init__(self,directory, mode = "full"):
		super(GazeHeadMovement,self).__init__()
		
		data, meta = loadArff(directory, mode)
		self.data = data
		self.size = len(data)
		self.head_cart = np.zeros((self.size,3,1))
		self.gaze_cart = np.zeros((self.size,3,1))
		self.gaze_fov = np.zeros((self.size,3,1))
		self.timestamp = data[:,0]/1000000.0 #time in second
		self.timestamp = self.timestamp.reshape(len(self.timestamp),1)

		self.meta = meta
		self.angle_head = Ang2Rads(data[:,6])
		self.label = data[:,7]

		self.ComputeCoord()
		

	def ComputeCoord(self):
		for i in range(self.size):
			(self.head_cart[i],self.gaze_cart[i],self.gaze_fov[i]) = PosTransform(self.data[i,:],self.meta)

	def getVelocity(self, target):
		disp = self.getDisp(target)
		timediff = self.timeDiff()

		velocity = np.zeros(disp.shape)
		for i in range(self.size):
			if i != 0:
				velocity[i] = disp[i]/timediff[i]
		return velocity

	def getAngularVelocity(self, target):
		angdisp = self.getAngularDisp(target)
		timediff = self.timeDiff()
		velocity = np.zeros(angdisp.shape)
		for i in range(self.size):
			if i != 0:
				velocity[i] = angdisp[i]/timediff[i]
		return velocity

	def getAcc(self, target):
		v = self.getVelocity(target)
		timediff = self.timeDiff()

		acc = np.zeros(v.shape)
		for i in range(self.size):
			if i > 1:
				acc[i] = v[i]/timediff[i]
		return acc

	def getAngularAcc(self, target):
		v = self.getAngularVelocity(target)
		timediff = self.timeDiff()
		acc = np.zeros(v.shape)
		for i in range(self.size):
			if i > 1:
				acc[i] = v[i]/timediff[i]
		return acc



	def getDisp(self,target):
		if target == 'head':
			data = self.head_cart
		elif target == 'gaze':
			data = self.gaze_cart
		else:
			print("only head and gaze velocity can be calculated")
			return None
		cartdisp = np.zeros(data.shape)
		for i in range(self.size):
			if i != 0:
				cartdisp[i] = data[i] - data[i-1]
		return cartdisp

	def getAngularDisp(self,target):
		if target == 'head':
			data = self.head_cart
		elif target == 'gaze':
			data = self.gaze_cart
		else:
			print("only head and gaze velocity can be calculated")
			return None

		angdisp = np.zeros(data.shape)
		for i in range(self.size):
			if i != 0:
				angdisp[i] = AngularDisp(data[i], last)
			last = data[i]
		return angdisp

	def timeDiff(self):
		timediff = np.zeros(self.timestamp.shape)
		for i in range(self.size):
			if i != 0:
				timediff[i] = self.timestamp[i] - self.timestamp[i-1]
		
		return timediff

	def getEquirectRes(self,filterV = False, filterA = True):
		x = self.data[:,4]
		y = self.data[:,5]
		t = self.data[:,0]

		v = np.zeros((len(self.label),2))
		a = np.zeros((len(self.label),2))
		for j in range(len(x)-1):
			j+=1
			v[j,0] = 1000000*(x[j]-x[j-1])/(t[j]-t[j-1])
			v[j,1] = 1000000*(y[j]-y[j-1])/(t[j]-t[j-1])


		v_filted = kalman_filter(v)
		for j in range(len(x)-2):
			j+=2
			a[j,0] = 1000000*(v_filted[j,0]-v_filted[j-1,0])/(t[j]-t[j-1])
			a[j,1] = 1000000*(v_filted[j,1]-v_filted[j-1,1])/(t[j]-t[j-1])
		a_filted = kalman_filter(a)

		if filterV:
			v = v_filted
		if filterA:
			a = a_filted
		return v, a


def kalman_filter(headV,Q = 1e-5):
	n_iter = len(headV)  
	sz = headV.shape#(n_iter,) # size of array  
	z = headV # observed value

	xhat=numpy.zeros(sz)      # x noise
	P=numpy.zeros(sz)         # covariance matrix
	xhatminus=numpy.zeros(sz) # x estimator
	Pminus=numpy.zeros(sz)    # estimated cov matrix 
	K=numpy.zeros(sz)         # kalman_gain  

	R = 0.1**2 # estimate of measurement variance, change to see effect  

	# intial guesses  
	xhat[0] = 0.0  
	P[0] = 1.0  

	for k in range(1,n_iter):  
	
		xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0  
		Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1  
	
		# update  
		K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1  
		xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  
		P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1  
	return xhat

def prepare_dataset(Directory = '/Users/louitech_zero/Desktop/360_em_dataset/ground_truth/'):


	directory = Directory+'test/'
	files = []
	for r, d, f in os.walk(directory):
		for file in f:
			files.append(os.path.join(r, file))
	directory = Directory + 'train/'
	for r, d, f in os.walk(directory):
		for file in f:
			files.append(os.path.join(r, file))
	
	count = 0

	for i in files:
		gazedata = GazeHeadMovement(i)
		gaze_equrect = np.zeros((len(gazedata.gaze_fov),2))
		v,a = gazedata.getEquirectRes()
		h = gazedata.angle_head
	for j in range(len(gazedata.gaze_fov)):
		Rads = Cart2Spher(gazedata.gaze_fov[j])
		gaze_equrect[j] = Spher2Equ(Rads[0],Rads[1],gazedata.meta['width_px'],gazedata.meta["height_px"])    
	if count == 0:
		equrect = gaze_equrect
		V = v
		A = a
		head_pose = h
		BreakPoint = [len(a)]

	else:
		equrect = np.append(equrect,gaze_equrect,axis = 0 )
		V = np.append(V,v,axis = 0 )
		A = np.append(A,a,axis = 0 )
		head_pose = np.append(head_pose,h,axis = 0 )
		BreakPoint.append(len(a))
		count+=1


	for i in range(len(V)):
		if abs(V[i,0]) > 2500:
			V[i] = V[i-1]
		if abs(A[i,0]) > 2500:
			A[i] = A[i-1]     
		if abs(V[i,1]) > 1500:
			V[i] = V[i-1]
		if abs(A[i,1]) > 1500:
			A[i] = A[i-1] 
	return V, A

if __name__ == "__main__":
	data = GazeHeadMovement("/Users/louitech_zero/Desktop/360_em_dataset/ground_truth/train/004_07_football_hFc9HUYRbKc.arff")



