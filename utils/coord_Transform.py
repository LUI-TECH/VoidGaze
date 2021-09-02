import numpy as np
from math import sin,cos,asin,tan,atan2, acos, pi


def Spher2Equ(horRads, verRads, widthPx, heightPx):
	xEq = (horRads / (2 * pi)) * widthPx
	yEq = (verRads / pi) * heightPx
	if yEq < 0:
		yEq = -yEq
		xEq = xEq + widthPx/2
	

	if yEq >= heightPx:
		yEq = 2 * heightPx - yEq -1
		xEq = xEq + widthPx/2


	if xEq < 0:
		xEq = widthPx + xEq - 1


	if xEq > widthPx:
		xEq = xEq - widthPx
	return np.array([xEq,yEq])

def Cart2Spher(CartVec):
	if all(CartVec) == 0:
		value = 0.0
	else: 
		value = CartVec[1]/(np.sqrt(CartVec[0]**2+CartVec[1]**2+CartVec[2]**2))
	verRads = acos(value)
	horRads = atan2(CartVec[2],CartVec[0])
	return np.array([horRads, verRads])


def Equ2Spher(x , y, width_px, height_px):
	horRads = (x *2 * pi) / width_px
	verRads = (y * pi) / height_px
	return np.array([horRads, verRads])




def Spher2Cart(Rads):
	horRads = Rads[0]
	verRads = Rads[1]
	cartVec = np.zeros((3,1))

	cartVec[0]= sin(verRads) * cos(horRads)
	cartVec[1]= cos(verRads)
	cartVec[2]= sin(verRads) * sin(horRads)
	return cartVec






def Ang2Rads(headAngle):
	return headAngle* pi / 180

def PosTransform(data,meta):
	t = data[0]
	x_gaze = data[1]
	y_gaze = data[2]
	x_head = data[4]
	y_head = data[5]
	angle_head = data[6]
	label = data[7]

	width_px = meta['width_px']
	height_px = meta['height_px']
	

	sphereHead = Equ2Spher(x_head,y_head,width_px,height_px)

	CartHead = Spher2Cart(sphereHead)

	angleHeadRads = Ang2Rads(angle_head)
	videoVec = np.array([-1, 0, 0])



	head2Ref = rotation(CartHead, -angleHeadRads);
	video2Ref = rotation(videoVec, 0);
	rotMatrix = np.transpose(video2Ref.dot(np.transpose(head2Ref)))


	sphereGaze = Equ2Spher(x_gaze,y_gaze,width_px,height_px)
	CartGaze = Spher2Cart(sphereGaze)


	GazeVec_FOV = pointRotation(rotMatrix,CartGaze)
	return (CartHead,CartGaze,GazeVec_FOV)

def pointRotation(rot,vec):
	return rot.dot(vec)

def rotation(vec, angle):

	theta = asin(vec[1])
	psi = 0
	if abs(theta)< pi/2 -0.01:
		psi = atan2(vec[2],vec[0])

	rotMatrix = np.zeros((3,3))
	rotMatrix[0,0] = cos(theta)*cos(psi)
	rotMatrix[0,1] = -sin(theta)
	rotMatrix[0,2] = cos(theta)*sin(psi)

	rotMatrix[1,0] = cos(angle)*sin(theta)*cos(psi) + sin(angle)*sin(psi)
	rotMatrix[1,1] = cos(angle)*cos(theta)
	rotMatrix[1,2] = cos(angle)*sin(theta)*sin(psi) - sin(angle)*cos(psi)

	rotMatrix[2,0] = sin(angle)*sin(theta)*cos(psi) - cos(angle)*sin(psi)
	rotMatrix[2,1] = sin(angle)*cos(theta)
	rotMatrix[2,2] = sin(angle)*sin(theta)*sin(psi) + cos(angle)*cos(psi)
	return rotMatrix


def AngularDisp(v1,v2):

	v1 = v1.reshape(3)
	v2 = v2.reshape(3)

	""" Returns the angle in radians between vectors 'v1' and 'v2'    """
	cosang = np.dot(v1, v2)
	sinang = np.linalg.norm(np.cross(v1, v2))
	return atan2(sinang, cosang)

	


