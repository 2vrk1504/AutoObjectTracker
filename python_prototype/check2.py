import numpy as np 
import glob
import cv2
from velocity_as_state import *


def _min(x, y):
	if x < y:
		return x
	else:
		return y

ground_truth_file = open('./BlurBody/groundtruth_rect.txt', 'r')
ground_truth_coords = []
for line in ground_truth_file:
	ground_truth_coords.append([int(x) for x in line.split()])

filename = './BlurBody/img/0001.jpg' 

frame = cv2.imread(filename)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame2 = np.copy(frame)
w,h = frame.shape[:2]
N = 100
sig_d = 10
sig_mse = 10
u,v,m,n = ground_truth_coords[0]
temp = frame[v:v+n,u:u+m]
cv2.imshow('temp', temp)
cv2.waitKey(0)
init_center = [u+m/2, v+n/2]
S = part_filt(N, temp, w, h, sig_d, sig_mse, init_center, frame)

frame_no = 0
list_of_files = glob.glob('./BlurBody/img/*.jpg')
list_of_files.sort()
frames = []
for filename in list_of_files:
	if frame_no != 0:
		frames.append(cv2.imread(filename))
	frame_no += 1
frame_no = 0

for frame in frames:
	x,y = S.xt_1[0].u, S.xt_1[0].v
	print
	print 'Frame: ', frame_no	
	frame1 = np.copy(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	S.sample(frame)
	print 'N = ', S.num
	for i in xrange(S.num):
		cv2.circle(frame1,(int(S.xt_1[i].u),int(S.xt_1[i].v)), 2, (0,0,_min(255,int(round(S.xt_1[i].wt*2550)))), -1)
		print 'weight ', i, ': ',S.xt_1[i].wt
	
	
	#u = np.sum([S.xt_1[i].u*S.xt_1[i].wt for i in xrange(S.num)])
	#v = np.sum([S.xt_1[i].v*S.xt_1[i].wt for i in xrange(S.num)])
	u = S.mean_best_ten[0]
	v = S.mean_best_ten[1]	
	#u = np.average([S.xt_1[i].u for i in xrange(1)])
	#v = np.average([S.xt_1[i].v for i in xrange(1)])
	n = S.n
	m = S.m
	
	#frame1 = cv2.GaussianBlur(frame1,(15,15),0)
	#frame2 = cv2.GaussianBlur(frame2,(15,15),0)
	
	'''
	dx, dy = cv2.spatialGradient(frame1)
	IxIy = np.append(dx.reshape((dx.size,1)), dy.reshape((dy.size,1)), axis = 1)
	It = (frame1 - frame2).reshape((frame1.size, 1))
	mat = np.dot(IxIy.T, IxIy)
	vel = np.dot(np.linalg.inv(mat), np.dot(IxIy.T, It))
	print 'velocity: vx = ', vel[0], ' vy = ', vel[1]
	'''
	
	#print 'cam vel: ', round(cv2.phaseCorrelate(frame1/255., frame2/255.)[0][0]),' ', round(cv2.phaseCorrelate(frame1/255., frame2/255.)[0][1])

	cv2.rectangle(frame1, (int(u-m/2),int(v-n/2)), (int(u+m/2), int(v+n/2)), (200,200,0), 2)
	u1,v1,m1,n1 = ground_truth_coords[frame_no+1]
	cv2.rectangle(frame1, (u1,v1), (u1+m1, v1+n1), (0, 500,200), 2)
	frame_no+=1
	cv2.imshow('frame',frame1)
	#cv2.imshow('frame_blur',frame1)
	#print 'frame shape: ', frame.shape
	#cv2.imshow('dif', (frame1 - frame2)/255.)
	#frame2 = np.copy(frame1)
	cv2.waitKey(0)

cv2.destroyAllWindows()
