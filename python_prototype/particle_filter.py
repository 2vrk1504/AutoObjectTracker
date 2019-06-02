import numpy as np
import cv2
import time
from mergeSort import *


#Particle Filter class
class part_filt:

	#COnstructor
	def __init__(self, num, temp, w, h, sig_d, sig_mse, init_center, frame, sigma_wm = 1, batch_size = 5, ff = 1, n_0 = 5, k = 10, alpha = 0.8, fps = 24):
		self.num = num
		self.n, self.m = temp.shape[:2]
		self.frames_passed = 0
		self.forget_factor = ff
		self.mu_data = temp.reshape((temp.size, 1))
		self.prev_frame = frame
		
		#number of eigenvectors 
		self.k = k

		self.batch_size = batch_size

		self.B = np.zeros((self.n*self.m, 0))
		self.sub_s = np.zeros((self.n*self.m, 0)) 
		self.sigma_svd = np.zeros(0)

		self.xt = []
		self.xt_1 = []

		self.n_0 = n_0
		self.prev_us = np.zeros((0,1))
		self.prev_vs = np.zeros((0,1))
		self.t_poly_weights = np.zeros((4,2))
		self.t_matrix = np.zeros((0,4))
		self.fps = fps
		self.sig_mse = sig_mse
		self.sig_d = sig_d
		self.alpha = alpha
		self.sigma_wm = sigma_wm

		self.weightmask = np.ones(self.mu_data.shape)
		for j in xrange(self.num):
			x = particle(init_center[0], init_center[1], 1.0/self.num)
			self.xt_1.append(x)
	
	#Sample only first 'n' particles with highest weight
	#This solves the issue. LOL :P
	def sample(self, frame):

		self.num = len(self.xt_1)
		total_p = 0 
		i = 0
		eta = 0.0

		#Store the mean of the ten best particles
		nmlz = np.sum([self.xt_1[i].wt for i in range(10)])
		self.mean_best_ten = (np.sum([self.xt_1[i].u*self.xt_1[i].wt for i in range(10)])/nmlz, 
							  np.sum([self.xt_1[i].v*self.xt_1[i].wt for i in range(10)])/nmlz)

		vx, vy = (0,0)#self.cam_vel(frame)
		print 'Camera velocity: ', vx, ' ', vy
		self.regress()
		u_t_plus_1 = self.get_new_u()
		v_t_plus_1 = self.get_new_v()
		
		n_vel = int(self.alpha*self.num) #n particles with the velocity
		while i < n_vel:
			p = int(round(self.xt_1[i].wt*n_vel, 0))
			total_p += p
			if(total_p < n_vel):
				#Create Gaussian Noise
				delt_u = np.random.normal(-vx + (u_t_plus_1 - self.mean_best_ten[0]), self.sig_d, p)
				delt_v = np.random.normal(-vy + (v_t_plus_1 - self.mean_best_ten[1]), self.sig_d, p)
				j = 0
				while j < p:
					new_u = self.xt_1[i].u + delt_u[j]
					new_v = self.xt_1[i].v + delt_v[j]
					new_wt = self.pzt(frame, new_u, new_v)
					eta+=new_wt
					self.xt.append(particle(new_u, new_v, new_wt))
					j+=1
			else:
				#Create Gaussian noise
				delt_u = np.random.normal(-vx + (u_t_plus_1 - self.mean_best_ten[0]), self.sig_d, n_vel - total_p + p)
				delt_v = np.random.normal(-vy + (v_t_plus_1 - self.mean_best_ten[1]), self.sig_d, n_vel - total_p + p)
				j = 0
				while j < n_vel - total_p + p:
					new_u = self.xt_1[i].u + delt_u[j]
					new_v = self.xt_1[i].v + delt_v[j]
					new_wt= self.pzt(frame,new_u, new_v)
					eta+=new_wt
					self.xt.append(particle(new_u, new_v, new_wt))
					j+=1
				break
			i+=1
		#If target of 'n' particles has not been reached, add particles with higher weights
		if(total_p < n_vel):
			delt_u = np.random.normal(-vx + (u_t_plus_1 - self.mean_best_ten[0]), self.sig_d, n_vel - total_p)
			delt_v = np.random.normal(-vy + (v_t_plus_1 - self.mean_best_ten[1]), self.sig_d, n_vel - total_p)
			j = 0
			while j < n_vel - total_p:
				#Create Gaussian noise
				new_u = self.xt_1[j].u + delt_u[j]
				new_v = self.xt_1[j].v + delt_v[j]
				new_wt = self.pzt(frame, new_u, new_v)
				eta+=new_wt
				self.xt.append(particle(new_u, new_v, new_wt))
				j+=1
		
		#Sample some amount of particles near the original place with noise too...
		#because vel can get haywire at times
		#CURRENT RATIO is 80:20
		n_wo_vel = self.num - n_vel
		delt_u = np.random.normal(-vx, self.sig_d, n_wo_vel)
		delt_v = np.random.normal(-vy, self.sig_d, n_wo_vel)
		i = 0
		while i < n_wo_vel:
			new_u = self.mean_best_ten[0] + delt_u[i]
			new_v = self.mean_best_ten[1] + delt_v[i]
			new_wt = self.pzt(frame, new_u, new_v)
			eta+=new_wt
			self.xt.append(particle(new_u, new_v, new_wt))
			i+=1

		i = 0
		
		while i < self.num:
			self.xt[i].wt /=eta
			i+=1
				
		
		#Merge sort, to sort particles by weight
		self.sort_by_weight()
		self.xt_1 = self.xt
		self.xt = []
				
		
		#self.weight_mask(frame)
		start = time.clock()
		self.update_temp(frame)
		self.disp_eig()
		self.frames_passed = self.frames_passed + 1
		self.prev_frame = frame
		print 'sample function time ',time.clock() - start 
	
	#Calculate P(Zt|Xt)
	def pzt(self, frame, u, v):
		h,w = frame.shape[:2]
		#start = time.clock()
		#Boundary Condtitions... :P
		if(u<=w-self.m/2 and u >= self.m/2 and v>= self.n/2 and v<=h-self.n/2):
			#All these if conditions to make sure we have same sized images to subtract
			if(self.n%2==0 and self.m%2 == 0):
				img2 = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2)]
			elif(self.n%2==0 and self.m%2 != 0):
				img2 = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2 )+ 1]
			elif(self.n%2!=0 and self.m%2 == 0):
				img2 = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2)]
			else:
				img2 = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2) + 1]
			
			img2 = img2.reshape((img2.size,1))

			#Real stuff happens here
			err = self.MSE(img2) 
			#weight = err
			weight = np.exp(-err/(2*self.sig_mse**2))
			#print 'err wt', err,' ',weight
			#print 'pzt time ', time.clock() - start

			return weight
		else:
			return 0

	#Mean Squared Error
	def MSE(self,img2):
		z = img2 - self.mu_data

		p = np.dot(self.sub_s, np.dot(self.sub_s.T,z))
		l = ((z-p)**2)
		#cv2.imshow('img', (z+self.mu_data).reshape(self.n, self.m)/255.)
		#cv2.imshow('z', z.reshape(self.n, self.m)/(255.))
		#cv2.imshow('l', l.reshape(self.n, self.m)/(255.**2))
		#cv2.waitKey(0)
		#print l
		#m = (l > ((60**2)*np.ones(l.shape))).astype(int)
		#err = np.sum(m*(l.astype(float)/(l+(60**2)*3)))	
		err = np.sum(l)/l.size
		return err

	def sort_by_weight(self):
		mergeSort(self.xt, 0, self.num-1)

		
	def update_temp(self, frame):
		u = self.xt_1[0].u
		v = self.xt_1[0].v

		if(self.n%2==0 and self.m%2 == 0):
			img2 = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2)]
		elif(self.n%2==0 and self.m%2 != 0):
			img2 = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2 )+ 1]
		elif(self.n%2!=0 and self.m%2 == 0):
			img2 = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2)]
		else:
			img2 = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2) + 1]
		img2 = img2.reshape((img2.size,1))

		if (self.frames_passed + 1)%self.batch_size != 0:
			self.B = np.append(self.B, img2, axis = 1)
		
		else:
			self.B = np.append(self.B, img2, axis = 1)
			factor = (self.frames_passed*1.0/(self.frames_passed + self.batch_size))**0.5

			mean_of_B = np.mean(self.B, axis = 1).reshape(self.mu_data.shape)

			cv2.imshow('mu_b', (mean_of_B/255.).reshape((self.n, self.m)))
			cv2.waitKey(0)
			
			self.B -= mean_of_B
			B_hat = np.append( self.B, (mean_of_B - self.mu_data) * factor , axis = 1)

			self.B = np.zeros((self.mu_data.size, 0))

			self.mu_data = (self.mu_data*self.frames_passed*self.forget_factor + mean_of_B*self.batch_size)*1./(self.frames_passed*self.forget_factor+self.batch_size)

			U_sigma = self.forget_factor*np.dot(self.sub_s, np.diag(self.sigma_svd)) #Matrix multiplication of U and Sigma
			QR_mat = np.append(U_sigma, B_hat, axis = 1) #This is the matrix whose QR factors we want

			U_B_tild, R = np.linalg.qr(QR_mat)
			
			U_tild, sig_tild, vh_tild = np.linalg.svd(R)

			U_new = np.dot(U_B_tild, U_tild )

			if(sig_tild.size > self.k):
				self.sigma_svd = sig_tild[ 0:self.k ]
				self.sub_s = U_new[:, 0:self.k ]
			else:
				j = 0 #iterator
				while j < self.sub_s.shape[1]:
					self.sub_s[:,j] = U_new[:,j]
					self.sigma_svd[j] = sig_tild[j]
					j = j+1
				self.sub_s = np.append(self.sub_s, U_new[:,j].reshape((self.sub_s.shape[0], 1)), axis = 1)
				self.sigma_svd = np.append(self.sigma_svd, sig_tild[j])

		
		
		
	def disp_eig(self):
		cv2.imshow('mean', self.mu_data.reshape((self.n, self.m))/255.0)
		for i in xrange(self.sub_s.shape[1]):
			sub_s = self.sub_s[:,i].reshape(self.mu_data.shape)
			temp = sub_s #+ self.mu_data)/255.0
			#temp = (self.sub_s[:,i])
			disp = temp.reshape(self.n,self.m)
			#cv2.imshow('disp2', disp)
			disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
			#stack = np.dstack((stack,disp))
			cv2.imshow('disp', disp)
			cv2.waitKey(0)

	#Occlusion handling
	def weight_mask(self, frame):
		u = np.sum([self.xt_1[i].u*self.xt_1[i].wt for i in xrange(self.num)])
		v = np.sum([self.xt_1[i].v*self.xt_1[i].wt for i in xrange(self.num)])
		It = 0
		D = np.zeros(self.weightmask.shape)
		#need to make It as a mn cross k matrix
		if(self.n%2==0 and self.m%2 == 0):
			It = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2)]
		elif(self.n%2==0 and self.m%2 != 0):
			It = frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2 )+ 1]
		elif(self.n%2!=0 and self.m%2 == 0):
			It = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2)]
		else:
			It = frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2) + 1]
			It = It.flatten()
			prod = It - np.matmul(np.matmul(self.sub_s, self.sub_s.T),It)
			#prod = prod.flatten()
			for i in xrange(prod.size):
				D[i] = prod[i]*self.weightmask[i]
				self.weightmask[i] = np.exp(-1*D[i]**2/self.sigma_wm**2)


	#some kind of cubic regression in temporal domain, 
	#predicts next point given the motion history
	#needs slight tweaks, slightly unstable model

	#OPEN TO SUGGESTIONS!!!! :P
	#Run and see
	def regress(self):
		print 'u(t) = ', self.mean_best_ten[0]
		print 'v(t) = ', self.mean_best_ten[1]

		self.prev_us = np.append(self.prev_us, np.ones((1,1))*self.mean_best_ten[0], axis = 0)		
		self.prev_vs = np.append(self.prev_vs, np.ones((1,1))*self.mean_best_ten[1], axis = 0)

		if(self.frames_passed >= self.n_0):
			self.prev_us = np.delete(self.prev_us, 0, axis = 0)
			self.prev_vs = np.delete(self.prev_vs, 0, axis = 0)

		if self.frames_passed == 0:
			t = np.zeros((1,4))
			t[0,0] = 1.
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			self.t_poly_weights[:,0] = np.array([self.prev_us[0],0,0,0])
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0],0,0,0])

		elif self.frames_passed == 1:
			t = np.array([(self.frames_passed**i) for i in range(4)]).reshape((1,4))	
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			self.t_poly_weights[:,0] = np.array([self.prev_us[0], (self.prev_us[1] - self.prev_us[0])*self.fps, 0, 0])
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0], (self.prev_vs[1] - self.prev_vs[0])*self.fps, 0, 0])
		
		elif self.frames_passed == 2:
			t = np.array([(self.frames_passed**i) for i in range(4)]).reshape((1,4))	
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)

			self.t_poly_weights[:,0] = np.array([self.prev_us[0], 
				(-self.prev_us[2]+4*self.prev_us[1]-3*self.prev_us[0])*self.fps/2., 
				(self.prev_us[2]-2*self.prev_us[1]+self.prev_us[0])*(self.fps**2)/2., 0])
			
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0], 
				(-self.prev_vs[2]+4*self.prev_vs[1]-3*self.prev_vs[0])/(2.*self.fps), 
				(self.prev_vs[2]-2*self.prev_vs[1]+self.prev_vs[0])/(2.*(self.fps**2)), 0])

		else:
			if self.frames_passed < self.n_0:
				t = np.array([(self.frames_passed*1./self.fps)**i for i in range(4)]).reshape((1,4))	
				self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			ata = np.dot(self.t_matrix.T, self.t_matrix)
			self.t_poly_weights[:,0] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_us)).reshape((4,))
			self.t_poly_weights[:,1] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_vs)).reshape((4,))


	def get_new_u(self):
		if(self.frames_passed >= self.n_0):
			tplusone = np.array([((1.*self.n_0/self.fps)**i) for i in range(4)])
		else:
			tplusone = np.array([((self.frames_passed * 1./self.fps)**i) for i in range(4)])
		new_u = np.dot(tplusone, self.t_poly_weights[:,0].reshape((4,1)))[0]
		print 'u(t+1) = ', new_u
		return new_u

	def get_new_v(self):
		if(self.frames_passed >= self.n_0):
			tplusone = np.array([((self.n_0*1./self.fps)**i) for i in range(4)])
		else:
			tplusone = np.array([((self.frames_passed*1./self.fps)**i) for i in range(4)])

		new_v = np.dot(tplusone, self.t_poly_weights[:,1].reshape((4,1)))[0]
		print 'v(t+1) = ', new_v
		return new_v

	def cam_vel(self, frame):
		'''
		dx, dy = cv2.spatialGradient(frame)
		IxIy = np.append(dx.reshape((dx.size,1)), dy.reshape((dy.size,1)), axis = 1)
		It = (frame - self.prev_frame).reshape((frame.size, 1))
		mat = np.dot(IxIy.T, IxIy)
		vel = np.dot(np.linalg.inv(mat), np.dot(IxIy.T, It))
		'''
		vel = cv2.phaseCorrelate(self.prev_frame/255., frame/255.)
		return (vel[0][0], vel[0][1])