import numpy as np
prsc = 'float64'
#input
prelbd = [2,2,1]
premu = [3,2,1]
predelta = [[1,4,9],[16,25,36],[49,60,81]]
#input end

#parameter
prea = [[0.8,0.1,0.1],[0.3,0.5,0.2],[0.3,0.4,0.3]]
N = 2000
ep = 0.000000001
gamma = 0.01
#parameter end

n1 = len(prelbd)
n2 = len(premu)
mu = np.array(premu, dtype=prsc)
lbd = np.array(prelbd, dtype=prsc)
a = np.array(prea, dtype=prsc)

delta = np.array(predelta, dtype=prsc)

for x in range(N):
	I = []
	for i in range(n1):
		for j in range(n2):
			if a[i][j] < ep:
				I.append(n2*i+j)	
	ps = 1
	ter = 0
	while ps == 1:
		NMT = np.array([[0]*(n1*n2)]*(n1+len(I)), dtype=prsc)
		for i in range(n1):
			for j in range(n2):
				NMT[i][n2*i+j]=1
		cnt = 0
		for k in I:
			NMT[n1+cnt][k]=1
			cnt = cnt+1
		NM = np.transpose(NMT)
		tmpM = np.linalg.inv(np.matmul(NMT,NM))
		
		PM = np.identity(n1*n2)-np.matmul(NM,np.matmul(tmpM,NMT))
		Del = np.array([0]*(n1*n2), dtype=prsc)
		for j in range(n2):
			lbd2 = 0
			for i in range(n1):
				lbd2 = lbd2 + lbd[i]*a[i][j]
			for i in range(n1):
				Del[n2*i+j] = lbd[i]*(mu[j]/((mu[j]-lbd2)**2)+delta[i][j])
		s = -np.matmul(PM,Del)
		lbd3 = np.matmul(np.matmul(tmpM, NMT), Del)
		ps = 0
		if np.linalg.norm(s)<ep:
			subI = I.copy()
			for i in range(len(I)):
				if lbd3[n1+i] < ep:
					subI.remove(I[i])
					ps = 1
			I = subI.copy()
			if ps==0:
				ter = 1
	if ter==1:
		break
	gamma2 = gamma
	norm = 0
	for i in range(n1*n2):
		norm = norm + s[i]*s[i]
	s = s / np.sqrt(norm)
	for i in range(n1):
		for j in range(n2):
			if s[n2*i+j] >= 0:
				continue
			else:
				if gamma2 >= -a[i][j]/s[n2*i+j]:
					gamma2 = -a[i][j]/s[n2*i+j]
	for j in range(n2):
		lbd2 = 0
		disp = 0
		for i in range(n1):
			lbd2 = lbd[i]*a[i][j]
			disp = s[n2*i+j]*lbd[i]
		if disp < ep:
			continue
		else:
			if gamma2 >= (mu[j]-lbd2-ep)/disp:
				gamma2 = (mu[j]-lbd2-ep)/disp
			
	a = a + gamma2 * np.reshape(s, (n1,n2))

F = 0
for j in range(n2):
	lbd2 = 0
	for i in range(n1):
		lbd2 = lbd2 + lbd[i]*a[i][j]
	F = F + lbd2 / (mu[j]-lbd2)
	
for i in range(n1):
	for j in range(n2):
		F = F + lbd[i]*delta[i][j]*a[i][j]
print(a)
print(F)
