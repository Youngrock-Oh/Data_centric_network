#Input variables
#Lambdas : list of output rate of previous layer
#Mus : list of service rate of next layer, increasing order
#Deltas : list of delay
#Initial : initial guess for aij. it should be strictly feasible
#eps1 : threshold of barrier method
#eps2 : threshold of Newton method in barrier method
#t0 : initial value of t in barrier method
#m : multiply to t at the last part of loop of barrier nethod. should be bigger than 1 
#alpha : in line search 0~0.5 / recommend 0.01~0.3
#beata : in line seach  0~1  / recommend 0.1 or 0.8

#Local variables
#t : barrier method t
#Nst : Newton step
#Ndec : square of Newton decrement
#s : something like learning rate in Newton

import numpy as np
import math 

def optimal_aij(Lambdas, Mus, Deltas, Initial, eps1, eps2, t0, m, alpha, beta):
    lambdas = np.array(Lambdas,dtype=np.float64)
    mus = np.array(Mus)
    deltas = np.array(Deltas)
    
    Ml = len(lambdas)
    Nl = len(mus)
    t = t0

    #F is an Nl by Nl matrix which represents the null-space of [1 1 ... 1]
    F = np.zeros((Nl, Nl))
    for i in range(Nl):
        F[i, i] = 1
    for i in range(Nl-1):
        F[i+1, i] = -1
    F[0, Nl-1] = -1   
    
    #Initial guess
    initial = np.array(Initial)
    Y = np.zeros((Ml, Nl))
    for i in range(Ml):
        Y[i,:] = np.linalg.lstsq(F, initial[i,:]-1/Nl*np.ones(Nl), rcond=None)[0]  
    #Define f, gradf, hessf        
    #f
    def f(k, Z):
        S11=0
        S12=0
        S21=0
        S22=0
        S31=0
        S32=0
        S41=0
        S42=0
        S51=0
        S52=0
        #1st sum
        for j in range(Nl-1):
            for i in range(Ml):
                S12 += (Z[i,j+1]-Z[i,j]+1/Nl)*lambdas[i]
            S11 += -1 + mus[j+1]/(mus[j+1]-S12)
            S12 *= 0
        for i in range(Ml):
            S12 += (Z[i,0]-Z[i,Nl-1]+1/Nl)*lambdas[i]
        S11 += -1+mus[0]/(mus[0]-S12)    
        #2nd sum        
        for j in range(Nl-1):
            for i in range(Ml):
                S22 += lambdas[i]*deltas[i, j+1]*(Z[i, j+1]-Z[i, j]+1/Nl)
            S21 += S22
            S22 *= 0
        for i in range(Ml):
            S22 += lambdas[i]*deltas[i,0]*(Z[i,0]-Z[i,Nl-1]+1/Nl)
        S21 += S22    
        #3rd sum
        for j in range(Nl-1):
            for i in range(Ml):
                S32 += math.log(Z[i,j+1]-Z[i,j]+1/Nl)
            S31 += S32
            S32 *= 0
        for i in range(Ml):
            S32 += math.log(Z[i,0]-Z[i, Nl-1]+1/Nl)
        S31 += S32
        #4th sum
        for j in range(Nl-1):
            for i in range(Ml):
                S42 += math.log(1-(Z[i,j+1]-Z[i,j]+1/Nl))
            S41 += S42
            S42 *= 0
        for i in range(Ml):
            S42 += math.log(1-(Z[i,0]-Z[i,Nl-1]+1/Nl))
        S41 += S42
        #5th sum
        for j in range(Nl-1):
            for i in range(Ml):
                S52 += Z[i, j+1]-Z[i, j]+1/Nl
            S51 += math.log(mus[j+1]-S52)
            S52 *= 0
        for i in range(Ml):
            S52 += Z[i,0]-Z[i,Nl-1]+1/Nl
        S51 += math.log(mus[0]-S52)
        #total
        return k*(S11+S21)-S31-S41-S51
    
    #gradf
    def gradf(k, Z):
        grad = np.zeros(Ml*Nl)
        T1=0
        T2=0
        T3=0
        T4=0
        #Z = np.c_[Z, Z[:, [0,1]]]
        for be in range(Nl-2):
            for al in range(Ml):
                for i in range(Ml):
                    T1 += ((Z[i, be+1]-Z[i, be]+1/Nl)*lambdas[i])
                for i in range(Ml):
                    T2 += ((Z[i, be+2]-Z[i, be+1]+1/Nl)*lambdas[i])
                for i in range(Ml):
                    T3 += Z[i, be+1]-Z[i, be]+1/Nl
                for i in range(Ml):
                    T4 += Z[i, be+2]-Z[i, be+1]+1/Nl
                grad[al*Nl+be+1] = k*(lambdas[al]*mus[be+1]/pow((mus[be+1]-T1),2)-lambdas[al]*mus[be+2]/pow((mus[be+2]-T2),2) \
                    +lambdas[al]*deltas[al,be+1]-lambdas[al]*deltas[al,be+2]) \
                    -(1/(Z[al,be+1]-Z[al,be]+1/Nl)-1/(Z[al,be+2]-Z[al,be+1]+1/Nl)) \
                    -(-1/(1-(Z[al,be+1]-Z[al,be]+1/Nl))+1/(1-(Z[al,be+2]-Z[al,be+1]+1/Nl))) \
                    -(-1/(mus[be+1]-T3)+1/(mus[be+2]-T4))
                T1 *= 0
                T2 *= 0
                T3 *= 0
                T4 *= 0
        #be=Nl-2
        for al in range(Ml):
            for i in range(Ml):
                T1 += ((Z[i, Nl-1]-Z[i, Nl-2]+1/Nl)*lambdas[i])
            for i in range(Ml):
                T2 += ((Z[i, 0]-Z[i, Nl-1]+1/Nl)*lambdas[i])
            for i in range(Ml):
                T3 += Z[i, Nl-1]-Z[i, Nl-2]+1/Nl
            for i in range(Ml):
                T4 += Z[i, 0]-Z[i, Nl-1]+1/Nl
            grad[al*Nl+Nl-1] = k*(lambdas[al]*mus[Nl-1]/pow((mus[Nl-1]-T1),2)-lambdas[al]*mus[0]/pow((mus[0]-T2),2) \
                +lambdas[al]*deltas[al,Nl-1]-lambdas[al]*deltas[al,0]) \
                -(1/(Z[al,Nl-1]-Z[al,Nl-2]+1/Nl)-1/(Z[al,0]-Z[al,Nl-1]+1/Nl)) \
                -(-1/(1-(Z[al,Nl-1]-Z[al,Nl-2]+1/Nl))+1/(1-(Z[al,0]-Z[al,Nl-1]+1/Nl))) \
                -(-1/(mus[Nl-1]-T3)+1/(mus[0]-T4))
            T1 *= 0
            T2 *= 0
            T3 *= 0
            T4 *= 0
        #be=Nl-1
        for al in range(Ml):
            for i in range(Ml):
                T1 += ((Z[i, 0]-Z[i, Nl-1]+1/Nl)*lambdas[i])
            for i in range(Ml):
                T2 += ((Z[i, 1]-Z[i, 0]+1/Nl)*lambdas[i])
            for i in range(Ml):
                T3 += Z[i, 0]-Z[i, Nl-1]+1/Nl
            for i in range(Ml):
                T4 += Z[i, 1]-Z[i, 0]+1/Nl
            grad[al*Nl] = k*(lambdas[al]*mus[0]/pow((mus[0]-T1),2)-lambdas[al]*mus[1]/pow((mus[1]-T2),2) \
                +lambdas[al]*deltas[al,0]-lambdas[al]*deltas[al,1]) \
                -(1/(Z[al,0]-Z[al,Nl-1]+1/Nl)-1/(Z[al,1]-Z[al,0]+1/Nl)) \
                -(-1/(1-(Z[al,0]-Z[al,Nl-1]+1/Nl))+1/(1-(Z[al,1]-Z[al,0]+1/Nl))) \
                -(-1/(mus[0]-T3)+1/(mus[1]-T4))
            T1 *= 0
            T2 *= 0
            T3 *= 0
            T4 *= 0
        return np.transpose(grad)        
                
    #hessf
    def hessf(k, Z):
        #대각성분 아닌거 먼저 만들고 자기자신의 transpose 더한 뒤에 대각 성분 넣기?
        hess = np.zeros((Ml*Nl, Ml*Nl))
        #
        R11=0
        R12=0
        for be in range(Nl-1):
            for al in range(Ml):
                for i in range(Ml):
                    R11 += (Z[i, be+1]-Z[i, be]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R12 += (Z[i, be+1]-Z[i, be]+1/Nl)
                hess[Nl*al+be+1,Nl*al+be]=k*(-2*pow(lambdas[al],2)*mus[be+1]/pow(mus[be+1]-R11,3)) \
                    -1/pow(Z[al,be+1]-Z[al,be]+1/Nl,2)-1/pow(1-(Z[al,be+1]-Z[al,be]+1/Nl),2) \
                    -1/pow(mus[be+1]-R12,2)
                R11 *= 0
                R12 *= 0                    
        for al in range(Ml):        
            for i in range(Ml):
                R11 += (Z[i, 0]-Z[i, Nl-1]+1/Nl)*lambdas[i]
            for i in range(Ml):
                R12 += (Z[i, 0]-Z[i, Nl-1]+1/Nl)
            hess[Nl*al, Nl*al+Nl-1]=k*(-2*pow(lambdas[al],2)*mus[0]/pow(mus[0]-R11,3)) \
                    -1/pow(Z[al,0]-Z[al,Nl-1]+1/Nl,2)-1/pow(1-(Z[al,0]-Z[al,Nl-1]+1/Nl),2) \
                    -1/pow(mus[0]-R12,2)
            R11 *= 0
            R12 *= 0
        #
        R21=0
        R22=0
        for be in range(Nl-1):
            for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R21 += (Z[i, be+1]-Z[i,be]+1/Nl)*lambdas[i]
                        for i in range(Ml):
                            R22 += Z[i, be+1]-Z[i,be]+1/Nl
                        hess[Nl*al+be+1, Nl*alp+be]=k*(-2*mus[be+1]*lambdas[al]*lambdas[alp]/pow(mus[be+1]-R21,3)) \
                            -1/pow(mus[be+1]-R22,2)
                        R21 *= 0
                        R22 *= 0
        for al in range(Ml):
            for alp in range(Ml):
                if al != alp:
                   for i in range(Ml):
                       R21 += (Z[i, 0]-Z[i,Nl-1]+1/Nl)*lambdas[i]
                   for i in range(Ml):
                       R22 += Z[i, 0]-Z[i,Nl-1]+1/Nl
                   hess[Nl*al, Nl*alp+Nl-1]=k*(-2*mus[0]*lambdas[al]*lambdas[alp]/pow(mus[0]-R21,3)) \
                       -1/pow(mus[0]-R22,2)    
                   R21 *= 0
                   R22 *= 0
        #traspose
        hess += np.transpose(hess)           
        #
        R31=0
        R32=0
        R33=0
        R34=0
        for be in range(Nl-2):
            for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R31 += (Z[i,be+1]-Z[i,be]+1/Nl)*lambdas[i]
                        for i in range(Ml):
                            R32 += (Z[i,be+2]-Z[i,be+1]+1/Nl)*lambdas[i]
                        for i in range(Ml):    
                            R33 += (Z[i,be+1]-Z[i,be]+1/Nl)
                        for i in range(Ml):
                            R34 += (Z[i,be+2]-Z[i,be+1]+1/Nl)
                        hess[Nl*al+be+1,Nl*alp+be+1]=k*(2*mus[be+1]*lambdas[al]*lambdas[alp]/pow(mus[be+1]-R31,3)+2*mus[be+2]*lambdas[al]*lambdas[alp]/pow(mus[be+2]-R32,3)) \
                            +(1/pow(mus[be+1]-R33,2)+1/pow(mus[be+2]-R34,2))
                        R31 *= 0
                        R32 *= 0
                        R33 *= 0
                        R34 *= 0
        #be=Nl-2
        for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R31 += (Z[i,Nl-1]-Z[i,Nl-2]+1/Nl)*lambdas[i]
                        for i in range(Ml):
                            R32 += (Z[i,0]-Z[i,Nl-1]+1/Nl)*lambdas[i]
                        for i in range(Ml):    
                            R33 += (Z[i,Nl-1]-Z[i,Nl-2]+1/Nl)
                        for i in range(Ml):
                            R34 += (Z[i,0]-Z[i,Nl-1]+1/Nl)
                        hess[Nl*al+Nl-1,Nl*alp+Nl-1]=k*(2*mus[Nl-1]*lambdas[al]*lambdas[alp]/pow(mus[Nl-1]-R31,3)+2*mus[0]*lambdas[al]*lambdas[alp]/pow(mus[0]-R32,3)) \
                            +(1/pow(mus[Nl-1]-R33,2)+1/pow(mus[0]-R34,2))
                        R31 *= 0
                        R32 *= 0
                        R33 *= 0
                        R34 *= 0
        #be=Nl-1
        for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R31 += (Z[i,0]-Z[i,Nl-1]+1/Nl)*lambdas[i]
                        for i in range(Ml):
                            R32 += (Z[i,1]-Z[i,0]+1/Nl)*lambdas[i]
                        for i in range(Ml):    
                            R33 += (Z[i,0]-Z[i,Nl-1]+1/Nl)
                        for i in range(Ml):
                            R34 += (Z[i,1]-Z[i,0]+1/Nl)
                        hess[Nl*al,Nl*alp]=k*(2*mus[0]*lambdas[al]*lambdas[alp]/pow(mus[0]-R31,3)+2*mus[1]*lambdas[al]*lambdas[alp]/pow(mus[1]-R32,3)) \
                            +(1/pow(mus[0]-R33,2)+1/pow(mus[1]-R34,2))
                        R31 *= 0
                        R32 *= 0
                        R33 *= 0
                        R34 *= 0                
        #
        R41=0
        R42=0
        R43=0
        R44=0
        for be in range(Nl-2):
            for al in range(Ml):
                for i in range(Ml):
                    R41 += (Z[i, be+1]-Z[i, be]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R42 += (Z[i, be+2]-Z[i,be+1]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R43 += Z[i,be+1]-Z[i,be]+1/Nl
                for i in range(Ml):
                    R44 += Z[i,be+2]-Z[i,be+1]+1/Nl
                hess[Nl*al+be+1, Nl*al+be+1] = k*(2*pow(lambdas[al],2)*mus[be+1]/pow(mus[be+1]-R41,3)+2*pow(lambdas[al],2)*mus[be+2]/pow(mus[be+2]-R42,3)) \
                    +(1/pow(Z[al,be+1]-Z[al,be]+1/Nl,2)+1/pow(Z[al,be+2]-Z[al,be+1]+1/Nl,2)) \
                    +(1/pow(1-(Z[al,be+1]-Z[al,be]+1/Nl),2)+1/pow(1-(Z[al,be+2]-Z[al,be+1]+1/Nl),2)) \
                    +(1/pow(mus[be+1]-R43,2)+1/pow(mus[be+2]-R44,2))                        
                R41 *= 0
                R42 *= 0
                R43 *= 0
                R44 *= 0
        #be=Nl-2
        for al in range(Ml):
                for i in range(Ml):
                    R41 += (Z[i, Nl-1]-Z[i, Nl-2]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R42 += (Z[i, 0]-Z[i,Nl-1]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R43 += Z[i,Nl-1]-Z[i,Nl-2]+1/Nl
                for i in range(Ml):
                    R44 += Z[i,0]-Z[i,Nl-1]+1/Nl
                hess[Nl*al+Nl-1, Nl*al+Nl-1] = k*(2*pow(lambdas[al],2)*mus[Nl-1]/pow(mus[Nl-1]-R41,3)+2*pow(lambdas[al],2)*mus[0]/pow(mus[0]-R42,3)) \
                    +(1/pow(Z[al,Nl-1]-Z[al,Nl-2]+1/Nl,2)+1/pow(Z[al,0]-Z[al,Nl-1]+1/Nl,2)) \
                    +(1/pow(1-(Z[al,Nl-1]-Z[al,Nl-2]+1/Nl),2)+1/pow(1-(Z[al,0]-Z[al,Nl-1]+1/Nl),2)) \
                    +(1/pow(mus[Nl-1]-R43,2)+1/pow(mus[0]-R44,2))                        
                R41 *= 0
                R42 *= 0
                R43 *= 0
                R44 *= 0
        #be=Nl-1
        for al in range(Ml):
                for i in range(Ml):
                    R41 += (Z[i, 0]-Z[i, Nl-1]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R42 += (Z[i, 1]-Z[i,0]+1/Nl)*lambdas[i]
                for i in range(Ml):
                    R43 += Z[i,0]-Z[i,Nl-1]+1/Nl
                for i in range(Ml):
                    R44 += Z[i,1]-Z[i,0]+1/Nl
                hess[Nl*al, Nl*al] = k*(2*pow(lambdas[al],2)*mus[0]/pow(mus[Nl-1]-R41,3)+2*pow(lambdas[al],2)*mus[1]/pow(mus[1]-R42,3)) \
                    +(1/pow(Z[al,0]-Z[al,Nl-1]+1/Nl,2)+1/pow(Z[al,1]-Z[al,0]+1/Nl,2)) \
                    +(1/pow(1-(Z[al,0]-Z[al,Nl-1]+1/Nl),2)+1/pow(1-(Z[al,1]-Z[al,0]+1/Nl),2)) \
                    +(1/pow(mus[0]-R43,2)+1/pow(mus[1]-R44,2))                        
                R41 *= 0
                R42 *= 0
                R43 *= 0
                R44 *= 0                        
        return hess       
    
    #barrier method    
    while 1/t > eps1:
        Nst = -np.matmul(np.linalg.inv(hessf(t,Y)), gradf(t,Y))
        Ndec = -np.matmul(np.transpose(gradf(t,Y)), Nst)
        while Ndec/2 > eps2:
            s=1
            while f(t, Y+s*np.reshape(Nst,(Ml,Nl))) > f(t, Y)+alpha*s*(-Ndec):
                s *= beta
            Y+=s*np.reshape(Nst, (Ml, Nl))
            Nst = -np.matmul(np.linalg.inv(hessf(t,Y)), gradf(t,Y))
            Ndec = -np.matmul(np.transpose(gradf(t,Y)), Nst)
        t*=m
    #aij    
    A = np.zeros((Ml, Nl))
    for i in range(Ml):
        A[i,:]=np.matmul(F,Y[i,:])+1/Nl*np.transpose(np.ones(Nl))    
    
    #lambda_hat
    lambda_hats = np.zeros(Nl)
    for j in range(Nl):
        for i in range(Ml):
            lambda_hats[j] += A[i,j]*lambdas[i]
    #service_time
    service_time = 0
    service1 = 0
    service2 = 0
    for i in range(Nl):
        service1 += lambda_hats[i]/(mus[i]-lambda_hats[i])
    for i in range(Ml):
        for j in range(Nl):
            service2 += lambdas[i]*deltas[i,j]*A[i,j]
    service_time = service1 + service2        
    #condition check
    D=np.zeros((Ml,Nl))
    for i in range(Ml):    
        for j in range(Nl):
            D[i,j]=mus[j]/pow(mus[j]-lambda_hats[j],2)+deltas[i,j]
    
    return A, lambda_hats, service_time, D     
        
A, lambda_hats, service_time, D = optimal_aij([20000,25000], [30000,40000], [[0,0],[0,0]], [[0.5,0.5],[0.5,0.5]], 0.0000001, 0.0000001, 1, 1.1, 0.01, 0.1)        
print(A)
print(lambda_hats)
print(service_time)        
print(D)        
#If something is wrong, change Initial or m.        
        
        
        
        
        
        