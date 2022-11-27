#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import cmath
from numpy import sin
import matplotlib.pyplot as plt
#from random import seed
from random import random
import time
import multiprocessing as mp

numcores=2#mp.cpu_count()

nd=10
n=300
J=1
assmJ=0.0001 #asymmetry in the hopping J
W=2
k0=1.4
sigx=10
tf=60
tstep=0.2
ntplot=100
reflect=0.000001
neglectpsi=0.0000001

tic = time.perf_counter()
showfig=0


def prob_id(iidt,ndt,Ht,psit,ntemp,tstept,trstept,reflectt,Nt):
    nttemp=ntemp
    for itemp in Nt:
        Ht[(itemp,itemp)]=(random()-0.5)*W
    ittemp=0
    probxt=np.zeros( (ntplot+1, n) )
    for itemp in Nt:
        probxt[(0,itemp)]=np.real(psit[(itemp)]*np.conj(psit[(itemp)]))
    #reflected=0
    psix=psit
    while ittemp<1+ntemp:
        ittemp=ittemp+1
        k1=-1j*np.matmul(Ht,psix)
        k2=-1j*np.matmul(Ht,psix+(tstept/2)*k1)
        k3=-1j*np.matmul(Ht,psix+(tstept/2)*k2)
        k4=-1j*np.matmul(Ht,psix+tstept*k3)
        psix = psix + (tstept/6)*(k1 + 2*k2 + 2*k3 + k4)
        if abs(ittemp*trstept-round(ittemp*trstept))<10.0**(-5):
            for itemp in Nt:
                probxt[(round(ittemp*trstept),itemp)]=np.real(psix[(itemp)]*np.conj(psix[(itemp)]))
            border=probxt[(round(ittemp*trstept),0)]+probxt[(round(ittemp*trstept),1)]
            border=border+probxt[(round(ittemp*trstept),n-2)]+probxt[(round(ittemp*trstept),n-1)]
            if border > reflectt:
                #if ittemp<ntemp:
                    #print('Reflected at ittemp=',ittemp)
                nttemp=min(ntemp,ittemp)
                ittemp=ntemp+1
    if abs((iidt+1)*10/ndt-round((iidt+1)*10/ndt))<10.0**(-5):
        print ('+10%')
    return (nttemp,probxt)#(ntt,probxt)


pool=mp.Pool(numcores)
#pool=mp.Pool(mp.cpu_count())


nt=round(tf/tstep)
#reflected=0
tplotstep=tf/ntplot
trstep=(ntplot*tstep)/tf
normalize = 0
ndd=nd
N=np.arange(0,n,1)
x=np.array([-(n/2) + i for i in N])

for i in N:
    normalize=normalize+math.exp(-(x[(i)]**2)/(sigx**2))
normalize=math.sqrt(normalize)
psi=np.array([0+1j*0 for i in N])
for i in N:
    if math.exp(-(x[(i)]**2)/(sigx**2))>neglectpsi:
        psi[(i)]=(cmath.exp(-((x[(i)]**2)/(2*sigx**2)) +1j*k0*x[(i)]))/normalize
        
Mprobxt=np.zeros( (ntplot+1, n) )
H=np.zeros( (n, n) )
for i in N:
    for l in N:
        if i==l+1:
            H[(i,l)]=-J*math.exp(assmJ)
        if i==l-1:
            H[(i,l)]=-J*math.exp(-assmJ)

results=[]

# without parallelize
# for iid in range(nd):
#     results.append(prob_id(iid,nd,H,psi,nt,tstep,trstep,reflect,N))


# parallelize with apply
# results = [pool.apply(prob_id, args=(iid,nd,H,psi,nt,tstep,trstep,reflect,N)) for iid in range(nd)]
##pool.close()

# parallelize with apply_async and callback
# def collect_result(result):
#     global results
#     results.append(result)
# for iid in range(nd):
#     pool.apply_async(prob_id, args=(iid,nd,H,psi,nt,tstep,trstep,reflect,N), callback=collect_result)
# pool.close()
# pool.join()

# parallelize with apply_async without callback
# result_objects = [pool.apply_async(prob_id, args=(iid,nd,H,psi,nt,tstep,trstep,reflect,N)) for iid in range(nd)]
# results = [r.get() for r in result_objects]
# pool.close()
# pool.join()


# parallelize with starmap_async
results=pool.starmap_async(prob_id,[(iid,nd,H,psi,nt,tstep,trstep,reflect,N) for iid in range(nd)]).get()
pool.close()
pool.join()

ntt=nt
for iid in range(nd):
    ntt=min(ntt,results[(iid)][(0)])
    Mprobxt=Mprobxt+results[(iid)][(1)]

if ntt<nt:
    print('reflected, ntt=',ntt,'instead of',nt)

probxt=np.resize(results[(nd-1)][(1)],(1+round(ntt*trstep),n))
Mprobxt=np.resize(Mprobxt,(1+round(ntt*trstep),n))
Mprobxt=(1/ndd)*Mprobxt

v0 = 2*J*sin(k0)
cm=np.array([0.0 for it in range(1+round(ntt*trstep))])
var2=np.array([0.0 for it in range(1+round(ntt*trstep))])
Mcm=np.array([0.0 for it in range(1+round(ntt*trstep))])
Mvar2=np.array([0.0 for it in range(1+round(ntt*trstep))])

for it in range(1+round(ntt*trstep)):
    for i in N:
        cm[(it)]=cm[(it)]+ x[(i)]*probxt[(it,i)]
        var2[(it)]=var2[(it)]+ (x[(i)]**2)*probxt[(it,i)]
        Mcm[(it)]=Mcm[(it)]+ x[(i)]*Mprobxt[(it,i)]
        Mvar2[(it)]=Mvar2[(it)]+ (x[(i)]**2)*Mprobxt[(it,i)]

dMvar2=np.array([(Mvar2[(it+1)]-Mvar2[(it)])/tplotstep for it in range(round(ntt*trstep))])
T1=np.array([it*tplotstep for it in range(1+round(ntt*trstep))])
T2=np.array([it*tplotstep for it in range(round(ntt*trstep))])

toc = time.perf_counter()
print('code runned in', toc - tic, 'seconds')

np.savetxt("Probxt.out", probxt, fmt="%s")
np.savetxt("MProbxt.out", Mprobxt, fmt="%s")


f1=plt.figure(figsize=(7,4))
plt.xlabel("t J")
plt.ylabel("Xcm/a")
plt.plot(T1, cm,'b-')
plt.savefig('f1.png')
if showfig==1:
    plt.show()

f2=plt.figure(figsize=(7,4))
plt.xlabel("t J")
plt.ylabel("<Xcm>/a")
plt.plot(T1, Mcm,'b-')
plt.savefig('f2.png')
if showfig==1:
    plt.show()

f3=plt.figure(figsize=(7,4))
plt.xlabel("t J")
plt.ylabel("var/a")
plt.plot(T1, [math.sqrt(var2[(it)]) for it in range(1+round(ntt*trstep))],'b-')
plt.savefig('f3.png')
if showfig==1:
    plt.show()

f4=plt.figure(figsize=(7,4))
plt.xlabel("t J")
plt.ylabel("<var>/a")
plt.plot(T1, [math.sqrt(Mvar2[(it)]) for it in range(1+round(ntt*trstep))],'b-')
plt.savefig('f4.png')
if showfig==1:
    plt.show()

f5, ax=plt.subplots(figsize=(7,4))
ax.plot(T2, [dMvar2[(it)] for it in range(round(ntt*trstep))],'b-', label='d<var2>/dt')
ax.plot(T2, [2*v0*Mcm[(it)] for it in range(round(ntt*trstep))],'r--', label='2v0<Xcm>')
legend = ax.legend(loc='upper right')
plt.xlabel("t J")
plt.ylabel("a.u.")
plt.savefig('f5.png')
if showfig==1:
    plt.show()

f6=plt.figure(figsize=(7,4))
plt.imshow(np.transpose(probxt),extent=[0,tplotstep*round(ntt*trstep),x[(0)],x[(n-1)]], aspect='auto',origin='lower')
plt.colorbar()
plt.title('|psi(x,t)|^2', fontweight ="bold",loc='right')
plt.xlabel("t J")
plt.ylabel("1/a")
plt.savefig('f6.png')
if showfig==1:
    plt.show()

f7=plt.figure(figsize=(7,4))
plt.imshow(np.transpose(Mprobxt),extent=[0,tplotstep*round(ntt*trstep),x[(0)],x[(n-1)]], aspect='auto',origin='lower')
plt.colorbar()
plt.title('<|psi(x,t)|^2>', fontweight ="bold",loc='right')
plt.xlabel("t J")
plt.ylabel("1/a")
plt.savefig('f7.png')
if showfig==1:
    plt.show()
