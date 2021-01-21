import numpy as np
import pickle as pkl
import sys
from numpy.linalg import norm

def load(fileName):
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput

def save(obj,fileName):
    file = open(fileName,'wb')
    pkl.dump(obj,file)
    file.close()

with open(sys.argv[1],'r') as f:
    lines = f.readlines()

N = int(lines[-22].split()[1])+2

h = np.zeros((N,20))

for m in range(N):
    split = lines[m].split()
    for n in range(20):
        h[m,n] = float(split[n])

J = np.zeros((N,N,21,21))

for i in range(N):
    for j in range(i+1,N):
        m = m+1
        assert(lines[m] == '# ' + str(i) + ' ' + str(j) + '\n')
        for k in range(21):
            m = m+1
            split = lines[m].split()
            for l in range(21):
                J[i,j,k,l] = J[j,i,l,k] = float(split[l])

data = load(sys.argv[2])
data['h'] = h
data['J'] = J
fn = norm(J,axis=(2,3))
data['frobenius_norm'] = fn
cn = fn - np.mean(fn,axis=0)[np.newaxis,:]*np.mean(fn,axis=1)[:,np.newaxis]/np.mean(fn)
data['corrected_norm'] = cn


save(data,sys.argv[2])