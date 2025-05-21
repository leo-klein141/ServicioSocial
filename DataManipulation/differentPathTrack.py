import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import cupy as cp
import cupyx


N = 4235
hstep = 0.001
sepFrames = 10
timeSimulated = 25.0
numberOfFrames = 1+int(timeSimulated/hstep)//sepFrames

parentFileDiretory = 'D:/MyFiles/Documentos/CodigosServicioSocial/NumbaCuda/SerialLeapfrog/'

X_CPU = np.fromfile(parentFileDiretory+'serialPositions.bin', dtype=np.float64, count=2*N*numberOfFrames)
V_CPU = np.fromfile(parentFileDiretory+'serialVelocities.bin', dtype=np.float64, count=2*N*numberOfFrames)
T_CPU = np.fromfile(parentFileDiretory+'serialTime.bin', dtype=np.float64, count=numberOfFrames)

X_GPU = np.fromfile(parentFileDiretory+'parallelPositions.bin', dtype=np.float64, count=2*N*numberOfFrames)
V_GPU = np.fromfile(parentFileDiretory+'parallelVelocities.bin', dtype=np.float64, count=2*N*numberOfFrames)
T_GPU = np.fromfile(parentFileDiretory+'parallelTime.bin', dtype=np.float64, count=numberOfFrames)

assert len(T_CPU) == len(T_GPU), "Cantidad de tiempos distintos!\n"

X_CPU = X_CPU.reshape((numberOfFrames,N,2))
V_CPU = V_CPU.reshape((numberOfFrames,N,2))

X_GPU = X_GPU.reshape((numberOfFrames,N,2))
V_GPU = V_GPU.reshape((numberOfFrames,N,2))

boxSize = 183.6

leftUpperCorner = (-boxSize*0.5, boxSize)
rightLoerCorner = ( boxSize*0.5, 0.0)

imageResolutionPixels = (2048,2048)
blockPixels = (16,16)

for i in range(numberOfFrames):
    print('Diferencia en tiempos: ', (T_CPU[i] - T_GPU[i]))
