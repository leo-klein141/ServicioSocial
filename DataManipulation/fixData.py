import numpy as np

N = 4235

firstsIterations = 914

parentDirectory = 'D:/MyFiles/Documentos/CodigosServicioSocial/NumbaCuda/SerialLeapfrog/'

firstChunckX = np.fromfile(parentDirectory+'firstIterations/serialPositions.bin', dtype=np.float64, count=2*firstsIterations*N)
firstChunckV = np.fromfile(parentDirectory+'firstIterations/serialVelocities.bin', dtype=np.float64, count=2*firstsIterations*N)
firstChunckT = np.fromfile(parentDirectory+'firstIterations/serialTime.bin', dtype=np.float64, count=firstsIterations)

firstChunckX = firstChunckX.reshape((firstsIterations, N, 2))
firstChunckV = firstChunckV.reshape((firstsIterations, N, 2))
#firstChunckX = firstChunckX.reshape((firstsIterations, N, 2))

totalIterations = 2500
otherIterations = 1 + totalIterations - firstsIterations

otherChunckX = np.fromfile(parentDirectory+'laterIterations/serialPositions.bin', dtype=np.float64, count=2*N*(1+totalIterations-firstsIterations))
otherChunckV = np.fromfile(parentDirectory+'laterIterations/serialVelocities.bin', dtype=np.float64, count=2*N*(1+totalIterations-firstsIterations))
otherChunckT = np.fromfile(parentDirectory+'laterIterations/serialTime.bin', dtype=np.float64, count=(1+totalIterations-firstsIterations))

otherChunckX = otherChunckX.reshape((otherIterations, N, 2))
otherChunckV = otherChunckV.reshape((otherIterations, N, 2))

dataX = np.concatenate(firstChunckX, otherChunckX[1:])
dataV = np.concatenate(firstChunckV, otherChunckV[1:])
dataT = np.concatenate(firstChunckT, otherChunckT[1:])

np.save()

