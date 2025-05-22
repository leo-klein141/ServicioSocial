import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import cupy as cp
import cupyx
import matplotlib as mpl



def getPosition(pos:np.ndarray, LUC:tuple[float], RLC:tuple[float], pixelsImage:tuple[int])->tuple[int]:
    assert len(pos)==2 and pos.shape[0] == 2, 'Formato de entrada para la posición incorrecto!\n'

    dxPixel = (RLC[0]-LUC[0])/pixelsImage[0]
    dyPixel = (LUC[1]-RLC[1])/pixelsImage[1]

    ### Encontrar en que pixel esta
    px = int((pos[0] - LUC[0])/dxPixel)
    py = int((LUC[1] - pos[1])/dyPixel)

    if(py >= pixelsImage[1]):
        if(py - pixelsImage[1]  > 1):
            exit('Particula demasiado lejos de los margenes')
        py = pixelsImage[1]-1

    if(px >= pixelsImage[0]):
        if(px - pixelsImage[0]  > 1):
            exit('Particula demasiado lejos de los margenes')
        px = pixelsImage[0]-1
    
    if(py < 0):
        if(py < -1):
            exit('Particula demasiado lejos de los margenes')
        py = 0

    if(px < 0):
        if(px < -1):
            exit('Particula demasiado lejos de los margenes')
        px = 0

    # assert (px > 0 and py > 0) and (px < pixelsImage[0] and py < pixelsImage[1]), 'ERROR: Los pixeles se salen de la imagen!\n'+f'Pos:  {(px,py)}'
    # print(f'Posicion = {pos} -> pixelPos = {(px,py)}')
    return (px, py)

def getBlock(pixelPos:tuple[int], pixelsImage:tuple[int], pixelsBlock:tuple[int])->tuple[int]:
    #Ahora tenemos la posicion en pixeles, ahora queremos hacer en bloques
    # print(f'pixelPos = {pixelPos}')
    assert pixelPos[0] < pixelsImage[0] and pixelPos[1] < pixelsImage[1], 'Tenemos una particula fuera de la imagen!\nNo se puede procesar!!!\n'
    assert pixelPos[0] >= 0 and pixelPos[1] >= 0, 'Tenemos una particula fuera de la imagen!\nNo se puede procesar!!!\n'
    
    return (pixelPos[0]//pixelsBlock[0], pixelPos[1]//pixelsBlock[1])

def getPixelDistance(pixPosI:tuple[int], pixPosJ:tuple[int], pixelsImage:tuple[int], blockPixel:tuple[int])->int|np.int64:
    assert (0 <= pixPosI[0] and pixPosI[0] < pixelsImage[0]), 'Posicion de pixel (en X) no es valida para la primer entrada!\n'
    assert (0 <= pixPosI[1] and pixPosI[1] < pixelsImage[1]), 'Posicion de pixel (en Y) no es valida para la primer entrada!\n'

    assert (0 <= pixPosJ[0] and pixPosJ[0] < pixelsImage[0]), 'Posicion de pixel (en X) no es valida para la segunda entrada!\n'
    assert (0 <= pixPosJ[1] and pixPosJ[1] < pixelsImage[1]), 'Posicion de pixel (en Y) no es valida para la segunda entrada!\n'
    
    return np.int64(abs(pixPosI[0] - pixPosJ[0])//blockPixel[0]+abs(pixPosI[1] - pixPosJ[1])//blockPixel[1])

N = 4235
hstep = 0.001
sepFrames = 10
timeSimulated = 25.0
numberOfFrames = 1+2500

parentFileDiretory = '/home/leonardocuda/Documentos/Servicio_Social/CavityProblem/dataManipulationPrograms/'

X_CPU = np.fromfile(parentFileDiretory+'dataSerial/serialPositions.bin', dtype=np.float64, count=2*N*numberOfFrames)
V_CPU = np.fromfile(parentFileDiretory+'dataSerial/serialVelocities.bin', dtype=np.float64, count=2*N*numberOfFrames)
T_CPU = np.fromfile(parentFileDiretory+'dataSerial/serialTime.bin', dtype=np.float64, count=numberOfFrames)

X_GPU = np.fromfile(parentFileDiretory+'dataParallel/positions.bin', dtype=np.float64, count=2*N*numberOfFrames)
V_GPU = np.fromfile(parentFileDiretory+'dataParallel/velocities.bin', dtype=np.float64, count=2*N*numberOfFrames)
T_GPU = np.fromfile(parentFileDiretory+'dataParallel/time.bin', dtype=np.float64, count=numberOfFrames)

assert len(T_CPU) == len(T_GPU), "Cantidad de tiempos distintos!\n"

X_CPU = X_CPU.reshape((numberOfFrames,N,2))
V_CPU = V_CPU.reshape((numberOfFrames,N,2))

X_GPU = X_GPU.reshape((numberOfFrames,N,2))
V_GPU = V_GPU.reshape((numberOfFrames,N,2))

boxSize = 183.6

leftUpperCorner = (-boxSize*0.5, boxSize)
rightLowerCorner = ( boxSize*0.5, 0.0)

imageResolutionPixels = (16*512, 16*512)
blockPixels = (16,16)
# barDistance = np.linspace(0.0, imageResolutionPixels[0]+imageResolutionPixels[1], (blockPixels[0]+blockPixels[1])//2 )
mycmap = mpl.cm.viridis
mynorm = mpl.colors.Normalize(vmin=0, vmax=imageResolutionPixels[0]//blockPixels[0]+imageResolutionPixels[1]//blockPixels[1] )


for i in range(numberOfFrames):
    print('Iteracion numero: ', i, ', tiempo: ', T_CPU[i])
    print('Diferencia en tiempos: ', (T_CPU[i] - T_GPU[i]))
    VM_CPU = V_CPU[i,:,0]**2 + V_CPU[i,:,1]**2
    VM_GPU = V_GPU[i,:,0]**2 + V_GPU[i,:,1]**2
    print('Diferencias en las velocidades promedio = ', (np.mean(VM_CPU) - np.mean(VM_GPU)))
    print('Promedio de las diferencias en las velocidades = ', (np.mean(np.sqrt( (V_CPU[i,:,0]-V_GPU[i,:,0])**2 + (V_CPU[i,:,1] - V_GPU[i,:,1])**2 ))))
    ######## Hasta aquí hemos impreso la información de las velocidades.
    print('Error acumulado de las distancias = ', (np.sum(np.sqrt( (X_CPU[i,:,0]-X_GPU[i,:,0])**2 + (X_CPU[i,:,1] - X_GPU[i,:,1])**2 ))))

    ### Ahora a establecer los bloques de pixeles
    particlesInBlockCPU = np.zeros((imageResolutionPixels[0]//blockPixels[0], imageResolutionPixels[1]//blockPixels[1]), dtype=np.int64)
    distancesWithPixels = np.zeros_like(particlesInBlockCPU, dtype=np.int64)
    for p in range(N):
        pixCPU = getPosition(X_CPU[i,p], leftUpperCorner,rightLowerCorner, imageResolutionPixels)
        pixGPU = getPosition(X_GPU[i,p], leftUpperCorner,rightLowerCorner, imageResolutionPixels)
        particlesInBlockCPU[getBlock(pixCPU,imageResolutionPixels,blockPixels)]+=1
        # particlesInBlock[getBlock(pixCPU,imageResolutionPixels,blockPixels)]+=1
        distancesWithPixels[getBlock(pixCPU, imageResolutionPixels, blockPixels)] += getPixelDistance(pixCPU, pixGPU, imageResolutionPixels, blockPixels)
    
    plt.figure(1,figsize=(9,8),dpi=200)
    imgplot = plt.imshow(distancesWithPixels, cmap='viridis', norm=mynorm)
    imgplot.axes.set_title(f'Comparacion en distancia de Manhatan Pixels t = {T_CPU[i]} ps')
    imgplot.axes.set_xlabel(f'Bloques en X')
    imgplot.axes.set_ylabel(f'Bloques en Y')
    plt.colorbar(imgplot)
    plt.savefig(parentFileDiretory+'imageDiferences/'+f'difPathImg_{i*sepFrames}.png')
    plt.clf()
    
    
