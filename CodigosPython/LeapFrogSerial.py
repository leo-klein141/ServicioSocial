### CODIGO MACISO PARA LLEVAR EN EL CPU

import numpy as np
import numba as nb
from math import sqrt
import matplotlib.pyplot as plt

from numba import float32, float64, guvectorize
############ ant

h = 0.001 # espacio
boxSide = 183.6
cuadrantLength = boxSide * 0.5
totalSteps = 25000 #total de iteraciones
frameIterSep = 10

G = 160330.0 * 818.9 #ante que indicará nuestro potencial de atracción
H = 160330.0 #ante que indicará nuestro potencial de repulsión
p = 13 #Potencia para nuestro potencial de atracción (-G*r^(-p))
q = 7 #Potencia para nuestro potencial de repulsión (H*r^(-q))
damp = 0.75 #ante de fricción al golpear una caja
v_tapa = -12.0 #Velocidad que adquiere la particula al chocar con la tapa
vp = 6.23 #Velocidad inicial de las particulas

rInt = np.float64(3.39) #Distancia máxima de interacción, esta depende de si queremos ver vorticidad o no
rMin = np.float64(2.75) #Valor minimo de nuestro radio para evitar velocidades demasiado grandes (overflow)
softFactor = 0.001 #Valor que nos ayudará a mantener nuestra fuerza acta

### ------- Opcional Up to Configuration ---------

row1_part = int(61)
row1_dx = 3.06
row2_part = 60
row2_dx = 3.06
rows_dy = 2.65
blockParticles = int(121)
totalParticles = int(4235)

#### Hora de definir funciones
@guvectorize(
    [(float64[:], float64[:], float64[:])],
    '(n), (n) -> (n)',
    nopython=True,
    target='parallel'
)
def pairwise_lennarJonnes_gu(body1:np.ndarray[np.float64], body2:np.ndarray[np.float64], outForce:np.ndarray[np.float64]):
    """
    Calculates the gravitational force between two individual bodies.

    Args:
    body1 (array-like): [x1, y1] of the first body.
    body2 (array-like): [x2, y2] of the second body.
    outForce (array-like): Output array to store the force on body1.
    G (float): Gravitational constant.
    """
    x1, y1 = body1[0], body1[1]
    x2, y2 = body2[0], body2[1]
    dx = x2 - x1
    dy = y2 - y1
    distance_sq = dx*dx + dy*dy
    if distance_sq < 1e-12 or distance_sq > rInt*rInt:
        outForce[0] = 0.0
        outForce[1] = 0.0
        return

    distance_sq = distance_sq**(0.5)

    force_x = dx/distance_sq
    force_y = dy/distance_sq
    upperPot = -G

    i = 1
    dist = np.float64(1.0)
    if distance_sq < rMin:
        distance_sq = rMin

    dist *= distance_sq
    while i < p - q:
        i+=1
        dist *= distance_sq
    ### En este punto dist = (distance_sq**(p-q))

    upperPot += H*dist

    while i<p:
        i+=1
        dist *= distance_sq
    ### En este punto dist = distance_sq**p
    outForce[0] = force_x*upperPot/dist
    outForce[1] = force_y*upperPot/dist
    return


def compute_net_forces_parallel(bodies):
    """Computes the net gravitational force on each body using guvectorize."""
    num_bodies = bodies.shape[0]
    forces = np.zeros((num_bodies, 2), dtype=np.float64)

    for i in range(num_bodies):
        # Calculate forces exerted by all other bodies on body i
        other_bodies = np.delete(bodies, i, axis=0)
        assert other_bodies.shape[0] == (num_bodies-1) and other_bodies.shape[1]==2, "Error, no se eliminó nada, auxilio!\n"
        current_body = np.tile(bodies[i], (num_bodies-1, 1)) # Reshape for guvectorize
        #current_body = bodies[i]

        forces_on_i = pairwise_lennarJonnes_gu(
        current_body, other_bodies, np.zeros_like(other_bodies)
        )
        forces[i, :] = np.sum(forces_on_i, axis=0)

    return forces

@nb.guvectorize(
    [(float64[:],float64[:],float64[:],float64[:])],
    '(n),(n) -> (n),(n)',
    target='parallel',
    nopython=True
)
def fitInsideParticle(position, velocity, newPosition, newVelocity):
    newPosition[0] = position[0]
    newPosition[1] = position[1]
    newVelocity[0] = velocity[0]
    newVelocity[1] = velocity[1]

    marginError = np.float64(1e-6)

    if((-cuadrantLength < newPosition[0] and newPosition[0] < cuadrantLength)
    and (0 < newPosition[1] and newPosition[1] < boxSide)):
        return
    
    contCycles = 0
    isInside = False

    while not(isInside) and contCycles < 25:
        contCycles += 1

        if( - cuadrantLength > newPosition[0] ):
        # Tenemos una particula a la izquierda
            newPosition[0] = -2.0*cuadrantLength - newPosition[0]
            newVelocity[0] *= -damp
            newVelocity[1] *= damp

        if( newPosition[1] < 0):
            newPosition[1] = -newPosition[1]
            newVelocity[0] *= damp
            newVelocity[1] *= -damp
        
        if( cuadrantLength < newPosition[0] ):
            newPosition[0] = 2.0*cuadrantLength - newPosition[0]
            newVelocity[0] *= -damp
            newVelocity[1] *= damp

        if( boxSide < newPosition[1] ):
            newPosition[1] = 2.0*boxSide - newPosition[1]
            newVelocity[0] = newVelocity[0]*damp + v_tapa
            newVelocity[1] = -newVelocity[1]*damp

        if((-cuadrantLength < newPosition[0] and newPosition[0] < cuadrantLength)
        and (0 < newPosition[1] and newPosition[1] < boxSide)):
            isInside = True

    return


@nb.guvectorize(
    '(float64[:], float64[:], float64, float64[:])',
    '(n),(n),() -> (n)',
    target='parallel',
    nopython=True
)
def integration(yn, dy, hstep, y):
    y[0] = yn[0] + hstep*dy[0]
    y[1] = yn[1] + hstep*dy[1]
    return

def getInitialPositions(firsRowPos:tuple, secondRowPos:tuple, sepX:float|np.float64, sepY:float|np.float64
    , numFirstRow:int, N:int):
    positions = np.zeros((N, 2),dtype=np.float64)
    blockP = int(2*numFirstRow - 1)

    firsRowPos = np.array(firsRowPos, dtype=np.float64)
    secondRowPos = np.array(secondRowPos, dtype=np.float64)
    sepX = np.float64(sepX)
    sepY = np.float64(sepY)


    positions[0] = firsRowPos
    for i in range(1, numFirstRow):
        positions[i][0] = positions[i-1][0] + sepX
        positions[i][1] = positions[i-1][1]

    positions[numFirstRow] = secondRowPos
    for i in range(numFirstRow+1, blockP):
        positions[i][0] = positions[i-1][0] + sepX
        positions[i][1] = positions[i-1][1]

    for i in range(blockP, N):
        positions[i][0] = positions[i-blockP][0]
        positions[i][1] = positions[i-blockP][1] + 2.0*sepY

    return positions

def getInitialVelocities(v0, N):
    angles = np.random.uniform(-np.pi,np.pi,N)
    velocities = v0*np.transpose([np.cos(angles), np.sin(angles)])
    return velocities

if __name__ == '__main__':
    # Example usage:
    import time
    num_bodies = totalParticles
    hstep = h
    bodies_dataX = getInitialPositions((-cuadrantLength, 0.0), (-cuadrantLength + row1_dx*0.5, rows_dy),\
    row1_dx, rows_dy, row1_part, num_bodies)
    bodies_dataV = getInitialVelocities(vp, num_bodies)
    # Run once for compilation
    bodies_dataA = compute_net_forces_parallel(bodies_dataX)


    print("Tiempo actual en la simulacion! : ", (0.0))
    print("Maximo valor en la acelaracion: ", (np.max(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
    print("Minimo valor en la acelaracion: ", (np.min(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
    velocidades = np.sqrt(bodies_dataV[:,0]**2 + bodies_dataV[:,1]**2)
    velocidadPromedio = (1.0/num_bodies)*np.sum(velocidades)
    print("Energia promedio del sistema: ", (3.0103e-24)*velocidadPromedio, " Joules.")
    ### Aqui hacemos una mascara, en particular queremos ver quienes son las que son particulas muy rapidas
    mask = velocidades>=0.9*velocidadPromedio
    elementosRapidosX = []
    elementosRapidosV = []
    magnitude = []
    for i in range(len(mask)):
        if mask[i]:
            elementosRapidosX.append(bodies_dataX[i])
            elementosRapidosV.append(bodies_dataV[i])
            magnitude.append(velocidades[i])
    index = int(0)
    filename = 'SerialLeapfrog/fastParticlesImages/plot_quiver'+str(index)+'.png'

    elementosRapidosV = np.array(elementosRapidosV, dtype=np.float64)
    elementosRapidosX = np.array(elementosRapidosX, dtype=np.float64)

    plt.figure(figsize=(12, 13))
    plt.plot([-boxSide/2.0, boxSide/2.0], [0.0, 0.0], color='k')
    plt.plot([boxSide/2.0, boxSide/2.0], [0.0, boxSide], color='k')
    plt.plot([boxSide/2.0, -boxSide/2.0], [boxSide, boxSide], color='k')
    plt.plot([-boxSide/2.0, -boxSide/2.0], [boxSide, 0.0], color='k')
    plt.quiver(elementosRapidosX[:,0], elementosRapidosX[:,1],
    elementosRapidosV[:,0], elementosRapidosV[:,1],
    magnitude, cmap='viridis')
    plt.colorbar(label='Velocidad (Angstroms/ps)')
    plt.xlabel("X-axis (Angstroms)")
    plt.ylabel("Y-axis (Angstroms)")
    plt.title(f"Simulacion al tiempo {0.0} ps. Particulas veloces")
    plt.savefig(filename)
    plt.clf()
    print("\n")


    with open('serialPositions.bin', 'wb') as sp:
        with open('serialVelocities.bin', 'wb') as sv:
            with open('serialTime.bin', 'wb') as st:
                frames = 0
                st.write(np.float64(0.0).tobytes())
                sp.write(bodies_dataX.tobytes())
                sv.write(bodies_dataV.tobytes())

                #ahora calculo
                bodies_dataA = compute_net_forces_parallel(bodies_dataX)

                for s in range(totalSteps):
                    if frames == frameIterSep:
                        st.write(np.float64(s*hstep).tobytes())
                        sp.write(bodies_dataX.tobytes())
                        sv.write(bodies_dataV.tobytes())
                        print("Tiempo actual en la simulacion! : ", (s*hstep))
                        print("Maximo valor en la acelaracion: ", (np.max(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
                        print("Minimo valor en la acelaracion: ", (np.min(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
                        velocidades = np.sqrt(bodies_dataV[:,0]**2 + bodies_dataV[:,1]**2)
                        velocidadPromedio = (1.0/num_bodies)*np.sum(velocidades)
                        print("Energia promedio del sistema: ", (3.0103e-24)*velocidadPromedio, " Joules.")
                        ### Aqui hacemos una mascara, en particular queremos ver quienes son las que son particulas muy rapidas
                        mask = velocidades>=0.8*velocidadPromedio
                        elementosRapidosX = []
                        elementosRapidosV = []
                        magnitude = []
                        for i in range(len(mask)):
                            if mask[i]:
                                elementosRapidosX.append(bodies_dataX[i])
                                elementosRapidosV.append(bodies_dataV[i])
                                magnitude.append(velocidades[i])
                        index = s
                        filename = 'SerialLeapfrog/fastParticlesImages/plot_quiver'+str(index)+'.png'

                        elementosRapidosV = np.array(elementosRapidosV, dtype=np.float64)
                        elementosRapidosX = np.array(elementosRapidosX, dtype=np.float64)

                        plt.plot([-boxSide/2.0, boxSide/2.0], [0.0, 0.0], color='k')
                        plt.plot([boxSide/2.0, boxSide/2.0], [0.0, boxSide], color='k')
                        plt.plot([boxSide/2.0, -boxSide/2.0], [boxSide, boxSide], color='k')
                        plt.plot([-boxSide/2.0, -boxSide/2.0], [boxSide, 0.0], color='k')
                        plt.quiver(elementosRapidosX[:,0], elementosRapidosX[:,1],
                        elementosRapidosV[:,0], elementosRapidosV[:,1],
                        magnitude, cmap='viridis')
                        plt.colorbar(label='Velocidad (Angstroms/ps)')
                        plt.xlabel("X-axis (Angstroms)")
                        plt.ylabel("Y-axis (Angstroms)")
                        plt.title(f"Simulacion al tiempo {hstep*s} ps. Particulas veloces")
                        plt.savefig(filename)
                        plt.clf()
                        print("\n")
                        frames = 0

                    bodies_dataV = bodies_dataV + 0.5*hstep*bodies_dataA
                    bodies_dataX = bodies_dataX + hstep*bodies_dataV
                    bodies_dataX, bodies_dataV = fitInsideParticle(bodies_dataX, bodies_dataV, np.zeros_like(bodies_dataX), np.zeros_like(bodies_dataV))
                    bodies_dataA = compute_net_forces_parallel(bodies_dataX)

                    bodies_dataV = bodies_dataV + 0.5*hstep*bodies_dataA
                    frames += 1
                    st.write(np.float64(totalSteps*hstep).tobytes())
                    sp.write(bodies_dataX.tobytes())
                    sv.write(bodies_dataV.tobytes())
                    pass

    
    print("Tiempo actual en la simulacion! : ", (h*totalSteps))
    print("Maximo valor en la acelaracion: ", (np.max(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
    print("Minimo valor en la acelaracion: ", (np.min(bodies_dataA[:,0]**2 + bodies_dataA[:,1]**2)))
    velocidades = np.sqrt(bodies_dataV[:,0]**2 + bodies_dataV[:,1]**2)
    velocidadPromedio = (1.0/num_bodies)*np.sum(velocidades)
    print("Energia promedio del sistema: ", (3.0103e-24)*velocidadPromedio, " Joules.")
    ### Aqui hacemos una mascara, en particular queremos ver quienes son las que son particulas muy rapidas
    mask = velocidades>=0.9*velocidadPromedio
    elementosRapidosX = []
    elementosRapidosV = []
    magnitude = []
    for i in range(len(mask)):
        if mask[i]:
            elementosRapidosX.append(bodies_dataX[i])
            elementosRapidosV.append(bodies_dataV[i])
            magnitude.append(velocidades[i])
    index = int(0)
    filename = 'SerialLeapfrog/fastParticlesImages/plot_quiver'+str(totalSteps)+'.png'

    elementosRapidosV = np.array(elementosRapidosV, dtype=np.float64)
    elementosRapidosX = np.array(elementosRapidosX, dtype=np.float64)

    plt.figure(figsize=(12, 13))
    plt.plot([-boxSide/2.0, boxSide/2.0], [0.0, 0.0], color='k')
    plt.plot([boxSide/2.0, boxSide/2.0], [0.0, boxSide], color='k')
    plt.plot([boxSide/2.0, -boxSide/2.0], [boxSide, boxSide], color='k')
    plt.plot([-boxSide/2.0, -boxSide/2.0], [boxSide, 0.0], color='k')
    plt.quiver(elementosRapidosX[:,0], elementosRapidosX[:,1],
    elementosRapidosV[:,0], elementosRapidosV[:,1],
    magnitude, cmap='viridis')
    plt.colorbar(label='Velocidad (Angstroms/ps)')
    plt.xlabel("X-axis (Angstroms)")
    plt.ylabel("Y-axis (Angstroms)")
    plt.title(f"Simulacion al tiempo {totalSteps*h} ps. Particulas veloces")
    plt.savefig(filename)
    plt.clf()
    print("\n")
    
    # Time the parallel execution
    start_time = time.time()
    bodies_dataA = compute_net_forces_parallel(bodies_dataX)
    end_time = time.time()
    print(f"Time taken with guvectorize (parallel): {end_time - start_time:.6f} seconds")
