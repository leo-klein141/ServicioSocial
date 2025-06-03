#include <hip/hip_runtime.h>

#include <fstream>
#include <stdio.h>
#include <cmath>
#include <string>

#define BLOCK_SIZE 32
#define EPS_RADIUS 1e-9f

#define NUMBER_OF_PARTICLES 4235

///////////////////////
//      Los siguientes son valores que nos servirán para nuestra forma de correr la simulación
//      En particular los valores de la diferencia de paso en la integración
//      Así cómo del potencial que estaremos trabajando
////////////////// Constantes para el codigo
const double h = 0.001L; // espacio
const double boxSide = 183.6L;
const double cuadrantLength = boxSide * 0.5L;
const int totalSteps = 250000; //total de iteraciones
const int frameIterSep =    10;
const double DMAX = 3.39L;                  //Distancia máxima de interacción, esta depende de si queremos ver vorticidad o no
const double G = 160330.0L * 818.9L;     //Constante que indicará nuestro potencial de atracción
const double H = 160330.0L;              //Constante que indicará nuestro potencial de repulsión
const int p = 13;                       //Potencia para nuestro potencial de atracción     (-G*r^(-p))
const int q = 7;                        //Potencia para nuestro potencial de repulsión      (H*r^(-q))
const double damp = 0.75L;               //Constante de fricción al golpear una caja
double vtapa = -12.0L;                  //Velocidad que adquiere la particula al chocar con la tapa
const double vini = 6.23L;                 //Velocidad inicial de las particulas
//const double g = 0.0L;                      //Constante de gravedad (no es necesaria a considerar en nuestro problema)
const double DMIN = 2.75L;                 //Valor minimo de nuestro radio para evitar velocidades demasiado grandes (overflow)
// const double softFactor = 0.001L;

/// ------- Opcional Up to Configuration ---------

const int row1_part = 61;
const double rows_dx = 3.06L;
const int row2_part = 60;
//const double row2_dx = 3.06L;
const double rows_dy = 2.65L;
const int blockParticles = row1_part + row2_part;
// const int totalParticles = 4235;

/// --------------------------------------

//////// BLOCKS / THREADS CONFIGURATION
const int NUMBLOCKS = 1 + (NUMBER_OF_PARTICLES / BLOCK_SIZE);

///// Constantes en la GPU :

__device__ const double dt = 0.001L; // espacio
__device__ const double dboxSide = 183.6L;
//__device__ const double dhalf = boxSide * 0.5L;
//__device__ const int totalSteps = 250000; //total de iteraciones
//__device__ const int frameIterSep =    10;
__device__ const double DDMAX = 3.39L;                  //Distancia máxima de interacción, esta depende de si queremos ver vorticidad o no
__device__ const double dG = 160330.0L * 818.9L;     //Constante que indicará nuestro potencial de atracción
__device__ const double dH = 160330.0L;              //Constante que indicará nuestro potencial de repulsión
__device__ const int dp = 13;                       //Potencia para nuestro potencial de atracción     (-G*r^(-p))
__device__ const int dq = 7;                        //Potencia para nuestro potencial de repulsión      (H*r^(-q))
__device__ const double ddamp = 0.75L;               //Constante de fricción al golpear una caja
__device__ double dvtapa = -12.0L;                  //Velocidad que adquiere la particula al chocar con la tapa
//__device__ const double vini = 6.23L;                 //Velocidad inicial de las particulas
//const double g = 0.0L;                      //Constante de gravedad (no es necesaria a considerar en nuestro problema)
__device__ const double DDMIN = 2.75L;                 //Valor minimo de nuestro radio para evitar velocidades demasiado grandes (overflow)

__device__ double3 calculateForces(double3 myPosition, double3 currAcc, int shift){
    __shared__ double blockPositions[3*BLOCK_SIZE];
    double3 sumAcc = make_double3(0.0, 0.0, 0.0);
    double aj[3];
    double force;
    double dxy;
    double radius;
    //double3 a;
    for(int i=0, j; i<BLOCK_SIZE ; i++){
        radius = 1.0;
        j = (shift + i)%BLOCK_SIZE;
        aj[0] = blockPositions[j]               - myPosition.x;
        aj[1] = blockPositions[j + BLOCK_SIZE]  - myPosition.y;
        aj[2] = blockPositions[j + 2*BLOCK_SIZE]- myPosition.z;

        dxy = aj[0]*aj[0] + aj[1]*aj[1] + aj[2]*aj[2];
        if(dxy < EPS_RADIUS || DDMAX * DDMAX < dxy ){ continue; }

        aj[0] /= sqrt(dxy);
        aj[1] /= sqrt(dxy);
        aj[2] /= sqrt(dxy);

        dxy = max(dxy, DDMIN * DDMIN);

        for(j = 0; j < dq ; j++){   radius *= dxy;  }

        force += dH/sqrt(radius);

        for(     ; j < dp ; j++){  radius *= dxy;   }

        force -= dG/sqrt(radius);

        aj[0] *= force;
        aj[1] *= force;
        aj[2] *= force;

        sumAcc = make_double3(sumAcc.x + aj[0], sumAcc.y + aj[1], sumAcc.z + aj[2]);
    }

    currAcc = make_double3(currAcc.x + sumAcc.x, currAcc.y + sumAcc.y, currAcc.z + sumAcc.z);

    return currAcc;
}

__device__ void updateForces(double* posBodies, double* accBodies){
    __shared__ double blockPositions[3*BLOCK_SIZE];
    double3 piBody;
    double3 acc = make_double3(0.0, 0.0, 0.0);
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid;

    if(gtid < NUMBER_OF_PARTICLES){
        piBody = make_double3(
            posBodies[gtid],
            posBodies[gtid + NUMBER_OF_PARTICLES],
            0.0//posBodies[gtid + 2*NUMBER_OF_PARTICLES]
        );
    } else {
        piBody = make_double3(
            dboxSide*2.0,
            dboxSide*2.0,
            0.0
        );
    }

    for(int i = 0, tile = 0; i < NUMBER_OF_PARTICLES ; i+=BLOCK_SIZE, tile++){
        tid = blockDim.x * tile + threadIdx.x;
        if(tid < NUMBER_OF_PARTICLES){
            blockPositions[threadIdx.x] = posBodies[tid];
            blockPositions[threadIdx.x +   BLOCK_SIZE] = posBodies[tid +   NUMBER_OF_PARTICLES];
            blockPositions[threadIdx.x + 2*BLOCK_SIZE] = 0.0;//posBodies[tid + 2*NUMBER_OF_PARTICLES];
        }else{
            blockPositions[threadIdx.x] = 2.0*dboxSide;
            blockPositions[threadIdx.x +   BLOCK_SIZE] = -2.0*dboxSide;
            blockPositions[threadIdx.x + 2*BLOCK_SIZE] = 0.0;//posBodies[tid + 2*NUMBER_OF_PARTICLES];
        }
        __syncthreads();

        acc = calculateForces(piBody, acc, threadIdx.x);

        __syncthreads();
    }

    if(gtid < NUMBER_OF_PARTICLES){
        accBodies[gtid] = acc.x;
        accBodies[gtid + NUMBER_OF_PARTICLES] = acc.y;
        //accBodies[gtid + 2*NUMBER_OF_PARTICLES] = acc.z;
    }

    return;

}

__device__ void integrationStep(double* Y, double* dY, double factor){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    if( gtid < NUMBER_OF_PARTICLES ){
        Y[gtid] += factor * dt * dY[gtid];
        Y[gtid + NUMBER_OF_PARTICLES] += factor * dt * dY[gtid + NUMBER_OF_PARTICLES];
        //Y[gtid + 2*NUMBER_OF_PARTICLES] += factor * dt * dY[gtid + 2*NUMBER_OF_PARTICLES];
    }
    return;
}

__device__ void fitInsideBox(double* posBodies, double* velBodies){
    int i_part = blockIdx.x * blockDim.x + threadIdx.x; // Particula indice
    int s = 0;
    if (i_part >= NUMBER_OF_PARTICLES) return;
    

    while (s < 25 && ((posBodies[i_part] < -0.5*dboxSide || posBodies[i_part] > 0.5*dboxSide)
        || (posBodies[i_part + NUMBER_OF_PARTICLES] < 0.0 || posBodies[i_part + NUMBER_OF_PARTICLES] > dboxSide) ) ) {
        if (posBodies[i_part] < -0.5*dboxSide) {
            velBodies[i_part] *= -ddamp; /// x
            velBodies[i_part + NUMBER_OF_PARTICLES] *= ddamp; /// y
            posBodies[i_part] = -posBodies[i_part] - dboxSide;
        }        

        // Modify if is under the cuadrant
        if (posBodies[i_part + NUMBER_OF_PARTICLES] < 0) {
            velBodies[i_part] *= ddamp; //velocidad x
            velBodies[i_part + NUMBER_OF_PARTICLES] *= -ddamp; //velocidad y
            posBodies[i_part + NUMBER_OF_PARTICLES] = -posBodies[i_part + NUMBER_OF_PARTICLES];
        }
        
        if (posBodies[i_part] > 0.5*dboxSide) {
            velBodies[i_part] *= -ddamp; // x
            velBodies[i_part + NUMBER_OF_PARTICLES] *= ddamp; // y
            posBodies[i_part] = 2 * 0.5*dboxSide - posBodies[i_part];
        }

        //Choca con la tapa
        if (posBodies[i_part + NUMBER_OF_PARTICLES] > dboxSide) {
            velBodies[i_part]        =  velBodies[i_part]*ddamp + d_vtapa;//Velocidad en X
            velBodies[i_part + NUMBER_OF_PARTICLES] = -velBodies[i_part + NUMBER_OF_PARTICLES]*ddamp; //Velocidad en Y
            posBodies[i_part + NUMBER_OF_PARTICLES] = 2.0 * dboxSide - posBodies[i_part + NUMBER_OF_PARTICLES];
        }

        s++;
    }
    
    return;
}

void storeInformation(std::string filename, int dimSpace, int ){
    return;
}

int main(int argc, char* argv[]){
    
    return 0;
}
