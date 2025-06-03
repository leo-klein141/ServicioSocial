//Objetivo cambiar de double a double para mejorar la eficiencia de tiempo
// Simulacion para el vapor de agua

// Este programa se basa principalmente en la forma de calcular la fuerza de interacción de acuerdo
// al articulo de Nvidia GEMS3 https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
// Aquí intentaremos explicar el funcionamiento del programa

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <math.h>
#include <iostream>

#include <random>

#include <hip/hip_runtime.h>

///////////////////////
//      Los siguientes son valores que nos servirán para nuestra forma de correr la simulación
//      En particular los valores de la diferencia de paso en la integración
//      Así cómo del potencial que estaremos trabajando
////////////////// Constantes para el codigo
const double h = 0.0001L; // espacio
const double boxSide = 183.6L;
const double cuadrantLength = boxSide * 0.5L;
const int totalSteps = 250000; //total de iteraciones
const int frameIterSep =    100;
const double D = 3.39L;                  //Distancia máxima de interacción, esta depende de si queremos ver vorticidad o no
const double G = 160330.0L * 818.9L;     //Constante que indicará nuestro potencial de atracción
const double H = 160330.0L;              //Constante que indicará nuestro potencial de repulsión
const int p = 13;                       //Potencia para nuestro potencial de atracción     (-G*r^(-p))
const int q = 7;                        //Potencia para nuestro potencial de repulsión      (H*r^(-q))
const double damp = 0.75L;               //Constante de fricción al golpear una caja
double v_tapa = -12.0L;                  //Velocidad que adquiere la particula al chocar con la tapa
const double vp = 6.23L;                 //Velocidad inicial de las particulas
const double g = 0.0L;                      //Constante de gravedad (no es necesaria a considerar en nuestro problema)
const double Reps = 2.75L;                 //Valor minimo de nuestro radio para evitar velocidades demasiado grandes (overflow)
const double softFactor = 0.001L;

/// ------- Opcional Up to Configuration ---------

const int row1_part = 61;
const double row1_dx = 3.06L;
const int row2_part = 60;
//const double row2_dx = 3.06L;
const double rows_dy = 2.65L;
const int blockParticles = 121;
const int totalParticles = 4235;

/// --------------------------------------

//////// BLOCKS / THREADS CONFIGURATION

const int NUM_BLOCKS = 80;
const int NUM_THREADS = 64;

const int tileBLOCK = 32;

////////////////////////

bool cudaCheckErrors(std::string action) {
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        printf("Lastima, error en %s\n\tCon error: '%s'\n", action.c_str(), hipGetErrorString(e));
        return true;
    }

    return false;
}

////////////////////////////////////
//////   ---------   Functions

__constant__ int d_tp; //constante que guarda el total de part�culas

__constant__ double d_R; //constante que guarda el radio de interaccion
__constant__ double d_minR; //constante que guarda el radio d�nde la aceleracion
//se va alv, y puede causar problemas con salirse de los margenes

__constant__ double d_G;
__constant__ double d_H;
__constant__ int d_p;
__constant__ int d_q;
__constant__ double EPS2; //El valor que nos permitirá mantener suave nuestro potencial
__constant__ double d_lsquare;

__device__ double3 bodyBodyInteraction(double3 bi, double3 bj, double3 ai){
    /**
     * Calcula la fuerza de interacción entre cada uno de los dos cuerpos, y almacena la aceleración acumulada
     * de la particula i, en nuestro caso el bodyBodyInteraction se calcula usando el potencial de 
     * Lennard-Jhones por lo que ocupamos nuestra información de G,H,p,q
     * Consideremos r_ij = bj - bi (vector)
     * Luego || r_ij || = sqrt( dot(r_ij, r_ij) )
     * Pero podemos quedarnos con que distSqr = || r_ij ||*|| r_ij ||
     * 
     * Haciendo ai  =  ( -G + H*sqrt(distSqr^(p-q)) )/sqrt(distSqr^p) * ( r_ij/sqrt(distSqr) )
     * 
     */
    double3 r;
    double upper_potential = 0.0L;
    int i;
    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // Distancia al cuadrado Donde tenemos 3 mul, 3 sums, 1 sqrt = > [ 7 FLOPS ]
    double distR = r.x*r.x + r.y*r.y + r.z*r.z + EPS2; // distR := sqrt( dot(r_ij, r_ij) )
    double distP = 1.0L;
    //int firstStop = d_p - d_q;

    if(abs(distR - EPS2) < 1e-10 || distR > d_R*d_R ){
        return ai;
    }

    //distR = sqrtf(distR);

    //Ahora normalizamos la distancia en cada componente 3 divs [3 FLOPS].
    r.x *= 1.0L/sqrt(distR);
    r.y *= 1.0L/sqrt(distR);
    r.z *= 1.0L/sqrt(distR);

    //Ya que tenemos normalizada la distancia lo siguiente es obtener el valor la distancia (p-q)
    //Pero antes haremos un pequeño ajuste para las cantidades que sean mayores a nuestro radio
    // de interacción

    //Ahora si la distancia es mayor al radio de estudio, entonces tenemos que hacer la fuerza 0, una forma
    //sencilla es simplemente convertir a las componentes del radio a 0.
    if(distR < d_minR*d_minR){
        distR = d_minR*d_minR;
    }
    
    for(i=0 ; i < d_q ; i++){
        distP *= distR;
    }

    upper_potential += d_H/sqrt(distP);

    for(; i < d_p ; i++){
        distP *= distR;
    }

    upper_potential -= d_G/sqrt(distP);

    //Ahora agregamos la aceleracion generada por este potencial (interaccion cuerpo a cuerpo) [3 FLOPS]
    ai.x += upper_potential*r.x;
    ai.y += upper_potential*r.y;
    ai.z += upper_potential*r.z;
    
    
    return ai;
    
}

__constant__ int pBLOCK;
__device__ double3
tile_calculation(double3 myPosition, double3 accel, int pos){
    /**
     * Calcula la aceleración resultante para la particula actual, pero lo hace de la siguiente forma
     * Considera que tiene todo un bloque de particulas que vamos a poder almacenar, en particular un
     * bloque de pTILE particulas (en nuestro caso sería 32, 64, 128 o 264 ), es decir,
     * actualmente queremos calcular la acelaración para myPosition, pero solo vamos a centrarnos para
     * pTILE particulas dentro de esta función.
     */
    int i, j;
    extern __shared__ double3 shPosition[];
    for (i = 0; i < pBLOCK ; i++) {
        j = (i + pos)%pBLOCK;
        accel = bodyBodyInteraction(myPosition, shPosition[j], accel);
    }
    return accel;
}


__global__ void
calculate_forces(double *globalX, double *globalA){
    extern __shared__ double3 shPosition[];
    double3 myPosition;
    int i, tile;
    double3 acc = make_double3(0.0L, 0.0L, 0.0L);
    int gtid = blockIdx.x * blockDim.x + threadIdx.x; //Posicion global del thread en el grid
    if(gtid < d_tp){
        myPosition.x = globalX[gtid];         //Nuestra posicion X en el thread
        myPosition.y = globalX[gtid + d_tp];  //Nuestra posicion Y en el thread
        myPosition.z = 0.0L;                  //Nuestra posicion Z en el thread
    }
  
    for (i = 0, tile = 0; i < d_tp ; i += pBLOCK, tile++) {
        /**
         * El codigo se seguira ejecutando mientras haya cuerpos aún por evaluar dentro
         * de la cantidad de bloques que declaramos, por ejemplo, si escogimos bloques de 32 threads
         * Y nosotros queremos calcular 1250 particulas, entonces veremos que tendríamos int(1250/32) + 1
         * de bloques que realizarán las instrucciones 
         */
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < d_tp){
            shPosition[threadIdx.x].x = globalX[idx]; //Guarda un conjunto de posiciones sobre este bloque
            shPosition[threadIdx.x].y = globalX[idx + d_tp]; //Guarda un conjunto de posiciones sobre este bloque
            shPosition[threadIdx.x].z = 0.0L; //Guarda un conjunto de posiciones sobre este bloque
        }else{
            shPosition[threadIdx.x].x = 4.0*d_lsquare; //Guarda un conjunto de posiciones sobre este bloque
            shPosition[threadIdx.x].y = 4.0*d_lsquare; //Guarda un conjunto de posiciones sobre este bloque
            shPosition[threadIdx.x].z = 0.0L; //Guarda un conjunto de posiciones sobre este bloque
        }
        
        __syncthreads();

        if(gtid < d_tp){ acc = tile_calculation(myPosition, acc, threadIdx.x); }

        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    
    if(gtid < d_tp){
        globalA[gtid]        = acc.x;
        globalA[gtid + d_tp] = acc.y;
    }
}

__global__ void getVelocitiesSquare(double* V_, double* VS){

    int gtid = blockIdx.x*blockDim.x + threadIdx.x;

    VS[gtid] = V_[gtid]*V_[gtid] + V_[gtid + d_tp]*V_[gtid + d_tp];
}


double getAverage(double* deviceArray){
    double res = 0;
    double* hostArray;
    HIP
    for(int i=0 ; i<totalParticles ; i++){
        res += hostArray[i];
    }
    return res/totalParticles;
}



__constant__ double d_hsquare;
__constant__ double d_damp;
__constant__ double d_vtapa;
__global__ void notScapingBox(double* X_, double* V_) {
    //Solo queremos comprobar que no se va a salir de nuestro cuadrado.!!!!
    int i_part = blockIdx.x * blockDim.x + threadIdx.x; // Particula indice
    int s = 0;
    if (i_part >= d_tp) return;
    

    while (s < 25 && ((X_[i_part] < -d_hsquare || X_[i_part] > d_hsquare)
        || (X_[i_part + d_tp] < 0.0 || X_[i_part + d_tp] > d_lsquare) ) ) {
        if (X_[i_part] < -d_hsquare) {
            V_[i_part] *= -d_damp; /// x
            V_[i_part + d_tp] *= d_damp; /// y
            X_[i_part] = -X_[i_part] - 2.0 * d_hsquare;
        }        

        // Modify if is under the cuadrant
        if (X_[i_part + d_tp] < 0) {
            V_[i_part] *= d_damp; //velocidad x
            V_[i_part + d_tp] *= -d_damp; //velocidad y
            X_[i_part + d_tp] = -X_[i_part + d_tp];
        }
        
        if (X_[i_part] > d_hsquare) {
            V_[i_part] *= -d_damp; // x
            V_[i_part + d_tp] *= d_damp; // y
            X_[i_part] = 2 * d_hsquare - X_[i_part];
        }

        //Choca con la tapa
        if (X_[i_part + d_tp] > d_lsquare) {
            V_[i_part]        =  V_[i_part]*d_damp + d_vtapa;//Velocidad en X
            V_[i_part + d_tp] = -V_[i_part + d_tp]*d_damp; //Velocidad en Y
            X_[i_part + d_tp] = 2 * d_lsquare - X_[i_part + d_tp];
        }

        s++;
    }
    return;
}

__constant__ double d_timeStep;

__global__ void halfVelocity(double* V_, double* A_) {
    int i_part = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_part < d_tp) {
        V_[i_part] += 0.5 * d_timeStep * A_[i_part];
        V_[i_part + d_tp] += 0.5 * d_timeStep * A_[i_part + d_tp];
        //V_[i_part + 2*d_tp] += 0.5*d_timeStep*A_[i_part + 2*d_tp];
    }
    return;
}

__global__ void nextStepP(double* X_, double* V_) {
    int i_part = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_part < d_tp) {
        X_[i_part] += d_timeStep * V_[i_part];
        X_[i_part + d_tp] += d_timeStep * V_[i_part + d_tp];
        // X_[i_part+2*d_tp] += d_timeStep*V_[i_part+2*d_tp];
    }
    return;
}

__global__ void fillWith0(double* Arre, int t) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < t) {
        Arre[tid] = 0.0;
        Arre[tid + t] = 0.0;
    }


}

int main(void) {
    ///////////////////////
    //// Initial conditions //////

    printf("Starting...\n");

    hipSetDevice(0);
    if (cudaCheckErrors("No se inicia el Device :c")) return -1;

    double* X;
    double* V;
    double* A;
    double* auxV;

    double hA[totalParticles*2];
    //double* masses;
    //cudaMallocManaged(&masses, totalParticles*sizeof(double));
    hipMalloc(&X, 2 * totalParticles * sizeof(double));
    hipMalloc(&V, 2 * totalParticles * sizeof(double));
    hipMalloc(&A, 2 * totalParticles * sizeof(double));
    hipMalloc(&auxV, 2 * totalParticles * sizeof(double));
    if (cudaCheckErrors("Error despu�s de alojar memoria.\n")) return -1;
    hipDeviceSynchronize();
    if (cudaCheckErrors("No pudo sincronizar al inicio del c�digo!\n")) return -1;

    //// Aqui guardaremos la informaci�n para cierta cantidad de frames
    double frameX[2][2 * totalParticles];
    double frameV[2][2 * totalParticles];

    /// Ahora inicializaremos la informaci�n de nuestra info en la tarjeta.
    //cudaMemset(A, 0.0, 2*totalParticles*sizeof(double));

    double* Y0 = (double*)malloc(2 * totalParticles * sizeof(double));

    // Initial conditions
        // First Position (x,y)
    Y0[0] = -cuadrantLength; // lower left corner particle x
    Y0[totalParticles] = 0.0; // lower left corner particle y
    // Y0[posIdxZ(0)] = ???
    for (int i = 1; i < row1_part; i++) {
        Y0[i] = Y0[i - 1] + row1_dx; // x - coor
        Y0[i + totalParticles] = 0.0; // y - coor
    }
    Y0[row1_part - 1] = cuadrantLength;
    // Second Row type
    Y0[row1_part] = -cuadrantLength + row1_dx * 0.5; // upper left corner particle x
    Y0[row1_part + totalParticles] = rows_dy; // upper left corner particle y
    for (int i = row1_part + 1; i < blockParticles; i++) {
        Y0[i] = Y0[i - 1] + row1_dx; // x - coor
        Y0[i + totalParticles] = rows_dy; // y - coor
    }

    // Filling the info
    for (int i = blockParticles; i < totalParticles; i++) {
        Y0[i] = Y0[i - blockParticles]; // x - coor
        Y0[i + totalParticles] = Y0[i - blockParticles + totalParticles] + 2.0 * rows_dy; // y - coor
    }

    hipMemcpy(X, Y0, 2 * totalParticles * sizeof(double), hipMemcpyHostToDevice);
    if (cudaCheckErrors("Posiciones dentro del cuadro, pero fallo al copiar!\n")) return -1;

    double* xrand = (double*)malloc(totalParticles * sizeof(double));
    double* yrand = (double*)malloc(totalParticles * sizeof(double));
    double* magxy = (double*)malloc(totalParticles * sizeof(double));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);


    for (int i = 0; i < totalParticles; i++) {
        xrand[i] = dis(gen);
        yrand[i] = dis(gen);
        magxy[i] = sqrt(xrand[i] * xrand[i] + yrand[i] * yrand[i]);
        xrand[i] *= vp / magxy[i];
        yrand[i] *= vp / magxy[i];
    }

    // Ahora copiamos las velocidades en nuestra velocidad inicial
    hipMemcpy(V, xrand, totalParticles * sizeof(double), hipMemcpyHostToDevice);
    if (cudaCheckErrors("Error despues de intentar copiar las velocidades de x en V")) return -1;
    //cudaMemcpy(yrand,Y0+4*totalParticles,totalParticles*sizeof(double), cudaMemcpyHostToDevice);
    hipMemcpy(V + totalParticles, yrand, totalParticles * sizeof(double), hipMemcpyHostToDevice);
    if (cudaCheckErrors("Error despues de intentar copiar las velocidades de y en V")) return -1;

    for (int i = 0; i < totalParticles; i++) {
        frameX[0][i] = Y0[i];
        frameX[0][i + totalParticles] = Y0[i + totalParticles];
        frameV[0][i] = xrand[i];
        frameV[0][i + totalParticles] = yrand[i];
    }

    free(xrand); free(yrand); free(magxy); free(Y0);

    /**
    7

    *********************

        Ahora seguiremos con poner todas las constantes que declaramos como variables globales para nuestros kernels

    ********************
    */
    
    hipMemcpyToSymbol(HIP_SYMBOL(d_tp), &totalParticles, sizeof(int));
    hipMemcpyToSymbol(HIP_SYMBOL(d_damp), &damp, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_G), &G, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_H), &H, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_hsquare), &cuadrantLength, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_lsquare), &boxSide, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_timeStep), &h, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_minR), &Reps, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_R), &D, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(d_vtapa), &v_tapa, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(EPS2), &softFactor, sizeof(double));
    hipMemcpyToSymbol(HIP_SYMBOL(pBLOCK), &tileBLOCK, sizeof(int));
    

    ///Ahora declaramos nuestros archivos de salida que vamos a utilizar

    std::ofstream outT("time.bin", std::ofstream::out | std::ofstream::binary);
    std::ofstream outPositions("positions.bin", std::ofstream::out | std::ofstream::binary);
    std::ofstream outVelocities("velocities.bin", std::ofstream::out | std::ofstream::binary);
    std::ofstream outVelocitiesMean("velocitiesMean.bin", std::ofstream::out | std::ofstream::binary);
    std::ofstream outREADME("README.txt");
    //std::ofstream outIdxs("idxs.bin", std::ofstream::out | std::ofstream::binary);
    //std::ofstream outY("par_y.bin", std::ofstream::out | std::ofstream::binary);
    //std::ofstream outZ("par_z.bin", std::ios::out | std::ios::binary);
    
    double currTimeElapsed;
    ///Declaramos el tiempo 0
    currTimeElapsed = 0.0;
    outT.write((char*)&currTimeElapsed, sizeof(double));
    //Ahora guardamos las posiciones en el tiempo 0
    for (int p = 0; p < totalParticles; p++) {
        outPositions.write((char*)&frameX[0][p], sizeof(double));
        outPositions.write((char*)&frameX[0][p + totalParticles], sizeof(double));
        outVelocities.write((char*)&frameV[0][p], sizeof(double));
        outVelocities.write((char*)&frameV[0][p + totalParticles], sizeof(double));
        outVelocitiesMean.write((char*)&frameV[0][p], sizeof(double));
        outVelocitiesMean.write((char*)&frameV[0][p + totalParticles], sizeof(double));
    }

    
    /// Listo
    /*******************************************
    *********************************************
            Precalculos antes de iniciar la simulacion
    ****************************************************
    ************************************************/

    ///Limpiamos las aceleraciones
    //cudaMemset(A, 0.0, 2 * totalParticles * sizeof(double));
    //fillWith0 <<< 192, 64 >>> (A, totalParticles);
    //if (cudaCheckErrors("No Pudimos establecer la aceleraci�n en 0")) return -1;
    //hipDeviceSynchronize();
    //+if (cudaCheckErrors("Error de syn despues de llenar 0 la A!\n")) return -1;

    //Iterations of the system
    dim3 tForce(tileBLOCK);
    dim3 bForce((totalParticles + tileBLOCK - 1)/tileBLOCK);
    size_t shareMemorySize = 2*tileBLOCK*sizeof(double3);

    calculate_forces<<<bForce, tForce, shareMemorySize >>>(X, A);
    hipDeviceSynchronize();
    if (cudaCheckErrors("No Sincroniza :(, after de primera fuerza\n")) return -1;

    double vxmean, vymean;

    hipMemcpy(hA, A, 2*totalParticles*sizeof(double), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    double aimax = 0.0f;
    double aimin = 1e20f;
    double aihost;
    int positionMin = -1;
    int positionMax = -1;
    for(int i=0 ; i<totalParticles ; i++){
        aihost = sqrt(hA[i+totalParticles]*hA[i+totalParticles] + hA[i]*hA[i]);
        if(aihost < 0) std::cout << "Error, encontramos valores negativos para la raíz cuadrada!, posicion : " << i << "\n";
        if(aimax < aihost){
            aimax = aihost;
            positionMax = i;
        }
        if(aimin > aihost){
            aimin = aihost;
            positionMin = i;
        }        
    }

    ///Ahora hacemos el for loop
    int countSteps = 0;
    for (int step = 0; step < totalSteps; step++) {
        //En este punto es d�nde tenemos que ver si tenemos que a�adir valores a los frames
        if (countSteps == frameIterSep) {
            printf("-----------Iteracion: %d / %d!\n", step, totalSteps);
            
            //Entonces copiamos la informacion actual en nuestros frames
            hipMemcpy(frameX[1], X, 2 * totalParticles * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(frameV[1], V, 2 * totalParticles * sizeof(double), hipMemcpyDeviceToHost);
            if (cudaCheckErrors("Error al copiar nuestros valores de la tarjeta a donde deberian de ir!!!\n")) return -15;
            countSteps = 0;

            hipDeviceSynchronize();

            //Ahora copiamos nuestra informaci�n en la salida!
            currTimeElapsed = step * h;
            printf("Tiempo elapsado: %.6f\n\n", currTimeElapsed);
            outT.write((char*)&currTimeElapsed, sizeof(double));
            //Ahora guardamos las posiciones en el tiempo 0
            for (int p = 0; p < totalParticles; p++) {
                outPositions.write((char*)&frameX[1][p], sizeof(double));
                outPositions.write((char*)&frameX[1][p + totalParticles], sizeof(double));
                outVelocities.write((char*)&frameV[1][p], sizeof(double));
                outVelocities.write((char*)&frameV[1][p + totalParticles], sizeof(double));
                
                vxmean = (frameV[1][p] - frameV[0][p]) / (h * frameIterSep);
                vymean = (frameV[1][p+totalParticles] - frameV[0][p+totalParticles]) / (h*frameIterSep);
                
                frameX[0][p] = frameX[1][p];
                frameV[0][p] = frameV[1][p];
                frameX[0][p + totalParticles] = frameX[1][p + totalParticles];
                frameV[0][p + totalParticles] = frameV[1][p + totalParticles];

                outVelocitiesMean.write((char*)&vxmean, sizeof(double));
                outVelocitiesMean.write((char*)&vymean, sizeof(double));
            }
        }

        halfVelocity<<<NUM_BLOCKS,NUM_THREADS>>>(V, A);
        hipDeviceSynchronize();
        nextStepP<<<NUM_BLOCKS, NUM_THREADS>>>(X, V);
        hipDeviceSynchronize();
        if (cudaCheckErrors("Error de sync despu�s de calcular el siguiente paso!\n")) return -10;
        
        /// Hora de checar que ninguna particula se exceda del tama�o de la caja!
        notScapingBox<<<NUM_BLOCKS, NUM_THREADS>>>(X, V);
        if (cudaCheckErrors("Ruptura inesperada, no pudimos sincronizar al calcular el siguiente paso!!!\n")) return -10;
        //Hacemos las aceleracion en 0.
        
        if (cudaCheckErrors("Error de syn despues de llenar 0 la A!\n")) return -1;
        calculate_forces<<<bForce, tForce, shareMemorySize >>>(X, A);
        hipDeviceSynchronize();
        if (cudaCheckErrors("Error al esperar por obtener la fuerza resultante!\n")) return -3;
        halfVelocity<<< NUM_BLOCKS, NUM_THREADS>>>(V, A);
        hipDeviceSynchronize();
        if (cudaCheckErrors("Error al esperar por obtener la fuerza resultante!\n")) return -3;

        countSteps++;
    }

    hipMemcpy(frameX[1], X, 2 * totalParticles * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(frameV[1], V, 2 * totalParticles * sizeof(double), hipMemcpyDeviceToHost);
    if (cudaCheckErrors("Error al copiar nuestros valores de la tarjeta a donde deberian de ir!!!\n")) return -15;

    ///Ahora liberamos la memoria de la tarjeta grafica

    hipFree(X);
    hipFree(V);
    hipFree(A);

    //Terminamos de copiar nuestra informacion!!!
    currTimeElapsed = totalSteps * h;
    outT.write((char*)&currTimeElapsed, sizeof(double));
    //Ahora guardamos las posiciones en el tiempo 0
    for (int p = 0; p < totalParticles; p++) {
        outPositions.write((char*)&frameX[1][p], sizeof(double));
        outPositions.write((char*)&frameX[1][p + totalParticles], sizeof(double));
        outVelocities.write((char*)&frameV[1][p], sizeof(double));
        outVelocities.write((char*)&frameV[1][p + totalParticles], sizeof(double));

        vxmean = (frameV[1][p] - frameV[0][p]) / (h * frameIterSep);
        vymean = (frameV[1][p + totalParticles] - frameV[0][p + totalParticles]) / (h * frameIterSep);
        
        outVelocitiesMean.write((char*)&vxmean, sizeof(double));
        outVelocitiesMean.write((char*)&vymean, sizeof(double));
    }

    outT.close();
    outPositions.close();
    outVelocities.close();
    outVelocitiesMean.close();


    ////Por ultimo elaboramos el README que guarda la informaci�n de lo simulado.

    outREADME << "Resumen de la simulacion:\n--------------------------------------------------\n";
    outREADME << "Particulas simuladas : " << totalParticles << "\n";
    outREADME << "Tiempo elapsado de la simulacion: " << totalSteps * h << "picosegundos!\n";

    outREADME << "Formato de los binarios:\n";
    outREADME << "En las posiciones y velocidades tenemos tuplas de los valores correspondientes (x,y)\n";

    outREADME << "La acelaracion maxima y minima en el momento inicial!\n";
    outREADME << "Valor minimo :" << aimin << ",  posicion " << positionMin << "\n";
    outREADME << "Valor maximo :" << aimax << ",  posicion " << positionMax << "\n";


    outREADME.close();

    return 0;
}
