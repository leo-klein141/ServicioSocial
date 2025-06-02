
// Simulacion para el vapor de agua
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <math.h>

#include <random>

#include  <cuda_runtime.h>

////////////////// Constantes para el codigo
const double h = 0.001;
const double boxSide = 183.6L;
const double cuadrantLength = boxSide * 0.5;
const int totalSteps = 1000000;
const int frameIterSep = 10;
const double D = 3.39L; //3.0 * 2.725L;//3.39L;
const double G = 160330.0L * 818.9L;
const double H = 160330.0L;
const int p = 13;
const int q = 7;
const double damp = 0.75L;
double v_tapa = -12.0L;
const double vp = 6.23L;
const double g = 0;
const double Reps = 2.75;

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

const int NUM_THREADS = 32;
const int NUM_BLOCKS = (totalParticles + NUM_THREADS - 1) / NUM_THREADS;

////////////////////////

bool cudaCheckErrors(std::string action) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Lastima, error en %s\n\tCon error: '%s'\n", action.c_str(), cudaGetErrorString(e));
        return true;
    }

    return false;
}

////////////////////////////////////
//////   ---------   Functions

__constant__ int d_tp; //constante que guarda el total de partículas

__constant__ double d_R; //constante que guarda el radio de interaccion
__constant__ double d_minR; //constante que guarda el radio dónde la aceleracion
//se va alv, y puede causar problemas con salirse de los margenes
__constant__ double d_G;
__constant__ double d_H;
__constant__ int blockThreadSize;
__constant__ int d_p;
__constant__ int d_q;

__device__ double2 calculateBlockInteractions(double2 posBody, double2 currAcc, int shift) {
    /**
        Función auxiliar que nos regresa la aceleración para los cuerpos que están
        dentro de nuestro bloque de threads.
    **/
    extern __shared__ double posOtherParticles[];
    double sumAX = 0; double sumAY = 0;
    double xj,yj;
    double ax, ay;
    double force;
    double r;
    double a;
    double distSqr;
    int p;
    int j;

    for (int i = 0; i < blockThreadSize; i++) {
        //Aquí haremos la parte de caclular la fuerza generada por la interacción
        //Entre cada una de las particulas del bloque

        j = (shift + i) % blockThreadSize;
        xj = posOtherParticles[j];
        yj = posOtherParticles[j + blockThreadSize];
        ax = 0;
        ay = 0;
        r = 1.0L;
        force = 0.0L;

        distSqr = (xj - posBody.x) * (xj - posBody.x) + (yj - posBody.y) * (yj - posBody.y);
        if (distSqr > d_R * d_R || distSqr < 1e-10) {
            continue;
        }
        ax = (xj - posBody.x) / sqrt(distSqr);
        ay = (yj - posBody.y) / sqrt(distSqr);

        if (distSqr < d_minR*d_minR) {
            distSqr = d_minR*d_minR;
        }

        for (p = 0; p < d_q ; p++) {
            r *= distSqr;
        }

        force += d_H / sqrt(r);

        for (; p < d_p; p++) {
            r *= distSqr;
        }

        force -= d_G / sqrt(r);

        ax *= force;
        ay *= force;

        sumAX += ax;
        sumAY += ay;
    }

    currAcc = make_double2(currAcc.x + sumAX, currAcc.y + sumAY);

    return currAcc;
}

__global__ void force(double* X_, double* A_) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x; // indice particula i
    extern __shared__ double posOtherParticles[];
    //Ahora tenemos la cantidad total de particulas 
    // Y queremos hacer que una contra otra se golpee o interaccione
    //Ahora dividimos el trabajo sobre las particulas
    //pero antes copiamos la información del cuerpo que queremos obtener.

    double2 myPosition = make_double2(1000.0L, 1000.0L);
    double2 acc = make_double2(0.0L, 0.0L);
    int i; //Nos ayudará en el for, nos dice cuantas particulas hemos comparado contra
    int tile; //Nos dice la cantidad de bloques de particulas que hemos procesado.
    int j;      //Indice de las otras particulas

    if (gtid < d_tp) {
        myPosition = make_double2(X_[gtid], X_[gtid + d_tp]);
    }

    for (i = 0, tile = 0; i < d_tp; i += blockThreadSize, tile++) {
        j = tile * blockDim.x + threadIdx.x;
        if (j < d_tp) {
            posOtherParticles[threadIdx.x] = X_[j];
            posOtherParticles[threadIdx.x + blockThreadSize] = X_[j + d_tp];
            //posOtherParticles[threadIdx.x + 2*blockThreadSize] = X_[j + 2*d_tp];
        } else {
            posOtherParticles[threadIdx.x] = 1000.0;
            posOtherParticles[threadIdx.x + blockThreadSize] = 1000.0;
            //posOtherParticles[threadIdx.x + 2*blockThreadSize] = 1000.0;
        }
        __syncthreads();

        //Ahora a calcular la información de la aceleración

        acc = calculateBlockInteractions(myPosition, acc, threadIdx.x);

        __syncthreads();

    }

    double2 nowAcc = acc;

    if (gtid < d_tp) {
        A_[gtid] = nowAcc.x;
        A_[gtid + d_tp] = nowAcc.y;
    }

    return;
}

__constant__ double d_lsquare;
__constant__ double d_hsquare;
__constant__ double d_damp;
__constant__ double d_vtapa;
__global__ void notScapingBox(double* X_, double* V_) {
    //Solo queremos comprobar que no se va a salir de nuestro cuadrado.!!!!
    int i_part = blockIdx.x * blockDim.x + threadIdx.x; // Particula indice

    if (i_part >= d_tp) return;
    int s = 0;

    while (++s <= 25 && ((X_[i_part] < -d_hsquare || X_[i_part] > d_hsquare)
        || (X_[i_part + d_tp] < 0.0 || X_[i_part + d_tp] > d_lsquare) ) ) {
        if (X_[i_part] < -d_hsquare) {
            V_[i_part] *= -d_damp; /// x
            V_[i_part + d_tp] *= d_damp; /// y
            X_[i_part] = -X_[i_part] - 2.0 * d_hsquare;
        }

        if (X_[i_part] > d_hsquare) {
            V_[i_part] *= -d_damp; // x
            V_[i_part + d_tp] *= d_damp; // y
            X_[i_part] = 2 * d_hsquare - X_[i_part];
        }

        // Modify if is under the cuadrant
        if (X_[i_part + d_tp] < 0) {
            V_[i_part] *= d_damp; //velocidad x
            V_[i_part + d_tp] *= -d_damp; //velocidad y
            X_[i_part + d_tp] = -X_[i_part + d_tp];
        }

        //Choca con la tapa
        if (X_[i_part + d_tp] > d_lsquare) {
            V_[i_part] = 0.0;
            V_[i_part + d_tp] += d_vtapa;
            X_[i_part + d_tp] = 2 * d_lsquare - X_[i_part + d_tp];
        }
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

    cudaSetDevice(0);
    if (cudaCheckErrors("No se inicia el Device :c")) return -1;

    double* X;
    double* V;
    double* A;
    //double* masses;
    //cudaMallocManaged(&masses, totalParticles*sizeof(double));
    cudaMalloc(&X, 2 * totalParticles * sizeof(double));
    cudaMalloc(&V, 2 * totalParticles * sizeof(double));
    cudaMalloc(&A, 2 * totalParticles * sizeof(double));
    if (cudaCheckErrors("Error después de alojar memoria.\n")) return -1;
    cudaDeviceSynchronize();
    if (cudaCheckErrors("No pudo sincronizar al inicio del código!\n")) return -1;

    //// Aqui guardaremos la información para cierta cantidad de frames
    double frameX[2][2 * totalParticles];
    double frameV[2][2 * totalParticles];

    /// Ahora inicializaremos la información de nuestra info en la tarjeta.
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
    Y0[row1_part - 1] = cuadrantLength - 1E-6;
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

    cudaMemcpy(X, Y0, 2 * totalParticles * sizeof(double), cudaMemcpyHostToDevice);
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
    cudaMemcpy(V, xrand, totalParticles * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaCheckErrors("Error despues de intentar copiar las velocidades de x en V")) return -1;
    //cudaMemcpy(yrand,Y0+4*totalParticles,totalParticles*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V + totalParticles, yrand, totalParticles * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaCheckErrors("Error despues de intentar copiar las velocidades de y en V")) return -1;

    for (int i = 0; i < totalParticles; i++) {
        frameX[0][i] = Y0[i];
        frameX[0][i + totalParticles] = Y0[i + totalParticles];
        frameV[0][i] = xrand[i];
        frameV[0][i + totalParticles] = yrand[i];
    }

    free(xrand); free(yrand); free(magxy); free(Y0);

    /**
    *********************

        Ahora seguiremos con poner todas las constantes que declaramos como variables globales para nuestros kernels

    ********************
    */

    cudaMemcpyToSymbol(d_tp, &totalParticles, sizeof(int));
    cudaMemcpyToSymbol(d_damp, &damp, sizeof(double));
    cudaMemcpyToSymbol(d_G, &G, sizeof(double));
    cudaMemcpyToSymbol(d_H, &H, sizeof(double));
    cudaMemcpyToSymbol(d_hsquare, &cuadrantLength, sizeof(double));
    cudaMemcpyToSymbol(d_lsquare, &boxSide, sizeof(double));
    cudaMemcpyToSymbol(d_timeStep, &h, sizeof(double));
    cudaMemcpyToSymbol(d_minR, &Reps, sizeof(double));
    cudaMemcpyToSymbol(d_R, &D, sizeof(double));
    cudaMemcpyToSymbol(d_vtapa, &v_tapa, sizeof(double));
    cudaMemcpyToSymbol(d_p, &p, sizeof(int));
    cudaMemcpyToSymbol(d_q, &q, sizeof(int));
    cudaMemcpyToSymbol(blockThreadSize, &NUM_THREADS, sizeof(int));
    
    if(cudaCheckErrors("No se pudo copiar las constantes a la memoria VRAM!\n")) return -1;

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
    //fillWith0 <<<NUM_BLOCKS , NUM_THREADS >>> (A, totalParticles);
    //if (cudaCheckErrors("No Pudimos establecer la aceleración en 0")) return -1;
    //cudaDeviceSynchronize();
    //if (cudaCheckErrors("Error de syn despues de llenar 0 la A!\n")) return -1;

    //Iterations of the system
    //dim3 tForce(32, 32);
    //dim3 bForce((totalParticles + 31) / 32, (totalParticles + 31) / 32);


    //Primero calculamos la primer fuerza.
    size_t sharedMemSize = 4 * NUM_THREADS * sizeof(double);
    force<<<NUM_BLOCKS, NUM_THREADS, sharedMemSize>>>(X, A);
    if (cudaCheckErrors("No se pudo llamar a la fuerza en su primera llamada\n")) return -1;

    cudaDeviceSynchronize();
    if (cudaCheckErrors("No Sincroniza :(, after de primera fuerza\n")) return -1;

    double vxmean, vymean;

    ///Ahora hacemos el for loop
    int countSteps = 0;
    for (int step = 0; step < totalSteps; step++) {
        //En este punto es dónde tenemos que ver si tenemos que añadir valores a los frames
        if (countSteps == frameIterSep) {
            printf("-----------Iteracion: %d / %d!\n", step, totalSteps);
            
            //Entonces copiamos la informacion actual en nuestros frames
            cudaMemcpy(frameX[1], X, 2 * totalParticles * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(frameV[1], V, 2 * totalParticles * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaCheckErrors("Error al copiar nuestros valores de la tarjeta a donde deberian de ir!!!\n")) return -15;
            countSteps = 0;

            //Ahora copiamos nuestra información en la salida!
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

        halfVelocity<<<NUM_BLOCKS, NUM_THREADS>>>(V, A);
        cudaDeviceSynchronize();
        nextStepP<<<NUM_BLOCKS, NUM_THREADS>>>(X, V);
        cudaDeviceSynchronize();
        if (cudaCheckErrors("Error de sync después de calcular el siguiente paso!\n")) return -10;
        
        /// Hora de checar que ninguna particula se exceda del tamaño de la caja!
        notScapingBox<<<NUM_BLOCKS, NUM_THREADS>>>(X, V);
        cudaDeviceSynchronize();
        if (cudaCheckErrors("Ruptura inesperada, no pudimos sincronizar al calcular el siguiente paso!!!\n")) return -10;
        //Hacemos las aceleracion en 0.
        //fillWith0 << < NUM_BLOCKS, NUM_THREADS >> > (A, totalParticles);
        //if (cudaCheckErrors("No Pudimos establecer la aceleración en 0")) return -1;
        //cudaDeviceSynchronize();
        //if (cudaCheckErrors("Error de sync despues de llenar 0 la A!\n")) return -1;
        force<<<NUM_BLOCKS, NUM_THREADS, sharedMemSize>>>(X, A);
        cudaDeviceSynchronize();
        if (cudaCheckErrors("Error al esperar por obtener la fuerza resultante!\n")) return -3;
        halfVelocity<<<NUM_BLOCKS, NUM_THREADS>>>(V, A);

        countSteps++;
    }

    cudaMemcpy(frameX[1], X, 2 * totalParticles * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(frameV[1], V, 2 * totalParticles * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaCheckErrors("Error al copiar nuestros valores de la tarjeta a donde deberian de ir!!!\n")) return -15;

    ///Ahora liberamos la memoria de la tarjeta grafica

    cudaFree(X);
    cudaFree(V);
    cudaFree(A);

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


    ////Por ultimo elaboramos el README que guarda la información de lo simulado.

    outREADME << "Resumen de la simulacion:\n--------------------------------------------------\n";
    outREADME << "Particulas simuladas : " << totalParticles << "\n";
    outREADME << "Frames simulados: " << (1 + totalSteps / frameIterSep) << "\n";
    outREADME << "Cada frame representa :" << frameIterSep*h << " picosegundos.\n";

    outREADME << "Tiempo elapsado de la simulacion: " << totalSteps * h << " picosegundos!\n";

    outREADME << "Formato de los binarios:\n";
    outREADME << "En las posiciones y velocidades tenemos tuplas de los valores correspondientes (x,y)\n";


    outREADME.close();

    return 0;
}
