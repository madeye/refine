#include <float.h>
#include <iostream>
#include <cutil_inline.h>
#include "ipoint.h"

using namespace std;

#define TNUM 16
#define DIM 64

float* des1;
float* d_des1;
IpVec* ipt;

//! Populate IpPairVec with matched ipts
__device__ int matchnum;

__global__ void match_kernel(float *des1, float *des2, int size1, int size2)
{

    float dist, d1, d2;
    __shared__ float des[DIM * TNUM];

    d1 = d2 = FLT_MAX;
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint tid = threadIdx.x;

    if (index >= size1)
        return;

    for (int i = 0; i < DIM; i++)
    {
        des[tid + i*TNUM] = des1[i + index*DIM]; // avoid bank conflict
    }

    for (uint j = 0; j < size2; j++)
    {
        float sum=0.f;

        for (uint i = 0; i < DIM; i++)
        {
            float v1 = des[tid + i*TNUM];
            /*float v1 = des1[index*DIM + i];*/
            float v2 = des2[j*DIM + i];
            sum += (v1 - v2) * (v1 - v2);
        }
        dist = sum;

        if (dist < d1) // if this feature matches better than current best
        {
            d2 = d1;
            d1 = dist;
        }
        else if (dist < d2) // this feature matches better than second best
        {
            d2 = dist;
        }
    }

    // If match has a d1:d2 ratio < 0.65 ipoints are a match
    if (d1 / d2 < 0.65f)
    {
        // add the matchnum
        atomicAdd(&matchnum, 1);
        /*matchnum++;*/
    }

}


float get_matches_gpu(IpVec &ipts1, IpVec &ipts2, int &pairs)
{

    pairs = 0;
    const int size1 = ipts1.size();
    const int size2 = ipts2.size();

    if (ipt != &ipts1) {

        free(des1);
        CUDA_SAFE_CALL(cudaFree(d_des1));

        des1 = (float*) malloc (sizeof(float) * DIM * size1);

        for(unsigned int i = 0; i < size1; i++) 
        {
            for (unsigned int j = 0; j < DIM; j++)
            {
                des1[i*DIM + j] = ipts1[i].descriptor[j];
            }
        }

        CUDA_SAFE_CALL(cudaMalloc((void **) &d_des1, size1 * DIM * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_des1, des1,
                    size1 * DIM * sizeof(float), cudaMemcpyHostToDevice));

        ipt = &ipts1;
    }

    float* des2 = (float*) malloc (sizeof(float) * DIM * size2);

    for(unsigned int i = 0; i < size2; i++) 
    {
        for (unsigned int j = 0; j < DIM; j++)
        {
            des2[i*DIM + j] = ipts2[i].descriptor[j];
        }
    }

    float *d_des2;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_des2, size2 * DIM * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_des2, des2,
                size2 * DIM * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(matchnum, &pairs, sizeof(int)));

    // GPU Timer
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    match_kernel<<< (size1 + TNUM - 1) / TNUM, TNUM >>>(d_des1, d_des2, size1, size2);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&pairs, matchnum, sizeof(int)));

    /*cout << pairs << endl;*/

    free(des2);
    CUDA_SAFE_CALL(cudaFree(d_des2));

    return time;

}
