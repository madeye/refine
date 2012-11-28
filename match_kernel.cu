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
__device__ int matchnums[1024] = {0};

__global__ void match_kernel_new(float *orig, float *comp, 
        int orig_size, int vec_offset, int vec_size, int vec_id)
{

    float dist, d1, d2;
    __shared__ float des[DIM * TNUM];

    d1 = d2 = FLT_MAX;
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint tid = threadIdx.x;

    if (index >= orig_size)
        return;

#pragma unroll
    for (int i = 0; i < DIM; i++)
    {
        des[tid + i*TNUM] = orig[i + index*DIM]; // avoid bank conflict
    }

    for (uint j = 0; j < vec_size; j++)
    {
        float sum=0.f;

        for (uint i = 0; i < DIM; i++)
        {
            float v1 = des[tid + i*TNUM];
            /*float v1 = des1[index*DIM + i];*/
            float v2 = comp[vec_offset*DIM + j*DIM + i];
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
        atomicAdd(&(matchnums[vec_id]), 1);
        /*matchnum++;*/
    }

}

__global__ void match_kernel(float *des1, float *des2, int size1, int size2)
{

    float dist, d1, d2;
    __shared__ float des[DIM * TNUM];

    d1 = d2 = FLT_MAX;
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    uint tid = threadIdx.x;

    if (index >= size1)
        return;

#pragma unroll
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

    cudaFuncSetCacheConfig(match_kernel, cudaFuncCachePreferL1);

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

float get_matches_gpu_new(IpVec &orig_vec, vector<IpVec> &top_vec, 
        vector<int> &top_index, Matches &matches) {
    float *orig, *d_orig;
    float *comp, *d_comp;
    int *index, *d_index;
    const int orig_size = orig_vec.size();

    orig = (float*) malloc (sizeof(float) * DIM * orig_size);
    for(unsigned int i = 0; i < orig_size; i++) 
    {
        for (unsigned int j = 0; j < DIM; j++)
        {
            orig[i*DIM + j] = orig_vec[i].descriptor[j];
        }
    }

    CUDA_SAFE_CALL(cudaMalloc((void **) &d_orig, orig_size * DIM * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_orig, orig,
                orig_size * DIM * sizeof(float), cudaMemcpyHostToDevice));

    int comp_size = 0;
    index = (int *) malloc (sizeof(int) * DIM * top_vec.size());
    for (int i = 0; i < top_vec.size(); i++) {
        IpVec vec = top_vec[i];
        comp_size += vec.size();
        index[i] = vec.size();
    }
    comp = (float *) malloc (sizeof(float) * DIM * comp_size);
    for (int i = 0, n = 0; i < top_vec.size(); i++) {
        for (int j = 0; j < index[i]; j++) {
            for (int k = 0; k < DIM; k++, n++) {
                comp[n] = top_vec[i][j].descriptor[k];
            }
        }
    }

    CUDA_SAFE_CALL(cudaMalloc((void **) &d_comp, comp_size * DIM * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_comp, comp,
                comp_size * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_index, top_vec.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_index, index,
                top_vec.size() * sizeof(int), cudaMemcpyHostToDevice));


    // GPU Timer
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int vec_offset = 0;

    for (int i = 0; i < top_vec.size(); i++) {
        const int vec_size = index[i];
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        match_kernel_new<<< (orig_size + TNUM - 1) / TNUM, TNUM, 0, stream >>>(d_orig, d_comp,
                orig_size, vec_offset, vec_size, i);

        vec_offset += vec_size;

    }

    int *match_nums = (int *) malloc (top_vec.size() * sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(match_nums, matchnums, top_vec.size() * sizeof(int)));

    for (int i = 0; i < top_vec.size(); i++) {
        ImageMatch match;
        match.index = top_index[i];
        match.pairs = match_nums[i];
        matches.push_back(match);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    free(orig);
    free(comp);
    free(index);

    CUDA_SAFE_CALL(cudaFree(d_orig));
    CUDA_SAFE_CALL(cudaFree(d_comp));
    CUDA_SAFE_CALL(cudaFree(d_index));

    return time;

}
