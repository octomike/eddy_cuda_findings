//Example 1. Application Using C and cuBLAS: 1-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define K 3 //1196032
#define N 3 //54
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float *devPtrA, *devPtrC;
    float *a=0, *c=0;
    float alpha = 1.0, beta = 1.0;

    a = (float *)malloc (N * K * sizeof (*a));
    c = (float *)malloc (N * N * sizeof (*c));
    if (!a || !c) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= K; i++) {
            a[IDX2F(i,j,K)] = 1.0f;
        }
    }
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= N; i++) {
            c[IDX2F(i,j,N)] = 1.0f;
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, N*K*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for A");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc ((void**)&devPtrC, N*N*sizeof(*c));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed for C");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (K, N, sizeof(*a), a, K, devPtrA, K);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (N, N, sizeof(*c), c, N, devPtrC, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSsyrk (handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, N, K, &alpha, devPtrA, K, &beta, devPtrC, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("%s: %s\n", "cublasSsyrk failed", _cudaGetErrorEnum(stat));
        cudaFree (devPtrA);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasGetMatrix (N, N, sizeof(*c), devPtrC, N, c, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cudaFree (devPtrC);
    cublasDestroy(handle);
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= j; i++) {
            printf ("%7.0f", c[IDX2F(i,j,N)]);
        }
        printf ("\n");
    }
    free(a);
    free(c);
    return EXIT_SUCCESS;
}
