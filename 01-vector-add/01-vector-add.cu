#include <stdio.h>

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  
  for(int i = idx; i < N; i+=stride)
  {
    result[i] = a[i] + b[i];
  }
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

#define CUDA_STATUS_CHECK(_err) { if (_err != cudaSuccess) printf("Cuda error: %s\n", cudaGetErrorString(_err)); }

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);
  cudaError_t err;
  int threads_per_block = 256;
  int num_blocks = 40;
  
  float *a;
  float *b;
  float *c;

  err = cudaMallocManaged(&a, size);
  CUDA_STATUS_CHECK(err);
  err = cudaMallocManaged(&b, size);
  CUDA_STATUS_CHECK(err);
  err = cudaMallocManaged(&c, size);
  CUDA_STATUS_CHECK(err);
  
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto<<<num_blocks,threads_per_block>>>(c, a, b, N);
  err = cudaGetLastError();
  CUDA_STATUS_CHECK(err);
  
  err = cudaDeviceSynchronize();
  CUDA_STATUS_CHECK(err);
  
  checkElementsAre(7, c, N);

  err = cudaFree(a);
  CUDA_STATUS_CHECK(err);
  err = cudaFree(b);
  CUDA_STATUS_CHECK(err);
  err = cudaFree(c);
  CUDA_STATUS_CHECK(err);
}
