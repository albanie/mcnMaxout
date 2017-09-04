// @file maxout_gpu.cu
// @brief Maxout block 
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#include "maxout.hpp"
#include <assert.h>
#include <stdio.h>
#include <float.h>
#include <bits/datacu.hpp>
#include <bits/data.hpp>
#include <sm_20_atomic_functions.h>


template<typename T> __global__ void
maxout_kernel(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int numUnits,
 const int numPieces)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    /*int px = pooledIndex ;*/
    /*int py = px / pooledWidth ;*/
    /*int pz = py / pooledHeight ;*/
    /*px %= pooledWidth ;*/
    /*py %= pooledHeight ;*/
    /*data += pz * (pooledWidth * pooledHeight) ;*/

    printf("pooledIndex: %d\n", pooledIndex) ;
    /*printf("pooledVolume: %d\n", pooledVolume) ;*/
    int area = pooledWidth * pooledHeight ;
    int s = pooledIndex % area ;  // spatial offset
    int u  = (pooledIndex / area) % numUnits ; // unit
    int t = pooledIndex / (area * numUnits) ; // trial 
    int offset = area * (u + t * numUnits * numPieces) ; // channel offset
    printf("s: %d\n", s) ;
    printf("u: %d\n", u) ;
    printf("t: %d\n", t) ;
    printf("offset: %d\n", offset) ;
    printf("numPieces: %d\n", numPieces) ;
    T bestValue = data[offset + s] ;  
    for (int k = 0; k < numPieces ; ++k) {     
       int idx = area*(u + k*numUnits + t*numUnits*numPieces) + s ;
       bestValue = max(bestValue, data[idx]) ;
       printf("k: %d, idx: %d, p: %d, best: %g\n", k, idx, pooledIndex, bestValue) ;
    }
    pooled[pooledIndex] = bestValue ;
    printf("storing: %g at pooledIndex: %d \n", pooled[pooledIndex], pooledIndex) ;
  }
}

template<typename T> __global__ void
maxout_backward_kernel(T* derData,
 const T* data,
 const T* derPooled,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int numUnits,
 const int numPieces)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  printf("pooledIndex: %d\n", pooledIndex) ;
  if (pooledIndex < pooledVolume) {
    int thx = pooledIndex % (pooledWidth*pooledHeight);  // which element in pooled
    int ut  = (pooledIndex / (pooledWidth*pooledHeight)) % numUnits; //which unit
    int ntr = pooledIndex / (pooledWidth*pooledHeight*numUnits); // which trial   
    T bestValue = data[thx + pooledWidth*pooledHeight*(ut +  ntr*numUnits*numPieces)];  // GET value in data
    int bestindex = 0;    
    for (int k = 0; k < numPieces ; ++k) {
      //T value = data[thx + pooledWidth*pooledHeight*(ut*numPieces+k)];
      T value = data[thx + pooledWidth*pooledHeight*(ut + k*numUnits +  ntr*numUnits*numPieces)];
      if (value > bestValue) {
        bestValue = value ;
        bestindex = k;    
      }
    }
    /*
     Comment (from original pooling implementation): 
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    int dain = thx + pooledWidth*pooledHeight*(ut + bestindex*numUnits +  ntr*numUnits*numPieces);
    atomicAdd(derData + dain, derPooled[pooledIndex]) ;
    //derData[dain] = derPooled[pooledIndex];
  }
}

/* ---------------------------------------------------------------- */
/*                                                   maxout_forward */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

    template<typename T>
    struct maxout<vl::VLDT_GPU,T>
    {

    static vl::ErrorCode
    forward(float* pooled,
            float const* data,
            size_t height, size_t width, size_t depth,
            size_t numUnits, size_t numPieces)
    {
      int pooledWidth = width;
      int pooledHeight = height;
      int pooledVolume = pooledWidth * pooledHeight * depth / numPieces ;
      maxout_kernel<float>
        <<< vl::divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        (pooled, data, pooledHeight, pooledWidth, pooledVolume, numUnits, numPieces) ;
      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

/* ---------------------------------------------------------------- */
/*                                                  maxout_backward */
/* ---------------------------------------------------------------- */ static vl::ErrorCode
    backward(float* derData,
             float const* data,
             float const* derPooled,
             size_t height, size_t width, 
             size_t depth, size_t numUnits, 
             size_t numPieces)
    {
      int pooledWidth = width;
      int pooledHeight = height;
      int pooledVolume = pooledWidth * pooledHeight * depth /  numPieces;
      maxout_backward_kernel<float>
      <<< vl::divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
          (derData, data, derPooled,
           pooledHeight, pooledWidth, pooledVolume,
            numUnits, numPieces);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
} ;

} } // namespace vl::impl

template struct vl::impl::maxout<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::maxout<vl::VLDT_GPU, double> ;
#endif
