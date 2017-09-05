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
    int area = pooledWidth * pooledHeight ;
    int s = pooledIndex % area ;  // spatial offset
    int u  = (pooledIndex / area) % numUnits ; // unit index
    int b = pooledIndex / (area * numUnits) ; // batch index 
    int offset = area * (u * numPieces + b * numUnits * numPieces) ; 
    T bestValue = data[offset + s] ;  
    for (int k = 0; k < numPieces ; ++k) {     
       int idx = area*(k + u*numPieces + b*numUnits*numPieces) + s ;
       bestValue = max(bestValue, data[idx]) ;
    }
    pooled[pooledIndex] = bestValue ;
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
  if (pooledIndex < pooledVolume) {
    int area = pooledWidth * pooledHeight ;
    int s = pooledIndex % area ;  // spatial offset
    int u  = (pooledIndex / area) % numUnits ; // unit index
    int b = pooledIndex / (area * numUnits) ; // batch index 
    int offset = area * (u * numPieces + b * numUnits * numPieces) ; 
    T bestValue = data[offset + s] ;  
    int bestIndex = 0;    
    for (int k = 0; k < numPieces ; ++k) {
      int idx = area*(k + u*numPieces + b*numUnits*numPieces) + s ;
      T value = data[idx];
      if (value > bestValue) {
        bestValue = value ;
        bestIndex = k;    
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
    int dain = offset + s + area * bestIndex ;
    atomicAdd(derData + dain, derPooled[pooledIndex]) ;
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
    forward(T* pooled,
            T const* data,
            size_t height, size_t width, size_t depth,
            size_t numUnits, size_t numPieces)
    {
      int pooledWidth = width;
      int pooledHeight = height;
      int pooledVolume = pooledWidth * pooledHeight * depth / numPieces ;
      maxout_kernel<T><<< vl::divideAndRoundUp(pooledVolume, 
        VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>(pooled, data, 
          pooledHeight, pooledWidth, pooledVolume, numUnits, numPieces) ;
      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

/* ---------------------------------------------------------------- */
/*                                                  maxout_backward */
/* ---------------------------------------------------------------- */ 
    static vl::ErrorCode
    backward(T* derData,
             T const* data,
             T const* derPooled,
             size_t height, size_t width, 
             size_t depth, size_t numUnits, 
             size_t numPieces)
    {
      int pooledWidth = width;
      int pooledHeight = height;
      int pooledVolume = pooledWidth * pooledHeight * depth /  numPieces;
      maxout_backward_kernel<T><<< vl::divideAndRoundUp(pooledVolume, 
        VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>(derData, data, 
        derPooled, pooledHeight, pooledWidth, pooledVolume, numUnits, 
          numPieces) ;
      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::maxout<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::maxout<vl::VLDT_GPU, double> ;
#endif
