// @file nnmaxout.cu
// @brief Maxout block
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Based on the MatConvNet vl_nnpooling.cu function 
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#include "nnmaxout.hpp"
#include "impl/maxout.hpp"

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

/* ---------------------------------------------------------------- */
/*                                                 nnmaxout_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, T) \
status = vl::impl::maxout<deviceType, T>::forward \
((T*)output.getMemory(), \
(T const*)data.getMemory(), \
data.getHeight(), data.getWidth(), \
data.getDepth() * data.getSize(), \
numUnits, numPieces) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnmaxout_forward(vl::Context& context,
                     vl::Tensor output,
                     vl::Tensor data,
                     int numUnits,
                     int numPieces)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnmaxout_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                                nnmaxout_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

#define DISPATCH(deviceType, type) \
status = vl::impl::maxout<deviceType, type>::backward \
((type*)derData.getMemory(), \
(type const*)data.getMemory(), \
(type*)derOutput.getMemory(), \
derData.getHeight(), \
derData.getWidth(), \
derData.getDepth() * derData.getSize(), \
numUnits, numPieces) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return vl::VLE_Unknown ; \
}

vl::ErrorCode
vl::nnmaxout_backward(vl::Context& context,
                      vl::Tensor derData,
                      vl::Tensor data,
                      vl::Tensor derOutput,
                      int numUnits, 
                      int numPieces)
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }

  return context.passError(status, "nnmaxout_backward") ;
}
