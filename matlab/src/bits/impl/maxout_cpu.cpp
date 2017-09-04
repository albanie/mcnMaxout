// @file maxout_cpu.cpp
// @brief Maxout block 
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#include <bits/data.hpp>
#include <bits/mexutils.h>
#include "maxout.hpp"

#include <limits>
#include <algorithm>
#include <cmath>
// ------------------------------------------------------------------- //                                                             Helpers
// -------------------------------------------------------------------

template <typename type>
struct acc_max {
  inline acc_max(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                                   maxout_forward */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

   template<typename T>
   struct maxout<vl::VLDT_CPU,T>
   {

   static vl::ErrorCode
   forward(float* pooled,
           float const* data,
           size_t height, size_t width, size_t depth,
           size_t numUnits, size_t numPieces)
   {
     vlmxError(VLMXE_IllegalArgument, "CPU mode not implemented") ;
   }

/* ---------------------------------------------------------------- */
/*                                                  maxout_backward */
/* ---------------------------------------------------------------- */

   static vl::ErrorCode
   backward(float* derData,
            float const* data,
            float const* derPooled,
            size_t height, size_t width, 
            size_t depth, size_t numUnits, 
            size_t numPieces)
    {
     vlmxError(VLMXE_IllegalArgument, "CPU mode not implemented") ;
    }
  } ;
} } // namespace vl::impl

template struct vl::impl::maxout<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::maxout<vl::VLDT_CPU, double> ;
#endif
