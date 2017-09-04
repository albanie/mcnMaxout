// @file maxout.hpp
// @brief Maxout unit block MEX wrapper
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#ifndef VL_MAXOUT_H
#define VL_MAXOUT_H

#include <bits/data.hpp>
#include <cstddef>

namespace vl { namespace impl {

    template<vl::DeviceType dev, typename T>
    struct maxout {

    static vl::ErrorCode
    forward(T* maxed,
            T const* data,
            size_t height, size_t width, size_t depth,
            size_t numunit, size_t numpiece) ;


    static vl::ErrorCode
    backward(T* derData,
             T const* data,
             T const* derPooled,
             size_t height, size_t width, size_t depth,
             size_t numunit, size_t numpiece) ;
    } ;
  } 
}
#endif /* defined(VL_MAXOUT_H) */

//#if ENABLE_GPU
  //template<> vl::ErrorCode
  //maxout_forward<vl::GPU, float>(float* pooled,
                                 //float const* data,
                                 //size_t height, size_t width, size_t depth,
                                 //size_t numunit, size_t numpiece);

  //template<> vl::ErrorCode
  //maxout_backward<vl::GPU, float>(float* derData,
                                  //float const* data,
                                  //float const* derPooled,
                                  //size_t height, size_t width, size_t depth,
                                  //size_t numunit, size_t numpiece);

//#endif

