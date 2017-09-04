// @file nnmaxout.hpp
// @brief Maxout unit block
// @author Jia-Ren Chang
// @author Samuel Albanie

/*
Copyright (C) 2017 Jia-Ren Chang and Samuel Albanie.
Licensed under The MIT License [see LICENSE.md for details]
*/

#ifndef __vl__nnmaxout__
#define __vl__nnmaxout__

#include <bits/data.hpp>
#include <stdio.h>

namespace vl {

    vl::ErrorCode
    nnmaxout_forward(vl::Context& context,
                     vl::Tensor output,
                     vl::Tensor data,
                     int numUnits, int numPieces) ;

    vl::ErrorCode
    nnmaxout_backward(vl::Context& context,
                      vl::Tensor derData,
                      vl::Tensor data,
                      vl::Tensor derOutput,
                      int numUnits, int numPieces);
}

#endif /* defined(__vl__nnmaxout__) */
